"""Fault boundaries for durable GPU switch reconciliation (no GPU or ops dispatch)."""
import asyncio

import pytest

from llm_bawt.media.gpu_handoff import GpuHandoff, START_OPERATIONS, STOP_OPERATIONS
from llm_bawt.media.gpu_handoff_store import GpuHandoffStore, HandoffConflict
from tests.sqlite_support import fast_sqlite_engine
from tests.test_gpu_handoff_executor import FakeAdapter, confirm, offer


@pytest.fixture
def setup(tmp_path):
    engine = fast_sqlite_engine(tmp_path / "recovery.db", check_same_thread=False)
    store, adapter = GpuHandoffStore(engine), FakeAdapter()
    yield store, adapter, GpuHandoff(store, adapter, poll_interval=0, job_timeout=0.1)
    engine.dispose()


def test_lost_start_response_requires_receipt_and_skips_successful_start(setup):
    store, adapter, controller = setup
    asyncio.run(confirm(controller, asyncio.run(offer(controller))))

    original = adapter.dispatch
    async def lost_response(snapshot, *, user, generation):
        result = await original(snapshot, user=user, generation=generation)
        if snapshot["operation"]["slug"] == START_OPERATIONS[0]:
            raise RuntimeError("response lost after operation succeeded")
        return result

    adapter.dispatch = lost_response

    async def interrupted():
        proposed = await controller.offer_restore(user="nick", expected_generation=store.status().generation)
        generation = await controller.accept_restore(user="nick", token=proposed["token"],
            expected_generation=proposed["generation"], approved=True)
        with pytest.raises(RuntimeError, match="response lost"):
            await controller.execute_restore(user="nick", generation=generation)

    asyncio.run(interrupted())
    fenced = store.status()
    assert fenced.phase == "recovery_required" and fenced.pending_action == "start_moshi_stt"
    assert fenced.last_job_id is None  # The ops store, not the HTTP response, owns the receipt.
    adapter.jobs.pop("stt")
    with pytest.raises(HandoffConflict, match="unverified"):
        asyncio.run(controller.offer_restore(user="nick", expected_generation=fenced.generation))
    assert store.status() == fenced
    # The idempotent ops key resolves the actual successful job; do not replay it.
    adapter.jobs["stt"] = {"id": "stt", "operation": START_OPERATIONS[0], "terminal": True,
                           "state": "succeeded", "exit_code": 0}
    adapter.dispatch = original
    prior_starts = adapter.effects.count(START_OPERATIONS[0])

    async def recover():
        proposed = await controller.offer_restore(user="nick", expected_generation=fenced.generation)
        assert proposed["operations"] == [START_OPERATIONS[1]]
        generation = await controller.accept_restore(user="nick", token=proposed["token"],
            expected_generation=proposed["generation"], approved=True)
        await controller.execute_restore(user="nick", generation=generation)

    asyncio.run(recover())
    assert adapter.effects.count(START_OPERATIONS[0]) == prior_starts
    assert store.status().owner == "voice"


def test_interrupted_drain_keeps_claims_and_refuses_busy_worker_reset(setup):
    store, adapter, controller = setup
    asyncio.run(confirm(controller, asyncio.run(offer(controller))))
    fenced = store.recover_orphaned_video(worker_restarted=True)
    store.resume_video(expected_generation=fenced.generation)
    store.claim_video("ongoing")
    adapter.render_active = True

    async def begin():
        proposed = await controller.offer_restore(user="nick", expected_generation=store.status().generation)
        await controller.accept_restore(user="nick", token=proposed["token"],
            expected_generation=proposed["generation"], approved=True)
    asyncio.run(begin())
    assert store.status().phase == "draining"
    fence = store.mark_interrupted()
    assert fence.phase == "recovery_required" and store.active_video_jobs() == 1
    store.validate_video_claim("ongoing")  # Accepted queue remains runnable after app restart.
    with pytest.raises(HandoffConflict):
        store.claim_video("late")
    with pytest.raises(HandoffConflict, match="still running"):
        asyncio.run(controller.offer_restore(user="nick", expected_generation=fence.generation))
    assert "reset_video" not in adapter.effects


def test_lost_lease_and_active_voice_block_restore_without_effects(setup):
    store, adapter, controller = setup
    asyncio.run(confirm(controller, asyncio.run(offer(controller))))
    effects = list(adapter.effects)
    adapter.voice["park_id"] = "b" * 32
    with pytest.raises(HandoffConflict, match="lease was lost"):
        asyncio.run(controller.offer_restore(user="nick", expected_generation=store.status().generation))
    assert adapter.effects == effects
    adapter.voice["park_id"] = "a" * 32
    adapter.voice["parked"] = False
    with pytest.raises(HandoffConflict, match="lease was lost"):
        asyncio.run(controller.offer_restore(user="nick", expected_generation=store.status().generation))
    adapter.voice["parked"] = True
    adapter.voice["active_gpu_work"] = 1
    with pytest.raises(HandoffConflict, match="Active voice"):
        asyncio.run(controller.offer_restore(user="nick", expected_generation=store.status().generation))
    assert adapter.effects == effects


def test_restore_rejects_stale_gpu_without_reset_or_ops(setup):
    store, adapter, controller = setup
    asyncio.run(confirm(controller, asyncio.run(offer(controller))))
    effects = list(adapter.effects)
    adapter.stale = True
    with pytest.raises(HandoffConflict, match="telemetry is stale"):
        asyncio.run(controller.offer_restore(user="nick", expected_generation=store.status().generation))
    assert adapter.effects == effects


def test_interrupted_stop_with_unknown_receipt_blocks_new_consent(setup):
    store, adapter, controller = setup
    adapter.failure = RuntimeError("ops response lost")
    with pytest.raises(RuntimeError):
        asyncio.run(confirm(controller, asyncio.run(offer(controller))))
    fenced = store.status()
    assert fenced.pending_action == "stop_moshi_stt"
    with pytest.raises(HandoffConflict, match="unverified"):
        asyncio.run(controller.offer_restore(user="nick", expected_generation=fenced.generation))
    assert store.status() == fenced
    assert STOP_OPERATIONS[0] in adapter.effects
