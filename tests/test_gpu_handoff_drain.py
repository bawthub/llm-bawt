"""Consented video drain: queue claims remain valid, but no new claims enter."""
import asyncio

import pytest

from llm_bawt.media.gpu_handoff import GpuHandoff
from llm_bawt.media.gpu_handoff_store import GpuHandoffStore, HandoffConflict
from tests.sqlite_support import fast_sqlite_engine
from tests.test_gpu_handoff_executor import FakeAdapter, confirm, offer


@pytest.fixture
def setup(tmp_path):
    engine = fast_sqlite_engine(tmp_path / "drain.db", check_same_thread=False)
    store, adapter = GpuHandoffStore(engine), FakeAdapter()
    controller = GpuHandoff(store, adapter, poll_interval=0.01, job_timeout=0.1, drain_timeout=0.2)

    async def establish():
        await confirm(controller, await offer(controller))
        # The first calibrated render has finished and validated the profile.
        # Test only ordinary video claims here, not GPU rendering.
        fenced = store.recover_orphaned_video(worker_restarted=True)
        assert fenced.phase == "recovery_required"
        store.resume_video(expected_generation=fenced.generation)

    asyncio.run(establish())
    yield store, adapter, controller
    engine.dispose()


def test_accepted_queue_drains_before_reset_and_start(setup):
    store, adapter, controller = setup
    controller.drain_timeout = 2  # Leave room for SQLite thread scheduling under the full suite.
    store.claim_video("active")
    store.claim_video("queued")
    adapter.render_active = True

    async def run():
        offered = await controller.offer_restore(user="nick", expected_generation=store.status().generation)
        assert not any(e in adapter.effects for e in ("reset_video", *controller_start_ops))
        store.claim_video("accepted-during-offer")
        generation = await controller.accept_restore(user="nick", token=offered["token"],
            expected_generation=offered["generation"], approved=True)
        assert store.status().phase == "draining"
        with pytest.raises(HandoffConflict):
            store.claim_video("arrived-after-confirm")
        task = asyncio.create_task(controller.execute_restore(user="nick", generation=generation))
        await asyncio.sleep(0.02)
        assert store.status().phase == "draining"
        assert "reset_video" not in adapter.effects
        for job in ("active", "queued", "accepted-during-offer"):
            store.validate_video_claim(job)
            store.release_video(job)
        await asyncio.sleep(0.02)  # The last claim released before worker status turns idle.
        assert store.status().phase == "draining" and "reset_video" not in adapter.effects
        adapter.render_active = False
        await task

    controller_start_ops = ("bawthub.start-moshi-stt", "bawthub.start-moshi-tts")
    asyncio.run(run())
    assert store.status().owner == "voice"
    assert adapter.effects[-4:] == ["reset_video", *controller_start_ops, "unpark"]


def test_drain_timeout_preserves_claims_and_requires_recovery(setup):
    store, adapter, controller = setup
    store.claim_video("slow")

    async def run():
        offered = await controller.offer_restore(user="nick", expected_generation=store.status().generation)
        generation = await controller.accept_restore(user="nick", token=offered["token"],
            expected_generation=offered["generation"], approved=True)
        with pytest.raises(HandoffConflict, match="timed out"):
            await controller.execute_restore(user="nick", generation=generation)

    asyncio.run(run())
    assert store.status().phase == "recovery_required"
    assert store.active_video_jobs() == 1
    assert "reset_video" not in adapter.effects


def test_denied_or_duplicate_restore_does_not_drain(setup):
    store, adapter, controller = setup
    store.claim_video("accepted")

    async def run():
        offered = await controller.offer_restore(user="nick", expected_generation=store.status().generation)
        with pytest.raises(HandoffConflict):
            await controller.accept_restore(user="nick", token=offered["token"],
                expected_generation=offered["generation"], approved=False)
        assert store.status().phase == "offered"
        store.claim_video("accepted-after-decline")
        generation = await controller.accept_restore(user="nick", token=offered["token"],
            expected_generation=offered["generation"], approved=True)
        with pytest.raises(HandoffConflict):
            await controller.accept_restore(user="nick", token=offered["token"],
                expected_generation=offered["generation"], approved=True)
        assert store.status().generation == generation

    asyncio.run(run())
    assert "reset_video" not in adapter.effects
