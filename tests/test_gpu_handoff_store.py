"""GPU handoff consent must remain scoped, one-use and crash-safe."""

import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from llm_bawt.media.gpu_handoff_store import GpuHandoffStore, HandoffConflict
from llm_bawt.ops.seeds import SEEDS


@pytest.fixture
def store():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    result = GpuHandoffStore(engine)
    yield result
    engine.dispose()


def test_scoped_one_use_offer_and_generation_fencing(store):
    assert store.status().owner == "unknown"
    token, offered = store.offer(user="nick", target="video", actions=("park_voice", "stop_stt"), expected_generation=0)
    assert offered.generation == 1
    with pytest.raises(HandoffConflict):
        store.offer(user="nick", target="voice", actions=("start_tts",), expected_generation=0)
    with pytest.raises(HandoffConflict):
        store.confirm(user="someone_else", token=token, expected_generation=1)
    with pytest.raises(HandoffConflict):
        store.confirm(user="nick", token="wrong", expected_generation=1)
    started = store.confirm(user="nick", token=token, expected_generation=1)
    assert (started.phase, started.generation, started.actions) == ("transitioning", 2, ("park_voice", "stop_stt"))
    with pytest.raises(HandoffConflict):
        store.confirm(user="nick", token=token, expected_generation=1)
    with pytest.raises(HandoffConflict):
        store.begin_action(generation=2, action="stop_stt")
    assert store.begin_action(generation=2, action="park_voice").pending_action == "park_voice"
    with pytest.raises(HandoffConflict):
        store.begin_action(generation=2, action="park_voice")
    assert store.record_job(generation=2, action="park_voice", job_id="ops-123").last_job_id == "ops-123"
    with pytest.raises(HandoffConflict):
        store.record_job(generation=2, action="park_voice", job_id="ops-456")
    with pytest.raises(HandoffConflict):
        store.complete(generation=2, target="video")
    store.finish_action(generation=2, action="park_voice")
    store.begin_action(generation=2, action="stop_stt")
    store.finish_action(generation=2, action="stop_stt")
    assert store.complete(generation=2, target="video").owner == "video"


def test_video_claim_requires_owner_and_blocks_switch(store):
    with pytest.raises(HandoffConflict, match="does not belong"):
        store.claim_video("render-1")
    token, offer = store.offer(user="nick", target="video", actions=("verify",), expected_generation=0)
    transition = store.confirm(user="nick", token=token, expected_generation=offer.generation)
    store.begin_action(generation=transition.generation, action="verify")
    store.finish_action(generation=transition.generation, action="verify")
    store.complete(generation=transition.generation, target="video")
    store.claim_video("render-1")
    assert store.active_video_jobs() == 1
    store.claim_video("render-2")
    assert store.active_video_jobs() == 2
    with pytest.raises(HandoffConflict, match="Active video"):
        store.offer(user="nick", target="voice", actions=("unload_wan",), expected_generation=store.status().generation)
    store.release_video("render-1")
    store.release_video("render-2")
    assert store.active_video_jobs() == 0
    token, offer = store.offer(user="nick", target="voice", actions=("unload_wan",), expected_generation=store.status().generation)
    assert offer.target == "voice" and token
    with pytest.raises(HandoffConflict, match="does not belong"):
        store.claim_video("render-2")


def test_orphaned_video_claim_needs_recovery_not_replay(store):
    token, offer = store.offer(user="nick", target="video", actions=("verify",), expected_generation=0)
    transition = store.confirm(user="nick", token=token, expected_generation=offer.generation)
    store.begin_action(generation=transition.generation, action="verify")
    store.finish_action(generation=transition.generation, action="verify")
    store.complete(generation=transition.generation, target="video")
    store.claim_video("lost-render")
    recovered = GpuHandoffStore(store.engine).recover_orphaned_video()
    assert (recovered.owner, recovered.phase) == ("unknown", "recovery_required")
    assert store.recover_orphaned_video() == recovered
    with pytest.raises(HandoffConflict):
        store.claim_video("new-render")
    with pytest.raises(HandoffConflict):
        store.offer(user="nick", target="video", actions=("stop_moshi_tts",),
                    expected_generation=recovered.generation)
    # Restoring voice is the recovery path; its worker reset clears the claims.
    token, offer = store.offer(user="nick", target="voice", actions=("reset_video_worker",),
                               expected_generation=recovered.generation)
    assert offer.target == "voice" and store.active_video_jobs() == 1


def test_video_worker_restart_loses_owner_even_without_claim(store):
    token, offer = store.offer(user="nick", target="video", actions=("verify",), expected_generation=0)
    transition = store.confirm(user="nick", token=token, expected_generation=offer.generation)
    store.begin_action(generation=transition.generation, action="verify")
    store.finish_action(generation=transition.generation, action="verify")
    store.complete(generation=transition.generation, target="video")
    assert store.active_video_jobs() == 0
    recovered = store.recover_orphaned_video(worker_restarted=True)
    assert recovered.owner == "unknown" and recovered.phase == "recovery_required"
    assert "residency" in recovered.last_error
    with pytest.raises(HandoffConflict):
        store.claim_video("after-restart")


def test_stop_requires_receipt_before_advance(store):
    token, offer = store.offer(user="nick", target="video", actions=("stop_moshi_stt",), expected_generation=0)
    transition = store.confirm(user="nick", token=token, expected_generation=offer.generation)
    store.begin_action(generation=transition.generation, action="stop_moshi_stt")
    with pytest.raises(HandoffConflict, match="receipt"):
        store.finish_action(generation=transition.generation, action="stop_moshi_stt")
    store.record_job(generation=transition.generation, action="stop_moshi_stt", job_id="ops-job")
    assert store.finish_action(generation=transition.generation, action="stop_moshi_stt").next_action == 1


def test_new_offer_invalidates_prior_consent(store):
    old, _ = store.offer(user="nick", target="video", actions=("stop_stt",), expected_generation=0)
    new, state = store.offer(user="nick", target="video", actions=("stop_stt", "stop_tts"), expected_generation=1)
    with pytest.raises(HandoffConflict):
        store.confirm(user="nick", token=old, expected_generation=1)
    assert store.confirm(user="nick", token=new, expected_generation=state.generation).phase == "transitioning"


def test_expired_offer_fails_closed(store, monkeypatch):
    import llm_bawt.media.gpu_handoff_store as module
    monkeypatch.setattr(module.time, "time", lambda: 1000)
    token, state = store.offer(user="nick", target="video", actions=("stop_stt",), expected_generation=0, ttl=1)
    monkeypatch.setattr(module.time, "time", lambda: 1001)
    with pytest.raises(HandoffConflict):
        store.confirm(user="nick", token=token, expected_generation=state.generation)


def test_crash_recovery_never_replays_operation(store):
    token, state = store.offer(user="nick", target="video", actions=("stop_stt",), expected_generation=0)
    started = store.confirm(user="nick", token=token, expected_generation=state.generation)
    store.begin_action(generation=started.generation, action="stop_stt")
    store.record_job(generation=started.generation, action="stop_stt", job_id="ops-123")
    recovered = GpuHandoffStore(store.engine).mark_interrupted()
    assert (recovered.owner, recovered.phase, recovered.last_job_id) == ("unknown", "recovery_required", "ops-123")
    assert store.mark_interrupted() == recovered
    with pytest.raises(HandoffConflict):
        store.offer(user="nick", target="video", actions=("stop_stt",), expected_generation=recovered.generation)


def test_moshi_operations_are_exact_and_disabled():
    for action in ("stop", "start"):
        for service in ("stt", "tts"):
            seed = next(row for row in SEEDS if row["slug"] == f"bawthub.{action}-moshi-{service}")
            assert seed["enabled"] is False
            assert json.loads(seed["command_script"]) == {
                "action": action, "compose_project": "bawthub", "compose_service": service,
            }
            assert json.loads(seed["args_schema_json"])["additionalProperties"] is False


def test_invalid_offer_does_not_mutate(store):
    with pytest.raises(ValueError):
        store.offer(user="nick", target="video", actions=(), expected_generation=0)
    assert store.status().generation == 0


def test_state_columns_fit_every_owner_and_target():
    """SQLite ignores varchar lengths; PostgreSQL enforces them (TASK-927)."""
    from llm_bawt.media.gpu_handoff_store import _state

    for name in ("unknown", "voice", "video", "video_calibration", "recovery_required"):
        assert len(name) <= _state.c.owner.type.length
        assert len(name) <= _state.c.offer_target.type.length
