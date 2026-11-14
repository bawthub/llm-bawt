"""TTS idle sleep and wake-on-use through the GPU handoff fence (TASK-1678).

Same world as the TASK-1171 restart tests: the real ops service, handoff
store, adapter and controller over SQLite, with a fake executor and a fake
reconciler health route. No containers, network or GPU.
"""
import json
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_bawt.media.gpu_handoff_store import (
    STOP_OPERATIONS, TTS_SLEEP_OPERATION, TTS_WAKE_OPERATION, GpuHandoffFence, HandoffConflict, _video_jobs,
)
from llm_bawt.ops import JOB_QUEUED, OpsDispatchError
from llm_bawt.ops.executor import ExecutorError
from llm_bawt.service.routes import media_handoff
from llm_bawt.service.routes.service_auth import require_service_token
from tests.test_gpu_tts_restart import TTS_SPEC, instance, job_count, run, set_state, world  # noqa: F401


@pytest.fixture
def power(world):  # noqa: F811
    world.ops_store.create_operation({"slug": TTS_SLEEP_OPERATION, "enabled": True,
                                      "command_script": json.dumps({**TTS_SPEC, "action": "stop"}),
                                      "args_schema_json": {"type": "object", "properties": {},
                                                           "additionalProperties": False}}, actor="t")
    return world


def stopped():
    return [instance(state="exited", health=None, started=datetime.now(UTC) - timedelta(hours=1))]


def sleep(world, request_id="s1"):
    return run(world.controller.tts_power(action="sleep", request_id=request_id))


def wake(world, request_id="w1"):
    return run(world.controller.tts_power(action="wake", request_id=request_id))


def status(world, action, request_id):
    return run(world.controller.tts_power_status(action=action, request_id=request_id))


def finish(world, key, state, *, exit_code=None):
    job = world.ops_store.get_job_by_key(key)
    world.ops_store.mark_terminal(job.id, state=state, exit_code=exit_code)
    return job


def asleep(world):
    sleep(world)
    finish(world, "tts-sleep:s1", "succeeded", exit_code=0)
    world.health.instances = stopped()
    assert status(world, "sleep", "s1")["outcome"] == "released"
    world.health.instances = None


# ── Sleep ───────────────────────────────────────────────────────────────────

def test_sleep_fences_and_dispatches_the_exact_tts_stop(power):
    result = sleep(power)
    state = power.handoff.status()
    assert (state.owner, state.phase, state.tts_restart_key) == ("voice", "tts_sleeping", "tts-sleep:s1")
    assert result["outcome"] == "in_progress" and result["owner"] == "voice"
    job = power.ops_store.get_job_by_key("tts-sleep:s1")
    assert (job.operation_slug, job.caller_actor, job.caller_backend) == (
        TTS_SLEEP_OPERATION, "bawthub-tts-idle", "http-operator")
    assert power.executor.dispatches == [job.id]
    assert power.handoff.admits("tts-sleep:s1", TTS_SLEEP_OPERATION)
    assert not power.handoff.admits("tts-sleep:s1", TTS_WAKE_OPERATION)


def test_proven_stop_leaves_the_gpu_asleep(power):
    sleep(power)
    finish(power, "tts-sleep:s1", "succeeded", exit_code=0)
    power.health.instances = [instance()]  # still running: not yet proven
    held = status(power, "sleep", "s1")
    assert (held["outcome"], held["power"]["reason"]) == ("held", "awaiting_stop_evidence")
    power.health.instances = stopped()
    result = status(power, "sleep", "s1")
    assert (result["outcome"], result["owner"]) == ("released", "asleep")
    state = power.handoff.status()
    assert (state.owner, state.phase, state.tts_restart_key) == ("asleep", "idle", None)
    assert status(power, "sleep", "s1")["outcome"] == "released"  # idempotent


@pytest.mark.parametrize(("state", "code"), [
    ({"owner": "video", "phase": "idle"}, "gpu_not_voice"),
    ({"owner": "unknown", "phase": "idle"}, "gpu_not_voice"),
    ({"owner": "voice", "phase": "transitioning"}, "gpu_not_voice"),
    ({"owner": "voice", "phase": "tts_restarting", "tts_restart_key": "tts-restart:r"}, "tts_power_in_progress"),
    ({"owner": "asleep", "phase": "idle"}, "tts_asleep"),
])
def test_sleep_refuses_outside_an_idle_voice_gpu(power, state, code):
    set_state(power.engine, **state)
    with pytest.raises(HandoffConflict) as exc:
        sleep(power)
    assert exc.value.code == code and job_count(power.engine) == 0


def test_open_video_claims_refuse_sleep(power):
    with power.engine.begin() as conn:
        conn.execute(_video_jobs.insert().values(id="v1", generation=0))
    with pytest.raises(HandoffConflict) as exc:
        sleep(power)
    assert exc.value.code == "active_video_claims" and job_count(power.engine) == 0


def test_never_dispatched_sleep_returns_the_gpu_to_voice(power):
    power.executor.on_preflight = lambda: (_ for _ in ()).throw(ExecutorError("worker image missing"))
    sleep(power)
    finish(power, "tts-sleep:s1", "failed")
    power.health.error = "must not be read"
    result = status(power, "sleep", "s1")
    assert (result["outcome"], result["owner"]) == ("released", "voice")
    assert power.health.urls == []


@pytest.mark.parametrize("state", ["lost", "timed_out", "cancelled", "failed"])
def test_uncertain_sleep_holds_until_the_stop_is_seen(power, state):
    sleep(power)
    finish(power, "tts-sleep:s1", state)
    assert status(power, "sleep", "s1")["outcome"] == "held"
    with pytest.raises(HandoffConflict):
        wake(power)
    power.health.instances = stopped()
    assert status(power, "sleep", "s1")["owner"] == "asleep"


def test_replayed_sleep_reads_the_same_job(power):
    first = sleep(power)
    assert sleep(power)["power"]["job_id"] == first["power"]["job_id"]
    with pytest.raises(HandoffConflict) as exc:
        sleep(power, "s2")
    assert exc.value.code == "tts_power_in_progress" and len(power.executor.dispatches) == 1


# ── Wake ────────────────────────────────────────────────────────────────────

def test_wake_starts_tts_and_returns_the_gpu_to_voice(power):
    asleep(power)
    wake(power)
    state = power.handoff.status()
    assert (state.owner, state.phase) == ("asleep", "tts_waking")
    job = power.ops_store.get_job_by_key("tts-wake:w1")
    assert job.operation_slug == TTS_WAKE_OPERATION and power.executor.dispatches[-1] == job.id
    finish(power, "tts-wake:w1", "succeeded", exit_code=0)
    # Docker health is still "starting": the backend waits for Moshi itself.
    power.health.instances = [instance(health="starting")]
    result = status(power, "wake", "w1")
    assert (result["outcome"], result["owner"]) == ("released", "voice")
    assert power.handoff.status().phase == "idle"


@pytest.mark.parametrize(("instances", "reason"), [
    ([instance(started=datetime.now(UTC) - timedelta(hours=1))], "awaiting_start_evidence"),
    ([instance(restarts=1)], "policy_restart_observed"),
    ([instance(health="unhealthy")], "unhealthy"),
    ([instance(), instance(iid="b")], "ambiguous_instances"),
])
def test_wake_without_fresh_start_evidence_holds(power, instances, reason):
    asleep(power)
    wake(power)
    finish(power, "tts-wake:w1", "succeeded", exit_code=0)
    power.health.instances = instances
    result = status(power, "wake", "w1")
    assert (result["outcome"], result["power"]["reason"], result["owner"]) == ("held", reason, "asleep")


def test_never_dispatched_wake_stays_asleep(power):
    asleep(power)
    power.executor.on_preflight = lambda: (_ for _ in ()).throw(ExecutorError("worker image missing"))
    wake(power)
    finish(power, "tts-wake:w1", "failed")
    assert status(power, "wake", "w1")["owner"] == "asleep"


def test_wake_refuses_unless_asleep(power):
    with pytest.raises(HandoffConflict) as exc:
        wake(power)
    assert exc.value.code == "tts_not_asleep" and job_count(power.engine) == 0


def test_wake_fence_lost_before_dispatch_keeps_the_job_unsubmitted(power):
    asleep(power)
    power.executor.on_preflight = lambda: set_state(power.engine, phase="idle", tts_restart_key=None)
    wake(power)
    assert power.ops_store.get_job_by_key("tts-wake:w1").state == JOB_QUEUED
    assert len(power.executor.dispatches) == 1  # only the sleep ran


# ── Everything else refuses while asleep ────────────────────────────────────

def test_asleep_blocks_restart_handoffs_and_reconciles(power):
    asleep(power)
    with pytest.raises(HandoffConflict) as exc:
        run(power.controller.tts_restart(request_id="r1"))
    assert exc.value.code == "tts_asleep"
    for call in (power.controller.offer(user="u", expected_generation=2, calibration=True),
                 power.controller.offer_restore(user="u", expected_generation=2)):
        with pytest.raises(HandoffConflict) as exc:
            run(call)
        assert exc.value.code == "tts_asleep"


def test_handoffs_refuse_while_a_sleep_holds(power):
    sleep(power)
    with pytest.raises(HandoffConflict) as exc:
        run(power.controller.offer(user="u", expected_generation=1, calibration=True))
    assert exc.value.code == "tts_power_in_progress"
    with pytest.raises(HandoffConflict) as exc:
        run(power.controller.tts_restart(request_id="r1"))
    assert exc.value.code == "tts_power_in_progress"


def test_a_stop_needs_a_stop_or_sleep_fence(power):
    snapshot = power.ops.prepare_invocation(TTS_SLEEP_OPERATION, {})
    with pytest.raises(OpsDispatchError) as exc:
        power.ops.dispatch_job(operation_slug=TTS_SLEEP_OPERATION, args={}, approved_snapshot=snapshot,
                               idempotency_key="k", create_fence=GpuHandoffFence.tts_wake("k"))
    assert exc.value.code == "gpu_handoff_required"
    assert STOP_OPERATIONS[1] == TTS_SLEEP_OPERATION


def test_ids_and_actions_are_validated(power):
    with pytest.raises(HandoffConflict) as exc:
        status(power, "sleep", "../x")
    assert exc.value.code == "args_invalid"
    with pytest.raises(HandoffConflict) as exc:
        status(power, "nap", "s1")
    assert exc.value.code == "args_invalid"
    assert status(power, "sleep", "s1")["outcome"] == "refused"


# ── Routes ──────────────────────────────────────────────────────────────────

@pytest.fixture
def client(power, monkeypatch):
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def factory():
        yield power.controller

    monkeypatch.setattr(media_handoff, "_controller", factory)
    app = FastAPI()
    app.include_router(media_handoff.router)
    app.dependency_overrides[require_service_token] = lambda: None
    with TestClient(app) as test_client:
        yield test_client


def test_routes_sleep_and_answer_status(client, power):
    url = "/v1/media/local-video/handoff/tts-power"
    first = client.post(url, json={"action": "sleep", "request_id": "s1"})
    assert first.status_code == 200 and first.json()["outcome"] == "in_progress"
    assert client.post(url, json={"action": "nap", "request_id": "s1"}).status_code == 422
    other = client.post(url, json={"action": "wake", "request_id": "w1"})
    assert other.status_code == 409 and other.json()["detail"]["code"] == "tts_power_in_progress"
    finish(power, "tts-sleep:s1", "succeeded", exit_code=0)
    power.health.instances = stopped()
    done = client.get(f"{url}/sleep/s1").json()
    assert (done["outcome"], done["owner"]) == ("released", "asleep")
