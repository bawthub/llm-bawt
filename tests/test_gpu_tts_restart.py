"""GPU voice guard and the fenced TTS restart (TASK-1171).

No containers, network or GPU: a fake executor and a fake reconciler health
route behind the real ops service, handoff store, adapter and controller.
"""
import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select

from llm_bawt.media.gpu_handoff import GpuHandoff
from llm_bawt.media.gpu_handoff_adapter import LocalHandoffAdapter
from llm_bawt.media.gpu_handoff_store import (
    START_OPERATIONS, TTS_RESTART_OPERATION, GpuHandoffFence, GpuHandoffStore, HandoffConflict, _state,
)
from llm_bawt.ops import JOB_QUEUED, OpsDispatchError, OpsService, OpsStore
from llm_bawt.ops.executor import (
    CONFIG_CHECKED_ACTIONS, ROUTING_ACTION, SUPPORTED_ACTIONS, DispatchResult, Executor, ExecutorError,
    ReconcileResult,
)
from llm_bawt.ops.gpu_guard import UNGATED_ACTIONS, GpuGuardUnverifiable, gpu_voice_target
from llm_bawt.ops.image_deploy import IMAGE_ACTIONS
from llm_bawt.ops.models import OpsJob
from llm_bawt.ops.plan_reconcile import ACTION as PLAN_ACTION
from llm_bawt.service.routes import media_handoff
from llm_bawt.service.routes.ops import DISPATCH_ERROR_STATUS
from llm_bawt.service.routes.service_auth import require_service_token
from tests.sqlite_support import fast_sqlite_engine

TTS_SPEC = {"action": "restart", "compose_project": "bawthub", "compose_service": "tts"}


class Inspector:
    """Executor stand-in for the guard: answers a container's Compose labels."""

    def __init__(self, identity=("other", "web"), error=None):
        self.identity, self.error, self.calls = identity, error, []

    def target_compose_identity(self, spec, args):
        self.calls.append((spec, args))
        if self.error:
            raise ExecutorError(self.error)
        return self.identity


# ── The guard predicate ─────────────────────────────────────────────────────

EVERY_ACTION = sorted(SUPPORTED_ACTIONS | IMAGE_ACTIONS | CONFIG_CHECKED_ACTIONS | {ROUTING_ACTION})
#: Actions whose own spec names the target. A service plan is gated by its bound
#: targets instead (see the plan tests below).
LITERAL_ACTIONS = [action for action in EVERY_ACTION if action != PLAN_ACTION]


@pytest.mark.parametrize("action", LITERAL_ACTIONS)
def test_every_action_but_stop_and_pull_is_gated_for_tts(action):
    spec = {"action": action, "compose_project": "bawthub", "compose_service": "tts",
            **({} if action in ("start", "restart", "stop", "pull") else {"container_name": "bawthub-tts-1"})}
    gated = gpu_voice_target(spec, {}, Inspector(("bawthub", "tts")))
    assert gated == (None if action in ("stop", "pull") else "tts")


def test_a_new_action_is_gated_until_classified():
    spec = {"action": "some_future_action", "compose_project": "bawthub", "compose_service": "stt"}
    assert gpu_voice_target(spec, {}, Inspector()) == "stt"
    assert UNGATED_ACTIONS >= {"stop", "pull"}
    assert not UNGATED_ACTIONS & (IMAGE_ACTIONS | CONFIG_CHECKED_ACTIONS | {ROUTING_ACTION, "start", "restart"})


@pytest.mark.parametrize("action", LITERAL_ACTIONS)
def test_other_services_are_not_gated(action):
    spec = {"action": action, "compose_project": "bawthub", "compose_service": "frontend",
            "container_name": "bawthub-frontend-1"}
    assert gpu_voice_target(spec, {}, Inspector(("bawthub", "frontend")), live=True) is None


def test_named_start_reads_the_live_labels():
    spec = {"action": "start", "container_name": "anything"}
    assert gpu_voice_target(spec, {}, Inspector(("bawthub", "tts"))) == "tts"
    assert gpu_voice_target(spec, {}, Inspector(("bawthub", "web"))) is None


PLAN = {"action": PLAN_ACTION, "targets": ["a", "b"], "records_target": "a", "probes": []}


def _target(service, action="reconcile_app_config"):
    return {"action": action, "compose_project": "bawthub" if service in ("tts", "stt") else "llm-bawt",
            "compose_service": service, "container_name": f"x-{service}"}


def test_a_service_plan_is_gated_by_its_bound_targets():
    # TASK-1476: a plan names no container; it recreates each bound target.
    quiet = [_target("app"), _target("crawl4ai", "reconcile_tagged_config")]
    assert gpu_voice_target(PLAN, {}, Inspector(), plan_targets=quiet) is None
    assert gpu_voice_target(PLAN, {}, Inspector(), plan_targets=[*quiet, _target("tts")]) == "tts"
    inspector = Inspector(("bawthub", "stt"))
    named_only = {"action": "reconcile_tagged_config", "container_name": "renamed"}
    assert gpu_voice_target(PLAN, {}, inspector, live=True, plan_targets=[named_only]) == "stt"


@pytest.mark.parametrize("targets", [None, []])
def test_an_unbound_service_plan_refuses(targets):
    with pytest.raises(GpuGuardUnverifiable, match="not bound"):
        gpu_voice_target(PLAN, {}, Inspector(), plan_targets=targets)


def test_a_service_plan_cannot_hide_a_target_behind_a_nested_plan():
    with pytest.raises(GpuGuardUnverifiable, match="another plan"):
        gpu_voice_target(PLAN, {}, Inspector(), plan_targets=[PLAN])


def test_name_from_arg_is_resolved_before_inspect():
    spec = {"action": "restart", "container_name_from_arg": "target"}
    inspector = Inspector(("bawthub", "stt"))
    assert gpu_voice_target(spec, {"target": "moshi-a"}, inspector) == "stt"
    assert inspector.calls == [(spec, {"target": "moshi-a"})]


def test_label_selector_needs_no_inspect():
    inspector = Inspector(error="no docker")
    spec = {"action": "start", "compose_project": "bawthub", "compose_service_from_arg": "svc"}
    assert gpu_voice_target(spec, {"svc": "tts"}, inspector) == "tts"
    assert inspector.calls == []


def test_unreadable_identity_fails_closed():
    with pytest.raises(GpuGuardUnverifiable):
        gpu_voice_target({"action": "restart", "container_name": "x"}, {}, Inspector(error="gone"))
    with pytest.raises(GpuGuardUnverifiable):
        gpu_voice_target({"action": "deploy_image"}, {}, Inspector())


@pytest.mark.parametrize("identity", [(None, None), ("bawthub", None), (None, "tts"), ("", "tts"), None])
@pytest.mark.parametrize("live", [False, True])
def test_named_container_without_complete_labels_fails_closed(identity, live):
    """Codex r1 B1: a named container whose Compose labels are missing or
    partial could still be a voice service, so it refuses like an unreadable one."""
    with pytest.raises(GpuGuardUnverifiable):
        gpu_voice_target({"action": "restart", "container_name": "bawthub-tts-1"}, {},
                         Inspector(identity), live=live)


def test_partial_literal_identity_reads_the_container():
    spec = {"action": "deploy_image", "container_name": "c", "compose_project": "bawthub"}
    assert gpu_voice_target(spec, {}, Inspector(("bawthub", "tts"))) == "tts"
    with pytest.raises(GpuGuardUnverifiable):
        gpu_voice_target(spec, {}, Inspector((None, None)))


def test_selector_with_an_empty_service_fails_closed():
    spec = {"action": "start", "compose_project": "bawthub", "compose_service_from_arg": "svc"}
    with pytest.raises(GpuGuardUnverifiable):
        gpu_voice_target(spec, {"svc": ""}, Inspector())


def test_live_recheck_catches_a_relabelled_container():
    spec = {"action": "deploy_image", "container_name": "c", "compose_project": "bawthub",
            "compose_service": "web"}
    assert gpu_voice_target(spec, {}, Inspector(("bawthub", "tts"))) is None
    assert gpu_voice_target(spec, {}, Inspector(("bawthub", "tts")), live=True) == "tts"


def test_guard_codes_answer_409():
    assert DISPATCH_ERROR_STATUS["gpu_handoff_required"] == 409
    assert DISPATCH_ERROR_STATUS["gpu_guard_unverifiable"] == 409


# ── Ops service + handoff store on one database ─────────────────────────────

class FakeExecutor(Executor):
    def __init__(self):
        self.dispatches, self.identity, self.on_preflight = [], ("bawthub", "tts"), None

    def kind(self):
        return "docker"

    def available(self):
        return True

    def preflight(self, snapshot):
        if self.on_preflight:
            self.on_preflight()

    def target_compose_identity(self, spec, args):
        return self.identity

    def dispatch(self, **kwargs):
        self.dispatches.append(kwargs["job_id"])
        return DispatchResult("worker-" + kwargs["job_id"])

    def reconcile(self, **kwargs):
        return ReconcileResult(None)


class Policies:
    def __init__(self):
        self.policies = []

    def compile_bundle(self):
        return SimpleNamespace(policies=self.policies)


class Health:
    """The reconciler's /v1/health/services route."""

    def __init__(self):
        self.instances = None  # None: one healthy instance started now
        self.error = None
        self.urls = []

    async def get(self, url, timeout):
        self.urls.append(url)
        if self.error:
            raise httpx.ConnectError(self.error)
        instances = self.instances if self.instances is not None else [instance()]
        body = {"schemaVersion": 1, "fresh": True,
                "service": {"id": url.rsplit("/", 1)[1], "instances": instances}}
        return httpx.Response(200, json=body)


def instance(*, started=None, restarts=0, state="running", health="healthy", iid="a"):
    started = started or datetime.now(UTC) + timedelta(seconds=5)
    return {"instanceId": iid, "state": state, "health": health,
            "startedAt": started.isoformat(), "restartCount": restarts}


@pytest.fixture
def world(tmp_path):
    engine = fast_sqlite_engine(tmp_path / "ops.db", check_same_thread=False)
    ops_store = OpsStore(None, engine=engine)
    executor = FakeExecutor()
    ops = OpsService(ops_store, executor=executor)
    for slug, spec in ((TTS_RESTART_OPERATION, TTS_SPEC),
                       (START_OPERATIONS[1], {**TTS_SPEC, "action": "start"})):
        ops_store.create_operation({"slug": slug, "enabled": True, "command_script": json.dumps(spec),
                                    "args_schema_json": {"type": "object", "properties": {},
                                                         "additionalProperties": False}}, actor="t")
    handoff = GpuHandoffStore(engine)
    set_state(engine, owner="voice", phase="idle")
    health, policies = Health(), Policies()
    adapter = LocalHandoffAdapter(config=SimpleNamespace(OPS_HEALTH_URL="http://reconciler"), http=health,
                                  video=None, ops=ops, policies=policies)
    yield SimpleNamespace(engine=engine, ops=ops, ops_store=ops_store, executor=executor, handoff=handoff,
                          health=health, policies=policies, adapter=adapter,
                          controller=GpuHandoff(handoff, adapter))
    engine.dispose()


def set_state(engine, **values):
    with engine.begin() as conn:
        conn.execute(_state.update().where(_state.c.id == 1).values(**values))


def job_count(engine):
    with engine.connect() as conn:
        return len(conn.execute(select(OpsJob.id)).all())


def run(coro):
    return asyncio.run(coro)


def test_no_path_runs_a_gated_operation_without_the_handoff_fence(world):
    snapshot = world.ops.prepare_invocation(TTS_RESTART_OPERATION, {})
    for kwargs in ({}, {"approved_snapshot": snapshot, "approval_request_id": "apr",
                        "create_fence": lambda session: None}):
        with pytest.raises(OpsDispatchError) as exc:
            world.ops.dispatch_job(operation_slug=TTS_RESTART_OPERATION, args={}, idempotency_key="k",
                                   caller_backend="http-operator", **kwargs)
        assert exc.value.code == "gpu_handoff_required"
    assert job_count(world.engine) == 0
    assert world.handoff.status().phase == "idle"


def test_unverifiable_target_refuses_before_any_job(world):
    world.ops_store.create_operation({"slug": "named", "enabled": True,
                                      "command_script": '{"action":"restart","container_name":"x"}',
                                      "args_schema_json": {"type": "object", "properties": {}}}, actor="t")
    world.executor.target_compose_identity = lambda spec, args: (_ for _ in ()).throw(ExecutorError("gone"))
    with pytest.raises(OpsDispatchError) as exc:
        world.ops.dispatch_job(operation_slug="named", args={}, idempotency_key="k")
    assert exc.value.code == "gpu_guard_unverifiable"
    assert job_count(world.engine) == 0


def test_unlabelled_named_container_refuses_before_any_job(world):
    """Codex r1 B1 at the service: no job row for a container with no Compose labels."""
    world.ops_store.create_operation({"slug": "named", "enabled": True,
                                      "command_script": '{"action":"restart","container_name":"bawthub-tts-1"}',
                                      "args_schema_json": {"type": "object", "properties": {}}}, actor="t")
    world.executor.identity = (None, None)
    with pytest.raises(OpsDispatchError) as exc:
        world.ops.dispatch_job(operation_slug="named", args={}, idempotency_key="k")
    assert exc.value.code == "gpu_guard_unverifiable"
    assert job_count(world.engine) == 0 and world.executor.dispatches == []


def test_labels_lost_before_dispatch_keep_the_job_unsubmitted(world):
    """Codex r1 B1 at dispatch: labels gone by the live re-check → queued, no worker."""
    world.ops_store.create_operation({"slug": "web.restart", "enabled": True,
        "command_script": '{"action":"restart","container_name":"web"}',
        "args_schema_json": {"type": "object", "properties": {}}}, actor="t")
    world.executor.identity = ("bawthub", "web")
    world.executor.on_preflight = lambda: setattr(world.executor, "identity", (None, None))
    world.ops.dispatch_job(operation_slug="web.restart", args={}, idempotency_key="k")
    assert world.ops_store.get_job_by_key("k").state == JOB_QUEUED and world.executor.dispatches == []


def test_fence_and_job_commit_together(world):
    result = run(world.controller.tts_restart(request_id="r1"))
    state = world.handoff.status()
    assert (state.phase, state.tts_restart_key, state.generation) == ("tts_restarting", "tts-restart:r1", 1)
    assert result["outcome"] == "in_progress" and result["restart"]["job_state"] in ("dispatching", "accepted")
    job = world.ops_store.get_job_by_key("tts-restart:r1")
    assert (job.caller_actor, job.caller_bot_id, job.caller_backend) == (
        "bawthub-tts-admin", None, "http-operator")
    assert world.executor.dispatches == [job.id]


@pytest.mark.parametrize(("state", "code"), [
    ({"owner": "video", "phase": "idle"}, "gpu_not_voice"),
    ({"owner": "voice", "phase": "transitioning"}, "gpu_not_voice"),
    ({"owner": "unknown", "phase": "idle"}, "gpu_not_voice"),
])
def test_a_refused_fence_creates_no_job(world, state, code):
    set_state(world.engine, **state)
    with pytest.raises(HandoffConflict) as exc:
        run(world.controller.tts_restart(request_id="r1"))
    assert exc.value.code == code
    assert job_count(world.engine) == 0
    assert world.handoff.status().phase == state["phase"]


def test_open_video_claims_refuse_the_restart(world):
    from llm_bawt.media.gpu_handoff_store import _video_jobs
    with world.engine.begin() as conn:
        conn.execute(_video_jobs.insert().values(id="v1", generation=0))
    with pytest.raises(HandoffConflict) as exc:
        run(world.controller.tts_restart(request_id="r1"))
    assert exc.value.code == "active_video_claims" and job_count(world.engine) == 0


def test_replay_reads_the_same_job_and_another_request_is_refused(world):
    first = run(world.controller.tts_restart(request_id="r1"))
    again = run(world.controller.tts_restart(request_id="r1"))
    assert again["restart"]["job_id"] == first["restart"]["job_id"]
    with pytest.raises(HandoffConflict) as exc:
        run(world.controller.tts_restart(request_id="r2"))
    assert exc.value.code == "tts_restart_in_progress"
    assert job_count(world.engine) == 1 and len(world.executor.dispatches) == 1


def test_fence_lost_before_dispatch_keeps_the_job_unsubmitted(world):
    world.executor.on_preflight = lambda: set_state(world.engine, phase="idle", tts_restart_key=None)
    run(world.controller.tts_restart(request_id="r1"))
    job = world.ops_store.get_job_by_key("tts-restart:r1")
    assert job.state == JOB_QUEUED and world.executor.dispatches == []
    assert "GPU handoff" in (job.error_text or "") or "admit" in (job.error_text or "")


def test_dispatch_recheck_reads_live_labels(world):
    """A job whose literal spec was not GPU at creation still gates on the live container."""
    world.ops_store.create_operation({"slug": "web.deploy", "enabled": True,
        "command_script": '{"action":"restart","container_name":"web"}',
        "args_schema_json": {"type": "object", "properties": {}}}, actor="t")
    world.executor.identity = ("bawthub", "web")
    world.executor.on_preflight = lambda: setattr(world.executor, "identity", ("bawthub", "tts"))
    world.ops.dispatch_job(operation_slug="web.deploy", args={}, idempotency_key="k")
    assert world.ops_store.get_job_by_key("k").state == JOB_QUEUED and world.executor.dispatches == []


def test_stale_moshi_start_is_refused_at_creation(world):
    snapshot = world.ops.prepare_invocation(START_OPERATIONS[1], {})
    with pytest.raises(HandoffConflict) as exc:
        world.ops.dispatch_job(operation_slug=START_OPERATIONS[1], args={}, approved_snapshot=snapshot,
                               idempotency_key=f"gpu-handoff:0:{START_OPERATIONS[1]}",
                               create_fence=GpuHandoffFence.voice_start(
                                   generation=0, operation_slug=START_OPERATIONS[1]))
    assert exc.value.code == "gpu_handoff_stale" and job_count(world.engine) == 0


def test_restore_start_is_admitted_while_pending(world):
    set_state(world.engine, owner="video", phase="transitioning", offer_target="voice",
              pending_action="start_moshi_tts", generation=4)
    snapshot = world.ops.prepare_invocation(START_OPERATIONS[1], {})
    run(world.adapter.dispatch(snapshot, user="u", generation=4))
    assert len(world.executor.dispatches) == 1


def test_policy_deny_refuses_before_any_job(world):
    from agent_bridge.approval import PolicyAction
    world.policies.policies = [SimpleNamespace()]
    import llm_bawt.media.gpu_handoff_adapter as adapter_module
    original = adapter_module.evaluate
    adapter_module.evaluate = lambda *a, **k: SimpleNamespace(action=PolicyAction.DENY)
    try:
        with pytest.raises(HandoffConflict) as exc:
            run(world.controller.tts_restart(request_id="r1"))
    finally:
        adapter_module.evaluate = original
    assert exc.value.code == "policy_denied" and job_count(world.engine) == 0


def test_offers_are_refused_while_the_restart_holds(world):
    run(world.controller.tts_restart(request_id="r1"))
    with pytest.raises(HandoffConflict) as exc:
        run(world.controller.offer(user="u", expected_generation=1, calibration=True))
    assert exc.value.code == "tts_restart_in_progress"
    with pytest.raises(HandoffConflict) as exc:
        run(world.controller.offer_restore(user="u", expected_generation=1))
    assert exc.value.code == "tts_restart_in_progress"


# ── Status: the only releaser ───────────────────────────────────────────────

def finish(world, state, *, exit_code=None, dispatched=True):
    job = world.ops_store.get_job_by_key("tts-restart:r1")
    if not dispatched:
        assert job.dispatched_at is None
    world.ops_store.mark_terminal(job.id, state=state, exit_code=exit_code)
    return job


def status(world):
    return run(world.controller.tts_restart_status(request_id="r1"))


def test_success_with_start_evidence_releases(world):
    run(world.controller.tts_restart(request_id="r1"))
    assert status(world)["restart"]["reason"] == "awaiting_job"
    finish(world, "succeeded", exit_code=0)
    result = status(world)
    assert result["outcome"] == "released" and result["restart"]["observed_restart_count"] == 0
    state = world.handoff.status()
    assert (state.phase, state.tts_restart_key, state.generation) == ("idle", None, 2)
    assert status(world)["outcome"] == "released"  # idempotent
    assert world.health.urls[0] == "http://reconciler/v1/health/services/bawthub.tts"


@pytest.mark.parametrize(("instances", "error", "reason"), [
    ([instance(started=datetime.now(UTC) - timedelta(hours=1))], None, "awaiting_start_evidence"),
    ([instance(restarts=1)], None, "policy_restart_observed"),
    ([instance(), instance(iid="b")], None, "ambiguous_instances"),
    ([instance(health="starting")], None, "unhealthy"),
    ([instance(state="exited")], None, "unhealthy"),
    (None, "refused", "observation_unavailable"),
])
def test_success_without_start_evidence_holds(world, instances, error, reason):
    run(world.controller.tts_restart(request_id="r1"))
    finish(world, "succeeded", exit_code=0)
    world.health.instances, world.health.error = instances, error
    result = status(world)
    assert (result["outcome"], result["restart"]["reason"]) == ("held", reason)
    assert world.handoff.status().phase == "tts_restarting"


@pytest.mark.parametrize("state", ["lost", "timed_out", "cancelled", "failed"])
def test_uncertain_end_holds_until_start_evidence(world, state):
    """Loopy's barrier case: a lost job's restart can still land later."""
    run(world.controller.tts_restart(request_id="r1"))
    finish(world, state)
    world.health.instances = [instance(started=datetime.now(UTC) - timedelta(hours=1))]
    assert status(world)["outcome"] == "held"
    with pytest.raises(HandoffConflict):
        run(world.controller.tts_restart(request_id="r2"))
    world.health.instances = None
    result = status(world)
    assert (result["outcome"], result["restart"]["job_state"]) == ("released", state)


def test_preflight_refusal_releases_without_evidence(world):
    world.executor.on_preflight = lambda: (_ for _ in ()).throw(ExecutorError("worker image missing"))
    run(world.controller.tts_restart(request_id="r1"))
    finish(world, "failed", dispatched=False)
    world.health.error = "must not be read"
    result = status(world)
    assert (result["outcome"], result["restart"]["dispatched_at"]) == ("released", None)
    assert world.health.urls == []


def test_succeeded_with_nonzero_exit_holds(world):
    run(world.controller.tts_restart(request_id="r1"))
    finish(world, "succeeded", exit_code=3)
    assert status(world)["restart"]["reason"] == "awaiting_start_evidence"


def test_unknown_request_is_refused_and_ids_are_validated(world):
    assert status(world)["outcome"] == "refused"
    with pytest.raises(HandoffConflict) as exc:
        run(world.controller.tts_restart_status(request_id="../x"))
    assert exc.value.code == "args_invalid"


def test_a_release_does_not_free_another_requests_fence(world):
    run(world.controller.tts_restart(request_id="r1"))
    finish(world, "succeeded", exit_code=0)
    set_state(world.engine, tts_restart_key="tts-restart:other")
    result = status(world)
    assert result["outcome"] == "released"  # this job's fence is gone
    assert world.handoff.status().tts_restart_key == "tts-restart:other"


# ── Routes ──────────────────────────────────────────────────────────────────

@pytest.fixture
def client(world, monkeypatch):
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def factory():
        yield world.controller

    monkeypatch.setattr(media_handoff, "_controller", factory)
    app = FastAPI()
    app.include_router(media_handoff.router)
    with TestClient(app) as test_client:
        yield test_client, app


def test_routes_require_the_service_token(client, monkeypatch):
    from llm_bawt.bawthub_auth.service_token import TokenCheck
    from llm_bawt.service.routes import service_auth
    test_client, _ = client
    monkeypatch.setattr(service_auth, "service_token_source",
                        lambda: SimpleNamespace(check=lambda header: TokenCheck.INVALID))
    assert test_client.post("/v1/media/local-video/handoff/tts-restart", json={"request_id": "r1"}
                            ).status_code == 401
    assert test_client.get("/v1/media/local-video/handoff/tts-restart/r1").status_code == 401


def test_routes_dispatch_once_and_answer_coded_conflicts(client, world):
    test_client, app = client
    app.dependency_overrides[require_service_token] = lambda: None
    url = "/v1/media/local-video/handoff/tts-restart"
    first = test_client.post(url, json={"request_id": "r1"})
    assert first.status_code == 200 and first.json()["outcome"] == "in_progress"
    assert test_client.post(url, json={"request_id": "r1"}).json()["restart"]["job_id"] == (
        first.json()["restart"]["job_id"])
    other = test_client.post(url, json={"request_id": "r2"})
    assert other.status_code == 409 and other.json()["detail"]["code"] == "tts_restart_in_progress"
    assert test_client.post(url, json={"request_id": "r1", "extra": 1}).status_code == 422
    assert test_client.get(f"{url}/r1").json()["restart"]["reason"] == "awaiting_job"
    assert len(world.executor.dispatches) == 1


def test_route_maps_ops_refusals(client, world):
    test_client, app = client
    app.dependency_overrides[require_service_token] = lambda: None
    world.ops_store.update_operation(TTS_RESTART_OPERATION, {"enabled": False}, actor="t")
    response = test_client.post("/v1/media/local-video/handoff/tts-restart", json={"request_id": "r1"})
    assert response.status_code == 409 and response.json()["detail"]["code"] == "operation_disabled"
