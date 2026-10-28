"""TASK-1169 S4c: self-management requires isolated mode (hermetic)."""
from __future__ import annotations

import uuid

import pytest

from llm_bawt.ops import JOB_ACCEPTED, JOB_QUEUED, OpsDispatchError, OpsService
from llm_bawt.ops.execution_mode import SELF_TARGET_CARD_LINE, with_card_lines
from llm_bawt.ops.config_reconcile import ACTION as CONFIG_ACTION
from llm_bawt.ops.executor import ExecutorError
from llm_bawt.ops.image_deploy import DEPLOY_ACTION, ROLLBACK_ACTION
from llm_bawt.ops.plan_reconcile import ACTION as PLAN_ACTION
from llm_bawt.ops.self_target import (SELF_IDENTITY_UNKNOWN, SELF_REQUIRES_ISOLATED, SELF_STOP_UNRECOVERABLE,
                                      SELF_TARGET_UNVERIFIABLE, ContainerIdentity, SelfTargetError, self_target)
from tests.test_ops_executor import JOB as DOCKER_JOB, setup as docker_setup
from tests.test_ops_service import FakeExecutor, _op_data, _store

OWN_ID = "f" * 64
OWN = ContainerIdentity(OWN_ID, "ops", "reconciler")
OTHER = ContainerIdentity("e" * 64, "stack", "web")
# A name alias and an abbreviated ID resolve to the same full ID.
CONTAINERS = {"ops-reconciler-1": OWN, "reconciler-alias": OWN, OWN_ID[:12]: OWN, "web": OTHER,
              "reconciler-recreated": ContainerIdentity("d" * 64, "ops", "reconciler"),
              "other-project-reconciler": ContainerIdentity("c" * 64, "elsewhere", "reconciler")}


def _inspect(name):
    if name not in CONTAINERS:
        raise SelfTargetError(SELF_TARGET_UNVERIFIABLE, f"cannot inspect target container {name!r}")
    return CONTAINERS[name]


def _snapshot(spec, mode="isolated", args=None, plan=None):
    snapshot = {"spec": spec, "resolved_args": args or {},
                "execution": {"executor_kind": "docker", "execution_mode": mode},
                "operation": {"execution_mode": mode}}
    if plan is not None:
        snapshot["plan"] = {"targets": [{"spec": target} for target in plan]}
    return snapshot


def _classify(snapshot, own=lambda: OWN):
    return self_target(snapshot, own=own, inspect=_inspect)


def _refused(snapshot, **kwargs):
    with pytest.raises(SelfTargetError) as caught:
        _classify(snapshot, **kwargs)
    return caught.value.code


@pytest.mark.parametrize("spec, args", [
    ({"action": "restart", "container_name": "reconciler-alias"}, {}),
    ({"action": "restart", "container_name_from_arg": "target"}, {"target": OWN_ID[:12]}),
    ({"action": "restart", "container_name": "reconciler-recreated"}, {}),  # new ID, same labels
    ({"action": "restart", "compose_project": "ops", "compose_service": "reconciler"}, {}),
    ({"action": "restart", "compose_project": "ops", "compose_service_from_arg": "service"},
     {"service": "reconciler"}),
])
def test_a_process_job_on_the_ops_service_is_refused_however_it_names_it(spec, args):
    assert _refused(_snapshot(spec, "process", args)) == SELF_REQUIRES_ISOLATED


def test_the_same_service_name_in_another_project_is_not_the_ops_service():
    spec = {"action": "restart", "container_name": "other-project-reconciler"}
    assert _classify(_snapshot(spec, "process")) is None
    spec = {"action": "restart", "compose_project": "elsewhere", "compose_service": "reconciler"}
    assert _classify(_snapshot(spec, "process")) is None


@pytest.mark.parametrize("spec", [
    {"action": "stop", "container_name": "reconciler-alias"},
    {"action": "remove", "container_name_from_arg": "target", "compose_project": "ops"},
])
def test_an_isolated_job_that_would_leave_the_ops_service_down_is_refused(spec):
    assert _refused(_snapshot(spec, "isolated", {"target": "ops-reconciler-1"})) == SELF_STOP_UNRECOVERABLE


@pytest.mark.parametrize("spec", [
    {"action": "restart", "container_name": "ops-reconciler-1"},
    {"action": DEPLOY_ACTION, "container_name": "ops-reconciler-1", "compose_project": "ops",
     "compose_service": "reconciler"},
    {"action": ROLLBACK_ACTION, "container_name": "ops-reconciler-1", "compose_project": "ops",
     "compose_service": "reconciler"},
])
def test_isolated_self_restart_and_image_actions_are_allowed_and_named(spec):
    assert _classify(_snapshot(spec)) == "Compose service ops/reconciler"


def test_a_service_plan_is_classified_through_its_bound_targets():
    plan = {"action": PLAN_ACTION}
    targets = [{"action": CONFIG_ACTION, "container_name": "web"},
               {"action": CONFIG_ACTION, "container_name": "reconciler-alias"}]
    assert _refused(_snapshot(plan, "process", plan=targets)) == SELF_REQUIRES_ISOLATED
    assert _classify(_snapshot(plan, plan=targets)) == "Compose service ops/reconciler"
    assert _classify(_snapshot(plan, "process", plan=targets[:1])) is None
    assert _refused(_snapshot(plan, "process")) == SELF_TARGET_UNVERIFIABLE


def test_a_job_on_another_container_is_not_a_self_target():
    assert _classify(_snapshot({"action": "restart", "container_name": "web"}, "process")) is None


def test_pull_touches_no_container():
    assert _classify(_snapshot({"action": "pull", "container_name": "reconciler-alias"}, "process")) is None


def test_an_uninspectable_target_and_an_unknown_own_identity_both_fail_closed():
    assert _refused(_snapshot({"action": "restart", "container_name": "gone"}, "process")) == SELF_TARGET_UNVERIFIABLE

    def unknown():
        raise SelfTargetError(SELF_IDENTITY_UNKNOWN, "cannot inspect own container")
    # Even a job on another container: nothing can be proven not to be the ops service.
    assert _refused(_snapshot({"action": "restart", "container_name": "web"}), own=unknown) == SELF_IDENTITY_UNKNOWN


# --- DockerExecutor: its own identity, and the dispatch preflight ---------------

def _docker(tmp_path, monkeypatch, *, own_labels):
    executor, client, snapshot = docker_setup(tmp_path, monkeypatch)
    own = client.containers.rows["fake-app"]
    own.id, own.labels = OWN_ID, own_labels
    target = client.containers.rows.setdefault("app", type(own)())
    target.id = "a" * 64
    snapshot["execution"]["execution_mode"] = "process"
    snapshot["operation"] = {"execution_mode": "process"}
    return executor, client, snapshot


def test_the_docker_executor_reads_its_own_identity_once(tmp_path, monkeypatch):
    executor, client, snapshot = _docker(tmp_path, monkeypatch, own_labels={
        "com.docker.compose.project": "ops", "com.docker.compose.service": "reconciler"})
    assert executor.self_target(snapshot) is None
    del client.containers.rows["fake-app"]  # Its full ID cannot change while it lives.
    assert executor.self_target(snapshot) is None
    # The target, relabelled as the ops service after approval, is re-inspected.
    client.containers.rows["app"].labels = dict(client.containers.rows["app"].labels,
                                                **{"com.docker.compose.project": "ops",
                                                   "com.docker.compose.service": "reconciler"})
    with pytest.raises(ExecutorError, match=SELF_REQUIRES_ISOLATED):
        executor.self_target(snapshot)


def test_the_docker_preflight_refuses_a_self_target_by_its_full_id(tmp_path, monkeypatch):
    executor, client, snapshot = _docker(tmp_path, monkeypatch, own_labels={})
    snapshot["spec"] = {"action": "restart", "container_name": "fake-app"}
    with pytest.raises(ExecutorError, match=SELF_REQUIRES_ISOLATED):
        executor.self_target(snapshot)
    snapshot["spec"] = {"action": "restart", "container_name": "app"}
    assert executor.self_target(snapshot) is None


def test_the_docker_executor_fails_closed_without_its_own_identity(tmp_path, monkeypatch):
    executor, client, snapshot = _docker(tmp_path, monkeypatch, own_labels={})
    client.containers.rows["fake-app"].id = None
    with pytest.raises(ExecutorError, match=SELF_IDENTITY_UNKNOWN):
        executor.self_target(snapshot)
    monkeypatch.delenv("LLM_BAWT_OPS_APP_CONTAINER")
    monkeypatch.delenv("HOSTNAME", raising=False)
    with pytest.raises(ExecutorError, match=SELF_IDENTITY_UNKNOWN):
        executor.self_target(snapshot)


# --- The service: refused at prepare, named on the card, re-checked at claim -----

class SelfAwareExecutor(FakeExecutor):
    """Classifies like the Docker executor, from a table of fake containers."""

    def __init__(self, containers=None):
        super().__init__()
        self.containers = dict(CONTAINERS, **(containers or {}))

    def self_target(self, snapshot):
        return self_target(snapshot, own=lambda: OWN, inspect=lambda name: self.containers[name])

    def preflight(self, snapshot):
        super().preflight(snapshot)
        self.self_target(snapshot)


def _service(mode, script, containers=None):
    store = _store()
    store.create_operation(_op_data(slug="test.self", execution_mode=mode, command_script=script))
    executor = SelfAwareExecutor(containers)
    return OpsService(store, executor=executor), store, executor


@pytest.mark.parametrize("mode, script, code", [
    ("process", '{"action":"restart","container_name":"reconciler-alias"}', SELF_REQUIRES_ISOLATED),
    ("isolated", '{"action":"stop","container_name":"reconciler-alias"}', SELF_STOP_UNRECOVERABLE),
])
def test_prepare_refuses_a_self_target_before_any_approval_card(mode, script, code):
    service, _, _ = _service(mode, script)
    with pytest.raises(OpsDispatchError) as caught:
        service.prepare_invocation("test.self", {"target": "a"})
    assert caught.value.code == code


def test_an_isolated_self_restart_is_prepared_and_the_card_names_it():
    service, _, _ = _service("isolated", '{"action":"restart","container_name":"reconciler-alias"}')
    snapshot = service.prepare_invocation("test.self", {"target": "a"})
    assert snapshot["self_target"] == "Compose service ops/reconciler"
    assert SELF_TARGET_CARD_LINE in with_card_lines("Restart it?", snapshot)
    other, _, _ = _service("isolated", '{"action":"restart","container_name":"web"}')
    assert "self_target" not in other.prepare_invocation("test.self", {"target": "a"})


def test_a_target_that_became_the_ops_service_after_prepare_is_refused_at_claim():
    service, store, executor = _service("process", '{"action":"restart","container_name":"web"}')
    calls = []

    def renamed(name):  # Not the ops service at prepare; it is by dispatch.
        calls.append(name)
        return OTHER if len(calls) == 1 else OWN
    executor.self_target = lambda snapshot: self_target(snapshot, own=lambda: OWN, inspect=renamed)
    job = service.dispatch_job(operation_slug="test.self", args={"target": "a"},
                               idempotency_key=uuid.uuid4().hex)
    row = store.get_job(job["id"])
    assert row.state == JOB_QUEUED and row.worker_slot_at is None
    assert row.error_text.startswith(SELF_REQUIRES_ISOLATED)
    assert executor.dispatch_calls == []
    executor.self_target = lambda snapshot: None  # Back to another container: it runs.
    service.get_job_status(job["id"])
    assert store.get_job(job["id"]).state == JOB_ACCEPTED


def test_a_created_worker_is_not_started_by_recovery_once_its_target_is_refused(tmp_path, monkeypatch):
    executor, client, snapshot = docker_setup(tmp_path, monkeypatch)
    original = client.containers.create

    def create(*args, **kwargs):
        original(*args, **kwargs)
        raise TimeoutError("response lost")
    client.containers.create = create
    with pytest.raises(ExecutorError):
        executor.dispatch(job_id=DOCKER_JOB, snapshot=snapshot)
    worker = client.containers.rows[executor.worker_name(DOCKER_JOB)]

    def refuse(_snapshot):
        raise SelfTargetError(SELF_STOP_UNRECOVERABLE, "now the ops service")
    executor.self_target = refuse
    result = executor.reconcile(job_id=DOCKER_JOB, snapshot=snapshot)
    # No start was ever attempted, so it is certain nothing ran.
    assert result.state == "failed" and SELF_STOP_UNRECOVERABLE in result.error
    assert worker.starts == 0


def test_a_created_worker_with_an_earlier_start_attempt_stays_unconfirmed_not_refused(tmp_path, monkeypatch):
    executor, client, snapshot = docker_setup(tmp_path, monkeypatch)
    original = client.containers.create

    def create(*args, **kwargs):
        original(*args, **kwargs)
        raise TimeoutError("response lost")
    client.containers.create = create
    with pytest.raises(ExecutorError):
        executor.dispatch(job_id=DOCKER_JOB, snapshot=snapshot)
    worker = client.containers.rows[executor.worker_name(DOCKER_JOB)]
    directory = tmp_path / DOCKER_JOB
    (directory / "start-attempt.json").write_text('{"start_attempted": true}')
    checks = []

    def refuse(_snapshot):
        checks.append("checked")
        raise SelfTargetError(SELF_STOP_UNRECOVERABLE, "now the ops service")
    executor.self_target = refuse
    result = executor.reconcile(job_id=DOCKER_JOB, snapshot=snapshot)
    # That earlier start may still complete: lost, never a certain "failed".
    assert result.state == "lost" and "without confirmation" in result.error
    assert checks == [] and worker.starts == 0 and not (directory / "receipt.json").exists()


def test_an_inspected_target_without_a_full_id_fails_closed():
    containers = dict(CONTAINERS, anonymous=ContainerIdentity(None, None, None))
    snapshot = _snapshot({"action": "restart", "container_name": "anonymous"})
    with pytest.raises(SelfTargetError) as caught:
        self_target(snapshot, own=lambda: OWN, inspect=containers.__getitem__)
    assert caught.value.code == SELF_TARGET_UNVERIFIABLE


def test_an_abbreviated_id_cannot_prove_a_target_is_not_the_ops_service():
    containers = dict(CONTAINERS, short=ContainerIdentity(OWN_ID[:12], None, None))
    snapshot = _snapshot({"action": "restart", "container_name": "short"})
    with pytest.raises(SelfTargetError) as caught:
        self_target(snapshot, own=lambda: OWN, inspect=containers.__getitem__)
    assert caught.value.code == SELF_TARGET_UNVERIFIABLE


def test_the_docker_executor_needs_its_own_full_id(tmp_path, monkeypatch):
    executor, client, snapshot = _docker(tmp_path, monkeypatch, own_labels={})
    client.containers.rows["fake-app"].id = OWN_ID[:12]
    with pytest.raises(ExecutorError, match=SELF_IDENTITY_UNKNOWN):
        executor.self_target(snapshot)
