"""TASK-1179: same-image route repair, no live Docker or network calls."""
from __future__ import annotations

import copy
import fcntl
import json
import struct
from types import SimpleNamespace

import pytest

from llm_bawt.ops.executor import DockerExecutor, ExecutorError, validate_spec
from llm_bawt.ops.health_probe import WORKER_LABEL as HEALTH_PROBE_LABEL
from llm_bawt.ops.image_deploy import DeployFailed, clone_create_body
from llm_bawt.ops.routing_reconcile import (
    SCHEMA, WORKER_CAPABILITY_LABEL, RoutingReconciler,
    patched_rule, routing_binding, valid_routing_binding,
)
from llm_bawt.ops.seeds import SEEDS
from llm_bawt.ops.worker import run as worker_run
from tests.test_ops_image_deploy import (
    FakeDocker, JOB_ID, NAME, OLD_ID, REPO, SOURCE, _labels, _worker_request,
)

SEED = next(s for s in SEEDS if s["slug"] == "bawthub.reconcile-prod-routing")
SPEC = json.loads(SEED["command_script"])
ROUTER = "traefik.http.routers.frontprod-api"
RULE = "(Host(`app.bawthub.com`) || Host(`echo.ferreri.us`)) && (PathPrefix(`/api/chat`) || PathPrefix(`/api/clog`))"
REF = REPO + "@sha256:" + "a" * 64
REPORT = {"status": "down", "summary": {"ok": 1, "degraded": 0, "down": 1},
          "checks": [{"slug": "bawthub.database", "status": "ok"}, {"slug": "bridge.codex", "status": "down"}]}


class RoutingDocker(FakeDocker):
    def __init__(self, *, route_bad=False, route_delay=0, **kwargs):
        super().__init__(**kwargs)
        self.route_bad, self.route_delay, self.route_attempts = route_bad, route_delay, 0
        self.route_exec = False
        self.exec_bodies = []
        image = self.images[OLD_ID]
        image["RepoDigests"] = [REF]
        image["Config"]["Labels"].update(_labels())
        image["_release"] = {"version": "1.4.2", "sourceSha": SOURCE, "workflowRunId": "123456"}
        old = self.by_name(NAME)
        old["Config"]["Image"] = REF
        old["Config"]["Labels"].update({
            ROUTER + ".rule": RULE, ROUTER + ".service": "frontend-prod",
            ROUTER + ".entrypoints": "web", ROUTER + ".priority": "200",
            "traefik.http.services.frontend-prod.loadbalancer.server.port": "3000",
            "traefik.enable": "true",
        })
        old["HostConfig"].update(Binds=["/var/run/docker.sock:/var/run/docker.sock"], GroupAdd=None)

    def request(self, method, path, *, body=None, headers=None, raw=False):
        if path.endswith("/exec") and method == "POST":
            self.exec_bodies.append(body)
            self.route_exec = "http://traefik/api/system/health" in body["Cmd"]
        if path.startswith("/exec/") and self.route_exec:
            self.calls.append((method, path))
            if path.endswith("/start"):
                self.route_attempts += 1
                payload = json.dumps({"detail": "Not Found"} if self.route_bad or self.route_attempts <= self.route_delay else REPORT).encode()
                return struct.pack(">BxxxI", 1, len(payload)) + payload
            return {"ExitCode": 0}
        return super().request(method, path, body=body, headers=headers, raw=raw)


def snapshot(api):
    return {"spec": SPEC, "resolved_args": {},
            "routing": routing_binding(SPEC, api.by_name(NAME), api.images[OLD_ID]),
            "execution": {"start_delay_seconds": 0, "timeout_seconds": 60, "max_output_bytes": 65536}}


def reconciler(api, tmp_path, snap=None):
    return RoutingReconciler(api, snap or snapshot(api), JOB_ID, lock_root=tmp_path / "locks",
                             sleep=lambda _: None, clock=iter(range(0, 10000, 50)).__next__)


def test_seed_disabled_fixed_target_and_no_arguments():
    assert not SEED["enabled"] and SEED["max_concurrent"] == 1
    assert json.loads(SEED["args_schema_json"]) == {"type": "object", "properties": {}, "additionalProperties": False}
    assert validate_spec(SEED["command_script"]) == SPEC
    for key, value in (("command", "sh"), ("group_add", [123]), ("add_prefixes", ["/api"]),
                       ("container_name_from_arg", "target"), ("hostname", "bad\nHost: other")):
        with pytest.raises(ValueError):
            validate_spec(json.dumps({**SPEC, key: value}))


@pytest.mark.parametrize("rule", [
    "Host(`app.bawthub.com`) || PathPrefix(`/api/chat`)",
    "(Host(`evil.com`)) && (PathPrefix(`/api/chat`))",
    RULE + " || Host(`evil.com`)", "HostRegexp(`.*`)",
])
def test_unsafe_rule_refused(rule):
    with pytest.raises(ValueError):
        patched_rule(SPEC, rule)


def test_rule_preserves_hosts_paths_and_is_idempotent():
    after = patched_rule(SPEC, RULE)
    assert after == RULE[:-1] + " || PathPrefix(`/api/system`))"
    assert patched_rule(SPEC, after) == after


def test_repair_preserves_image_config_and_accepts_report_with_down_service(tmp_path):
    api = RoutingDocker(route_delay=1)
    old = copy.deepcopy(api.by_name(NAME))
    expected = clone_create_body(old, api.images[OLD_ID], api.images[OLD_ID], REF, JOB_ID)
    expected["Labels"][ROUTER + ".rule"] = patched_rule(SPEC, RULE)
    result = json.loads(reconciler(api, tmp_path).run())
    assert result["schema"] == SCHEMA and result["phase"] == "done"
    assert result["previous"]["image_id"] == result["deployed"]["image_id"] == OLD_ID
    assert api.create_bodies == [expected]
    assert api.by_name(NAME)["Config"]["Image"] == REF
    assert result["health"]["routing"]["summary"]["down"] == 1
    assert api.route_attempts == 2
    assert not any("/images/create" in p for _, p in api.calls)
    assert api.exec_bodies[-1]["Cmd"] == ["curl", "-fsS", "--max-time", "5", "-H", "Host: app.bawthub.com", "http://traefik/api/system/health"]


@pytest.mark.parametrize("field", ["env", "host", "rule", "identity", "network", "image"])
def test_stale_config_or_identity_refused_before_create(tmp_path, field):
    api = RoutingDocker()
    snap = snapshot(api)
    old = api.by_name(NAME)
    if field == "env":
        old["Config"]["Env"].append("NEW=secret")
    elif field == "host":
        old["HostConfig"]["GroupAdd"] = [123]
    elif field == "rule":
        old["Config"]["Labels"][ROUTER + ".rule"] = patched_rule(SPEC, RULE)
    elif field == "identity":
        old["Id"] = "b" * 64
    elif field == "network":
        old["NetworkSettings"]["Networks"]["bawthub_default"]["Aliases"].append("new-alias")
    else:
        old["Image"] = "sha256:" + "9" * 64
    with pytest.raises(DeployFailed, match="changed since approval"):
        reconciler(api, tmp_path, snap).run()
    assert not api.create_bodies and not api.side_effect_started
    assert "secret" not in json.dumps(snap)


def test_route_failure_restores_original_container(tmp_path):
    api = RoutingDocker(route_bad=True)
    with pytest.raises(DeployFailed) as caught:
        reconciler(api, tmp_path).run()
    assert caught.value.outcome_known and caught.value.detail["restored"]
    assert api.by_name(NAME)["Id"] == api.old_id
    assert api.by_name(NAME)["Config"]["Labels"][ROUTER + ".rule"] == RULE


def test_restore_failure_reports_unknown(tmp_path):
    api = RoutingDocker(route_bad=True, old_restart_fails=True)
    with pytest.raises(DeployFailed) as caught:
        reconciler(api, tmp_path).run()
    assert not caught.value.outcome_known


def test_shared_target_lock_blocks_routing_during_deployment(tmp_path):
    api = RoutingDocker()
    snap = snapshot(api)
    (tmp_path / "locks").mkdir()
    with (tmp_path / "locks" / f"{NAME}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with pytest.raises(DeployFailed, match="holds this target"):
            reconciler(api, tmp_path, snap).run()
    assert not api.calls


def test_correct_route_is_verified_without_recreating(tmp_path):
    api = RoutingDocker()
    api.by_name(NAME)["Config"]["Labels"][ROUTER + ".rule"] = patched_rule(SPEC, RULE)
    result = json.loads(reconciler(api, tmp_path).run())
    assert result["unchanged"] and not api.create_bodies and not api.side_effect_started


def test_snapshot_validation_rejects_smuggling_and_forged_delta():
    api = RoutingDocker()
    snap = snapshot(api)
    assert valid_routing_binding(snap)
    snap["routing"]["after_rule"] = "Host(`evil.com`)"
    assert not valid_routing_binding(snap)
    assert not valid_routing_binding({"spec": {"action": "restart"}, "routing": {}})
    assert valid_routing_binding({"spec": {"action": "restart"}})


def test_legacy_worker_capability_fails_before_dispatch(monkeypatch):
    images = SimpleNamespace(get=lambda _: SimpleNamespace(attrs={"Config": {"Labels": {}}}))
    executor = DockerExecutor(client_factory=lambda: SimpleNamespace(images=images))
    monkeypatch.setattr(executor, "_check_prerequisites", lambda _: None)
    monkeypatch.setattr(executor, "self_target", lambda _s: None)  # Not under test (TASK-1169 S4c).
    with pytest.raises(ExecutorError, match="deploy the tested worker first"):
        executor.routing_approval(SPEC)
    with pytest.raises(ExecutorError, match="deploy the tested worker first"):
        executor.preflight({"spec": SPEC, "execution": {"worker_image": "old"}})
    images.get = lambda _: SimpleNamespace(attrs={"Config": {"Labels": {WORKER_CAPABILITY_LABEL: "1"}}})
    # TASK-1801: the seeded row carries a health probe, which the worker must also run.
    with pytest.raises(ExecutorError, match=HEALTH_PROBE_LABEL):
        executor.preflight({"spec": SPEC, "execution": {"worker_image": "new"}})
    images.get = lambda _: SimpleNamespace(attrs={"Config": {"Labels": {WORKER_CAPABILITY_LABEL: "1",
                                                                         HEALTH_PROBE_LABEL: "1"}}})
    executor.preflight({"spec": SPEC, "execution": {"worker_image": "new"}})


def test_approval_snapshot_is_bound_and_rejects_rehashed_tampering():
    from llm_bawt.ops.service import OpsDispatchError
    from tests.test_ops_image_deploy import _rehash, _service

    api = RoutingDocker()
    service, store, executor = _service()
    store.create_operation({**SEED, "enabled": True})
    executor.routing_approval = lambda spec: routing_binding(spec, api.by_name(NAME), api.images[OLD_ID])
    approved = service.prepare_invocation(SEED["slug"], {})
    assert approved["routing"] == snapshot(api)["routing"]
    assert service._verify_snapshot(approved, SEED["slug"], {}) == approved
    approved["routing"]["after_rule"] = "Host(`other.com`)"
    with pytest.raises(OpsDispatchError, match="routing binding mismatch"):
        service._verify_snapshot(_rehash(approved), SEED["slug"], {})


def test_disabled_operation_and_user_patch_arguments_are_refused():
    from llm_bawt.ops.service import OpsDispatchError
    from tests.test_ops_image_deploy import _service

    service, store, _executor = _service()
    store.create_operation(SEED)
    with pytest.raises(OpsDispatchError, match="disabled"):
        service.prepare_invocation(SEED["slug"], {})
    # Seed schema rejects user-selected images, config and hosts independently.
    from llm_bawt.ops.validation import ArgValidationError, validate_args
    for args in ({"image": "evil"}, {"hostname": "other.com"}, {"labels": {}}, {"command": "sh"}):
        with pytest.raises(ArgValidationError):
            validate_args(args, SEED["args_schema_json"])


@pytest.mark.parametrize("field", ["Name", "Image", "Labels", "RepoDigests"])
def test_wrong_target_or_unpinned_image_refused(field):
    api = RoutingDocker()
    old, image = api.by_name(NAME), api.images[OLD_ID]
    if field == "Name":
        old["Name"] = "/other"
    elif field == "Image":
        old["Config"]["Image"] = REPO + ":latest"
    elif field == "Labels":
        old["Config"]["Labels"][ROUTER + ".service"] = "voice"
    else:
        image["RepoDigests"] = []
    with pytest.raises(ValueError):
        routing_binding(SPEC, old, image)


def test_stopped_frontend_is_not_started_as_side_effect(tmp_path):
    api = RoutingDocker()
    api.by_name(NAME)["State"]["Running"] = False
    with pytest.raises(DeployFailed, match="running healthy"):
        reconciler(api, tmp_path).run()
    assert not api.create_bodies


def test_worker_writes_durable_receipt_and_does_not_replay(tmp_path):
    api = RoutingDocker()
    path, digest = _worker_request(tmp_path, snapshot(api))
    assert worker_run(path, digest, api_factory=lambda _: api, sleep=lambda _: None) == 0
    receipt = json.loads((path.parent / "receipt.json").read_text())
    assert receipt["state"] == "succeeded" and receipt["exit_code"] == 0
    assert json.loads(receipt["output_tail"])["schema"] == SCHEMA
    before = len(api.calls)
    assert worker_run(path, digest, api_factory=lambda _: api) == 0
    assert len(api.calls) == before
