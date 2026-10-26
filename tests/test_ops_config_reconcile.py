"""TASK-1518: same-image Compose configuration reconcile, no live Docker or network."""
from __future__ import annotations

import copy
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_bawt.ops.compose_render import ComposeRenderer
from llm_bawt.ops.config_reconcile import (
    SCHEMA, WORKER_CAPABILITY_LABEL, ConfigReconciler, config_binding, valid_config_binding,
)
from llm_bawt.ops.executor import ConfigCheckError, DockerExecutor, validate_spec
from llm_bawt.ops.image_deploy import DeployFailed
from llm_bawt.ops.seeds import SEEDS
from llm_bawt.ops.worker import run as worker_run
from tests.test_ops_image_deploy import (
    CONFIG_HASH, DIGEST, JOB_ID, NAME, OLD_ID, RENDERER, REPO, SOURCE, SPEC as DEPLOY_SPEC, TOKEN_TARGET,
    FakeDocker, _labels, _worker_request, compose_project,
)

SEED = next(s for s in SEEDS if s["slug"] == "bawthub.reconcile-prod-config")
SPEC = json.loads(SEED["command_script"])
REF = REPO + "@sha256:" + "a" * 64
WORKER = "sha256:" + "0" * 64
SECRET = "sk-live-never-shown"
DRIFTED = compose_project(environment={"NODE_ENV": "production", "BAWTHUB_AUTH_SESSIONS": "on",
                                       "XAI_API_KEY": SECRET},
                          networks={"default": None, "llm-bawt": None})
EXECUTION = {"start_delay_seconds": 0, "timeout_seconds": 60, "max_output_bytes": 65536}


class ConfigDocker(FakeDocker):
    """A pinned, release-labelled target plus in-container ``stat``."""

    def __init__(self, *, stat_exit=0, fail_new_start=False, **kwargs):
        super().__init__(**kwargs)
        self.stat_exit, self.fail_new_start = stat_exit, fail_new_start
        self.stat_cmds = []
        image = self.images[OLD_ID]
        image["RepoDigests"] = [REF]
        image["Config"]["Labels"].update(_labels())
        image["_release"] = {"version": "1.4.2", "sourceSha": SOURCE, "workflowRunId": "123456"}
        self.by_name(NAME)["Config"]["Image"] = REF

    def request(self, method, path, *, body=None, headers=None, raw=False):
        if method == "POST" and path.endswith("/exec") and body["Cmd"][0] == "stat":
            self.calls.append((method, path))
            self.stat_cmds.append(body["Cmd"])
            return {"Id": "stat1"}
        if path.startswith("/exec/stat1/"):
            self.calls.append((method, path))
            if path.endswith("/start"):
                payload = b"440 1000 1000\n"
                return struct.pack(">BxxxI", 1, len(payload)) + payload
            return {"ExitCode": self.stat_exit}
        parts = path.split("/")
        if (self.fail_new_start and method == "POST" and path.endswith("/start")
                and parts[1] == "containers" and parts[2] != self.old_id and not self.containers.get(parts[2], {}).get("_render")):
            raise RuntimeError("Docker HTTP 500: cannot start")
        return super().request(method, path, body=body, headers=headers, raw=raw)


def binding(api, spec=SPEC):
    return config_binding(spec, api.by_name(NAME), api.images[OLD_ID], ComposeRenderer(api, spec, tag="app1"))[0]


def snapshot(api):
    return {"spec": SPEC, "resolved_args": {}, "config": binding(api), "execution": EXECUTION}


def reconciler(api, tmp_path, snap=None):
    return ConfigReconciler(api, snap or snapshot(api), JOB_ID, lock_root=tmp_path / "locks",
                            sleep=lambda _: None, clock=iter(range(0, 10000, 5)).__next__)


# ── catalog seed ────────────────────────────────────────────────────────────

def test_seed_disabled_fixed_target_and_no_arguments():
    assert not SEED["enabled"] and SEED["max_concurrent"] == 1 and SEED["risk_level"] == "high"
    assert json.loads(SEED["args_schema_json"]) == {"type": "object", "properties": {}, "additionalProperties": False}
    assert validate_spec(SEED["command_script"]) == SPEC
    assert SPEC["compose_config"] == DEPLOY_SPEC["compose_config"]
    for key, value in (("command", "sh"), ("container_name_from_arg", "target"),
                       ("display_values", {"XAI_API_KEY": ["x" * 40]}),
                       ("display_values", {"lower": ["on"]}),
                       ("stat_paths", ["/run/../etc/shadow"]), ("health_timeout_seconds", 5),
                       ("image_repository", "docker.io/evil/frontend")):
        with pytest.raises(ValueError):
            validate_spec(json.dumps({**SPEC, key: value}))


# ── worker: reconcile ───────────────────────────────────────────────────────

def test_drift_is_reconciled_at_the_same_image_and_values_stay_private(tmp_path):
    api = ConfigDocker(compose=DRIFTED)
    snap = snapshot(api)
    assert snap["config"]["drift"] == ["env:+BAWTHUB_AUTH_SESSIONS", "env:+XAI_API_KEY",
                                       "network:+bawthub_llm-bawt"]
    assert snap["config"]["display"] == {"BAWTHUB_AUTH_SESSIONS": "on"}
    assert snap["config"]["mounts"] == [f"/srv/compose/bawthub/token->{TOKEN_TARGET}:ro"]
    output = reconciler(api, tmp_path, snap).run()
    result = json.loads(output)
    assert result["schema"] == SCHEMA and result["phase"] == "done"
    assert result["previous"]["image_id"] == result["deployed"]["image_id"] == OLD_ID
    assert result["config"]["drift"] == snap["config"]["drift"]
    assert result["health"]["stat"] == {TOKEN_TARGET: "440 1000 1000"}
    assert api.stat_cmds == [["stat", "-c", "%a %u %g", TOKEN_TARGET]]
    body = api.create_bodies[0]
    assert body["Image"] == REF and f"XAI_API_KEY={SECRET}" in body["Env"]
    assert "BAWTHUB_AUTH_SESSIONS=on" in body["Env"]
    assert body["Labels"]["com.docker.compose.config-hash"] == CONFIG_HASH
    assert set(body["NetworkingConfig"]["EndpointsConfig"]) == {"bawthub_default", "bawthub_llm-bawt"}
    assert SECRET not in json.dumps(snap) and SECRET not in output
    assert not any("/images/create" in p for _, p in api.calls)
    assert api.by_name(NAME)["Id"] != api.old_id and not api.renderers_left()


def test_no_drift_verifies_only_and_recreates_nothing(tmp_path):
    api = ConfigDocker()
    result = json.loads(reconciler(api, tmp_path).run())
    assert result["unchanged"] and result["config"]["drift"] == []
    assert result["health"]["stat"] == {TOKEN_TARGET: "440 1000 1000"}
    assert not api.create_bodies and not api.side_effect_started


def test_stat_reports_missing_without_reading_contents(tmp_path):
    api = ConfigDocker(stat_exit=1)
    assert json.loads(reconciler(api, tmp_path).run())["health"]["stat"] == {TOKEN_TARGET: "missing"}


@pytest.mark.parametrize("change", ["container_env", "compose", "container_id", "image"])
def test_any_change_since_approval_refuses_before_create(tmp_path, change):
    api = ConfigDocker(compose=DRIFTED)
    snap = snapshot(api)
    old = api.by_name(NAME)
    if change == "container_env":
        old["Config"]["Env"].append("LATE=1")
    elif change == "compose":
        api.compose = compose_project(environment={"NODE_ENV": "production", "BAWTHUB_AUTH_SESSIONS": "off",
                                                   "XAI_API_KEY": SECRET},
                                      networks={"default": None, "llm-bawt": None})
    elif change == "container_id":
        old["Id"] = "b" * 64
    else:
        old["Image"] = "sha256:" + "9" * 64
    with pytest.raises(DeployFailed, match="changed.since.approval") as caught:
        reconciler(api, tmp_path, snap).run()
    assert caught.value.outcome_known and not api.create_bodies and not api.side_effect_started
    assert SECRET not in str(caught.value) and not api.renderers_left()


def test_unhealthy_target_is_not_reconciled(tmp_path):
    api = ConfigDocker(compose=DRIFTED)
    snap = snapshot(api)
    api.by_name(NAME)["State"]["Health"]["Status"] = "unhealthy"
    with pytest.raises(DeployFailed, match="running healthy"):
        reconciler(api, tmp_path, snap).run()
    assert not api.create_bodies


def test_render_failure_at_execution_changes_nothing(tmp_path):
    api = ConfigDocker(compose=DRIFTED)
    snap = snapshot(api)
    api.render_status = 1
    with pytest.raises(DeployFailed, match="renderer_failed") as caught:
        reconciler(api, tmp_path, snap).run()
    assert caught.value.outcome_known and not api.create_bodies


@pytest.mark.parametrize("old_restart_fails, known", [(False, True), (True, False)])
def test_failed_swap_restores_previous_container(tmp_path, old_restart_fails, known):
    api = ConfigDocker(compose=DRIFTED, fail_new_start=True, old_restart_fails=old_restart_fails)
    with pytest.raises(DeployFailed) as caught:
        reconciler(api, tmp_path).run()
    assert caught.value.outcome_known is known
    if known:
        assert caught.value.detail["restored"] and api.by_name(NAME)["Id"] == api.old_id


def test_worker_routes_config_action_and_writes_a_receipt(tmp_path):
    api = ConfigDocker(compose=DRIFTED)
    path, digest = _worker_request(tmp_path, snapshot(api))
    assert worker_run(path, digest, api_factory=lambda _: api, sleep=lambda _: None) == 0
    receipt = json.loads((path.parent / "receipt.json").read_text())
    assert receipt["state"] == "succeeded" and receipt["exit_code"] == 0
    assert json.loads(receipt["output_tail"])["schema"] == SCHEMA
    assert SECRET not in receipt["output_tail"]


# ── snapshot binding ────────────────────────────────────────────────────────

@pytest.mark.parametrize("tamper", [
    lambda s: s["config"].update(env={"XAI_API_KEY": SECRET}),
    lambda s: s["config"].update(drift=[1]),
    lambda s: s["config"]["source"].update(compose_files=["/etc/compose.yml"]),
    lambda s: s["config"].update(display={}),
    lambda s: s["config"].update(desired_hash="x"),
    lambda s: s.update(resolved_args={"image": "evil"}),
    lambda s: s["config"]["release"].update(version=""),
])
def test_binding_shape_is_validated(tamper):
    snap = snapshot(ConfigDocker(compose=DRIFTED))
    assert valid_config_binding(snap)
    tamper(snap)
    assert not valid_config_binding(snap)


def test_other_operations_may_not_carry_a_config_binding():
    assert valid_config_binding({"spec": {"action": "restart"}})
    assert not valid_config_binding({"spec": {"action": "restart"}, "config": {}})


def test_approval_snapshot_is_bound_and_rejects_smuggling():
    from llm_bawt.ops.service import OpsDispatchError
    from tests.test_ops_image_deploy import ARGS, _rehash, _service

    api = ConfigDocker(compose=DRIFTED)
    service, store, executor = _service()
    store.create_operation({**SEED, "enabled": True})
    executor.config_approval = lambda spec: binding(api, spec)
    approved = service.prepare_invocation(SEED["slug"], {})
    assert approved["config"] == binding(api)
    assert service._verify_snapshot(approved, SEED["slug"], {}) == approved
    approved["config"]["env"] = {"XAI_API_KEY": SECRET}
    with pytest.raises(OpsDispatchError, match="config binding mismatch"):
        service._verify_snapshot(_rehash(approved), SEED["slug"], {})
    deploy = service.prepare_invocation("bawthub.deploy-prod-image", ARGS)
    deploy["config"] = binding(api)
    with pytest.raises(OpsDispatchError, match="config binding mismatch"):
        service._verify_snapshot(_rehash(deploy), "bawthub.deploy-prod-image", ARGS)


def test_approval_refusal_codes_reach_the_caller():
    from llm_bawt.ops.service import OpsDispatchError
    from tests.test_ops_image_deploy import _service

    service, store, executor = _service()
    store.create_operation({**SEED, "enabled": True})

    def refuse(_spec):
        raise ConfigCheckError("unset_compose_variable", "config target cannot be bound: unset_compose_variable: XAI_API_KEY")

    executor.config_approval = refuse
    with pytest.raises(OpsDispatchError) as caught:
        service.prepare_invocation(SEED["slug"], {})
    assert caught.value.code == "unset_compose_variable" and "XAI_API_KEY" in str(caught.value)


# ── app executor: capability label, renderer image, read-only checks ───────

class _NotFound(Exception):
    status_code = 404


def _executor(api, *, worker_labels=None, renderer_present=True):
    pulls = []
    present = {"renderer": renderer_present}

    def get_image(ref):
        if ref == WORKER:
            labels = {WORKER_CAPABILITY_LABEL: "1"} if worker_labels is None else worker_labels
            return SimpleNamespace(attrs={"Config": {"Labels": labels}})
        if ref == RENDERER and not present["renderer"]:
            raise _NotFound(ref)
        return SimpleNamespace(attrs=api._find_image(ref))

    def pull(ref, auth_config=None):
        pulls.append((ref, auth_config))
        present["renderer"] = True

    def get_container(name):
        return SimpleNamespace(attrs=copy.deepcopy(api.by_name(name)), reload=lambda: None)

    client = SimpleNamespace(images=SimpleNamespace(get=get_image, pull=pull),
                             containers=SimpleNamespace(get=get_container), ping=lambda: True)
    executor = DockerExecutor(client_factory=lambda: client, worker_image=WORKER, receipt_volume="receipts",
                              receipt_root="/receipts", render_api_factory=lambda: api)
    executor._check_prerequisites = lambda _settings: None
    return executor, pulls


def test_executor_binds_the_same_public_material_and_pulls_the_renderer_anonymously():
    api = ConfigDocker(compose=DRIFTED)
    executor, pulls = _executor(api, renderer_present=False)
    assert executor.config_approval(SPEC) == binding(api)
    assert pulls == [(RENDERER, None)] and not api.renderers_left()


def test_executor_config_check_refuses_drift_by_name_at_the_given_ref():
    api = ConfigDocker(compose=DRIFTED)
    executor, _pulls = _executor(api)
    with pytest.raises(ConfigCheckError) as caught:
        executor.config_check(DEPLOY_SPEC, f"{REPO}@{DIGEST}")
    assert caught.value.code == "config_drift" and "env:+XAI_API_KEY" in str(caught.value)
    assert SECRET not in str(caught.value)
    assert api.render_bodies[0]["Env"] == [f"BAWTHUB_FRONTEND_IMAGE={REPO}@{DIGEST}"]
    api.compose = compose_project()
    assert executor.config_check(DEPLOY_SPEC, f"{REPO}@{DIGEST}") is None
    bare = {k: v for k, v in DEPLOY_SPEC.items() if k != "compose_config"}
    with pytest.raises(ConfigCheckError) as caught:
        executor.config_check(bare)
    assert caught.value.code == "config_check_unconfigured"


def test_legacy_worker_without_the_capability_fails_before_render_or_dispatch():
    api = ConfigDocker(compose=DRIFTED)
    executor, _pulls = _executor(api, worker_labels={"llm-bawt.ops.routing-reconcile": "1"})
    calls = (lambda: executor.config_approval(SPEC), lambda: executor.config_check(DEPLOY_SPEC),
             lambda: executor.preflight({"spec": SPEC, "execution": executor.execution_settings()}))
    for call in calls:
        with pytest.raises(ConfigCheckError) as caught:
            call()
        assert caught.value.code == "config_reconcile_unavailable"
    assert not api.render_bodies


def test_preflight_checks_prerequisites_without_rendering():
    api = ConfigDocker(compose=DRIFTED)
    executor, pulls = _executor(api, renderer_present=False)
    executor.preflight({"spec": SPEC, "execution": executor.execution_settings()})
    assert pulls == [(RENDERER, None)] and not api.render_bodies


def test_worker_image_carries_the_modules_and_capability_label():
    dockerfile = (Path(__file__).resolve().parents[1] / "docker" / "Dockerfile.ops-worker").read_text()
    for module in ("compose_render.py", "compose_config.py", "config_reconcile.py"):
        assert f"src/llm_bawt/ops/{module}" in dockerfile
    assert f'LABEL {WORKER_CAPABILITY_LABEL}="1"' in dockerfile


# ── r1 B1: external error text never reaches a receipt ─────────────────────
def _leaky(api, *, create=False, new_start=False, restore=False):
    """Docker errors that quote a rendered value, as a rejected request can."""
    original = api.request

    def request(method, path, *, body=None, headers=None, raw=False):
        parts = path.split("/")
        if create and method == "POST" and path.startswith("/containers/create?") \
                and any(e.startswith("XAI_API_KEY=") for e in body.get("Env", [])):
            raise RuntimeError(f"Docker HTTP 400: invalid environment XAI_API_KEY={SECRET}")
        if method == "POST" and path.endswith("/start") and parts[1] == "containers" \
                and not api.containers.get(parts[2], {}).get("_render"):
            if (new_start and parts[2] != api.old_id) or (restore and parts[2] == api.old_id):
                raise RuntimeError(f"Docker HTTP 500: rejected {SECRET}")
        return original(method, path, body=body, headers=headers, raw=raw)

    api.request = request
    return api


@pytest.mark.parametrize("leak, state, text", [
    ({"create": True}, "failed", "create failed (RuntimeError: Docker HTTP 400); nothing changed"),
    ({"new_start": True}, "failed", "previous container restored and healthy"),
    ({"new_start": True, "restore": True}, "lost", "RESTORE FAILED (RuntimeError: Docker HTTP 500)"),
])
def test_docker_error_quoting_a_rendered_value_never_reaches_the_receipt(tmp_path, leak, state, text):
    api = ConfigDocker(compose=DRIFTED)
    path, digest = _worker_request(tmp_path, snapshot(api))
    _leaky(api, **leak)
    assert worker_run(path, digest, api_factory=lambda _: api, sleep=lambda _: None) == 1
    receipt = json.loads((path.parent / "receipt.json").read_text())
    assert receipt["state"] == state and text in receipt["error"]
    assert SECRET not in json.dumps(receipt)
