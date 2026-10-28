"""TASK-1439 T2c: remove ONE stopped Compose-orphan container. No live Docker or network."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_bawt.ops.container_remove import (
    BINDING_KEYS, SCHEMA, WORKER_CAPABILITY_LABEL, ContainerRemover, RemoveRefused, approval_text,
    removal_binding, valid_remove_binding, validate_remove_schema,
)
from llm_bawt.ops.compose_render import RENDER_LABEL, ComposeRenderer
from llm_bawt.ops.executor import ConfigCheckError, DockerExecutor, validate_spec
from llm_bawt.ops.gpu_guard import gpu_voice_target
from llm_bawt.ops.worker import run as worker_run
from tests.test_ops_image_deploy import RENDERER, FakeDocker, _worker_request

PROJECT = "llm-bawt"
PROJECT_DIR = "/srv/compose/llm-bawt"
FILES = ["docker-compose.yml", "docker-compose.dev.yml"]
TARGET = "llm-bawt-old-bridge"
TARGET_ID = "a1" * 32
JOB_ID = "f" * 32
WORKER = "sha256:" + "0" * 64
SPEC = {"action": "remove", "container_name_from_arg": "container", "compose_project": PROJECT,
        "compose_config": {"compose_files": FILES, "renderer_image": RENDERER}}
SCHEMA_JSON = {"type": "object", "required": ["container"], "additionalProperties": False,
               "properties": {"container": {"type": "string", "enum": [TARGET]}}}
ARGS = {"container": TARGET}
EXECUTION = {"start_delay_seconds": 0, "timeout_seconds": 60, "max_output_bytes": 65536}
ROW = {"slug": "test.remove-old-bridge", "title": "Remove the old bridge container",
       "description": "One stopped orphan.", "enabled": True, "executor_kind": "docker", "target_host": "",
       "working_directory": None, "command_script": json.dumps(SPEC), "args_schema_json": SCHEMA_JSON,
       "args_defaults_json": {}, "timeout_seconds": 300, "start_delay_seconds": 0,
       "max_output_bytes": 65536, "max_concurrent": 1, "risk_level": "high"}


def project(**services):
    """``compose config --format json`` of the whole project: the target's
    service is gone; the app and a profiled worker remain."""
    base = {"app": {"image": "llm-bawt-app", "container_name": "llm-bawt-app"},
            "local-model-bridge": {"image": "lmb", "profiles": ["gpu"]}}
    base.update(services)
    return {"name": PROJECT, "services": base}


class RemoveDocker(FakeDocker):
    """The fake Engine plus one stopped Compose orphan and a scripted DELETE."""

    def __init__(self, *, state="exited", labels=None, delete_error=None, compose=None, **kwargs):
        super().__init__(compose=project() if compose is None else compose, **kwargs)
        self.deletes: list[str] = []
        self.delete_error = delete_error
        self.containers[TARGET_ID] = {
            "Id": TARGET_ID, "Name": f"/{TARGET}", "Image": "sha256:" + "7" * 64,
            "Config": {"Labels": {"com.docker.compose.project": PROJECT,
                                  "com.docker.compose.service": "old-bridge",
                                  "com.docker.compose.project.working_dir": PROJECT_DIR,
                                  # Stale: only the files at its creation. Never used.
                                  "com.docker.compose.project.config_files": f"{PROJECT_DIR}/docker-compose.yml",
                                  **(labels or {})}},
            "HostConfig": {}, "State": {"Running": state == "running", "Status": state}}

    def attach_stream(self, container_id):
        from tests.test_ops_image_deploy import _Stream, _frame

        self.calls.append(("POST", f"/containers/{container_id}/attach"))
        return _Stream(_frame(1, json.dumps(self.compose)) + (
            _frame(2, self.render_stderr) if self.render_stderr else b""))

    def request(self, method, path, *, body=None, headers=None, raw=False):
        if method == "DELETE" and path.startswith(f"/containers/{TARGET_ID}"):
            self.calls.append((method, path))
            self.deletes.append(path)
            if self.delete_error:
                raise self.delete_error
            self.containers.pop(TARGET_ID)
            return None
        return super().request(method, path, body=body, headers=headers, raw=raw)

    def target(self):
        return self.containers[TARGET_ID]


def binding(api, spec=SPEC):
    return removal_binding(spec, TARGET, api.target(), ComposeRenderer(api, spec, tag="t1"))


def snapshot(api):
    return {"spec": SPEC, "resolved_args": ARGS, "remove": binding(api), "execution": EXECUTION}


def remover(api, tmp_path, snap=None):
    return ContainerRemover(api, snap or snapshot(api), JOB_ID, lock_root=tmp_path / "locks")


# ── spec and catalog row ────────────────────────────────────────────────────

@pytest.mark.parametrize("change", [
    lambda s: s.update(force=True),
    lambda s: s.update(volumes=True),
    lambda s: s.update(container_name=TARGET),
    lambda s: s.pop("compose_project"),
    lambda s: s.update(compose_project="Bad Project"),
    lambda s: s.update(container_name_from_arg="1bad"),
    lambda s: s["compose_config"].update(compose_files=[]),
    lambda s: s["compose_config"].update(compose_files=["../x.yml"]),
    lambda s: s["compose_config"].update(compose_files=["a.yml"] * 2),
    lambda s: s["compose_config"].update(compose_files=[f"f{i}.yml" for i in range(5)]),
    lambda s: s["compose_config"].update(renderer_image="docker:latest"),
    lambda s: s["compose_config"].update(image_variable="X"),
])
def test_spec_is_closed(change):
    spec = copy.deepcopy(SPEC)
    change(spec)
    with pytest.raises(ValueError):
        validate_spec(json.dumps(spec))
    assert validate_spec(json.dumps(SPEC)) == SPEC


@pytest.mark.parametrize("change", [
    lambda s: s.pop("additionalProperties"),
    lambda s: s.update(required=[]),
    lambda s: s["properties"]["container"].pop("enum"),
    lambda s: s["properties"]["container"].update(enum=[]),
    lambda s: s["properties"]["container"].update(enum=["bad name"]),
    lambda s: s["properties"]["container"].update(type="integer"),
])
def test_the_args_enum_is_the_required_allowlist(change):
    schema = copy.deepcopy(SCHEMA_JSON)
    validate_remove_schema(SPEC, schema)
    change(schema)
    with pytest.raises(ValueError, match="enum"):
        validate_remove_schema(SPEC, schema)


def test_store_rejects_a_row_without_the_allowlist():
    from tests.test_ops_image_deploy import _service

    _service_, store, _executor = _service()
    open_schema = {**SCHEMA_JSON, "properties": {"container": {"type": "string"}}}
    with pytest.raises(ValueError, match="enum"):
        store.create_operation({**ROW, "args_schema_json": open_schema})
    store.create_operation(ROW)


# ── the refusals (read-only, nothing deleted) ───────────────────────────────

@pytest.mark.parametrize("api_kwargs, code", [
    ({"state": "running"}, "target_running"),
    ({"state": "paused"}, "target_running"),
    ({"labels": {RENDER_LABEL: "1"}}, "ops_container"),
    ({"labels": {"llm-bawt.ops.job": JOB_ID}}, "ops_container"),
    ({"labels": {"com.docker.compose.project": "other"}}, "compose_project_mismatch"),
    ({"labels": {"com.docker.compose.project.working_dir": "/srv/../etc"}}, "compose_source_invalid"),
    # Still declared, by its own service key...
    ({"compose": project(**{"old-bridge": {"image": "x"}})}, "compose_declared"),
    # ...by a RENAMED service keeping the container_name, only in the 2nd overlay...
    ({"compose": project(bridge2={"image": "x", "container_name": TARGET})}, "compose_declared"),
    # ...or by Compose's default <project>-<service>-<n> name.
    ({"compose": project(old={"image": "x"}), "labels": {"com.docker.compose.service": "gone"}},
     "compose_declared"),
    ({"compose": project(bad={"container_name": 7})}, "compose_unverifiable"),
    ({"compose": {"name": PROJECT}}, "compose_unverifiable"),
    ({"render_status": 1}, "compose_unverifiable"),  # e.g. a missing overlay file
    ({"render_stderr": "something unexpected\n"}, "compose_unverifiable"),
])
def test_refusals_are_known_and_delete_nothing(api_kwargs, code):
    api = RemoveDocker(**api_kwargs)
    if code == "compose_declared" and "old" in api.compose["services"]:
        api.target()["Name"] = f"/{PROJECT}-old-1"
        name = f"{PROJECT}-old-1"
    else:
        name = TARGET
    with pytest.raises(RemoveRefused) as caught:
        removal_binding(SPEC, name, api.target(), ComposeRenderer(api, SPEC, tag="t1"))
    assert caught.value.code == code and caught.value.outcome_known
    assert not api.deletes and not api.side_effect_started and not api.renderers_left()


def test_missing_compose_labels_refuse_before_rendering():
    api = RemoveDocker()
    del api.target()["Config"]["Labels"]["com.docker.compose.project.working_dir"]
    with pytest.raises(RemoveRefused) as caught:
        binding(api)
    assert caught.value.code == "compose_labels_missing" and not api.render_bodies


def test_render_uses_the_rows_files_every_profile_and_no_injected_env():
    """Not the container's stale config_files label: the row's current files."""
    api = RemoveDocker()
    result = binding(api)
    assert result == {"container_id": TARGET_ID, "name": TARGET, "compose_project": PROJECT,
                      "project_dir": PROJECT_DIR, "compose_files": [f"{PROJECT_DIR}/{f}" for f in FILES],
                      "status": "exited"}
    (body,) = api.render_bodies
    assert body["Cmd"] == ["compose", "-p", PROJECT, "-f", f"{PROJECT_DIR}/docker-compose.yml",
                           "-f", f"{PROJECT_DIR}/docker-compose.dev.yml",
                           "--profile", "*", "config", "--format", "json"]
    assert body["Env"] == [] and body["Image"] == RENDERER
    assert body["Entrypoint"][:2] == ["env", "-i"] and body["Entrypoint"][-1] == "docker"


# ── the worker ──────────────────────────────────────────────────────────────

def test_removes_exactly_one_container_unforced_and_keeps_volumes(tmp_path):
    api = RemoveDocker()
    snap = snapshot(api)
    out = json.loads(remover(api, tmp_path, snap).run())
    assert out == {"schema": SCHEMA, "job_id": JOB_ID, "removed": TARGET_ID, "name": TARGET}
    assert api.deletes == [f"/containers/{TARGET_ID}?force=false&v=false&link=false"]
    # The target itself sees one inspect and one DELETE; nothing else is touched.
    touched = [(m, p) for m, p in api.calls if TARGET in p or TARGET_ID in p]
    assert touched == [("GET", f"/containers/{TARGET}/json"), ("DELETE", api.deletes[0])]
    assert not api.create_bodies and not api.renderers_left()  # only the sealed renderer was created
    assert (tmp_path / "locks" / f"{TARGET}.lock").exists()


@pytest.mark.parametrize("mutate, code", [
    (lambda api: api.target()["State"].update(Running=True, Status="running"), "target_running"),
    (lambda api: api.containers.pop(TARGET_ID), "not_found"),
    (lambda api: api.containers.update({"b2" * 32: {**api.containers.pop(TARGET_ID), "Id": "b2" * 32}}),
     "target_replaced"),
    (lambda api: api.compose["services"].update(back={"container_name": TARGET}), "compose_declared"),
    (lambda api: setattr(api, "render_status", 1), "compose_unverifiable"),
])
def test_worker_rechecks_everything_after_approval(tmp_path, mutate, code):
    api = RemoveDocker()
    snap = snapshot(api)
    mutate(api)
    with pytest.raises(RemoveRefused) as caught:
        remover(api, tmp_path, snap).run()
    assert caught.value.code == code
    assert not api.deletes and not api.side_effect_started


def test_a_held_target_lock_is_busy(tmp_path):
    import fcntl

    api = RemoveDocker()
    (tmp_path / "locks").mkdir()
    with (tmp_path / "locks" / f"{TARGET}.lock").open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        with pytest.raises(RemoveRefused) as caught:
            remover(api, tmp_path).run()
    assert caught.value.code == "target_busy" and not api.deletes


@pytest.mark.parametrize("error, code", [
    (RuntimeError("Docker HTTP 409: container is running"), "target_running"),
    (RuntimeError("Docker HTTP 404: no such container"), "not_found"),
])
def test_docker_refusing_the_delete_is_a_known_outcome(tmp_path, error, code):
    api = RemoveDocker(delete_error=error)
    with pytest.raises(RemoveRefused) as caught:
        remover(api, tmp_path).run()
    assert caught.value.code == code and caught.value.outcome_known


@pytest.mark.parametrize("delete_error, state, unknown", [
    (None, "succeeded", False),
    (RuntimeError("Docker HTTP 409: container is running"), "failed", False),
    (ConnectionResetError("reply lost"), "lost", True),
])
def test_worker_routes_remove_and_maps_outcomes(tmp_path, delete_error, state, unknown):
    api = RemoveDocker(delete_error=delete_error)
    path, digest = _worker_request(tmp_path, snapshot(api))
    worker_run(path, digest, api_factory=lambda _: api, sleep=lambda _: None)
    receipt = json.loads((path.parent / "receipt.json").read_text())
    assert receipt["state"] == state
    assert receipt.get("side_effect_unknown", False) is unknown
    if state == "succeeded":
        assert json.loads(receipt["output_tail"])["removed"] == TARGET_ID


# ── snapshot binding ────────────────────────────────────────────────────────

@pytest.mark.parametrize("tamper", [
    lambda s: s["remove"].update(container_id="x"),
    lambda s: s["remove"].update(name="llm-bawt-app"),
    lambda s: s.update(resolved_args={"container": "llm-bawt-app"}),
    lambda s: s["remove"].update(compose_project="other"),
    lambda s: s["remove"].update(compose_files=["/etc/compose.yml"]),
    lambda s: s["remove"].update(project_dir="/"),
    lambda s: s["remove"].update(status="running"),
    lambda s: s["remove"].update(force=True),
    lambda s: s.pop("remove"),
])
def test_binding_shape_is_validated(tamper):
    snap = snapshot(RemoveDocker())
    assert set(snap["remove"]) == BINDING_KEYS and valid_remove_binding(snap)
    tamper(snap)
    assert not valid_remove_binding(snap)


def test_worker_refuses_a_tampered_binding(tmp_path):
    api = RemoveDocker()
    snap = snapshot(api)
    snap["remove"]["name"] = "llm-bawt-app"
    with pytest.raises(RemoveRefused) as caught:
        remover(api, tmp_path, snap)
    assert caught.value.code == "args_invalid" and not api.deletes


def test_other_operations_may_not_carry_a_removal_binding():
    assert valid_remove_binding({"spec": {"action": "restart"}})
    assert not valid_remove_binding({"spec": {"action": "restart"}, "remove": {}})


def test_approval_snapshot_is_bound_and_rejects_smuggling():
    from llm_bawt.ops.service import OpsDispatchError
    from tests.test_ops_image_deploy import ARGS as DEPLOY_ARGS, _rehash, _service

    api = RemoveDocker()
    service, store, executor = _service()
    store.create_operation(ROW)
    executor.remove_approval = lambda spec, args: binding(api, spec)
    approved = service.prepare_invocation(ROW["slug"], ARGS)
    assert approved["remove"] == binding(api) and approved["gpu_target"] is None
    assert service._verify_snapshot(approved, ROW["slug"], ARGS) == approved
    approved["remove"]["container_id"] = "b2" * 32
    approved["remove"]["name"] = "llm-bawt-app"
    with pytest.raises(OpsDispatchError, match="removal binding mismatch"):
        service._verify_snapshot(_rehash(approved), ROW["slug"], ARGS)
    deploy = service.prepare_invocation("bawthub.deploy-prod-image", DEPLOY_ARGS)
    deploy["remove"] = binding(api)
    with pytest.raises(OpsDispatchError, match="removal binding mismatch"):
        service._verify_snapshot(_rehash(deploy), "bawthub.deploy-prod-image", DEPLOY_ARGS)
    with pytest.raises(OpsDispatchError) as caught:
        service.prepare_invocation(ROW["slug"], {"container": "llm-bawt-app"})
    assert caught.value.code == "args_invalid"


def test_approval_refusal_codes_reach_the_caller():
    from llm_bawt.ops.service import OpsDispatchError
    from tests.test_ops_image_deploy import _service

    service, store, executor = _service()
    store.create_operation(ROW)

    def refuse(_spec, _args):
        raise ConfigCheckError("compose_declared", "compose_declared: still declared by service bridge2")

    executor.remove_approval = refuse
    with pytest.raises(OpsDispatchError) as caught:
        service.prepare_invocation(ROW["slug"], ARGS)
    assert caught.value.code == "compose_declared"


def test_gpu_guard_leaves_remove_ungated():
    assert gpu_voice_target(SPEC, ARGS, executor=None) is None


def test_approval_text_names_the_one_container_and_the_files():
    text = approval_text(binding(RemoveDocker()), None)
    assert TARGET in text and TARGET_ID[:12] in text and "docker-compose.dev.yml" in text
    assert "volumes kept" in text


# ── app executor ────────────────────────────────────────────────────────────

class _NotFound(Exception):
    status_code = 404


def _executor(api, *, worker_labels=None, renderer_present=True, pulls=None):
    def get_image(ref):
        if ref == WORKER:
            labels = {WORKER_CAPABILITY_LABEL: "1"} if worker_labels is None else worker_labels
            return SimpleNamespace(attrs={"Config": {"Labels": labels}})
        if ref == RENDERER and not renderer_present:
            raise _NotFound(ref)
        return SimpleNamespace(attrs=api._find_image(ref))

    def pull(ref, *args, **kwargs):
        if pulls is None:
            raise AssertionError("a removal must never pull an image")
        pulls.append(ref)

    def get_container(name):
        try:
            return SimpleNamespace(attrs=copy.deepcopy(api.by_name(name)), reload=lambda: None)
        except RuntimeError:
            raise _NotFound(name) from None

    client = SimpleNamespace(images=SimpleNamespace(get=get_image, pull=pull),
                             containers=SimpleNamespace(get=get_container), ping=lambda: True)
    executor = DockerExecutor(client_factory=lambda: client, worker_image=WORKER, receipt_volume="receipts",
                              receipt_root="/receipts", render_api_factory=lambda: api)
    executor._check_prerequisites = lambda _settings: None
    executor.self_target = lambda _snapshot: None  # Not under test here (TASK-1169 S4c).
    return executor


def test_executor_binds_the_same_public_material():
    api = RemoveDocker()
    assert _executor(api).remove_approval(SPEC, ARGS) == binding(api)
    assert not api.renderers_left() and not api.deletes


def test_executor_refusals_carry_fixed_codes():
    api = RemoveDocker(state="running")
    with pytest.raises(ConfigCheckError) as caught:
        _executor(api).remove_approval(SPEC, ARGS)
    assert caught.value.code == "target_running"
    api.containers.pop(TARGET_ID)
    with pytest.raises(ConfigCheckError) as caught:
        _executor(api).remove_approval(SPEC, ARGS)
    assert caught.value.code == "not_found"


@pytest.mark.parametrize("state", ["exited", "running"])
def test_a_missing_renderer_refuses_without_pulling(state):
    """r1 B1: neither approval nor preflight may pull an image for a removal."""
    api = RemoveDocker(state=state)
    pulls = []
    executor = _executor(api, renderer_present=False, pulls=pulls)
    for call in (lambda: executor.remove_approval(SPEC, ARGS),
                 lambda: executor.preflight({"spec": SPEC, "execution": executor.execution_settings()})):
        with pytest.raises(ConfigCheckError) as caught:
            call()
        assert caught.value.code == "renderer_image_missing"
    assert pulls == [] and not api.render_bodies and not api.deletes


def test_an_old_worker_without_the_capability_is_refused_before_any_render():
    from llm_bawt.ops.executor import ExecutorError

    api = RemoveDocker()
    executor = _executor(api, worker_labels={"llm-bawt.ops.config-reconcile": "1"})
    for call in (lambda: executor.remove_approval(SPEC, ARGS),
                 lambda: executor.preflight({"spec": SPEC, "execution": executor.execution_settings()})):
        with pytest.raises(ExecutorError, match=WORKER_CAPABILITY_LABEL):
            call()
    assert not api.render_bodies


def test_worker_image_carries_the_module_and_capability_label():
    root = Path(__file__).resolve().parents[1]
    dockerfile = (root / "docker" / "Dockerfile.ops-worker").read_text()
    assert "src/llm_bawt/ops/container_remove.py" in dockerfile
    assert f'LABEL {WORKER_CAPABILITY_LABEL}="1"' in dockerfile

