"""TASK-1649: same-image reconcile of a Compose-built DEV service (the BawtHub
hot-reload frontend) with its memory cap. No live Docker or network."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from llm_bawt.ops.app_config_reconcile import PROCESS_PROFILE_LABEL
from llm_bawt.ops.compose_config import BUILT_PROFILE, DEV_PROFILE, apply_desired, translate
from llm_bawt.ops.compose_render import ComposeError, ComposeRenderer
from llm_bawt.ops.config_reconcile import WORKER_CAPABILITY_LABEL as CONFIG_WORKER_LABEL, valid_config_binding
from llm_bawt.ops.dev_config_reconcile import (
    ACTION, DEV_TARGET_LABEL, SCHEMA, WORKER_CAPABILITY_LABEL as DEV_WORKER_LABEL, DevConfigReconciler,
    approval_text,
    dev_config_binding, memory_text, valid_dev_binding,
)
from llm_bawt.ops.executor import ConfigCheckError, validate_spec
from llm_bawt.ops.image_deploy import DeployFailed
from llm_bawt.ops.seeds import SEEDS
from llm_bawt.ops.service import APPROVAL_ONLY_ACTIONS, OpsDispatchError, _valid_config_snapshot
from llm_bawt.ops.tagged_config_reconcile import valid_tagged_binding
from llm_bawt.ops.worker import run as worker_run
from tests.test_ops_app_config_reconcile import (  # noqa: F401  (source: autouse stub of the sealed git check)
    APP_NAME, EXECUTION, HEAD, REDIS, SEED as APP_SEED, AppDocker, PublishedVerifier, _executor, _image,
    _service, _write_request, source,
)
from tests.test_ops_image_deploy import JOB_ID, RENDERER, RENDERER_ID

SEED = next(s for s in SEEDS if s["slug"] == "bawthub.reconcile-dev-frontend-config")
SEED_SPEC = json.loads(SEED["command_script"])
NAME = "bawthub-frontend-1"
DEV_DIR = "/srv/compose/bawthub"
FILES = [f"{DEV_DIR}/docker-compose.yml"]
TAG = "bawthub-frontend:latest"
DEV_ID, NEWER = "sha256:" + "1" * 64, "sha256:" + "2" * 64  # never the renderer fixture's ID
GIB4 = 4 * 1024 ** 3
SECOND = 1_000_000_000
SECRET = "/srv/private/never-shown"
SPEC = {**SEED_SPEC, "compose_config": {"renderer_image": RENDERER}}
HEALTH = {"test": ["CMD-SHELL", "curl -s -o /dev/null -m 8 http://localhost:5173/; [ $? != 7 ]"],
          "interval": "30s", "timeout": "12s", "start_period": "2m0s", "retries": 3}
DOCKER_HEALTH = {"Test": list(HEALTH["test"]), "Interval": 30 * SECOND, "Timeout": 12 * SECOND,
                 "StartPeriod": 120 * SECOND, "Retries": 3}
DEV_WORKER = {CONFIG_WORKER_LABEL: "1", PROCESS_PROFILE_LABEL: "1", DEV_WORKER_LABEL: "1"}


def _bind(source_path, target):
    return {"type": "bind", "source": source_path, "target": target, "bind": {"create_host_path": True}}


def dev_compose(**service):
    """``docker compose config --format json frontend`` for a dev block: no drift by default."""
    body = {"build": {"context": DEV_DIR, "dockerfile": "Dockerfile.dev"}, "image": TAG, "pull_policy": "build",
            "command": ["pnpm", "dev"], "entrypoint": None, "environment": {"NODE_ENV": "development"},
            "labels": {DEV_TARGET_LABEL: "1"},
            "healthcheck": dict(HEALTH), "mem_limit": str(GIB4), "memswap_limit": str(GIB4),
            "networks": {"default": None}, "restart": "unless-stopped",
            "volumes": [_bind(DEV_DIR, "/app"), _bind("/srv/dev", "/repos")]}
    body.update(service)
    return {"name": "bawthub", "services": {"frontend": {k: v for k, v in body.items() if v is not None}},
            "networks": {"default": {"name": "bawthub_default"}}}


class DevDocker(AppDocker):
    """The dev frontend as Compose built it, plus the app fixture's redis."""

    def __init__(self, *, compose=None, tag_moved=False, memory=GIB4, swap=GIB4, **kwargs):
        super().__init__(compose=dev_compose() if compose is None else compose,
                         **{"health_status": "healthy", **kwargs})
        image = _image(DEV_ID, [] if tag_moved else [TAG])
        image["Config"] = {"Env": ["PATH=/usr/local/bin:/usr/bin", "NODE_VERSION=22.0.0"],
                           "Entrypoint": ["docker-entrypoint.sh"], "Cmd": ["node"], "Labels": {}}
        self.images = {DEV_ID: image, RENDERER_ID: {"Id": RENDERER_ID, "RepoDigests": [RENDERER], "Config": {}}}
        if tag_moved:
            self.images[NEWER] = _image(NEWER, [TAG])
        old = "de" * 32  # never a FakeDocker sequence ID
        self.containers = {old: {
            "Id": old, "Name": f"/{NAME}", "Image": DEV_ID,
            "Config": {"Image": TAG, "Env": ["PATH=/usr/local/bin:/usr/bin", "NODE_VERSION=22.0.0",
                                             "NODE_ENV=development"],
                       "Cmd": ["pnpm", "dev"], "Entrypoint": ["docker-entrypoint.sh"],
                       "Healthcheck": dict(DOCKER_HEALTH),
                       "Labels": {"com.docker.compose.project": "bawthub", "com.docker.compose.service": "frontend",
                                  "com.docker.compose.project.working_dir": DEV_DIR,
                                  "com.docker.compose.project.config_files": ",".join(FILES),
                                  DEV_TARGET_LABEL: "1"}},
            "HostConfig": {"RestartPolicy": {"Name": "unless-stopped"}, "Memory": memory, "MemorySwap": swap,
                           "Binds": [f"{DEV_DIR}:/app:rw", "/srv/dev:/repos:rw"]},
            "NetworkSettings": {"Networks": {"bawthub_default": {"Aliases": [NAME, "frontend", old[:12]]}}},
            "State": {"Running": True, "Status": "running", "Health": {"Status": "healthy"}}},
            REDIS: self.containers[REDIS]}
        self.old_id = old

    def request(self, method, path, *, body=None, headers=None, raw=False):
        # The seed's stat_paths: mode/owner of the service token, never its contents.
        if path.startswith("/exec/stat1/"):
            self.calls.append((method, path))
            if path.endswith("/start"):
                return struct.pack(">BxxxI", 1, 14) + b"600 1000 1000\n"
            return {"ExitCode": 0}
        return super().request(method, path, body=body, headers=headers, raw=raw)


def binding(api, spec=SPEC, *, moved=False):
    return dev_config_binding(spec, api.by_name(NAME), api.images[DEV_ID],
                              ComposeRenderer(api, spec, tag="dev1"), tag_moved=lambda _tag: moved)[0]


def snapshot(api, **kwargs):
    return {"spec": SPEC, "resolved_args": {}, "config": binding(api, **kwargs), "execution": EXECUTION}


def run(api, tmp_path, snap=None):
    return json.loads(DevConfigReconciler(api, snap or snapshot(api), JOB_ID, lock_root=tmp_path / "locks",
                                          sleep=lambda _: None, clock=iter(range(0, 10000, 5)).__next__).run())


def _with_env(**extra):
    return dev_compose(environment={"NODE_ENV": "development", **extra})


# ── catalog seed and spec ──────────────────────────────────────────────────

def test_seed_is_disabled_dev_only_and_validates():
    assert not SEED["enabled"] and SEED["max_concurrent"] == 1 and SEED["category"] == "configuration"
    assert validate_spec(SEED["command_script"]) == SEED_SPEC and SEED_SPEC["action"] == ACTION
    assert "image_tag" not in SEED_SPEC and SEED_SPEC["display_values"] == {}
    assert "dev only" in SEED["approval_prompt_prefix"] and "No build or pull" in SEED["description"]


@pytest.mark.parametrize("change", [
    {"image_tag": TAG}, {"source_repository": "bawthub/bawthub"}, {"env_names": []},
    {"health_url": "http://evil.example:5173/api/health"}, {"action": "reconcile_app_config"},
    {"compose_config": {"renderer_image": "docker:latest"}}, {"stat_paths": ["/../etc"]},
])
def test_spec_refuses_foreign_or_unsafe_fields(change):
    with pytest.raises(ValueError):
        validate_spec(json.dumps({**SPEC, **change}))


# ── the memory cap ─────────────────────────────────────────────────────────

def _memory(**service):
    return translate(dev_compose(**service), SPEC, TAG, DEV_PROFILE)["memory"]


def test_memory_is_both_or_neither_and_bounded():
    assert _memory() == {"Memory": GIB4, "MemorySwap": GIB4}
    assert _memory(memswap_limit=-1) == {"Memory": GIB4, "MemorySwap": -1}
    assert _memory(mem_limit=None, memswap_limit=None) == {"Memory": 0, "MemorySwap": 0}
    for service, names in (({"memswap_limit": None}, ["memswap_limit"]), ({"mem_limit": None}, ["mem_limit"]),
                           ({"mem_limit": "4g"}, ["mem_limit"]), ({"mem_limit": 0}, ["mem_limit"]),
                           ({"mem_limit": True}, ["mem_limit"]), ({"mem_limit": str(1 << 51)}, ["mem_limit"]),
                           ({"memswap_limit": str(GIB4 - 1)}, ["memswap_limit"]),
                           ({"memswap_limit": "-2"}, ["memswap_limit"])):
        with pytest.raises(ComposeError) as caught:
            _memory(**service)
        assert caught.value.code == "unsupported_compose_key" and caught.value.names == names


def test_only_the_dev_profile_models_memory():
    with pytest.raises(ComposeError) as caught:
        translate(dev_compose(), SPEC, TAG, BUILT_PROFILE)
    assert caught.value.code == "unsupported_compose_key" and set(caught.value.names) == {"mem_limit",
                                                                                          "memswap_limit"}
    assert "memory" not in translate(dev_compose(mem_limit=None, memswap_limit=None), SPEC, TAG, BUILT_PROFILE)


def test_memory_text():
    assert memory_text({"Memory": 0, "MemorySwap": 0}) == "unlimited"
    assert memory_text({"Memory": GIB4, "MemorySwap": -1}) == f"{GIB4} bytes, swap unlimited"
    assert memory_text({"Memory": GIB4, "MemorySwap": GIB4}) == f"{GIB4} bytes, swap {GIB4} bytes"


# ── binding ─────────────────────────────────────────────────────────────────

def test_no_drift_binds_the_running_image_and_verifies_only(tmp_path):
    api = DevDocker()
    snap = snapshot(api)
    bound = snap["config"]
    assert bound["drift"] == [] and bound["image_id"] == DEV_ID and bound["image_ref"] == TAG
    assert bound["memory"] == f"{GIB4} bytes, swap {GIB4} bytes" and bound["source"]["compose_files"] == FILES
    assert valid_dev_binding(snap) and _valid_config_snapshot(snap)
    result = run(api, tmp_path, snap)
    assert result["schema"] == SCHEMA and not api.create_bodies


def test_env_drift_recreates_at_the_running_image_and_keeps_the_cap(tmp_path):
    api = DevDocker(compose=_with_env(REPOS_HOST_ROOT=SECRET))
    snap = snapshot(api)
    assert snap["config"]["drift"] == ["env:+REPOS_HOST_ROOT"]
    result = run(api, tmp_path, snap)
    assert result["schema"] == SCHEMA and result["phase"] == "done" and result["deployed"]["image_id"] == DEV_ID
    body = api.create_bodies[0]
    assert body["Image"] == DEV_ID and f"REPOS_HOST_ROOT={SECRET}" in body["Env"]
    assert body["HostConfig"]["Memory"] == GIB4 and body["HostConfig"]["MemorySwap"] == GIB4
    assert body["Healthcheck"] == {**DOCKER_HEALTH, "StartInterval": 0} and body["Cmd"] == ["pnpm", "dev"]
    assert api.exec_cmds[-2:] == [["curl", "-fsS", "--max-time", "10", SPEC["health_url"]],
                                  ["stat", "-c", "%a %u %g", SPEC["stat_paths"][0]]]
    assert result["health"]["stat"] == {SPEC["stat_paths"][0]: "600 1000 1000"}
    assert not any("/images/create" in p or "/build" in p for _m, p in api.calls) and not api.renderers_left()


def test_a_changed_cap_is_drift_and_applied(tmp_path):
    api = DevDocker(memory=2 * 1024 ** 3, swap=2 * 1024 ** 3)
    snap = snapshot(api)
    assert snap["config"]["drift"] == ["memory:~"]
    run(api, tmp_path, snap)
    assert api.create_bodies[0]["HostConfig"]["Memory"] == GIB4


def test_apply_desired_clears_a_dropped_cap():
    desired = translate(dev_compose(mem_limit=None, memswap_limit=None), SPEC, TAG, DEV_PROFILE)
    old = DevDocker().by_name(NAME)
    body = {"Image": DEV_ID, "Env": [], "Labels": {}, "HostConfig": {"Memory": GIB4, "MemorySwap": GIB4}}
    out = apply_desired(body, old, desired, container_name=NAME, service="frontend",
                        rendered_hash="h" * 64, image=DevDocker().images[DEV_ID])
    assert out["HostConfig"]["Memory"] == 0 and out["HostConfig"]["MemorySwap"] == 0


def test_a_foreign_running_image_refuses():
    api = DevDocker()
    api.by_name(NAME)["Config"]["Image"] = "ghcr.io/x/frontend:1"
    with pytest.raises(ValueError, match="Compose-built image"):
        binding(api)


def test_card_shows_names_memory_and_no_values():
    bound = binding(DevDocker(compose=_with_env(REPOS_HOST_ROOT=SECRET)))
    text = approval_text(SPEC, bound, SEED["approval_prompt_prefix"])
    assert SECRET not in json.dumps(bound) + text and "env:+REPOS_HOST_ROOT" in text
    assert f"Memory after: {GIB4} bytes, swap {GIB4} bytes" in text and "no source check runs" in text
    assert f"DEV target (its Compose service declares {DEV_TARGET_LABEL})" in text
    assert text.startswith(SEED["approval_prompt_prefix"]) and "now names" not in text
    assert "now names a different local image" in approval_text(SPEC, {**bound, "image_tag_moved": True}, None)


def test_moved_tag_is_shown_and_the_running_image_still_runs(tmp_path):
    api = DevDocker(compose=_with_env(REPOS_HOST_ROOT=SECRET), tag_moved=True)
    approved = _executor(api, worker_labels=DEV_WORKER).config_approval(SPEC)
    assert approved["image_tag_moved"] is True and approved["drift"] == ["env:+REPOS_HOST_ROOT"]
    run(api, tmp_path, {"spec": SPEC, "resolved_args": {}, "config": approved, "execution": EXECUTION})
    assert api.create_bodies[0]["Image"] == DEV_ID


@pytest.mark.parametrize("tamper", [
    lambda s: s["config"].update(env={"REPOS_HOST_ROOT": SECRET}),
    lambda s: s["config"].update(memory="lots"),
    lambda s: s["config"].update(memory=GIB4),
    lambda s: s["config"].pop("memory"),
    lambda s: s["config"].update(image_tag_moved="no"),
    lambda s: s["config"].update(image_tag="ghcr.io/evil/frontend:1"),
    lambda s: s["config"].update(image_ref="evil-frontend"),
    lambda s: s["config"].update(tmpfs=["data"]),
    lambda s: s["config"]["source"].update(compose_files=["/etc/compose.yml"]),
    lambda s: s["config"].update(display={"X": "1"}),
    lambda s: s.update(resolved_args={"x": 1}),
    lambda s: s.update(spec={**SPEC, "image_tag": TAG}),
])
def test_binding_shape_is_validated(tamper):
    snap = snapshot(DevDocker())
    assert valid_dev_binding(snap) and _valid_config_snapshot(snap)
    tamper(snap)
    assert not valid_dev_binding(snap) and not _valid_config_snapshot(snap)


def test_other_kinds_never_accept_a_dev_binding():
    snap = snapshot(DevDocker())
    assert not valid_config_binding(snap) and not valid_tagged_binding(snap)


# ── confined to a declared dev target (Codex r1 Blocker) ──────────────────

@pytest.mark.parametrize("labels", [None, {DEV_TARGET_LABEL: "0"}, {DEV_TARGET_LABEL: "true"}, {"other": "1"}])
def test_an_unmarked_compose_service_refuses_at_approval(labels):
    api = DevDocker(compose=dev_compose(labels=labels))
    with pytest.raises(ComposeError) as caught:
        binding(api)
    assert caught.value.code == "dev_target_unmarked" and caught.value.names == [DEV_TARGET_LABEL]
    with pytest.raises(ConfigCheckError) as caught:
        _executor(api, worker_labels=DEV_WORKER).config_approval(SPEC)
    assert caught.value.code == "dev_target_unmarked" and not api.create_bodies


def test_a_marker_only_on_the_running_container_is_not_enough():
    api = DevDocker(compose=dev_compose(labels=None))
    assert api.by_name(NAME)["Config"]["Labels"][DEV_TARGET_LABEL] == "1"
    with pytest.raises(ComposeError, match="dev_target_unmarked"):
        binding(api)


def test_the_worker_rechecks_the_marker_before_create(tmp_path):
    # Approved while marked; the Compose file then drops the marker.
    api = DevDocker(compose=_with_env(REPOS_HOST_ROOT=SECRET))
    snap = snapshot(api)
    api.compose = dev_compose(environment={"NODE_ENV": "development", "REPOS_HOST_ROOT": SECRET}, labels=None)
    with pytest.raises(DeployFailed, match="dev_target_unmarked") as caught:
        run(api, tmp_path, snap)
    assert caught.value.outcome_known and not api.create_bodies


def test_a_production_target_repointed_at_the_dev_action_refuses():
    # A catalog edit swaps the app row's spec to this action: the app's own
    # Compose service carries no marker, so it never binds.
    app = AppDocker()
    spec = {**SPEC, "container_name": APP_NAME, "compose_project": "llm-bawt", "compose_service": "app",
            "health_url": "http://127.0.0.1:8642/health", "health_value": "ok"}
    with pytest.raises(ConfigCheckError) as caught:
        _executor(app, worker_labels=DEV_WORKER).config_approval(spec)
    assert caught.value.code in ("dev_target_unmarked", "compose_image_mismatch") and not app.create_bodies


def test_the_dev_action_is_approval_only():
    assert ACTION in APPROVAL_ONLY_ACTIONS
    service, store, _executor_ = _service(PublishedVerifier(HEAD))
    store.create_operation({**SEED, "enabled": True})
    with pytest.raises(OpsDispatchError) as caught:
        service.dispatch_job(operation_slug=SEED["slug"], args={}, idempotency_key="run-button")
    assert caught.value.code == "approval_required" and store.list_jobs() == []
    # The Blocker's path: an enabled production row PATCHed to this action.
    store.create_operation({**APP_SEED, "enabled": True, "command_script": SEED["command_script"]})
    with pytest.raises(OpsDispatchError) as caught:
        service.dispatch_job(operation_slug=APP_SEED["slug"], args={}, idempotency_key="patched")
    assert caught.value.code == "approval_required" and store.list_jobs() == []


# ── execution ──────────────────────────────────────────────────────────────

def test_change_since_approval_refuses_before_create(tmp_path):
    api = DevDocker(compose=_with_env(REPOS_HOST_ROOT=SECRET))
    snap = snapshot(api)
    api.compose = _with_env(REPOS_HOST_ROOT=SECRET, OTHER="1")
    with pytest.raises(DeployFailed, match="config_changed_since_approval") as caught:
        run(api, tmp_path, snap)
    assert caught.value.outcome_known and not api.create_bodies


def test_failed_health_restores_the_previous_container(tmp_path):
    api = DevDocker(compose=_with_env(REPOS_HOST_ROOT=SECRET), health_status="starting")
    with pytest.raises(DeployFailed, match="health status did not match expected 'healthy'") as caught:
        run(api, tmp_path)
    assert caught.value.detail["restored"] and api.by_name(NAME)["Id"] == api.old_id


# ── worker capability ──────────────────────────────────────────────────────

@pytest.mark.parametrize("missing", [CONFIG_WORKER_LABEL, PROCESS_PROFILE_LABEL, DEV_WORKER_LABEL])
def test_dev_reconcile_requires_the_dev_worker(missing):
    api = DevDocker()
    executor = _executor(api, worker_labels={k: v for k, v in DEV_WORKER.items() if k != missing})
    for call in (lambda: executor.config_approval(SPEC),
                 lambda: executor.preflight({"spec": SPEC, "execution": executor.execution_settings()})):
        with pytest.raises(ConfigCheckError) as caught:
            call()
        assert caught.value.code == "config_reconcile_unavailable" and missing in str(caught.value)
    assert not api.render_bodies


def test_worker_runs_the_dev_kind_from_a_receipt(tmp_path):
    api = DevDocker(compose=_with_env(REPOS_HOST_ROOT=SECRET))
    root = tmp_path / "receipts"
    snap = {**snapshot(api), "execution": {**EXECUTION, "receipt_root": str(root)}}
    path, digest = _write_request(root, snap)
    assert not worker_run(path, digest, api_factory=lambda _: api, sleep=lambda _: None)
    fresh = _executor(DevDocker(), worker_labels=DEV_WORKER)
    fresh._get_worker = lambda _name: None
    result = fresh.reconcile(job_id=JOB_ID, snapshot=snap)
    assert result.state == "succeeded" and json.loads(result.output_tail)["schema"] == SCHEMA
    assert api.create_bodies[0]["Image"] == DEV_ID


def test_worker_image_carries_the_module_and_the_label():
    dockerfile = (Path(__file__).resolve().parents[1] / "docker" / "Dockerfile.ops-worker").read_text()
    assert "src/llm_bawt/ops/dev_config_reconcile.py" in dockerfile
    assert f'LABEL {DEV_WORKER_LABEL}="1"' in dockerfile
