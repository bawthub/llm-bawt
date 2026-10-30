"""TASK-997: digest-bound BawtHub prod image deploy/rollback.

Worker side (ImageDeployer against a fake Docker API), app side (approval-time
release verification, rollback binding, snapshot consistency), catalog seeds
and the GitHub verifier against a fake opener. No network, no Docker.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import io
import json
import struct
import urllib.error
import zipfile
from urllib.parse import parse_qs, unquote, urlsplit

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine

from llm_bawt.ops.release_models import RELEASE_AWAITING_DEPLOY_APPROVAL, ReleaseRun

from llm_bawt.ops import OpsDispatchError, OpsService, OpsStore
from llm_bawt.ops.compose_render import RENDER_DOCKER_CONFIG, RENDER_LABEL, RENDER_PATH
from llm_bawt.ops.executor import (ConfigCheckError, DispatchResult, DockerExecutor, Executor, ExecutorError,
                                   ReconcileResult, validate_spec)
from llm_bawt.ops.image_deploy import (DEPLOY_SCHEMA, DeployFailed, ImageDeployer,
                                       clone_create_body, parse_deployment)
from llm_bawt.ops.release import (GitHubReleaseVerifier, ReleaseVerificationError, ReleaseVerifier,
                                  binding_matches_args)
from llm_bawt.ops.seeds import SEEDS
from llm_bawt.ops.validation import canonical_json
from llm_bawt.ops.worker import run as worker_run

REPO = "ghcr.io/bawthub/frontend"
OLD_ID, NEW_ID = "sha256:" + "a" * 64, "sha256:" + "b" * 64
DIGEST = "sha256:" + "d" * 64
SOURCE, TRIGGER = "1" * 40, "2" * 40
JOB_ID = "f" * 32
NAME = "bawthub-frontend-prod-1"
SEED = {s["slug"]: s for s in SEEDS}
SPEC = json.loads(SEED["bawthub.deploy-prod-image"]["command_script"])
VERIFY_ARGS = {"workflow_run_id": "123456", "digest": DIGEST, "source_sha": SOURCE, "version": "1.4.2"}
RELEASE_ID = "e" * 32
ARGS = {"release_run_id": RELEASE_ID}
# TASK-1518: the target's Compose project, as its own labels record it.
PROJECT_DIR = "/srv/compose/bawthub"
COMPOSE_FILES = [f"{PROJECT_DIR}/{name}" for name in SPEC["compose_config"]["compose_files"]]
RENDERER = SPEC["compose_config"]["renderer_image"]
RENDERER_ID = "sha256:" + "9" * 64
CONFIG_HASH = "c0" * 32
TOKEN_TARGET = "/run/bawthub-service-token/token"
IMAGE_VARIABLE = "${" + SPEC["compose_config"]["image_variable"] + "}"


def compose_project(**service):
    """What ``docker compose config --format json frontend-prod`` prints for the
    fixture target: no drift by default. Keys mirror real Compose output, and
    the image is interpolated from the renderer's one injected variable."""
    body = {"command": None, "entrypoint": None, "environment": {"NODE_ENV": "production"},
            "image": IMAGE_VARIABLE, "labels": {}, "networks": {"default": None},
            "pull_policy": "missing", "restart": "unless-stopped",
            "volumes": [{"type": "bind", "source": f"{PROJECT_DIR}/token", "target": TOKEN_TARGET,
                         "read_only": True, "bind": {"create_host_path": True}}]}
    body.update(service)
    return {"name": "bawthub", "services": {"frontend-prod": body},
            "networks": {"default": {"name": "bawthub_default"}}}


def _frame(stream, text):
    payload = text.encode()
    return struct.pack(">BxxxI", stream, len(payload)) + payload


class _Stream:
    def __init__(self, raw):
        self.raw, self.closed = raw, False

    def read(self, limit):
        return self.raw[:limit]

    def close(self):
        self.closed = True


def _labels(source=SOURCE, version="1.4.2", run="123456"):
    return {"com.bawthub.release.schema": "1", "org.opencontainers.image.version": version,
            "org.opencontainers.image.revision": source, "com.bawthub.release.workflow-run-id": run}


def _release(**over):
    rel = {"workflow_run_id": "123456", "workflow_run_url": "https://github.com/x/actions/runs/123456",
           "source_sha": SOURCE, "version": "1.4.2", "tag": "v1.4.2", "digest": DIGEST,
           "image_ref": f"{REPO}@{DIGEST}", "expected_current_image_id": OLD_ID}
    rel.update(over)
    return rel


class FakeDocker:
    """Just enough of the Engine API for ImageDeployer, keyed like dockerd."""

    def __init__(self, *, new_health="healthy", new_labels=None, repo_digests=None,
                 local_new=True, old_restart_fails=False, compose=None, render_status=0, render_stderr=""):
        self.side_effect_started = False
        self._seq = 13  # created container ids: "e"*64, "f"*64, ...
        self.calls: list[tuple[str, str]] = []
        self.create_bodies: list[dict] = []
        self.render_bodies: list[dict] = []
        self.compose = compose_project() if compose is None else compose
        self.render_status, self.render_stderr = render_status, render_stderr
        self.old_restart_fails = old_restart_fails
        self.images = {OLD_ID: {"Id": OLD_ID, "RepoDigests": [], "_health": "healthy",
                                "_release": None,
                                "Config": {"Labels": {"org.opencontainers.image.title": "old"},
                                           "Env": ["PATH=/usr/bin"], "Cmd": ["node", "server"],
                                           "Healthcheck": {"Test": ["CMD", "curl"]}}},
                       RENDERER_ID: {"Id": RENDERER_ID, "RepoDigests": [RENDERER], "Config": {}}}
        self._new = {"Id": NEW_ID, "RepoDigests": [f"{REPO}@{DIGEST}"] if repo_digests is None else repo_digests,
                     "_health": new_health, "_refs": [f"{REPO}@{DIGEST}"],
                     "_release": {"version": "1.4.2", "sourceSha": SOURCE, "workflowRunId": "123456"},
                     "Config": {"Labels": _labels() if new_labels is None else new_labels,
                                "Env": ["PATH=/usr/bin"], "Cmd": ["node", "server"],
                                "Healthcheck": {"Test": ["CMD", "curl"]}}}
        if local_new:
            self.images[NEW_ID] = self._new
        old_id = "c" * 64
        self.containers = {old_id: {
            "Id": old_id, "Name": f"/{NAME}", "Image": OLD_ID,
            "Config": {"Image": "bawthub-frontend-prod:latest", "Env": ["PATH=/usr/bin", "NODE_ENV=production"],
                       "Cmd": ["node", "server"], "Healthcheck": {"Test": ["CMD", "curl"]},
                       "Labels": {"com.docker.compose.project": "bawthub",
                                  "com.docker.compose.service": "frontend-prod",
                                  "com.docker.compose.project.working_dir": PROJECT_DIR,
                                  "com.docker.compose.project.config_files": ",".join(COMPOSE_FILES),
                                  "org.opencontainers.image.title": "old"}},
            "HostConfig": {"RestartPolicy": {"Name": "unless-stopped"},
                           "Binds": [f"{PROJECT_DIR}/token:{TOKEN_TARGET}:ro"]},
            "NetworkSettings": {"Networks": {"bawthub_default": {"Aliases": [NAME, "frontend-prod", old_id[:12]]}}},
            "State": {"Running": True, "Status": "running", "Health": {"Status": "healthy"}}}}
        self.old_id = old_id

    # ── helpers ─────────────────────────────────────────────────────────
    def _find_container(self, key):
        for c in self.containers.values():
            if c["Id"] == key or c["Name"] == f"/{key}":
                return c
        raise RuntimeError("Docker HTTP 404: no such container")

    def _find_image(self, ref):
        for img in self.images.values():
            if img["Id"] == ref or ref in img["RepoDigests"] or ref in img.get("_refs", ()):
                return img
        raise RuntimeError("Docker HTTP 404: no such image")

    def by_name(self, name):
        return self._find_container(name)

    def renderers_left(self):
        return [c for c in self.containers.values() if c.get("_render")]

    # ── the sealed Compose renderer (TASK-1518) ─────────────────────────
    def _create_renderer(self, name, body):
        self.render_bodies.append(body)
        cid = hashlib.sha256(name.encode()).hexdigest()
        self.containers[cid] = {"Id": cid, "Name": f"/{name}", "Image": RENDERER_ID, "_render": True,
                                "Config": {"Image": body["Image"], "Entrypoint": body["Entrypoint"],
                                           "Cmd": body["Cmd"], "Env": body["Env"],
                                           "Labels": body["Labels"]},
                                "HostConfig": body["HostConfig"], "State": {"Running": False, "Status": "created"}}
        return {"Id": cid}

    def attach_stream(self, container_id):
        self.calls.append(("POST", f"/containers/{container_id}/attach"))
        container = self.containers[container_id]
        argv = container["Config"]["Cmd"]
        if "--hash" in argv:
            out = f"{argv[-1]} {CONFIG_HASH}\n"
        else:
            # The renderer runs ``env -i PATH=… VAR=ref … docker`` (TASK-1476).
            env = dict(e.partition("=")[::2] for e in container["Config"]["Entrypoint"] if "=" in e)
            image = env.get(IMAGE_VARIABLE.strip("${}"), "")
            out = json.dumps(self.compose).replace(IMAGE_VARIABLE, image)
        return _Stream(_frame(1, out) + (_frame(2, self.render_stderr) if self.render_stderr else b""))

    # ── Engine API ──────────────────────────────────────────────────────
    def request(self, method, path, *, body=None, headers=None, raw=False):
        self.calls.append((method, path))
        url = urlsplit(path)
        parts = [unquote(p) for p in url.path.strip("/").split("/")]
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        if parts[0] == "images" and method == "GET":
            return copy.deepcopy(self._find_image(parts[1]))
        if parts[0] == "images" and parts[1] == "create":
            raise AssertionError("the worker must never pull; the app pulls in preflight")
        if parts[0] == "containers" and parts[1] == "create" and RENDER_LABEL in (body.get("Labels") or {}):
            return self._create_renderer(query["name"], body)
        if parts[0] == "containers" and parts[1] == "create":
            self.create_bodies.append(body)
            image = self._find_image(body["Image"])
            self._seq += 1
            new = {"Id": f"{self._seq:x}" * 64, "Name": f"/{query['name']}", "Image": image["Id"],
                   "Config": {"Image": body["Image"], "Labels": body["Labels"]},
                   "HostConfig": body["HostConfig"], "NetworkSettings": {"Networks": {}},
                   "State": {"Running": False, "Status": "created"}}
            self.containers[new["Id"]] = new
            return {"Id": new["Id"]}
        if parts[0] == "exec":
            if parts[2] == "start":
                payload = json.dumps({"status": "healthy", "release": self._exec_target["_release"]}).encode()
                return struct.pack(">BxxxI", 1, len(payload)) + payload
            return {"ExitCode": 0}
        container = self._find_container(parts[1])
        action = parts[2] if len(parts) > 2 else ""
        if action == "json":
            return copy.deepcopy(container)
        if action == "exec":
            self._exec_target = self.images[container["Image"]]
            return {"Id": "exec1"}
        if method == "DELETE":
            self.containers.pop(container["Id"], None)
            return None
        if container.get("_render"):
            if action == "start":
                container["State"] = {"Running": False, "Status": "exited"}
                return None
            if action == "wait":
                return {"StatusCode": self.render_status}
            raise AssertionError(f"unexpected renderer call {method} {path}")
        if action == "stop":
            container["State"] = {"Running": False, "Status": "exited"}
        elif action == "start":
            if container["Id"] == self.old_id and self.old_restart_fails and any(
                    p.endswith(f"{self.old_id}/stop?t=20") for _m, p in self.calls):
                raise RuntimeError("Docker HTTP 500: cannot start")
            health = self.images[container["Image"]]["_health"]
            container["State"] = {"Running": True, "Status": "running", "Health": {"Status": health}}
        elif action == "rename":
            if any(c["Name"] == f"/{query['name']}" and c is not container for c in self.containers.values()):
                raise RuntimeError("Docker HTTP 409: name in use")
            container["Name"] = f"/{query['name']}"
        return None


def _deployer(api, tmp_path, *, snapshot=None):
    snapshot = snapshot or {"spec": SPEC, "release": _release()}
    return ImageDeployer(api, snapshot, JOB_ID, lock_root=tmp_path / "locks",
                         sleep=lambda _s: None, clock=iter(range(0, 10_000, 5)).__next__)


# ── worker: deploy ─────────────────────────────────────────────────────────

def test_deploy_success_swaps_by_digest_and_records_last_known_good(tmp_path):
    api = FakeDocker()
    record = json.loads(_deployer(api, tmp_path).run())
    live = api.by_name(NAME)
    assert live["Image"] == NEW_ID and live["Config"]["Image"] == f"{REPO}@{DIGEST}"
    assert api.old_id not in api.containers  # previous container cleaned up
    assert record["schema"] == DEPLOY_SCHEMA and record["phase"] == "done"
    assert record["previous"]["image_id"] == OLD_ID and record["deployed"]["image_id"] == NEW_ID
    assert record["health"]["release"]["sourceSha"] == SOURCE
    body = api.create_bodies[0]
    assert api.old_id[:12] not in body["NetworkingConfig"]["EndpointsConfig"]["bawthub_default"]["Aliases"]
    assert body["Labels"]["com.docker.compose.image"] == NEW_ID
    assert body["Labels"]["llm-bawt.ops.deployed-by"] == JOB_ID
    assert "org.opencontainers.image.title" not in body["Labels"]  # old image label not carried
    assert body["Env"] == ["NODE_ENV=production"]  # compose env kept, image env dropped
    # TASK-1518: Compose's own hash at the new ref, so a later `compose up` keeps it.
    assert body["Labels"]["com.docker.compose.config-hash"] == CONFIG_HASH
    assert parse_deployment(json.dumps(record)) == record
    assert not api.renderers_left()


# ── worker: deploy refuses on configuration drift (TASK-1518) ──────────────

def test_deploy_renders_the_target_compose_service_sealed_at_the_new_ref(tmp_path):
    api = FakeDocker()
    _deployer(api, tmp_path).run()
    assert len(api.render_bodies) == 2  # the JSON config, then Compose's config hash
    body = api.render_bodies[0]
    assert body["Image"] == RENDERER and body["Entrypoint"] == [
        "env", "-i", f"PATH={RENDER_PATH}", f"DOCKER_CONFIG={RENDER_DOCKER_CONFIG}",
        f"BAWTHUB_FRONTEND_IMAGE={REPO}@{DIGEST}", "docker"]
    assert body["Cmd"] == ["compose", "-p", "bawthub", "-f", COMPOSE_FILES[0], "-f", COMPOSE_FILES[1],
                           "config", "--format", "json", "frontend-prod"]
    assert api.render_bodies[1]["Cmd"][-3:] == ["config", "--hash", "frontend-prod"]
    assert body["Env"] == [] and body["NetworkDisabled"] is True  # the image's env never reaches Compose
    host = body["HostConfig"]
    assert host["Binds"] == [f"{PROJECT_DIR}:{PROJECT_DIR}:ro"] and host["NetworkMode"] == "none"
    assert host["ReadonlyRootfs"] and host["CapDrop"] == ["ALL"] and host["Privileged"] is False
    assert host["LogConfig"]["Type"] == "none"
    assert host["Tmpfs"] == {RENDER_DOCKER_CONFIG: "rw,noexec,nosuid,nodev,size=1048576"}
    assert not api.renderers_left()


@pytest.mark.parametrize("compose, names", [
    (compose_project(environment={"NODE_ENV": "production", "BAWTHUB_AUTH_SESSIONS": "on"}),
     ["env:+BAWTHUB_AUTH_SESSIONS"]),
    (compose_project(environment={"NODE_ENV": "development"}), ["env:~NODE_ENV"]),
    (compose_project(volumes=[]), [f"mount:-{TOKEN_TARGET}"]),
    (compose_project(networks={"default": None, "llm-bawt": None}), ["network:+bawthub_llm-bawt"]),
    (compose_project(restart="always"), ["restart:~"]),
    (compose_project(labels={"traefik.enable": "true"}), ["label:+traefik.enable"]),
])
def test_deploy_refuses_config_drift_with_names_only_before_touching_production(tmp_path, compose, names):
    api = FakeDocker(compose=compose)
    with pytest.raises(DeployFailed, match="config_drift") as caught:
        _deployer(api, tmp_path).run()
    assert caught.value.outcome_known and not api.side_effect_started and not api.create_bodies
    assert caught.value.detail["config_drift"] == names
    assert "development" not in str(caught.value) and "=on" not in str(caught.value)
    assert api.by_name(NAME)["Id"] == api.old_id and not api.renderers_left()


@pytest.mark.parametrize("api_kwargs, message", [
    ({"render_stderr": 'time="2026-10-26T00:00:00Z" level=warning msg="The \\"XAI_API_KEY\\" variable '
                       'is not set. Defaulting to a blank string."\n'}, "unset_compose_variable: XAI_API_KEY"),
    ({"render_stderr": "something else entirely\n"}, "renderer_warning"),
    ({"render_status": 1}, "renderer_failed"),
    ({"compose": compose_project(privileged=True)}, "unsupported_compose_key: privileged"),
    ({"compose": compose_project(image="ghcr.io/bawthub/frontend:latest")}, "compose_image_mismatch"),
    ({"compose": compose_project(pull_policy="always")}, "unsupported_pull_policy"),
])
def test_deploy_refuses_when_the_compose_service_cannot_be_checked(tmp_path, api_kwargs, message):
    api = FakeDocker(**api_kwargs)
    with pytest.raises(DeployFailed, match=message) as caught:
        _deployer(api, tmp_path).run()
    assert caught.value.outcome_known and not api.side_effect_started and not api.create_bodies
    assert not api.renderers_left()


def test_deploy_refuses_a_target_created_from_other_compose_files(tmp_path):
    api = FakeDocker()
    api.by_name(NAME)["Config"]["Labels"]["com.docker.compose.project.config_files"] = COMPOSE_FILES[0]
    with pytest.raises(DeployFailed, match="compose_source_invalid") as caught:
        _deployer(api, tmp_path).run()
    assert caught.value.outcome_known and not api.render_bodies and not api.create_bodies


def test_deploy_without_compose_config_fails_closed(tmp_path):
    api = FakeDocker()
    spec = {k: v for k, v in SPEC.items() if k != "compose_config"}
    with pytest.raises(DeployFailed, match="no compose_config") as caught:
        _deployer(api, tmp_path, snapshot={"spec": spec, "release": _release()}).run()
    assert caught.value.outcome_known and not api.render_bodies and not api.create_bodies


def test_deploy_health_failure_restores_previous_container(tmp_path):
    api = FakeDocker(new_health="unhealthy")
    with pytest.raises(DeployFailed) as caught:
        _deployer(api, tmp_path).run()
    assert caught.value.outcome_known is True
    assert caught.value.detail["restored"] is True and caught.value.detail["failed_phase"] == "health"
    live = api.by_name(NAME)
    assert live["Id"] == api.old_id and live["State"]["Running"]
    assert "e" * 64 not in api.containers


def test_deploy_restore_failure_is_uncertain(tmp_path):
    api = FakeDocker(new_health="unhealthy", old_restart_fails=True)
    with pytest.raises(DeployFailed) as caught:
        _deployer(api, tmp_path).run()
    assert caught.value.outcome_known is False
    assert "RESTORE FAILED" in str(caught.value)


@pytest.mark.parametrize("api_kwargs, snapshot_release, message", [
    ({}, {"expected_current_image_id": "sha256:" + "9" * 64}, "changed since approval"),
    ({"new_labels": _labels(source="3" * 40)}, {}, "labels do not match"),
    ({"repo_digests": [f"ghcr.io/evil/frontend@{DIGEST}"]}, {}, "RepoDigest"),
])
def test_deploy_refuses_before_touching_production(tmp_path, api_kwargs, snapshot_release, message):
    api = FakeDocker(**api_kwargs)
    snapshot = {"spec": SPEC, "release": _release(**snapshot_release)}
    with pytest.raises(DeployFailed, match=message) as caught:
        _deployer(api, tmp_path, snapshot=snapshot).run()
    assert caught.value.outcome_known and not api.side_effect_started
    assert not api.create_bodies and api.by_name(NAME)["Id"] == api.old_id


def test_deploy_refuses_when_approved_image_absent_and_never_pulls(tmp_path):
    api = FakeDocker(local_new=False)
    with pytest.raises(DeployFailed, match="not on the daemon") as caught:
        _deployer(api, tmp_path).run()
    assert caught.value.outcome_known and not api.side_effect_started and not api.create_bodies
    assert not any("/images/create" in path for _m, path in api.calls)


def test_target_lock_excludes_concurrent_image_actions(tmp_path):
    api = FakeDocker()
    (tmp_path / "locks").mkdir()
    with (tmp_path / "locks" / f"{NAME}.lock").open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        with pytest.raises(DeployFailed, match="holds this target") as caught:
            _deployer(api, tmp_path).run()
    assert caught.value.outcome_known and not api.calls


def test_rollback_restores_recorded_image_without_pulling(tmp_path):
    api = FakeDocker(local_new=True)
    _deployer(api, tmp_path).run()
    # A rollback must stay possible under drift: it never renders or compares.
    api.compose, renders = compose_project(environment={"NODE_ENV": "x"}), len(api.render_bodies)
    snapshot = {"spec": {**SPEC, "action": "rollback_image"},
                "rollback": {"deploy_job_id": JOB_ID, "from_image_id": NEW_ID, "from_image_ref": f"{REPO}@{DIGEST}",
                             "to_image_id": OLD_ID, "to_image_ref": "bawthub-frontend-prod:latest",
                             "to_release": None}}
    record = json.loads(ImageDeployer(api, snapshot, "0" * 32, lock_root=tmp_path / "locks",
                                      sleep=lambda _s: None).run())
    assert record["action"] == "rollback" and record["rollback_of"] == JOB_ID
    assert api.by_name(NAME)["Image"] == OLD_ID
    assert len(api.render_bodies) == renders


def test_clone_keeps_compose_overrides_that_differ_from_image():
    api = FakeDocker()
    old = api.by_name(NAME)
    old["Config"]["Cmd"] = ["node", "custom"]
    body = clone_create_body(old, api.images[OLD_ID], api._new, f"{REPO}@{DIGEST}", JOB_ID)
    assert body["Cmd"] == ["node", "custom"] and "Healthcheck" not in body


def test_clone_keeps_compose_identity_even_when_old_image_was_compose_built():
    # `make rebuild-prod` images carry compose project/service labels too; they
    # must survive or every later deploy refuses the target's identity.
    api = FakeDocker()
    old = api.by_name(NAME)
    compose = {k: v for k, v in old["Config"]["Labels"].items() if k.startswith("com.docker.compose.")}
    api.images[OLD_ID]["Config"]["Labels"].update(compose)
    body = clone_create_body(old, api.images[OLD_ID], api._new, f"{REPO}@{DIGEST}", JOB_ID)
    assert body["Labels"]["com.docker.compose.project"] == SPEC["compose_project"]
    assert body["Labels"]["com.docker.compose.service"] == SPEC["compose_service"]


# ── worker.run receipt semantics ───────────────────────────────────────────

def _worker_request(tmp_path, snapshot):
    job_dir = tmp_path / "jobs" / JOB_ID
    job_dir.mkdir(parents=True)
    raw = json.dumps({"job_id": JOB_ID, "snapshot": snapshot}).encode()
    (job_dir / "request.json").write_bytes(raw)
    return job_dir / "request.json", hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("api_kwargs, state", [
    ({"local_new": False}, "failed"),  # approved image absent: known, nothing touched
    ({"local_new": True, "new_health": "unhealthy"}, "failed"),  # restored
    ({"local_new": True, "new_health": "unhealthy", "old_restart_fails": True}, "lost"),
])
def test_worker_maps_deploy_outcomes_to_receipts(tmp_path, api_kwargs, state):
    execution = {"start_delay_seconds": 0, "timeout_seconds": 60, "max_output_bytes": 65536}
    request, digest = _worker_request(tmp_path, {"spec": SPEC, "release": _release(),
                                                 "resolved_args": ARGS, "execution": execution})
    api = FakeDocker(**api_kwargs)
    assert worker_run(request, digest, api_factory=lambda _t: api, sleep=lambda _s: None) == 1
    receipt = json.loads((request.parent / "receipt.json").read_text())
    assert receipt["state"] == state
    assert receipt["side_effect_unknown"] is (state == "lost")
    assert parse_deployment(receipt["output_tail"])["schema"] == DEPLOY_SCHEMA
    assert (tmp_path / "jobs" / ".target-locks").is_dir()


# ── catalog seeds and spec validation ──────────────────────────────────────

def test_image_seeds_ship_disabled_with_fixed_targets():
    for slug in ("bawthub.deploy-prod-image", "bawthub.rollback-prod-image"):
        seed = SEED[slug]
        spec = validate_spec(seed["command_script"])
        assert seed["enabled"] is False and seed["max_concurrent"] == 1 and seed["max_output_bytes"] >= 65536
        assert spec["image_repository"] == REPO and spec["container_name"] == NAME
        assert json.loads(seed["args_schema_json"])["additionalProperties"] is False


def test_deploy_seed_requires_compose_config_and_rollback_does_not():
    assert SPEC["compose_config"] == json.loads(
        SEED["bawthub.rollback-prod-image"]["command_script"])["compose_config"]
    bare = {k: v for k, v in SPEC.items() if k != "compose_config"}
    with pytest.raises(ValueError, match="requires compose_config"):
        validate_spec(json.dumps(bare))
    assert validate_spec(json.dumps({**bare, "action": "rollback_image"}))["action"] == "rollback_image"


@pytest.mark.parametrize("mutate", [
    lambda s: s.pop("image_repository"),
    lambda s: s.update(image_repository="docker.io/evil/frontend"),
    lambda s: s.update(container_name_from_arg="container"),
    lambda s: s.update(health_timeout_seconds=5),
    lambda s: s["compose_config"].update(renderer_image="docker:27-cli"),
    lambda s: s["compose_config"].update(compose_files=["../etc/compose.yml"]),
    lambda s: s["compose_config"].update(extra="x"),
])
def test_image_spec_rejects_selector_or_registry_drift(mutate):
    spec = copy.deepcopy(SPEC)
    mutate(spec)
    with pytest.raises(ValueError):
        validate_spec(json.dumps(spec))


class _NotFound(Exception):
    status_code = 404


class _FakeImages:
    def __init__(self, present=False, pull_error=None):
        self.present, self.pull_error, self.pulls = present, pull_error, []

    def get(self, ref):
        if not self.present:
            raise _NotFound(ref)

    def pull(self, ref, auth_config=None):
        self.pulls.append((ref, auth_config))
        if self.pull_error:
            raise self.pull_error
        self.present = True


def _pull_executor(images, auth=lambda: {"username": "puller", "password": "s3cret"}):
    client = type("Client", (), {"images": images})()
    return DockerExecutor(client_factory=lambda: client, worker_image="sha256:" + "0" * 64,
                          receipt_volume="receipts", receipt_root="/receipts", pull_auth=auth)


def test_execution_settings_carry_no_credential_plumbing():
    executor = _pull_executor(_FakeImages())
    assert executor.execution_settings(SPEC) == executor.execution_settings() == {
        "worker_image": "sha256:" + "0" * 64, "receipt_volume": "receipts", "receipt_root": "/receipts"}


def test_preflight_pulls_absent_release_image_with_db_credential():
    images = _FakeImages()
    executor = _pull_executor(images)
    executor.ensure_release_image(f"{REPO}@{DIGEST}")
    assert images.pulls == [(f"{REPO}@{DIGEST}", {"username": "puller", "password": "s3cret"})]
    executor.ensure_release_image(f"{REPO}@{DIGEST}")  # idempotent: already local
    assert len(images.pulls) == 1


def test_preflight_pull_failures_are_prerequisite_errors_without_secrets():
    with pytest.raises(ExecutorError, match="connect the ghcr-pull"):
        _pull_executor(_FakeImages(), auth=lambda: None).ensure_release_image(f"{REPO}@{DIGEST}")
    with pytest.raises(ExecutorError, match="pull failed") as caught:
        _pull_executor(_FakeImages(pull_error=RuntimeError("denied"))).ensure_release_image(f"{REPO}@{DIGEST}")
    assert "s3cret" not in str(caught.value)


def test_preflight_only_pulls_for_deploy(monkeypatch):
    images = _FakeImages()
    executor = _pull_executor(images)
    monkeypatch.setattr(executor, "_check_settings", lambda _s: None)
    # The deploy's config-check prerequisites have their own tests (test_ops_config_reconcile).
    monkeypatch.setattr(executor, "_check_config_worker", lambda _s, _action=None: None)
    monkeypatch.setattr(executor, "ensure_renderer_image", lambda _s: None)
    monkeypatch.setattr(executor, "self_target", lambda _s: None)  # Not under test (TASK-1169 S4c).
    # Worker capabilities have their own tests (TASK-1801, test_ops_health_probe).
    monkeypatch.setattr(executor, "_check_spec_capabilities", lambda _s, _spec: None)
    executor.client.ping = lambda: True
    executor.preflight({"spec": {**SPEC, "action": "rollback_image"}, "execution": {}})
    executor.preflight({"spec": {"action": "restart"}, "execution": {}})
    assert images.pulls == []
    executor.preflight({"spec": SPEC, "execution": {}, "release": _release()})
    assert [ref for ref, _a in images.pulls] == [f"{REPO}@{DIGEST}"]


# ── app side: approval-time binding ───────────────────────────────────────

class FakeExecutor(Executor):
    def __init__(self):
        self.current_image = OLD_ID
        self.config_checks: list[tuple[str, str | None]] = []
        self.config_error: ConfigCheckError | None = None

    def kind(self):
        return "docker"

    def target_compose_identity(self, spec, args):
        return ("test", "fake")  # not a GPU voice service (TASK-1171)

    def available(self):
        return True

    def inspect_target_image(self, spec):
        return self.current_image

    def check_spec_capabilities(self, spec):
        pass  # a worker built with every capability (TASK-1801)

    def config_check(self, spec, image_ref=None):
        self.config_checks.append((spec["action"], image_ref))
        if self.config_error:
            raise self.config_error

    def dispatch(self, **kwargs):
        return DispatchResult(host_unit_name="w", status_file_path="s", log_file_path="l")

    def reconcile(self, **kwargs):
        return ReconcileResult(state="accepted")


class FakeVerifier(ReleaseVerifier):
    def __init__(self, error=None):
        self.error = error

    def verify(self, spec, args):
        if self.error:
            raise ReleaseVerificationError(self.error)
        return {**args, "tag": f"v{args['version']}", "image_repository": REPO,
                "workflow_run_attempt": "1", "trigger_sha": TRIGGER,
                "image_ref": f"{REPO}@{args['digest']}", "workflow_run_url": "u"}


def _service(verifier=None, *, enable=True):
    from datetime import datetime, timezone
    from llm_bawt.ops.release_store import ReleaseStore

    store = object.__new__(OpsStore)
    store.engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    store._ensure_tables_exist()
    releases = ReleaseStore(None, engine=store.engine)
    receipt = {"schema": "bawthub.release-receipt/v1", "repository": "bawthub/bawthub",
               "release_request_id": "release-test", "workflow_run_id": "123456",
               "workflow_run_attempt": "1", "base_sha": TRIGGER, "source_sha": SOURCE,
               "version": "1.4.2", "tag": "v1.4.2", "digest": DIGEST,
               "image_repository": REPO, "status": "complete"}
    with Session(store.engine) as session:
        session.add(ReleaseRun(id=RELEASE_ID, parent_job_id="0" * 32,
                               release_request_id="release-test", release_task="TASK-1030",
                               github_repository=SPEC["github_repository"],
                               workflow_path=SPEC["workflow_path"],
                               canonical_branch=SPEC["canonical_branch"],
                               state=RELEASE_AWAITING_DEPLOY_APPROVAL, deployable=True,
                               receipt_json=json.dumps(receipt),
                               receipt_verified_at=datetime.now(timezone.utc),
                               github_run_id="123456", github_run_attempt=1,
                               source_sha=SOURCE, version="1.4.2", tag="v1.4.2",
                               digest=DIGEST, image_repository=REPO))
        session.commit()
    for slug in ("bawthub.deploy-prod-image", "bawthub.rollback-prod-image"):
        store.create_operation({**SEED[slug], "enabled": enable})
    executor = FakeExecutor()
    return OpsService(store, executor=executor, release_store=releases,
                      release_verifier=verifier or FakeVerifier()), store, executor


def _rehash(snapshot):
    body = {k: v for k, v in snapshot.items() if k != "snapshot_hash"}
    snapshot["snapshot_hash"] = hashlib.sha256(canonical_json(body).encode()).hexdigest()
    return snapshot


def test_disabled_seed_cannot_be_prepared():
    service, _store, _ex = _service(enable=False)
    with pytest.raises(OpsDispatchError) as caught:
        service.prepare_invocation("bawthub.deploy-prod-image", ARGS)
    assert caught.value.code == "operation_disabled"


def test_deploy_snapshot_binds_verified_release_and_current_image():
    service, _store, _ex = _service()
    snapshot = service.prepare_invocation("bawthub.deploy-prod-image", ARGS)
    assert snapshot["release"]["digest"] == DIGEST
    assert snapshot["release"]["expected_current_image_id"] == OLD_ID
    assert binding_matches_args(snapshot)


def test_deploy_approval_checks_config_at_the_release_ref_and_refuses_drift():
    service, _store, executor = _service()
    service.prepare_invocation("bawthub.deploy-prod-image", ARGS)
    assert executor.config_checks == [("deploy_image", f"{REPO}@{DIGEST}")]
    executor.config_error = ConfigCheckError("config_drift", "config_drift: (env:+BAWTHUB_AUTH_SESSIONS)")
    with pytest.raises(OpsDispatchError) as caught:
        service.prepare_invocation("bawthub.deploy-prod-image", ARGS)
    assert caught.value.code == "config_drift" and "env:+BAWTHUB_AUTH_SESSIONS" in str(caught.value)
    executor.config_error = ConfigCheckError("config_reconcile_unavailable", "config_reconcile_unavailable: old")
    with pytest.raises(OpsDispatchError) as caught:
        service.prepare_invocation("bawthub.deploy-prod-image", ARGS)
    assert caught.value.code == "config_reconcile_unavailable"


def test_unverified_release_is_refused_before_approval():
    service, _store, _ex = _service(FakeVerifier("workflow run 123456 branch is 'evil'"))
    with pytest.raises(OpsDispatchError) as caught:
        service.prepare_invocation("bawthub.deploy-prod-image", ARGS)
    assert caught.value.code == "release_unverified"


@pytest.mark.parametrize("args", [
    {**ARGS, "digest": "sha256:" + "D" * 64},
    {"release_run_id": "12; rm -rf /"},
    {},
])
def test_malformed_deploy_args_never_reach_the_verifier(args):
    service, _store, _ex = _service(FakeVerifier("must not be called"))
    with pytest.raises(OpsDispatchError) as caught:
        service.prepare_invocation("bawthub.deploy-prod-image", args)
    assert caught.value.code == "args_invalid"


def test_unverified_or_foreign_release_id_is_refused_before_approval():
    service, _store, _ex = _service()
    with pytest.raises(OpsDispatchError) as caught:
        service.prepare_invocation("bawthub.deploy-prod-image", {"release_run_id": "0" * 32})
    assert caught.value.code == "release_unverified"
    with Session(service.releases.engine) as session:
        row = session.get(ReleaseRun, RELEASE_ID)
        row.image_repository = "ghcr.io/other/frontend"
        session.add(row)
        session.commit()
    with pytest.raises(OpsDispatchError, match="stored release binding mismatch"):
        service.prepare_invocation("bawthub.deploy-prod-image", ARGS)


def test_deploy_snapshot_is_rejected_if_release_record_changes_after_approval():
    service, _store, _ex = _service()
    snapshot = service.prepare_invocation("bawthub.deploy-prod-image", ARGS)
    with Session(service.releases.engine) as session:
        row = session.get(ReleaseRun, RELEASE_ID)
        row.digest = "sha256:" + "b" * 64
        session.add(row)
        session.commit()
    with pytest.raises(OpsDispatchError) as caught:
        service.dispatch_job(operation_slug="bawthub.deploy-prod-image", args=ARGS,
                             approved_snapshot=snapshot)
    assert caught.value.code == "snapshot_invalid"


def test_forged_release_binding_in_snapshot_is_rejected():
    service, _store, _ex = _service()
    snapshot = service.prepare_invocation("bawthub.deploy-prod-image", ARGS)
    snapshot["release"]["digest"] = "sha256:" + "e" * 64
    with pytest.raises(OpsDispatchError) as caught:
        service.dispatch_job(operation_slug="bawthub.deploy-prod-image", args=ARGS,
                             approved_snapshot=_rehash(snapshot))
    assert caught.value.code == "snapshot_invalid"


def test_ordinary_snapshot_may_not_smuggle_a_release():
    assert binding_matches_args({"spec": {"action": "restart"}, "resolved_args": {}})
    assert not binding_matches_args({"spec": {"action": "restart"}, "resolved_args": {}, "release": {}})


def _succeeded_deploy(service, store):
    job = service.dispatch_job(operation_slug="bawthub.deploy-prod-image", args=ARGS, idempotency_key="deploy-1")
    record = {"schema": DEPLOY_SCHEMA, "action": "deploy", "phase": "done", "target": NAME,
              "previous": {"image_id": OLD_ID, "image_ref": "bawthub-frontend-prod:latest", "release": None},
              "deployed": {"image_id": NEW_ID, "image_ref": f"{REPO}@{DIGEST}", "release": None}}
    store.mark_terminal(job["id"], state="succeeded", exit_code=0, output_tail=json.dumps(record))
    return job["id"]


def test_rollback_binds_last_known_good_from_succeeded_deploy():
    service, store, executor = _service()
    deploy_id = _succeeded_deploy(service, store)
    assert store.get_job(deploy_id).to_api()["deployment"]["previous"]["image_id"] == OLD_ID
    executor.current_image = NEW_ID
    checks = len(executor.config_checks)
    executor.config_error = ConfigCheckError("config_drift", "config_drift: (env:~X)")
    snapshot = service.prepare_invocation("bawthub.rollback-prod-image", {"deploy_job_id": deploy_id})
    assert snapshot["rollback"]["to_image_id"] == OLD_ID and snapshot["rollback"]["from_image_id"] == NEW_ID
    assert len(executor.config_checks) == checks  # rollback never checks config


def test_rollback_refuses_when_target_moved_on_or_job_not_a_deploy():
    service, store, executor = _service()
    deploy_id = _succeeded_deploy(service, store)
    executor.current_image = "sha256:" + "7" * 64
    with pytest.raises(OpsDispatchError, match="no longer runs"):
        service.prepare_invocation("bawthub.rollback-prod-image", {"deploy_job_id": deploy_id})
    with pytest.raises(OpsDispatchError, match="SUCCEEDED deploy"):
        service.prepare_invocation("bawthub.rollback-prod-image", {"deploy_job_id": "0" * 32})


def test_ordinary_jobs_expose_no_deployment():
    from llm_bawt.ops.models import OpsJob
    for tail in ("Docker restart completed", '{"schema": "other"}', None):
        assert OpsJob(id="1" * 32, operation_slug="x", output_tail=tail).to_api()["deployment"] is None


# ── GitHubReleaseVerifier against a fake GitHub ─────────────────────────────

class _Resp:
    def __init__(self, status=200, body=b"", location=None):
        self.status, self._body, self.headers = status, body, {"Location": location} if location else {}

    def read(self, _n=-1):
        return self._body


class FakeGitHub:
    def __init__(self, *, run=None, receipt=None, tag_commit=SOURCE, annotated=True, artifact="release-receipt"):
        api = "https://api.github.com/repos/bawthub/bawthub"
        self.seen: list[tuple[str, str | None]] = []
        run = {"status": "completed", "conclusion": "success", "path": SPEC["workflow_path"] + "@refs/heads/main",
               "head_branch": "main", "event": "workflow_dispatch", "head_sha": TRIGGER,
               "repository": {"full_name": "bawthub/bawthub"}, "html_url": "https://run", **(run or {})}
        receipt = {"schema": "bawthub.release-receipt/v1", "status": "complete", "repository": "bawthub/bawthub",
                   "workflow_run_id": "123456", "workflow_run_attempt": "1", "base_sha": TRIGGER,
                   "source_sha": SOURCE, "version": "1.4.2", "tag": "v1.4.2", "digest": DIGEST,
                   "image_repository": REPO, **(receipt or {})}
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("release.json", json.dumps(receipt))
        tag_obj = {"type": "tag", "sha": "9" * 40} if annotated else {"type": "commit", "sha": tag_commit}
        self.routes = {
            f"{api}/actions/runs/123456": _Resp(body=json.dumps(run).encode()),
            f"{api}/actions/runs/123456/artifacts?name={artifact}": _Resp(body=json.dumps({"artifacts": [
                {"name": artifact, "expired": False, "archive_download_url": f"{api}/zip"}]}).encode()),
            f"{api}/zip": _Resp(302, location="https://blob.example/signed"),
            "https://blob.example/signed": _Resp(body=archive.getvalue()),
            f"{api}/git/ref/tags/v1.4.2": _Resp(body=json.dumps({"object": tag_obj}).encode()),
            f"{api}/git/tags/{'9' * 40}": _Resp(body=json.dumps({"object": {"type": "commit", "sha": tag_commit}}).encode()),
        }

    def open(self, request, timeout):
        self.seen.append((request.full_url, request.get_header("Authorization")))
        resp = self.routes.get(request.full_url)
        if resp is None and "/artifacts?name=" in request.full_url:
            # Like GitHub: an unknown artifact name is an empty listing, not 404.
            return _Resp(body=b'{"total_count": 0, "artifacts": []}')
        if resp is None:
            raise urllib.error.HTTPError(request.full_url, 404, "nf", {}, None)
        return resp


def test_github_verifier_accepts_bound_release_and_strips_auth_on_redirect():
    gh = FakeGitHub()
    release = GitHubReleaseVerifier("tok", opener=gh).verify(SPEC, VERIFY_ARGS)
    assert release["image_ref"] == f"{REPO}@{DIGEST}" and release["trigger_sha"] == TRIGGER
    auth = dict(gh.seen)
    assert auth["https://blob.example/signed"] is None
    assert auth["https://api.github.com/repos/bawthub/bawthub/actions/runs/123456"] == "Bearer tok"


@pytest.mark.parametrize("gh_kwargs, message", [
    ({"run": {"head_branch": "feature"}}, "branch"),
    ({"run": {"event": "push"}}, "event"),
    ({"run": {"conclusion": "failure"}}, "conclusion"),
    ({"run": {"path": ".github/workflows/other.yml"}}, "workflow path"),
    ({"receipt": {"digest": "sha256:" + "e" * 64}}, "digest"),
    ({"receipt": {"status": "partial"}}, "status"),
    ({"receipt": {"base_sha": "4" * 40}}, "base_sha"),
    ({"tag_commit": "5" * 40}, "does not resolve"),
])
def test_github_verifier_rejects_mismatches(gh_kwargs, message):
    with pytest.raises(ReleaseVerificationError, match=message):
        GitHubReleaseVerifier("tok", opener=FakeGitHub(**gh_kwargs)).verify(SPEC, VERIFY_ARGS)


def test_github_verifier_reads_latest_attempt_receipt():
    # TASK-1030: re-run attempt 2 publishes release-receipt-2.
    gh = FakeGitHub(run={"run_attempt": 2}, artifact="release-receipt-2",
                    receipt={"workflow_run_attempt": "2"})
    release = GitHubReleaseVerifier("tok", opener=gh).verify(SPEC, VERIFY_ARGS)
    assert release["workflow_run_attempt"] == "2"
    assert release["receipt_status"] == "complete"
    # The attempt-specific artifact was used; the legacy name never consulted.
    urls = [u for u, _ in gh.seen]
    assert any(u.endswith("artifacts?name=release-receipt-2") for u in urls)
    assert not any(u.endswith("artifacts?name=release-receipt") for u in urls)


def test_github_verifier_rejects_attempt_receipt_for_other_attempt():
    gh = FakeGitHub(run={"run_attempt": 2}, artifact="release-receipt-2",
                    receipt={"workflow_run_attempt": "1"})
    with pytest.raises(ReleaseVerificationError, match="workflow_run_attempt"):
        GitHubReleaseVerifier("tok", opener=gh).verify(SPEC, VERIFY_ARGS)


def test_github_verifier_never_falls_back_to_an_earlier_attempt_artifact():
    # Attempt 2 has no receipt yet; attempt 1's artifact must not satisfy it.
    gh = FakeGitHub(run={"run_attempt": 2}, artifact="release-receipt-1",
                    receipt={"workflow_run_attempt": "1"})
    with pytest.raises(ReleaseVerificationError, match="release-receipt-2"):
        GitHubReleaseVerifier("tok", opener=gh).verify(SPEC, VERIFY_ARGS)


def test_github_verifier_legacy_receipt_still_verifies_when_attempt_known():
    gh = FakeGitHub(run={"run_attempt": 1})
    release = GitHubReleaseVerifier("tok", opener=gh).verify(SPEC, VERIFY_ARGS)
    assert release["receipt_status"] == "complete"


def test_github_verifier_fails_closed_without_token():
    with pytest.raises(ReleaseVerificationError, match="token not connected"):
        GitHubReleaseVerifier("", opener=FakeGitHub()).verify(SPEC, VERIFY_ARGS)


def test_compose_declared_value_equal_to_old_image_default_survives_new_default(tmp_path):
    # TASK-1518 r1 B2: the clone drops values the OLD image supplied; a value
    # Compose declares must stay explicit when the NEW image's default changes.
    api = FakeDocker(compose=compose_project(environment={"NODE_ENV": "production", "PATH": "/usr/bin"}))
    api._new["Config"]["Env"] = ["PATH=/newbin"]
    assert json.loads(_deployer(api, tmp_path).run())["phase"] == "done"
    assert "PATH=/usr/bin" in api.create_bodies[0]["Env"]
