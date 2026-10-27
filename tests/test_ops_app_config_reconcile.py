"""TASK-1474: same-image Compose config reconcile of the Compose-BUILT app. No live Docker or network."""
from __future__ import annotations

import copy
import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

from llm_bawt.ops.app_config_reconcile import (
    SCHEMA, SOURCE_GUARD_LABEL, WORKER_CAPABILITY_LABEL, AppConfigReconciler, app_config_binding,
    approval_text, valid_app_binding,
)
from llm_bawt.ops.compose_config import BUILT_PROFILE, PINNED_PROFILE, translate
from llm_bawt.ops.compose_render import ComposeError, ComposeRenderer
from llm_bawt.ops.config_reconcile import valid_config_binding
from llm_bawt.ops.executor import ConfigCheckError, DockerExecutor, validate_spec
from llm_bawt.ops.image_deploy import DeployFailed
from llm_bawt.ops.release import ReleaseVerificationError
from llm_bawt.ops.seeds import SEEDS
from llm_bawt.ops.source_state import SourceStateError
from llm_bawt.ops.validation import canonical_json
from llm_bawt.ops.service import OpsDispatchError
from llm_bawt.ops.worker import OPS_JOB_LABEL, run as worker_run
from tests.test_ops_image_deploy import (
    CONFIG_HASH, JOB_ID, RENDERER, RENDERER_ID, FakeDocker, FakeVerifier, _frame, _service, _Stream,
    compose_project,
)

SEED = next(s for s in SEEDS if s["slug"] == "llm-bawt.reconcile-app-config")
SPEC = json.loads(SEED["command_script"])
FRONTEND_SPEC = json.loads(next(s for s in SEEDS if s["slug"] == "bawthub.reconcile-prod-config")["command_script"])
APP_NAME = SPEC["container_name"]
TAG = f"{SPEC['compose_project']}-{SPEC['compose_service']}"
APP_DIR = "/srv/compose/llm-bawt"
APP_FILES = [f"{APP_DIR}/docker-compose.yml", f"{APP_DIR}/docker-compose.dev.yml"]
APP_ID, NEWER_BUILD = "sha256:" + "3" * 64, "sha256:" + "4" * 64
REDIS = "5" * 64
SECRET = "sk-live-never-shown"
CMD = ["llm-service", "--host", "0.0.0.0"]
PORT = {"mode": "ingress", "target": 8642, "published": "8642", "protocol": "tcp"}
WORKER = "sha256:" + "0" * 64
EXECUTION = {"start_delay_seconds": 0, "timeout_seconds": 60, "max_output_bytes": 65536}
HEAD = "ab" * 20
OPS_VOLUME = "/var/lib/llm-bawt-ops"
SOURCE_PLAN = {"paths": ["server.sh", "src"], "compose_files": ["docker-compose.yml", "docker-compose.dev.yml"],
               "bytecode_ok": False}


class StubChecker:
    """Stands in for the sealed git check (proven against real git in
    test_ops_source_state.py). Answers are used in order; the last repeats."""

    def __init__(self, *answers):
        self.answers = list(answers) or [HEAD]
        self.calls = []

    def check(self, project_dir, plan):
        self.calls.append((project_dir, plan))
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        return answer


@pytest.fixture(autouse=True)
def source(monkeypatch):
    """The executor's and the worker's sealed source check, stubbed."""
    stub = StubChecker()
    monkeypatch.setattr("llm_bawt.ops.executor.SourceChecker", lambda *a, **k: stub)
    monkeypatch.setattr("llm_bawt.ops.app_config_reconcile.SourceChecker", lambda *a, **k: stub)
    return stub


def _bind_mount(name):
    return {"type": "bind", "source": f"{APP_DIR}/{name}", "target": f"/app/{name}",
            "bind": {"create_host_path": True}}


def app_compose(**service):
    """What ``docker compose config --format json app`` prints: no drift by default."""
    body = {"build": {"context": APP_DIR, "dockerfile": "Dockerfile"}, "command": CMD,
            "container_name": APP_NAME, "entrypoint": None,
            "depends_on": {"redis": {"condition": "service_healthy", "required": True}},
            "environment": {"LLM_BAWT_SERVICE_PORT": "8642"}, "networks": {"default": None},
            "ports": [dict(PORT)], "restart": "unless-stopped", "stdin_open": True, "tty": True,
            "volumes": [_bind_mount("src"), _bind_mount("server.sh"),
                        {"type": "volume", "source": "receipts", "target": OPS_VOLUME, "volume": {}}]}
    body.update(service)
    return {"name": "llm-bawt", "services": {"app": body}, "networks": {"default": {"name": "llm-bawt_default"}},
            "volumes": {"receipts": {"name": "llm-bawt_receipts"}}}


def _image(image_id, refs):
    return {"Id": image_id, "RepoDigests": [], "_health": "healthy", "_refs": refs,
            "Config": {"Env": ["PATH=/usr/bin"], "Cmd": ["python3"], "ExposedPorts": {"8001/tcp": {}},
                       "Labels": {"org.opencontainers.image.title": "llm-bawt"},
                       "Healthcheck": {"Test": ["CMD", "curl", "-f", "http://localhost:8642/health"]}}}


class AppDocker(FakeDocker):
    """The app as Compose built it, a redis dependency, in-container curl."""

    def __init__(self, *, compose=None, tag_moved=False, health_status="ok", health_exit=0,
                 fail_new_start=False, restore_fails=False):
        super().__init__(compose=app_compose() if compose is None else compose)
        self.health_status, self.health_exit = health_status, health_exit
        self.fail_new_start, self.restore_fails = fail_new_start, restore_fails
        self.exec_cmds = []
        self.images = {APP_ID: _image(APP_ID, [] if tag_moved else [TAG]),
                       RENDERER_ID: {"Id": RENDERER_ID, "RepoDigests": [RENDERER], "Config": {}}}
        if tag_moved:
            self.images[NEWER_BUILD] = _image(NEWER_BUILD, [TAG])
        old_id = "c" * 64
        self.containers = {old_id: {
            "Id": old_id, "Name": f"/{APP_NAME}", "Image": APP_ID,
            "Config": {"Image": TAG, "Env": ["PATH=/usr/bin", "LLM_BAWT_SERVICE_PORT=8642"], "Cmd": list(CMD),
                       "OpenStdin": True, "Tty": True, "ExposedPorts": {"8001/tcp": {}, "8642/tcp": {}},
                       "Healthcheck": {"Test": ["CMD", "curl", "-f", "http://localhost:8642/health"]},
                       "Labels": {"com.docker.compose.project": "llm-bawt", "com.docker.compose.service": "app",
                                  "com.docker.compose.project.working_dir": APP_DIR,
                                  "com.docker.compose.project.config_files": ",".join(APP_FILES),
                                  "org.opencontainers.image.title": "llm-bawt"}},
            "HostConfig": {"RestartPolicy": {"Name": "unless-stopped"},
                           "Binds": [f"{APP_DIR}/src:/app/src:rw", f"{APP_DIR}/server.sh:/app/server.sh:rw"],
                           "Mounts": [{"Type": "volume", "Source": "llm-bawt_receipts", "Target": OPS_VOLUME,
                                       "ReadOnly": False}],
                           "PortBindings": {"8642/tcp": [{"HostIp": "", "HostPort": "8642"}]}},
            "NetworkSettings": {"Networks": {"llm-bawt_default": {"Aliases": [APP_NAME, "app", old_id[:12]]}}},
            "State": {"Running": True, "Status": "running", "Health": {"Status": "healthy"}}},
            REDIS: {"Id": REDIS, "Name": "/llm-bawt-redis-1", "Image": "sha256:" + "6" * 64,
                    "Config": {"Labels": {"com.docker.compose.project": "llm-bawt",
                                          "com.docker.compose.service": "redis"}},
                    "State": {"Running": True, "Status": "running", "Health": {"Status": "healthy"}}}}
        self.old_id = old_id

    def attach_stream(self, container_id):
        argv = self.containers[container_id]["Config"]["Cmd"]
        if "--hash" in argv:
            return super().attach_stream(container_id)
        assert self.containers[container_id]["Config"]["Env"] == []  # nothing to interpolate
        return _Stream(_frame(1, json.dumps(self.compose)))

    def request(self, method, path, *, body=None, headers=None, raw=False):
        url = urlsplit(path)
        if url.path == "/containers/json":
            self.calls.append((method, path))
            wanted = dict(item.split("=", 1) for item in json.loads(parse_qs(url.query)["filters"][0])["label"])
            return [{"Id": c["Id"], "Labels": c["Config"]["Labels"]} for c in self.containers.values()
                    if all((c["Config"].get("Labels") or {}).get(k) == v for k, v in wanted.items())]
        parts = path.split("/")
        if method == "POST" and path.endswith("/exec") and body["Cmd"][0] in ("curl", "stat"):
            self.calls.append((method, path))
            self.exec_cmds.append(body["Cmd"])
            return {"Id": "health1" if body["Cmd"][0] == "curl" else "stat1"}
        if path.startswith("/exec/health1/"):
            self.calls.append((method, path))
            if path.endswith("/start"):
                payload = json.dumps({"status": self.health_status, "version": "x"}).encode()
                return struct.pack(">BxxxI", 1, len(payload)) + payload
            return {"ExitCode": self.health_exit}
        if method == "POST" and path.endswith("/start") and parts[1] == "containers" \
                and not self.containers.get(parts[2], {}).get("_render"):
            if self.fail_new_start and parts[2] != self.old_id:
                raise RuntimeError("Docker HTTP 500: cannot start")
            if self.restore_fails and parts[2] == self.old_id:
                raise RuntimeError("Docker HTTP 500: cannot restart")
        return super().request(method, path, body=body, headers=headers, raw=raw)


def binding(api, spec=SPEC, *, moved=False, checker=None):
    return app_config_binding(spec, api.by_name(APP_NAME), api.images[APP_ID],
                              ComposeRenderer(api, spec, tag="app1"), image_tag_moved=moved,
                              source_checker=checker or StubChecker())[0]


def snapshot(api, **kwargs):
    return {"spec": SPEC, "resolved_args": {}, "config": binding(api, **kwargs), "execution": EXECUTION}


def reconciler(api, tmp_path, snap=None):
    """Its source check is the ``source`` fixture's stub."""
    return AppConfigReconciler(api, snap or snapshot(api), JOB_ID, lock_root=tmp_path / "locks",
                               sleep=lambda _: None, clock=iter(range(0, 10000, 5)).__next__)


# ── catalog seed ────────────────────────────────────────────────────────────

def test_seed_is_disabled_delayed_and_warns_it_cuts_turns():
    assert not SEED["enabled"] and SEED["max_concurrent"] == 1 and SEED["risk_level"] == "high"
    assert SEED["start_delay_seconds"] > 0 and "Cuts in-flight chat turns" in SEED["description"]
    assert json.loads(SEED["args_schema_json"]) == {"type": "object", "properties": {}, "additionalProperties": False}
    assert validate_spec(SEED["command_script"]) == SPEC
    assert SPEC["compose_config"] == {"renderer_image": FRONTEND_SPEC["compose_config"]["renderer_image"]}
    for key, value in (("compose_config", {**SPEC["compose_config"], "compose_files": ["docker-compose.yml"]}),
                       ("compose_config", {**SPEC["compose_config"], "image_variable": "X"}),
                       ("compose_config", {"renderer_image": "docker:latest"}),
                       ("health_url", "http://evil.example:8642/health"),
                       ("health_url", "http://127.0.0.1:8642/health?x=1"),
                       ("image_repository", "ghcr.io/x/y"), ("stat_paths", ["/../etc"])):
        with pytest.raises(ValueError):
            validate_spec(json.dumps({**SPEC, key: value}))


# ── rule 0: the frontend profile is unchanged ──────────────────────────────

@pytest.mark.parametrize("key, value", [
    ("build", {"context": "."}), ("ports", [dict(PORT)]), ("container_name", "bawthub-frontend-prod-1"),
    ("stdin_open", True), ("tty", True), ("depends_on", {"redis": {"condition": "service_started"}}),
])
def test_pinned_profile_still_refuses_process_keys(key, value):
    ref = "ghcr.io/bawthub/frontend@sha256:" + "a" * 64
    rendered = json.loads(json.dumps(compose_project(**{key: value})).replace("${BAWTHUB_FRONTEND_IMAGE}", ref))
    with pytest.raises(ComposeError) as caught:
        translate(rendered, FRONTEND_SPEC, ref, PINNED_PROFILE)
    assert caught.value.code == "unsupported_compose_key" and caught.value.names == [key]
    with pytest.raises(ComposeError) as caught:
        translate(json.loads(json.dumps(compose_project(command=["sh"])).replace(
            "${BAWTHUB_FRONTEND_IMAGE}", ref)), FRONTEND_SPEC, ref)
    assert caught.value.code == "unsupported_compose_key" and caught.value.names == ["command"]


def test_built_profile_refuses_an_image_key_and_requires_build():
    for service, code in (({"image": TAG}, "unsupported_compose_key"), ({"build": None}, "compose_image_mismatch")):
        rendered = app_compose(**service)
        with pytest.raises(ComposeError) as caught:
            translate(rendered, SPEC, TAG, BUILT_PROFILE)
        assert caught.value.code == code


# ── binding ─────────────────────────────────────────────────────────────────

def test_binding_shows_names_ports_and_dependencies_only():
    api = AppDocker(compose=app_compose(environment={"LLM_BAWT_SERVICE_PORT": "8642", "XAI_API_KEY": SECRET}))
    bound = binding(api)
    assert bound["drift"] == ["env:+XAI_API_KEY"]
    assert bound["ports"] == ["*:8642->8642/tcp"] and bound["depends_on"] == ["redis:service_healthy"]
    assert bound["source"] == {"project_dir": APP_DIR, "compose_files": APP_FILES}
    assert bound["image_ref"] == TAG and bound["image_id"] == APP_ID and bound["image_tag"] == TAG
    assert valid_app_binding(snapshot(api)) and SECRET not in json.dumps(bound)
    assert bound["source_commit"] == {"repository": SPEC["source_repository"], "branch": SPEC["source_branch"],
                                      "head": HEAD}
    text = approval_text(SPEC, bound, None)
    assert "cuts in-flight chat turns" in text and "redis:service_healthy" in text and SECRET not in text
    assert f"Source: github.com/{SPEC['source_repository']} {SPEC['source_branch']} @ {HEAD}" in text
    assert "now names a different local build" not in text


def test_binding_refuses_a_foreign_image_and_bad_label_files():
    api = AppDocker()
    api.by_name(APP_NAME)["Config"]["Image"] = "ghcr.io/x/app:latest"
    with pytest.raises(ValueError, match="Compose-built image"):
        binding(api)
    api = AppDocker()
    api.by_name(APP_NAME)["Config"]["Labels"]["com.docker.compose.project.config_files"] = "/etc/compose.yml"
    with pytest.raises(ComposeError, match="compose_source_invalid"):
        binding(api)


@pytest.mark.parametrize("tamper", [
    lambda s: s["config"].update(env={"XAI_API_KEY": SECRET}),
    lambda s: s["config"].update(image_tag_moved="no"),
    lambda s: s["config"].update(image_tag="evil-app"),
    lambda s: s["config"].update(image_ref="ghcr.io/evil/app:1"),
    lambda s: s["config"]["source"].update(compose_files=["/etc/compose.yml"]),
    lambda s: s["config"].update(ports=[1]),
    lambda s: s.update(resolved_args={"x": 1}),
    lambda s: s["config"].pop("source_commit"),
    lambda s: s["config"]["source_commit"].update(head="HEAD"),
    lambda s: s["config"]["source_commit"].update(repository="evil/llm-bawt"),
    lambda s: s["config"]["source_commit"].update(branch="feature"),
    lambda s: s["config"]["source_commit"].update(extra=1),
    lambda s: s.update(spec={**SPEC, "source_mounts": ["/app/src", "/app/src/llm_bawt"]}),
    lambda s: s.update(spec={**SPEC, "data_mounts": [*SPEC["data_mounts"], "/app"]}),
])
def test_binding_shape_is_validated(tamper):
    snap = snapshot(AppDocker())
    assert valid_app_binding(snap)
    tamper(snap)
    assert not valid_app_binding(snap)


def test_the_frontend_validator_never_accepts_an_app_binding():
    assert not valid_config_binding(snapshot(AppDocker()))


# ── rule 4: ports ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("ports, names, bindings", [
    ([dict(PORT), {**PORT, "target": 8001, "published": "8001"}], ["port:+8001/tcp"],
     {"8001/tcp": [{"HostIp": "", "HostPort": "8001"}], "8642/tcp": [{"HostIp": "", "HostPort": "8642"}]}),
    ([], ["port:-8642/tcp"], {}),
    ([{**PORT, "published": "9642"}], ["port:~8642/tcp"], {"8642/tcp": [{"HostIp": "", "HostPort": "9642"}]}),
    ([{**PORT, "host_ip": "127.0.0.1"}], ["port:~8642/tcp"],
     {"8642/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8642"}]}),
], ids=["map", "unmap", "remap", "host_ip"])
def test_port_changes_are_drift_and_applied(tmp_path, ports, names, bindings):
    api = AppDocker(compose=app_compose(ports=ports))
    snap = snapshot(api)
    assert snap["config"]["drift"] == names
    result = json.loads(reconciler(api, tmp_path, snap).run())
    assert result["schema"] == SCHEMA and result["phase"] == "done"
    body = api.create_bodies[0]
    assert body["HostConfig"]["PortBindings"] == bindings
    assert body["ExposedPorts"] == {"8001/tcp": {}, **{k: {} for k in bindings}}
    assert body["Image"] == APP_ID


@pytest.mark.parametrize("port", [
    {**PORT, "app_protocol": "http"}, {**PORT, "name": "web"}, {**PORT, "mode": "host"},
    {k: v for k, v in PORT.items() if k != "mode"}, {k: v for k, v in PORT.items() if k != "published"},
    {**PORT, "published": "8000-8001"}, {**PORT, "protocol": "sctp"}, {**PORT, "host_ip": "0.0.0.0/8"},
    "8642:8642",
])
def test_unmodelled_port_forms_refuse(port):
    with pytest.raises(ComposeError) as caught:
        translate(app_compose(ports=[port]), SPEC, TAG, BUILT_PROFILE)
    assert caught.value.code == "unsupported_compose_key" and caught.value.names[0].startswith("ports")


# ── rules 2, 3, 5: command, container_name, stdin_open/tty ─────────────────

def test_absent_command_is_the_image_default(tmp_path):
    api = AppDocker(compose=app_compose(command=None))
    api.by_name(APP_NAME)["Config"]["Cmd"] = ["python3"]
    assert binding(api)["drift"] == []
    api.by_name(APP_NAME)["Config"]["Cmd"] = list(CMD)
    snap = snapshot(api)
    assert snap["config"]["drift"] == ["command:~"]
    reconciler(api, tmp_path, snap).run()
    assert "Cmd" not in api.create_bodies[0]


def test_changed_command_is_applied_and_entrypoint_must_be_null(tmp_path):
    api = AppDocker(compose=app_compose(command=["llm-service", "--debug"]))
    snap = snapshot(api)
    assert snap["config"]["drift"] == ["command:~"]
    reconciler(api, tmp_path, snap).run()
    assert api.create_bodies[0]["Cmd"] == ["llm-service", "--debug"]
    for service in ({"entrypoint": ["sh", "-c"]}, {"command": "llm-service --debug"}):
        with pytest.raises(ComposeError, match="unsupported_compose_key"):
            translate(app_compose(**service), SPEC, TAG, BUILT_PROFILE)


def test_container_name_must_equal_the_spec():
    with pytest.raises(ComposeError, match="compose_container_name_mismatch"):
        translate(app_compose(container_name="other"), SPEC, TAG, BUILT_PROFILE)
    with pytest.raises(ComposeError, match="compose_container_name_mismatch"):
        translate(app_compose(container_name=None), SPEC, TAG, BUILT_PROFILE)


def test_absent_stdin_open_and_tty_mean_false(tmp_path):
    compose = app_compose()
    del compose["services"]["app"]["stdin_open"], compose["services"]["app"]["tty"]
    api = AppDocker(compose=compose)
    snap = snapshot(api)
    assert snap["config"]["drift"] == ["stdin_open:~", "tty:~"]
    reconciler(api, tmp_path, snap).run()
    assert api.create_bodies[0]["OpenStdin"] is False and api.create_bodies[0]["Tty"] is False


# ── rule 6: depends_on is a precondition ───────────────────────────────────

@pytest.mark.parametrize("dep", [
    {"condition": "service_healthy", "required": True, "restart": True},
    {"condition": "service_healthy", "required": False},
    {"condition": "service_completed_successfully", "required": True},
    {"condition": "service_healthy", "required": True, "extra": 1},
])
def test_unmodelled_dependencies_refuse(dep):
    with pytest.raises(ComposeError) as caught:
        translate(app_compose(depends_on={"redis": dep}), SPEC, TAG, BUILT_PROFILE)
    assert caught.value.code == "unsupported_compose_key" and caught.value.names == ["depends_on"]


DRIFT = app_compose(environment={"LLM_BAWT_SERVICE_PORT": "8642", "NEW": "1"})


@pytest.mark.parametrize("change, code", [
    ("missing", "dependency_unresolved: redis matched 0"),
    ("duplicate", "dependency_unresolved: redis matched 2"),
    ("stopped", "dependency_not_running: redis"),
    ("unhealthy", "dependency_unhealthy: redis"),
])
def test_dependency_must_be_up_right_before_the_swap(tmp_path, change, code):
    api = AppDocker(compose=DRIFT)
    snap = snapshot(api)
    redis = api.containers[REDIS]
    if change == "missing":
        redis["Config"]["Labels"]["com.docker.compose.service"] = "other"
    elif change == "duplicate":
        api.containers["7" * 64] = {**copy.deepcopy(redis), "Id": "7" * 64}
    elif change == "stopped":
        redis["State"] = {"Running": False, "Status": "exited"}
    else:
        redis["State"]["Health"]["Status"] = "unhealthy"
    with pytest.raises(DeployFailed, match=code) as caught:
        reconciler(api, tmp_path, snap).run()
    assert caught.value.outcome_known and not api.create_bodies and not api.side_effect_started
    assert not any(m != "GET" for m, p in api.calls if REDIS in p)  # a dependency is only read


def test_ops_workers_never_count_as_the_dependency_and_started_ignores_health(tmp_path):
    api = AppDocker(compose=app_compose(environment={"LLM_BAWT_SERVICE_PORT": "8642", "NEW": "1"},
                                        depends_on={"redis": {"condition": "service_started", "required": True}}))
    worker = copy.deepcopy(api.containers[REDIS])
    worker["Id"] = "8" * 64
    worker["Config"]["Labels"][OPS_JOB_LABEL] = "job"
    api.containers[worker["Id"]] = worker
    api.containers[REDIS]["State"]["Health"]["Status"] = "starting"
    json.loads(reconciler(api, tmp_path).run())
    assert len(api.create_bodies) == 1


# ── rule 1: the image is bound by ID; a moved tag is shown, never deployed ──

class _NotFound(Exception):
    status_code = 404


def _executor(api, *, worker_labels=None):
    def get_image(ref):
        if ref == WORKER:
            labels = ({WORKER_CAPABILITY_LABEL: "1", SOURCE_GUARD_LABEL: "1"} if worker_labels is None
                      else worker_labels)
            return SimpleNamespace(attrs={"Config": {"Labels": labels}})
        try:
            return SimpleNamespace(attrs=api._find_image(ref))
        except RuntimeError:
            raise _NotFound(ref) from None

    def get_container(name):
        return SimpleNamespace(attrs=copy.deepcopy(api.by_name(name)), reload=lambda: None)

    client = SimpleNamespace(images=SimpleNamespace(get=get_image, pull=lambda *a, **k: None),
                             containers=SimpleNamespace(get=get_container), ping=lambda: True)
    executor = DockerExecutor(client_factory=lambda: client, worker_image=WORKER, receipt_volume="receipts",
                              receipt_root="/receipts", render_api_factory=lambda: api)
    executor._check_prerequisites = lambda _settings: None
    return executor


def test_moved_tag_is_flagged_and_the_running_image_id_is_deployed(tmp_path):
    api = AppDocker(compose=DRIFT, tag_moved=True)
    approved = _executor(api).config_approval(SPEC)
    assert approved["image_tag_moved"] is True and approved["drift"] == ["env:+NEW"]
    assert "now names a different local build" in approval_text(SPEC, approved, None)
    snap = {"spec": SPEC, "resolved_args": {}, "config": approved, "execution": EXECUTION}
    result = json.loads(reconciler(api, tmp_path, snap).run())
    assert api.create_bodies[0]["Image"] == APP_ID and result["deployed"]["image_id"] == APP_ID
    assert result["config"]["image_tag_moved"] is True
    assert not any(TAG in p for _m, p in api.calls if p.startswith("/images/"))  # worker never resolves the tag


def test_unmoved_tag_and_a_target_already_running_by_id():
    api = AppDocker()
    assert _executor(api).config_approval(SPEC)["image_tag_moved"] is False
    api.by_name(APP_NAME)["Config"]["Image"] = APP_ID
    assert binding(api)["image_ref"] == APP_ID


@pytest.mark.parametrize("labels, missing", [
    ({"llm-bawt.ops.config-reconcile": "1"}, WORKER_CAPABILITY_LABEL),
    # TASK-1474: a worker that reconciles the app but predates the source guard.
    ({WORKER_CAPABILITY_LABEL: "1"}, SOURCE_GUARD_LABEL),
])
def test_legacy_worker_without_the_app_capability_refuses(source, labels, missing):
    api = AppDocker()
    executor = _executor(api, worker_labels=labels)
    for call in (lambda: executor.config_approval(SPEC),
                 lambda: executor.preflight({"spec": SPEC, "execution": executor.execution_settings()})):
        with pytest.raises(ConfigCheckError) as caught:
            call()
        assert caught.value.code == "config_reconcile_unavailable" and missing in str(caught.value)
    assert not api.render_bodies and not source.calls


# ── reconcile outcomes ──────────────────────────────────────────────────────

def test_drift_is_reconciled_and_values_stay_private(tmp_path):
    compose = app_compose(environment={"LLM_BAWT_SERVICE_PORT": "8642", "XAI_API_KEY": SECRET})
    api = AppDocker(compose=compose)
    snap = snapshot(api)
    output = reconciler(api, tmp_path, snap).run()
    result = json.loads(output)
    assert result["health"] == {"status": "ok", "stat": {}} and result["previous"]["image_id"] == APP_ID
    assert api.exec_cmds[-1] == ["curl", "-fsS", "--max-time", "10", SPEC["health_url"]]
    body = api.create_bodies[0]
    assert f"XAI_API_KEY={SECRET}" in body["Env"] and body["Cmd"] == CMD
    assert body["Labels"]["com.docker.compose.config-hash"] == CONFIG_HASH
    assert body["HostConfig"]["Binds"] == [f"{APP_DIR}/server.sh:/app/server.sh:rw", f"{APP_DIR}/src:/app/src:rw",
                                           f"llm-bawt_receipts:{OPS_VOLUME}:rw"]
    assert body["NetworkingConfig"]["EndpointsConfig"]["llm-bawt_default"]["Aliases"] == [APP_NAME, "app"]
    assert SECRET not in output and SECRET not in json.dumps(snap)
    assert not any("/images/create" in p for _m, p in api.calls) and not api.renderers_left()


def test_no_drift_verifies_only(tmp_path):
    api = AppDocker()
    result = json.loads(reconciler(api, tmp_path).run())
    assert result["unchanged"] and result["health"]["status"] == "ok"
    assert not api.create_bodies and not api.side_effect_started


@pytest.mark.parametrize("kwargs, text", [
    ({"health_exit": 22}, "in-container health request exited 22"),
    ({"health_status": "degraded"}, "health status did not match expected 'ok'"),
    ({"health_status": SECRET}, "health status did not match expected 'ok'"),
    ({"health_status": SECRET + " <x>"}, "health status did not match expected 'ok'"),
])
def test_failed_health_restores_the_previous_container(tmp_path, kwargs, text):
    api = AppDocker(compose=DRIFT, **kwargs)
    with pytest.raises(DeployFailed) as caught:
        reconciler(api, tmp_path).run()
    assert caught.value.outcome_known and caught.value.detail["restored"]
    assert text in str(caught.value) and "degraded" not in str(caught.value) and SECRET not in str(caught.value)
    assert SECRET not in json.dumps(caught.value.detail)
    assert api.by_name(APP_NAME)["Id"] == api.old_id


def test_returned_health_status_never_reaches_the_durable_receipt(tmp_path):
    api = AppDocker(compose=DRIFT, health_status=SECRET)
    root = tmp_path / "receipts"
    snap = {**snapshot(api), "execution": {**EXECUTION, "receipt_root": str(root)}}
    path, digest = _write_request(root, snap)
    assert worker_run(path, digest, api_factory=lambda _: api, sleep=lambda _: None)
    written = [p for p in (root / JOB_ID).iterdir() if p.name != "request.json"]
    assert written and all(SECRET not in p.read_text() for p in written)
    fresh = _executor(AppDocker())
    fresh._get_worker = lambda _name: None
    result = fresh.reconcile(job_id=JOB_ID, snapshot=snap)
    assert result.state == "failed" and SECRET not in (result.error or "") + (result.output_tail or "")


def test_change_since_approval_refuses_before_create(tmp_path):
    api = AppDocker(compose=DRIFT)
    snap = snapshot(api)
    api.compose = app_compose(ports=[{**PORT, "published": "9000"}])
    with pytest.raises(DeployFailed, match="config_changed_since_approval") as caught:
        reconciler(api, tmp_path, snap).run()
    assert caught.value.outcome_known and not api.create_bodies


# ── rule 7: the app recreates itself; the receipt outlives it ──────────────

def _write_request(root, snap):
    job_dir = root / JOB_ID
    job_dir.mkdir(parents=True)
    raw = canonical_json({"job_id": JOB_ID, "snapshot": snap}).encode()
    (job_dir / "request.json").write_bytes(raw)
    return job_dir / "request.json", hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("kwargs, state", [({}, "succeeded"), ({"fail_new_start": True}, "failed"),
                                           ({"fail_new_start": True, "restore_fails": True}, "lost")])
def test_worker_receipt_is_reconciled_by_a_fresh_app_process(tmp_path, kwargs, state):
    api = AppDocker(compose=DRIFT, **kwargs)
    root = tmp_path / "receipts"
    snap = {**snapshot(api), "execution": {**EXECUTION, "receipt_root": str(root)}}
    path, digest = _write_request(root, snap)
    assert worker_run(path, digest, api_factory=lambda _: api, sleep=lambda _: None) == (state != "succeeded")
    # The dispatching app was recreated: a fresh executor knows only the receipt.
    fresh = _executor(AppDocker())
    fresh._get_worker = lambda _name: None
    result = fresh.reconcile(job_id=JOB_ID, snapshot=snap)
    assert result.state == state
    if state == "lost":
        assert "RESTORE FAILED" in result.error
    if state == "succeeded":
        assert json.loads(result.output_tail)["schema"] == SCHEMA


def test_worker_image_carries_the_module_and_capability_label():
    dockerfile = (Path(__file__).resolve().parents[1] / "docker" / "Dockerfile.ops-worker").read_text()
    for module in ("app_config_reconcile", "source_state", "sealed_container"):
        assert f"src/llm_bawt/ops/{module}.py" in dockerfile
    assert f'LABEL {WORKER_CAPABILITY_LABEL}="1"' in dockerfile
    assert f'LABEL {SOURCE_GUARD_LABEL}="1"' in dockerfile


# ── app side: snapshot binding ─────────────────────────────────────────────

def test_approval_snapshot_is_bound_and_rejects_smuggling():
    from tests.test_ops_image_deploy import _rehash

    api = AppDocker(compose=DRIFT)
    service, store, executor = _service(PublishedVerifier(HEAD))
    store.create_operation({**SEED, "enabled": True})
    executor.config_approval = lambda spec: binding(api, spec)
    approved = service.prepare_invocation(SEED["slug"], {})
    assert approved["config"] == binding(api)
    assert service._verify_snapshot(approved, SEED["slug"], {}) == approved
    approved["config"]["env"] = {"XAI_API_KEY": SECRET}
    with pytest.raises(OpsDispatchError, match="config binding mismatch"):
        service._verify_snapshot(_rehash(approved), SEED["slug"], {})
    with pytest.raises(OpsDispatchError, match="unknown property 'x'"):
        service.prepare_invocation(SEED["slug"], {"x": 1})


# ── TASK-1474: the bind-mounted source is exactly the published commit ──────

class PublishedVerifier(FakeVerifier):
    """The forge's branch head, or an error reading it."""

    def __init__(self, head=None, *, error=None):
        super().__init__()
        self.head, self.branch_error, self.asked = head, error, []

    def branch_head(self, repo, branch):
        self.asked.append((repo, branch))
        if self.branch_error:
            raise ReleaseVerificationError(self.branch_error)
        return self.head


def test_the_checker_verifies_the_planned_source_paths(source):
    binding(AppDocker(), checker=source)
    assert source.calls == [(APP_DIR, SOURCE_PLAN)]


@pytest.mark.parametrize("prefix, ok", [
    (None, False), ("/tmp/llm-bawt-pycache", True), ("/app/src/.pycache", False),
    (f"{OPS_VOLUME}/pycache", False), ("/var/lib", False), ("tmp/pycache", False),
])
def test_bytecode_is_tolerated_only_with_a_pycache_prefix_outside_every_mount(source, prefix, ok):
    env = {"LLM_BAWT_SERVICE_PORT": "8642", **({"PYTHONPYCACHEPREFIX": prefix} if prefix else {})}
    binding(AppDocker(compose=app_compose(environment=env)), checker=source)
    assert source.calls[-1][1]["bytecode_ok"] is ok


@pytest.mark.parametrize("where", ["compose", "image"])
@pytest.mark.parametrize("name", ["PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "PYTHONSTARTUP", "LD_PRELOAD"])
def test_runtime_env_that_loads_code_from_elsewhere_refuses(source, where, name):
    api = AppDocker(compose=app_compose(environment={"LLM_BAWT_SERVICE_PORT": "8642", name: "/srv/x"})
                    if where == "compose" else None)
    if where == "image":
        api.images[APP_ID]["Config"]["Env"].append(f"{name}=/srv/x")
    with pytest.raises(SourceStateError) as caught:
        binding(api, checker=source)
    assert caught.value.code == "source_env_unsafe" and caught.value.names == [name]
    assert "/srv/x" not in str(caught.value) and not source.calls


def _volumes(*extra, drop=None):
    compose = app_compose()
    service = compose["services"]["app"]
    service["volumes"] = [v for v in service["volumes"] if v["target"] != drop] + list(extra)
    return compose


def _vol(target):
    return {"type": "volume", "source": "receipts", "target": target, "volume": {}}


def _bind(source, target="/app/server.sh"):
    return {"type": "bind", "source": source, "target": target, "bind": {"create_host_path": True}}


@pytest.mark.parametrize("compose, code, name", [
    (_volumes(_vol("/data")), "source_mount_unclassified", "/data"),
    (_volumes(_vol("/app/src/llm_bawt/plugins")), "source_mount_shadowed", "/app/src/llm_bawt/plugins"),
    (_volumes(drop="/app/server.sh"), "source_mount_missing", "/app/server.sh"),
    (_volumes(_bind("/srv/elsewhere/server.sh"), drop="/app/server.sh"), "source_mount_invalid", "/app/server.sh"),
    (_volumes(_bind(APP_DIR), drop="/app/server.sh"), "source_mount_invalid", "/app/server.sh"),
    (_volumes(_bind(f"{APP_DIR}/.git/hooks"), drop="/app/server.sh"), "source_mount_invalid", "/app/server.sh"),
    (_volumes(_vol("/app/server.sh"), drop="/app/server.sh"), "source_mount_invalid", "/app/server.sh"),
], ids=["unclassified", "shadowed", "missing", "outside", "project-root", "git-dir", "volume"])
def test_every_mount_is_classified_and_source_mounts_bind_the_project(source, compose, code, name):
    with pytest.raises(SourceStateError) as caught:
        binding(AppDocker(compose=compose), checker=source)
    assert caught.value.code == code and caught.value.names == [name] and not source.calls


@pytest.mark.parametrize("tamper, name", [
    (lambda api: api.by_name(APP_NAME)["HostConfig"].update(Tmpfs={"/run/scratch": ""}), "/run/scratch"),
    (lambda api: api.images[APP_ID]["Config"].update(Volumes={"/scratch": {}}), "/scratch"),
    (lambda api: api.by_name(APP_NAME)["HostConfig"]["Binds"].append("/srv/x:/opt/x:ro"), "/opt/x"),
], ids=["tmpfs", "image-volume", "current-bind"])
def test_mounts_compose_does_not_render_are_classified_too(source, tamper, name):
    api = AppDocker()
    tamper(api)
    with pytest.raises(SourceStateError) as caught:
        binding(api, checker=source)
    assert caught.value.code == "source_mount_unclassified" and caught.value.names == [name]


def test_an_unclean_checkout_refuses_approval_with_its_code(source):
    source.answers = [SourceStateError("source_dirty", ["src/llm_bawt/bots.py"])]
    with pytest.raises(ConfigCheckError) as caught:
        _executor(AppDocker()).config_approval(SPEC)
    assert caught.value.code == "source_dirty" and "src/llm_bawt/bots.py" in str(caught.value)


def test_checkout_moved_since_approval_refuses_before_create(tmp_path, source):
    api = AppDocker(compose=DRIFT)
    snap = snapshot(api)
    source.answers = ["cd" * 20]
    with pytest.raises(DeployFailed, match="source_moved_since_approval") as caught:
        reconciler(api, tmp_path, snap).run()
    assert caught.value.outcome_known and not api.create_bodies and not api.side_effect_started


@pytest.mark.parametrize("answer, code", [
    ("cd" * 20, "source_moved_since_approval"),
    (SourceStateError("source_bytecode_unsafe", ["src/llm_bawt/__pycache__/"]), "source_bytecode_unsafe"),
    (SourceStateError("source_unsupported_entry", ["src/llm_bawt/module.py"]), "source_unsupported_entry"),
    (SourceStateError("source_unverifiable", detail="sealed git unavailable"), "source_unverifiable"),
], ids=["moved", "pyc", "symlink", "unverifiable"])
def test_source_is_rechecked_as_the_last_read_before_the_swap(tmp_path, source, answer, code):
    api = AppDocker(compose=DRIFT)
    snap = snapshot(api)
    source.answers = [HEAD, answer]  # the bind passes; the pre-swap recheck sees the change
    with pytest.raises(DeployFailed, match=code) as caught:
        reconciler(api, tmp_path, snap).run()
    assert caught.value.outcome_known and not api.create_bodies and not api.side_effect_started
    assert len(source.calls) == 2 and source.calls[0] == source.calls[1] == (APP_DIR, SOURCE_PLAN)
    assert api.by_name(APP_NAME)["State"]["Running"]


def test_a_reconcile_records_the_commit_and_verify_only_checks_once(tmp_path, source):
    result = json.loads(reconciler(AppDocker(compose=DRIFT), tmp_path).run())
    assert result["config"]["source_commit"]["head"] == HEAD and len(source.calls) == 2  # bind, pre-swap
    source.calls.clear()
    reconciler(AppDocker(), tmp_path).run()
    assert len(source.calls) == 1  # bind only: nothing is swapped


def test_published_source_is_asked_of_the_forge_for_the_spec_repository():
    verifier = PublishedVerifier(HEAD)
    service, store, executor = _service(verifier)
    store.create_operation({**SEED, "enabled": True})
    api = AppDocker(compose=DRIFT)
    executor.config_approval = lambda spec: binding(api, spec)
    assert service.prepare_invocation(SEED["slug"], {})["config"]["source_commit"]["head"] == HEAD
    assert verifier.asked == [(SPEC["source_repository"], SPEC["source_branch"])]


@pytest.mark.parametrize("verifier, code", [
    (PublishedVerifier("cd" * 20), "source_not_published"),
    (PublishedVerifier(error="GitHub unreachable: URLError"), "source_unverifiable"),
    (FakeVerifier(), "source_unverifiable"),  # a verifier that cannot read branch heads
], ids=["unpublished", "forge-error", "no-branch-reader"])
def test_unpublished_source_is_refused_before_anyone_is_asked(verifier, code):
    service, store, executor = _service(verifier)
    store.create_operation({**SEED, "enabled": True})
    api = AppDocker(compose=DRIFT)
    executor.config_approval = lambda spec: binding(api, spec)
    with pytest.raises(OpsDispatchError) as caught:
        service.prepare_invocation(SEED["slug"], {})
    assert caught.value.code == code
    if code == "source_not_published":
        assert "publish that commit first" in str(caught.value) and HEAD[:12] in str(caught.value)
