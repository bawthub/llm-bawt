"""TASK-1474 step 4: the BawtHub backend as a second Compose-BUILT reconcile target.

The built profile gains ``image`` (the build's local tag), ``pull_policy``
(inert) and ``security_opt`` (compared and applied); health becomes one JSON
field matched against the spec's value. No live Docker or network.
"""
from __future__ import annotations

import json
import struct

import pytest

from llm_bawt.ops.app_config_reconcile import (
    SCHEMA, AppConfigReconciler, app_config_binding, approval_text, valid_app_binding,
)
from llm_bawt.ops.compose_config import (
    BUILT_PROFILE, PINNED_PROFILE, apply_desired, built_image_tag, canonical_hash, translate,
)
from llm_bawt.ops.compose_render import ComposeError, ComposeRenderer
from llm_bawt.ops.executor import ConfigCheckError, validate_spec
from llm_bawt.ops.image_deploy import DeployFailed
from llm_bawt.ops.seeds import SEEDS
from llm_bawt.ops.source_state import SourceStateError, reserved_env_name, source_plan
from tests.test_ops_app_config_reconcile import (
    APP_NAME, EXECUTION, OPERATION, SPEC as APP_SPEC, TAG as APP_TAG, AppDocker, StubChecker, _executor,
    app_compose, source,  # noqa: F401 (autouse: the sealed git source check, stubbed)
)
from tests.test_ops_image_deploy import JOB_ID, RENDERER_ID, compose_project

SEED = next(s for s in SEEDS if s["slug"] == "bawthub.reconcile-backend-config")
SPEC = json.loads(SEED["command_script"])
NAME = SPEC["container_name"]
BACKEND_DIR = "/srv/compose/bawthub"
BACKEND_FILES = [f"{BACKEND_DIR}/docker-compose.yml"]
BACKEND_ID, OTHER_BUILD = "sha256:" + "7" * 64, "sha256:" + "8" * 64
TAG = "bawthub-backend:latest"
CMD = ["uv", "run", "--no-dev", "uvicorn", "bawthub.app:app", "--host", "0.0.0.0", "--port", "80"]
SECRET = "xai-live-never-shown"
TOKEN = "/run/bawthub-service-token"
SOCK = "/var/run/docker.sock"
MOSHI = "/app/services/moshi-server/configs"
PYCACHE = "/tmp/bawthub-pycache"
IMAGE_ENV = ["PATH=/app/.venv/bin:/usr/bin", "UV_COMPILE_BYTECODE=1", "UV_LOCKED=1", "HOSTNAME=0.0.0.0"]
ENV = {name: f"v-{name.lower()}" for name in SPEC["env_names"]}
ENV["XAI_API_KEY"] = SECRET


def backend_compose(**service):
    """``docker compose config --format json backend`` for the product's own service: no drift."""
    body = {"build": {"context": BACKEND_DIR, "dockerfile": "Dockerfile", "target": "hot-reloading"},
            "command": CMD, "entrypoint": None, "environment": dict(ENV),
            "extra_hosts": ["host.docker.internal=host-gateway"], "image": "bawthub-backend",
            "labels": {"traefik.enable": "true"}, "networks": {"default": None}, "pull_policy": "build",
            "restart": "unless-stopped", "security_opt": ["apparmor=unconfined"],
            "volumes": [
                {"type": "bind", "source": f"{BACKEND_DIR}/bawthub", "target": "/app/bawthub",
                 "bind": {"create_host_path": True}},
                {"type": "bind", "source": f"{BACKEND_DIR}/services/moshi-server/configs", "target": MOSHI,
                 "bind": {"create_host_path": True}},
                {"type": "bind", "source": SOCK, "target": SOCK, "bind": {"create_host_path": True}},
                {"type": "volume", "source": "llm-bawt-service-token", "target": TOKEN, "read_only": True,
                 "volume": {}}]}
    body.update(service)
    return {"name": "bawthub", "services": {"backend": body}, "networks": {"default": {"name": "bawthub_default"}},
            "volumes": {"llm-bawt-service-token": {"name": "llm-bawt-service-token", "external": True}}}


def _image(image_id, refs):
    return {"Id": image_id, "RepoDigests": [], "_health": "healthy", "_refs": refs,
            "Config": {"Env": list(IMAGE_ENV), "Cmd": ["python3"], "ExposedPorts": {"80/tcp": {}},
                       "Labels": {}, "Healthcheck": {"Test": ["CMD", "curl", "-f", "http://localhost/v1/health"]}}}


class BackendDocker(AppDocker):
    """The backend as Compose built it, with in-container curl and stat."""

    def __init__(self, *, compose=None, tag_moved=False, health=None, security=("apparmor=unconfined",),
                 env=None, **kwargs):
        super().__init__(compose=backend_compose() if compose is None else compose, **kwargs)
        self.health_body = {"ok": True, "service": "bawthub"} if health is None else health
        self.images = {BACKEND_ID: _image(BACKEND_ID, [] if tag_moved else [TAG]),
                       RENDERER_ID: self.images[RENDERER_ID]}
        if tag_moved:
            self.images[OTHER_BUILD] = _image(OTHER_BUILD, [TAG])
        env = ENV if env is None else env
        old_id = "d" * 64
        self.containers = {old_id: {
            "Id": old_id, "Name": f"/{NAME}", "Image": BACKEND_ID,
            "Config": {"Image": TAG, "Env": [*IMAGE_ENV, *(f"{k}={v}" for k, v in env.items())],
                       "Cmd": list(CMD), "ExposedPorts": {"80/tcp": {}},
                       "Healthcheck": {"Test": ["CMD", "curl", "-f", "http://localhost/v1/health"]},
                       "Labels": {"com.docker.compose.project": "bawthub", "com.docker.compose.service": "backend",
                                  "com.docker.compose.project.working_dir": BACKEND_DIR,
                                  "com.docker.compose.project.config_files": ",".join(BACKEND_FILES),
                                  "traefik.enable": "true"}},
            "HostConfig": {"RestartPolicy": {"Name": "unless-stopped"},
                           "Binds": [f"{BACKEND_DIR}/bawthub:/app/bawthub:rw",
                                     f"{BACKEND_DIR}/services/moshi-server/configs:{MOSHI}:rw",
                                     f"{SOCK}:{SOCK}:rw"],
                           "Mounts": [{"Type": "volume", "Source": "llm-bawt-service-token", "Target": TOKEN,
                                       "ReadOnly": True}],
                           "ExtraHosts": ["host.docker.internal:host-gateway"],
                           "SecurityOpt": list(security) or None},
            "NetworkSettings": {"Networks": {"bawthub_default": {"Aliases": [NAME, "backend", old_id[:12]]}}},
            "State": {"Running": True, "Status": "running", "Health": {"Status": "healthy"}}}}
        self.old_id = old_id

    def request(self, method, path, *, body=None, headers=None, raw=False):
        if path.startswith("/exec/health1/") and path.endswith("/start"):
            self.calls.append((method, path))
            payload = json.dumps(self.health_body).encode()
            return struct.pack(">BxxxI", 1, len(payload)) + payload
        if path.startswith("/exec/stat1/"):
            self.calls.append((method, path))
            if path.endswith("/start"):
                payload = b"600 0 0\n"
                return struct.pack(">BxxxI", 1, len(payload)) + payload
            return {"ExitCode": 0}
        return super().request(method, path, body=body, headers=headers, raw=raw)


def bind(api, spec=SPEC, *, moved=False, checker=None):
    return app_config_binding(spec, api.by_name(NAME), api.images[BACKEND_ID],
                              ComposeRenderer(api, spec, tag="be1"), tag_moved=lambda _tag: moved,
                              source_checker=checker or StubChecker())


def snapshot(api, spec=SPEC, **kwargs):
    return {"spec": spec, "resolved_args": {}, "config": bind(api, spec, **kwargs)[0], "execution": EXECUTION,
            "newly_trusted": None}


def reconciler(api, tmp_path, snap=None):
    return AppConfigReconciler(api, snap or snapshot(api), JOB_ID, lock_root=tmp_path / "locks",
                               sleep=lambda _: None, clock=iter(range(0, 10000, 5)).__next__)


def card(bound, spec=SPEC):
    return approval_text(spec, bound, SEED["approval_prompt_prefix"], operation=OPERATION, newly_trusted=None)


# ── catalog seed ────────────────────────────────────────────────────────────

def test_seed_is_disabled_validates_and_warns_it_cuts_voice_sessions():
    assert not SEED["enabled"] and SEED["max_concurrent"] == 1 and SEED["risk_level"] == "high"
    assert SEED["start_delay_seconds"] == 0  # not a self-target: the app serves the approval
    assert "Cuts active voice sessions" in SEED["description"] and SEED["approval_prompt_prefix"]
    assert validate_spec(SEED["command_script"]) == SPEC
    assert SPEC["health_field"] == "ok" and SPEC["health_value"] is True
    assert SPEC["stat_paths"] == [f"{TOKEN}/token"]
    assert SPEC["compose_config"] == APP_SPEC["compose_config"]
    assert len(SPEC["env_names"]) == 21 and "BAWTHUB_TTS_RESTART_CMD" not in SPEC["env_names"]
    assert not any(reserved_env_name(n) for n in SPEC["env_names"])


@pytest.mark.parametrize("value", [1, 0, "true ", "", "x" * 33, None, False, ["ok"], {"a": 1}])
def test_health_value_is_a_short_token_or_true(value):
    with pytest.raises(ValueError):
        validate_spec(json.dumps({**SPEC, "health_value": value}))


@pytest.mark.parametrize("key, value", [("health_field", "Status"), ("health_field", "a.b"),
                                        ("health_field", ""), ("health_status", "ok")])
def test_health_field_shape_and_the_retired_key(key, value):
    with pytest.raises(ValueError):
        validate_spec(json.dumps({**SPEC, key: value}))


# ── change 1: image ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("declared, tag", [
    ("bawthub-backend", "bawthub-backend:latest"), ("bawthub-backend:latest", "bawthub-backend:latest"),
    ("bawthub-backend:dev-1.2", "bawthub-backend:dev-1.2"), ("nick/backend", "nick/backend:latest"),
])
def test_a_declared_image_is_the_builds_local_tag(declared, tag):
    assert built_image_tag(declared) == tag


@pytest.mark.parametrize("declared", [
    "ghcr.io/bawthub/backend:1", "localhost/backend", "localhost:5000/backend:1", "registry:5000/backend",
    "bawthub-backend@sha256:" + "a" * 64, "Bawthub-Backend", "", None, 1, "a:b:c",
])
def test_a_pulled_or_pinned_image_refuses(declared):
    with pytest.raises(ComposeError) as caught:
        built_image_tag(declared)
    assert caught.value.code == "unsupported_compose_key" and caught.value.names == ["image"]
    if declared is not None:
        with pytest.raises(ComposeError):
            translate(backend_compose(image=declared), SPEC, TAG, BUILT_PROFILE)


def test_binding_names_the_declared_tag_and_pins_the_running_image_id():
    api = BackendDocker()
    bound, desired = bind(api)
    assert bound["image_tag"] == TAG and bound["image_ref"] == TAG and bound["image_id"] == BACKEND_ID
    assert desired["image"] == TAG and bound["drift"] == []
    assert valid_app_binding(snapshot(api))


def test_a_container_created_by_the_untagged_name_or_the_id_binds():
    for ref in ("bawthub-backend", BACKEND_ID):
        api = BackendDocker()
        api.by_name(NAME)["Config"]["Image"] = ref
        bound = bind(api)[0]
        assert bound["image_ref"] == ref and valid_app_binding(snapshot(api))


@pytest.mark.parametrize("ref", ["bawthub-other:latest", "bawthub-backend-backend", "ghcr.io/x/backend:latest"])
def test_a_container_running_another_tag_refuses(ref):
    api = BackendDocker()
    api.by_name(NAME)["Config"]["Image"] = ref
    with pytest.raises(ValueError, match="Compose-built image"):
        bind(api)


def test_a_declared_image_other_than_the_running_tag_refuses():
    api = BackendDocker(compose=backend_compose(image="bawthub-backend:next"))
    with pytest.raises(ValueError, match="Compose-built image"):
        bind(api)


def test_the_moved_tag_is_resolved_by_the_caller_for_the_declared_tag(tmp_path):
    api = BackendDocker(compose=backend_compose(environment={**ENV, "PYTHONPYCACHEPREFIX": PYCACHE}),
                        tag_moved=True)
    asked = []
    bound = app_config_binding(SPEC, api.by_name(NAME), api.images[BACKEND_ID], ComposeRenderer(api, SPEC, tag="m"),
                               tag_moved=lambda tag: asked.append(tag) or True, source_checker=StubChecker())[0]
    assert asked == [TAG] and bound["image_tag_moved"] is True
    assert f"tag {TAG} now names a different local build" in card(bound)
    approved = _executor(api).config_approval(SPEC)
    assert approved["image_tag_moved"] is True
    snap = {"spec": SPEC, "resolved_args": {}, "config": approved, "execution": EXECUTION, "newly_trusted": None}
    result = json.loads(reconciler(api, tmp_path, snap).run())
    assert api.create_bodies[0]["Image"] == BACKEND_ID and result["deployed"]["image_id"] == BACKEND_ID
    assert not any(TAG in p for _m, p in api.calls if p.startswith("/images/"))  # never re-resolved


@pytest.mark.parametrize("tamper", [
    lambda c: c.update(image_tag="ghcr.io/evil/backend:1"),
    lambda c: c.update(image_tag="bawthub-backend"),  # the default form: the ref no longer matches
    lambda c: c.update(image_tag="bawthub-backend@sha256:" + "a" * 64),
    lambda c: c.update(image_ref="bawthub-other:latest"),
    lambda c: c.update(security_opt=[1]),
    lambda c: c.pop("security_opt"),
])
def test_the_binding_shape_is_validated(tamper):
    snap = snapshot(BackendDocker())
    assert valid_app_binding(snap)
    tamper(snap["config"])
    assert not valid_app_binding(snap)


# ── change 1: pull_policy ───────────────────────────────────────────────────

@pytest.mark.parametrize("policy", ["build", "never"])
def test_build_and_never_are_inert(policy):
    desired = translate(backend_compose(pull_policy=policy), SPEC, TAG, BUILT_PROFILE)
    assert desired == translate(backend_compose(), SPEC, TAG, BUILT_PROFILE)
    assert "pull_policy" not in desired


@pytest.mark.parametrize("policy", ["always", "missing", "if_not_present", "daily", None])
def test_a_pulling_policy_refuses(policy):
    with pytest.raises(ComposeError) as caught:
        translate(backend_compose(pull_policy=policy), SPEC, TAG, BUILT_PROFILE)
    assert caught.value.code == "unsupported_pull_policy"


def test_the_frontend_profile_still_refuses_build_and_its_own_pull_policies_are_unchanged():
    ref = "ghcr.io/bawthub/frontend@sha256:" + "a" * 64
    frontend = json.loads(next(s for s in SEEDS if s["slug"] == "bawthub.reconcile-prod-config")["command_script"])

    def rendered(**service):
        return json.loads(json.dumps(compose_project(**service)).replace("${BAWTHUB_FRONTEND_IMAGE}", ref))

    for policy in ("missing", "never"):
        translate(rendered(pull_policy=policy), frontend, ref, PINNED_PROFILE)
    for service, code in (({"pull_policy": "build"}, "unsupported_pull_policy"),
                          ({"pull_policy": "always"}, "unsupported_pull_policy"),
                          ({"security_opt": ["apparmor=unconfined"]}, "unsupported_compose_key")):
        with pytest.raises(ComposeError) as caught:
            translate(rendered(**service), frontend, ref, PINNED_PROFILE)
        assert caught.value.code == code
    assert "security_opt" not in translate(rendered(), frontend, ref, PINNED_PROFILE)


# ── change 1: security_opt ──────────────────────────────────────────────────

APP_DESIRED_GOLDEN = "08b892521b207cfa0de3e6a0198862d3a3f6612d2839f3e9d76996ca01e6d09c"


def test_the_app_desired_config_is_byte_identical_without_the_new_keys():
    # Computed with the profile before step 4: the live app's approvals and
    # receipts keep their fingerprints.
    desired = translate(app_compose(), APP_SPEC, APP_TAG, BUILT_PROFILE)
    assert "image" not in desired and "security_opt" not in desired
    assert canonical_hash(desired) == APP_DESIRED_GOLDEN
    for empty in (None, []):
        assert translate(app_compose(security_opt=empty), APP_SPEC, APP_TAG, BUILT_PROFILE) == desired


def test_the_app_card_and_create_body_carry_no_security_options(tmp_path):
    api = AppDocker(compose=app_compose(environment={"LLM_BAWT_SERVICE_PORT": "8642", "NEW": "1"}))
    from tests.test_ops_app_config_reconcile import reconciler as app_reconciler, snapshot as app_snapshot
    snap = app_snapshot(api)
    assert snap["config"]["security_opt"] == [] and snap["config"]["drift"] == ["env:+NEW"]
    app_reconciler(api, tmp_path, snap).run()
    assert api.create_bodies[0]["HostConfig"]["SecurityOpt"] is None
    assert api.by_name(APP_NAME)["HostConfig"]["SecurityOpt"] is None


@pytest.mark.parametrize("declared, running, after", [
    (["apparmor=unconfined"], (), ["apparmor=unconfined"]),                     # add
    ([], ("apparmor=unconfined",), None),                                       # remove
    (["apparmor=docker-default"], ("apparmor=unconfined",), ["apparmor=docker-default"]),  # change
    (["no-new-privileges:true", "apparmor=unconfined"], ("apparmor=unconfined",),
     ["no-new-privileges:true", "apparmor=unconfined"]),
], ids=["add", "remove", "change", "extend"])
def test_security_opt_drift_is_named_shown_and_applied(tmp_path, declared, running, after):
    compose = backend_compose(security_opt=declared, environment={**ENV, "PYTHONPYCACHEPREFIX": PYCACHE})
    api = BackendDocker(compose=compose, security=running)
    snap = snapshot(api)
    assert snap["config"]["drift"] == ["env:+PYTHONPYCACHEPREFIX", "security_opt:~"]
    assert snap["config"]["security_opt"] == declared
    assert f"Security options after: {', '.join(declared) or 'none'}" in card(snap["config"])
    result = json.loads(reconciler(api, tmp_path, snap).run())
    assert result["phase"] == "done"
    assert api.create_bodies[0]["HostConfig"]["SecurityOpt"] == after


def test_an_unchanged_security_opt_is_not_drift():
    bound = bind(BackendDocker())[0]
    assert bound["drift"] == [] and bound["security_opt"] == ["apparmor=unconfined"]
    assert "Security options after: apparmor=unconfined" in card(bound)


@pytest.mark.parametrize("value", [
    "apparmor=unconfined", ["seccomp=/etc/profile.json"], ["seccomp={\"a\":1}"], ["label=type:x y"],
    ["apparmor=unconfined", "apparmor=unconfined"], [1], {"apparmor": "unconfined"},
])
def test_an_unmodelled_security_opt_refuses(value):
    with pytest.raises(ComposeError) as caught:
        translate(backend_compose(security_opt=value), SPEC, TAG, BUILT_PROFILE)
    assert caught.value.code == "unsupported_compose_key" and caught.value.names == ["security_opt"]


def test_apply_desired_owns_security_opt_only_for_the_built_profile():
    frontend_desired = {"env": {}, "labels": {}, "mounts": {}, "networks": ["n"], "extra_hosts": [],
                        "restart": ["no", 0]}
    old = {"Config": {"Labels": {}}, "HostConfig": {"SecurityOpt": ["apparmor=unconfined"]}}
    body = {"Env": [], "Labels": {}, "HostConfig": {"SecurityOpt": ["apparmor=unconfined"], "NetworkMode": "n"}}
    out = apply_desired(json.loads(json.dumps(body)), old, frontend_desired, container_name="x", service="s",
                        rendered_hash="h")
    assert out["HostConfig"]["SecurityOpt"] == ["apparmor=unconfined"]  # the pinned profile never touches it


# ── change 2: health ────────────────────────────────────────────────────────

def _health_spec(field, value):
    return {**SPEC, "health_field": field, "health_value": value}


@pytest.mark.parametrize("field, value, body", [
    ("ok", True, {"ok": True}), ("status", "ok", {"status": "ok", "x": 1}),
])
def test_health_passes_on_the_exact_field_value(tmp_path, field, value, body):
    spec = _health_spec(field, value)
    api = BackendDocker(health=body)
    result = json.loads(reconciler(api, tmp_path, snapshot(api, spec)).run())
    assert result["unchanged"] and result["health"] == {"field": field, "status": "matched",
                                                       "stat": {f"{TOKEN}/token": "600 0 0"}}


@pytest.mark.parametrize("body", [
    {"ok": 1}, {"ok": "true"}, {"ok": "True"}, {"ok": False}, {}, {"OK": True}, [True], {"ok": [True]},
    {"ok": SECRET},
])
def test_true_is_matched_by_identity_and_the_body_is_never_echoed(tmp_path, body):
    api = BackendDocker(compose=backend_compose(environment={**ENV, "PYTHONPYCACHEPREFIX": PYCACHE}), health=body)
    with pytest.raises(DeployFailed) as caught:
        reconciler(api, tmp_path).run()
    assert "health ok did not match expected True" in str(caught.value)
    assert caught.value.detail["restored"] and SECRET not in str(caught.value) + json.dumps(caught.value.detail)
    assert api.by_name(NAME)["Id"] == api.old_id


@pytest.mark.parametrize("body", [{"status": 1}, {"status": True}, {"status": "OK"}, {"status": ["ok"]}])
def test_a_string_value_needs_that_string(tmp_path, body):
    spec = _health_spec("status", "ok")
    api = BackendDocker(compose=backend_compose(environment={**ENV, "PYTHONPYCACHEPREFIX": PYCACHE}), health=body)
    with pytest.raises(DeployFailed, match="health status did not match expected 'ok'"):
        reconciler(api, tmp_path, snapshot(api, spec)).run()


# ── change 3: product env ───────────────────────────────────────────────────

def _plan(image_env, env=None):
    api = BackendDocker()
    image = {**api.images[BACKEND_ID], "Config": {**api.images[BACKEND_ID]["Config"], "Env": image_env}}
    desired = translate(backend_compose(environment=ENV if env is None else env), SPEC, TAG, BUILT_PROFILE)
    return source_plan(SPEC, {"project_dir": BACKEND_DIR, "compose_files": BACKEND_FILES},
                       api.by_name(NAME), image, desired)


@pytest.mark.parametrize("entry", ["UV_COMPILE_BYTECODE=0", "UV_COMPILE_BYTECODE=1", "UV_COMPILE_BYTECODE=true",
                                   "UV_COMPILE_BYTECODE=false", "UV_LOCKED=1", "UV_LOCKED=true",
                                   "HOSTNAME=0.0.0.0", "HOSTNAME=backend.local", "HOSTNAME="])
def test_the_backend_image_env_is_product_env(entry):
    assert _plan(["PATH=/usr/bin", entry])["classification"]["opaque_env"] == sorted(SPEC["env_names"])


@pytest.mark.parametrize("entry", ["UV_LOCKED=0", "UV_LOCKED=false", "UV_LOCKED=", "UV_COMPILE_BYTECODE=yes",
                                   "HOSTNAME=a b", "HOSTNAME=$(id)", "UV_INDEX_URL=https://x.example/simple",
                                   "UV_FROZEN=1", "UV_PROJECT_ENVIRONMENT=/tmp/v"])
def test_lock_dropping_and_other_uv_settings_still_refuse(entry):
    with pytest.raises(SourceStateError):
        _plan(["PATH=/usr/bin", entry])


@pytest.mark.parametrize("name", ["UV_INDEX_URL", "UV_LOCKED", "UV_COMPILE_BYTECODE", "HOSTNAME"])
def test_product_and_uv_names_can_never_be_listed_as_opaque(name):
    assert reserved_env_name(name)
    with pytest.raises(ValueError):
        validate_spec(json.dumps({**SPEC, "env_names": [*SPEC["env_names"], name]}))


def test_an_unclassified_compose_env_name_fails_closed():
    # e.g. a compose file that still sets a retired command variable.
    with pytest.raises(SourceStateError) as caught:
        _plan(IMAGE_ENV, {**ENV, "BAWTHUB_TTS_RESTART_CMD": "x"})
    assert caught.value.code == "source_env_unclassified"


# ── backend end to end ──────────────────────────────────────────────────────

def test_the_bytecode_rule_needs_the_pycache_prefix():
    checker = StubChecker()
    bind(BackendDocker(), checker=checker)
    assert checker.calls[0][1]["bytecode_ok"] is False  # the source __pycache__ would refuse
    checker = StubChecker()
    compose = backend_compose(environment={**ENV, "PYTHONPYCACHEPREFIX": PYCACHE})
    bound = bind(BackendDocker(compose=compose), checker=checker)[0]
    project, plan = checker.calls[0]
    assert project == BACKEND_DIR and plan["bytecode_ok"] is True and plan["paths"] == ["bawthub"]
    assert bound["drift"] == ["env:+PYTHONPYCACHEPREFIX"]
    for value in ("/app/bawthub/.pyc", TOKEN):
        with pytest.raises(SourceStateError):
            bind(BackendDocker(compose=backend_compose(environment={**ENV, "PYTHONPYCACHEPREFIX": value})))


def test_backend_reconcile_end_to_end(tmp_path):
    compose = backend_compose(environment={**ENV, "PYTHONPYCACHEPREFIX": PYCACHE})
    api = BackendDocker(compose=compose)
    snap = snapshot(api)
    bound = snap["config"]
    assert bound["classification"] == {"source_mounts": ["/app/bawthub"],
                                       "data_mounts": sorted([MOSHI, SOCK, TOKEN]),
                                       "opaque_env": sorted(SPEC["env_names"])}
    assert bound["source_commit"]["repository"] == "bawthub/bawthub" and bound["source_commit"]["branch"] == "main"
    assert bound["ports"] == [] and bound["depends_on"] == []
    text = card(bound)
    assert text.startswith(SEED["approval_prompt_prefix"]) and SECRET not in text
    assert f"Target: {NAME}" in text and f"Compose tag {TAG}" in text
    output = reconciler(api, tmp_path, snap).run()
    result = json.loads(output)
    assert result["schema"] == SCHEMA and result["phase"] == "done"
    assert result["health"] == {"field": "ok", "status": "matched", "stat": {f"{TOKEN}/token": "600 0 0"}}
    assert ["stat", "-c", "%a %u %g", f"{TOKEN}/token"] in api.exec_cmds
    assert ["curl", "-fsS", "--max-time", "10", SPEC["health_url"]] in api.exec_cmds
    body = api.create_bodies[0]
    assert body["Image"] == BACKEND_ID and f"PYTHONPYCACHEPREFIX={PYCACHE}" in body["Env"]
    assert body["HostConfig"]["SecurityOpt"] == ["apparmor=unconfined"] and body["Cmd"] == CMD
    assert SECRET not in output and SECRET not in json.dumps(snap)
    assert not any("/images/create" in p for _m, p in api.calls) and not api.renderers_left()


def test_the_container_name_defaults_to_composes_own():
    api = BackendDocker()
    assert bind(api)[0]["drift"] == []
    with pytest.raises(ComposeError) as caught:
        translate(backend_compose(), {**SPEC, "container_name": "bawthub-backend"}, TAG, BUILT_PROFILE)
    assert caught.value.code == "compose_container_name_mismatch"
    with pytest.raises(ComposeError):
        translate(backend_compose(container_name="other"), SPEC, TAG, BUILT_PROFILE)


def test_an_old_worker_refuses_the_backend_row():
    from llm_bawt.ops.app_config_reconcile import SOURCE_GUARD_LABEL, WORKER_CAPABILITY_LABEL
    executor = _executor(BackendDocker(), worker_labels={WORKER_CAPABILITY_LABEL: "1", SOURCE_GUARD_LABEL: "1"})
    with pytest.raises(ConfigCheckError) as caught:
        executor.config_approval(SPEC)
    assert caught.value.code == "config_reconcile_unavailable"
