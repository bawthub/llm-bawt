"""TASK-1518: Compose-declared vs running configuration, pure functions and the
sealed renderer. No Docker, no network."""
from __future__ import annotations

import copy
import json

import pytest

from llm_bawt.ops.compose_config import (
    CONFIG_HASH_LABEL, UNEXPECTED_VALUE, UNSET_VALUE, apply_desired, display_values, drift, evaluate,
    mount_source_changes, normalized, running, translate,
)
from llm_bawt.ops.compose_render import (
    RENDER_PATH, ComposeError, ComposeRenderer, compose_source, demux_streams, renderer_env,
)
from llm_bawt.ops.config_reconcile import validate_compose_config
from tests.test_ops_image_deploy import (
    COMPOSE_FILES, NAME, PROJECT_DIR, RENDERER, REPO, SPEC, TOKEN_TARGET, FakeDocker, _frame, _Stream,
    compose_project,
)

REF = REPO + "@sha256:" + "a" * 64


def _rendered(**service):
    return json.loads(json.dumps(compose_project(**service)).replace("${BAWTHUB_FRONTEND_IMAGE}", REF))


def _desired(**service):
    return translate(_rendered(**service), SPEC, REF)


# ── translate: an allowlist, values kept, unknown keys refused ─────────────

def test_translate_projects_the_real_compose_shape():
    rendered = _rendered(environment={"A": "1"}, labels={"traefik.enable": "true"},
                         networks={"default": None, "llm-bawt": {}}, extra_hosts=["host.docker.internal=host-gateway"],
                         restart="on-failure:3",
                         volumes=[{"type": "bind", "source": "/srv/x", "target": "/x", "read_only": True,
                                   "bind": {"create_host_path": True}},
                                  {"type": "volume", "source": "data", "target": "/data", "volume": {}}])
    rendered["volumes"] = {"data": {"name": "bawthub_data"}}
    rendered["networks"]["llm-bawt"] = {"name": "llm-bawt", "external": True}
    assert translate(rendered, SPEC, REF) == {
        "env": {"A": "1"}, "labels": {"traefik.enable": "true"},
        "mounts": {"/x": {"type": "bind", "source": "/srv/x", "read_only": True},
                   "/data": {"type": "volume", "source": "bawthub_data", "read_only": False}},
        "networks": ["bawthub_default", "llm-bawt"],
        "extra_hosts": ["host.docker.internal:host-gateway"], "restart": ["on-failure", 3]}


@pytest.mark.parametrize("service, code", [
    ({"cap_add": ["SYS_ADMIN"]}, "unsupported_compose_key"),
    ({"ports": ["3000:3000"]}, "unsupported_compose_key"),
    ({"command": ["sh", "-c", "x"]}, "unsupported_compose_key"),
    ({"entrypoint": ["sh"]}, "unsupported_compose_key"),
    ({"pull_policy": "build"}, "unsupported_pull_policy"),
    ({"image": REPO + ":latest"}, "compose_image_mismatch"),
    ({"environment": {"XAI_API_KEY": None}}, "unset_compose_variable"),
    ({"environment": {"A": 1}}, "unsupported_compose_key"),
    ({"networks": {"default": {"ipv4_address": "10.0.0.9"}}}, "unsupported_compose_key"),
    ({"volumes": [{"type": "tmpfs", "target": "/tmp"}]}, "unsupported_compose_key"),
    ({"volumes": [{"type": "bind", "source": "relative", "target": "/x"}]}, "unsupported_compose_key"),
    ({"volumes": [{"type": "volume", "source": "undeclared", "target": "/x"}]}, "unsupported_compose_key"),
    ({"volumes": [{"type": "bind", "source": "/a", "target": "/x", "bind": {"propagation": "shared"}}]},
     "unsupported_compose_key"),
    ({"restart": "sometimes"}, "unsupported_compose_key"),
    ({"extra_hosts": ["nohost"]}, "unsupported_compose_key"),
])
def test_translate_refuses_what_it_does_not_model(service, code):
    with pytest.raises(ComposeError) as caught:
        _desired(**service)
    assert caught.value.code == code
    assert "10.0.0.9" not in str(caught.value)


def test_unset_variable_error_names_only_the_variable():
    with pytest.raises(ComposeError) as caught:
        _desired(environment={"XAI_API_KEY": None, "OTHER": "secret-value"})
    assert caught.value.names == ["XAI_API_KEY"] and "secret-value" not in str(caught.value)


def test_missing_service_is_refused():
    with pytest.raises(ComposeError, match="compose_service_missing"):
        translate({"services": {}}, SPEC, REF)


# ── running/normalized/drift: names only ──────────────────────────────────

def _container(**over):
    container = copy.deepcopy(FakeDocker().by_name(NAME))
    container.update(over)
    return container


IMAGE = {"Config": {"Env": ["PATH=/usr/bin"], "Labels": {"org.opencontainers.image.title": "old"}}}


def test_matching_container_has_no_drift_after_image_and_bookkeeping_normalisation():
    container = _container()
    container["Config"]["Labels"]["llm-bawt.ops.deployed-by"] = "f" * 32
    assert drift(normalized(_desired(), IMAGE), normalized(running(container), IMAGE)) == []


def test_drift_reports_every_field_by_name_never_value():
    desired = _desired(environment={"NODE_ENV": "production", "NEW": "s3cret"}, labels={"x": "1"},
                       networks={"default": None, "llm-bawt": None}, restart="always",
                       extra_hosts=["a=1.2.3.4"], volumes=[])
    container = _container()
    container["Config"]["Env"].append("OLD=gone")
    names = drift(normalized(desired, IMAGE), normalized(running(container), IMAGE))
    assert names == sorted(["env:+NEW", "env:-OLD", "label:+x", f"mount:-{TOKEN_TARGET}",
                            "network:+bawthub_llm-bawt", "restart:~", "extra_hosts:~"])
    assert "s3cret" not in json.dumps(names) and "1.2.3.4" not in json.dumps(names)


def test_image_supplied_value_overridden_by_compose_is_drift():
    image = {"Config": {"Env": ["NODE_ENV=development"], "Labels": IMAGE["Config"]["Labels"]}}
    desired = _desired()  # NODE_ENV=production
    container = _container()
    container["Config"]["Env"] = ["NODE_ENV=development"]  # running the image default
    assert drift(normalized(desired, image), normalized(running(container), image)) == ["env:+NODE_ENV"]


def test_running_parses_binds_and_mounts():
    container = _container()
    container["HostConfig"]["Binds"] = ["/a:/x:ro,z", "named:/data"]
    container["HostConfig"]["Mounts"] = [{"Type": "bind", "Source": "/b", "Target": "/y", "ReadOnly": False}]
    assert running(container)["mounts"] == {
        "/x": {"type": "bind", "source": "/a", "read_only": True},
        "/data": {"type": "volume", "source": "named", "read_only": False},
        "/y": {"type": "bind", "source": "/b", "read_only": False}}


def test_display_values_show_only_exact_closed_enum_matches():
    spec = {"display_values": {"MODE": ["on", "off"], "ABSENT": ["on"], "OTHER": ["on"]}}
    assert display_values(spec, {"MODE": "on", "OTHER": "sk-live-123"}) == {
        "ABSENT": UNSET_VALUE, "MODE": "on", "OTHER": UNEXPECTED_VALUE}


# ── apply_desired ──────────────────────────────────────────────────────────

def test_apply_desired_overlays_declared_fields_and_keeps_the_rest():
    container = _container()
    container["HostConfig"]["CapAdd"] = ["NET_ADMIN"]  # unmodelled: carried as a plain deploy would
    body = {"Env": ["STALE=1"], "Labels": {"com.docker.compose.project": "bawthub", "stale": "1",
                                          "llm-bawt.ops.deployed-by": "j"},
            "HostConfig": container["HostConfig"], "Image": REF,
            "NetworkingConfig": {"EndpointsConfig": {"bawthub_default": {"Aliases": ["keep"]}}}}
    desired = _desired(environment={"A": "1"}, labels={"x": "1"}, networks={"default": None, "llm-bawt": None},
                       restart="always")
    out = apply_desired(body, container, desired, container_name=NAME, service="frontend-prod", rendered_hash="h")
    assert out["Env"] == ["A=1"] and body["Env"] == ["STALE=1"]  # input untouched
    assert out["Labels"] == {"com.docker.compose.project": "bawthub", "llm-bawt.ops.deployed-by": "j",
                             "x": "1", CONFIG_HASH_LABEL: "h"}
    host = out["HostConfig"]
    assert host["Binds"] == [f"{PROJECT_DIR}/token:{TOKEN_TARGET}:ro"] and host["Mounts"] is None
    assert host["RestartPolicy"] == {"Name": "always", "MaximumRetryCount": 0} and host["CapAdd"] == ["NET_ADMIN"]
    assert out["NetworkingConfig"]["EndpointsConfig"] == {
        "bawthub_default": {"Aliases": ["keep"]},
        "bawthub_llm-bawt": {"Aliases": [NAME, "frontend-prod"]}}


# ── compose_source: the target's own labels, matched to the spec ───────────

@pytest.mark.parametrize("working_dir, files", [
    ("", COMPOSE_FILES),
    ("/", ["/docker-compose.yml", "/docker-compose.image.yml"]),
    ("relative/dir", COMPOSE_FILES),
    ("/srv/../etc", ["/srv/../etc/docker-compose.yml", "/srv/../etc/docker-compose.image.yml"]),
    ("/srv/compose/bawthub/", COMPOSE_FILES),
    ("/srv/with space", ["/srv/with space/docker-compose.yml", "/srv/with space/docker-compose.image.yml"]),
    (PROJECT_DIR, COMPOSE_FILES[::-1]),
    (PROJECT_DIR, COMPOSE_FILES + [f"{PROJECT_DIR}/docker-compose.override.yml"]),
    (PROJECT_DIR, ["/elsewhere/docker-compose.yml", f"{PROJECT_DIR}/docker-compose.image.yml"]),
])
def test_compose_source_refuses_paths_the_spec_does_not_name(working_dir, files):
    container = _container()
    container["Config"]["Labels"].update({"com.docker.compose.project.working_dir": working_dir,
                                          "com.docker.compose.project.config_files": ",".join(files)})
    with pytest.raises(ComposeError, match="compose_source_invalid"):
        compose_source(SPEC, container)


def test_compose_source_reads_the_target_labels():
    assert compose_source(SPEC, _container()) == {"project_dir": PROJECT_DIR, "compose_files": COMPOSE_FILES}


@pytest.mark.parametrize("mutate", [
    lambda c: c.update(extra=1),
    lambda c: c.update(compose_files=[]),
    lambda c: c.update(compose_files=["a.yml", "a.yml"]),
    lambda c: c.update(compose_files=["/abs/a.yml"]),
    lambda c: c.update(compose_files=["a.json"]),
    lambda c: c.update(renderer_image="docker:27-cli"),
    lambda c: c.update(image_variable="lower"),
    lambda c: c.update(interpolation_env={"HOME": "home/op"}),
    lambda c: c.update(interpolation_env={"HOME": "/home/../root"}),
    lambda c: c.update(interpolation_env={"HOME": "/"}),
    lambda c: c.update(interpolation_env={"HOME": "/home/op x"}),
    lambda c: c.update(interpolation_env={"PATH": "/bin"}),
    lambda c: c.update(interpolation_env={"lower": "x"}),
    lambda c: c.update(interpolation_env={c["image_variable"]: "x"}),
    # Codex r1 Blocker: loader/CLI execution-control names never reach `env -i`.
    lambda c: c.update(interpolation_env={"LD_PRELOAD": f"{PROJECT_DIR}/payload.so"}),
    lambda c: c.update(interpolation_env={"LD_LIBRARY_PATH": PROJECT_DIR}),
    lambda c: c.update(interpolation_env={"DOCKER_HOST": "tcp://evil:2375"}),
    lambda c: c.update(interpolation_env={"DOCKER_CONFIG": PROJECT_DIR}),
    lambda c: c.update(interpolation_env={"COMPOSE_FILE": "x.yml"}),
    lambda c: c.update(interpolation_env={"APP_SECRET": "x"}),
    lambda c: c.update(interpolation_env=["HOME"]),
])
def test_compose_config_validation(mutate):
    config = copy.deepcopy(SPEC["compose_config"])
    assert validate_compose_config(copy.deepcopy(config)) == config
    mutate(config)
    with pytest.raises(ValueError):
        validate_compose_config(config)


# ── renderer: demux and cleanup in every outcome ───────────────────────────

def test_demux_separates_stdout_and_stderr():
    assert demux_streams(_frame(1, "out") + _frame(2, "err") + _frame(1, "put")) == ("output", "err")


class _BrokenDocker(FakeDocker):
    def __init__(self, fail_on, **kwargs):
        super().__init__(**kwargs)
        self.fail_on = fail_on

    def attach_stream(self, container_id):
        if self.fail_on == "attach":
            raise OSError("socket closed")
        return super().attach_stream(container_id)

    def request(self, method, path, *, body=None, headers=None, raw=False):
        if self.fail_on == "start" and path.endswith("/start") and self.containers.get(path.split("/")[2], {}).get("_render"):
            raise RuntimeError("Docker HTTP 500: start")
        return super().request(method, path, body=body, headers=headers, raw=raw)


@pytest.mark.parametrize("fail_on", ["attach", "start"])
def test_renderer_container_is_removed_when_rendering_fails(fail_on):
    api = _BrokenDocker(fail_on)
    source = compose_source(SPEC, api.by_name(NAME))
    with pytest.raises(ComposeError, match="renderer_unavailable"):
        ComposeRenderer(api, SPEC, tag="t1").render(source, REF)
    assert api.render_bodies and not api.renderers_left()


def test_renderer_image_absent_is_a_fixed_code():
    api = FakeDocker()
    del api.images[next(k for k, v in api.images.items() if RENDERER in v["RepoDigests"])]
    with pytest.raises(ComposeError, match="renderer_image_missing"):
        ComposeRenderer(api, SPEC, tag="t1").render(compose_source(SPEC, api.by_name(NAME)), REF)
    assert not api.render_bodies


def test_renderer_output_is_bounded_and_stderr_never_echoed():
    api = FakeDocker(render_stderr="level=error msg=\"XAI_API_KEY=sk-live-secret\"\n")
    with pytest.raises(ComposeError) as caught:
        ComposeRenderer(api, SPEC, tag="t1").render(compose_source(SPEC, api.by_name(NAME)), REF)
    assert caught.value.code == "renderer_warning" and "sk-live" not in str(caught.value)
    with pytest.raises(ValueError):
        ComposeRenderer(api, SPEC, tag="Bad Tag")


# ── TASK-1476: the renderer's own HOME never reaches interpolation ─────────
# The cutover r3 rendered ${HOME} binds as /root/... because the renderer image
# runs as root: every home bind pointed at an empty directory the daemon made.

OPERATOR_HOME = "/home/op"
HOME_BIND = "/home/agent/.codex"  # the in-container target of a ${HOME} bind


class _ComposeShell(FakeDocker):
    """Emulates the renderer process environment: the image's default env
    (HOME=/root, as root) plus Docker ``Env``, unless the entrypoint is
    ``env -i``, which leaves only its own NAME=value arguments. Unset
    variables warn exactly as Compose does."""

    IMAGE_ENV = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/root"}

    def attach_stream(self, container_id):
        container = self.containers[container_id]
        argv, entrypoint = container["Config"]["Cmd"], container["Config"]["Entrypoint"]
        args = entrypoint[2:-1] if entrypoint[:2] == ["env", "-i"] else []
        env = dict(e.partition("=")[::2] for e in args) if args else {
            **self.IMAGE_ENV, **dict(e.partition("=")[::2] for e in container["Config"]["Env"])}
        if "--hash" in argv:
            return super().attach_stream(container_id)
        text, err = json.dumps(self.compose), ""
        for name in ("HOME", SPEC["compose_config"]["image_variable"]):
            if "${" + name + "}" in text:
                if name in env:
                    text = text.replace("${" + name + "}", env[name])
                else:
                    err += (f'time="2026-10-28T00:00:00Z" level=warning msg="The \\"{name}\\" '
                            'variable is not set. Defaulting to a blank string."\n')
                    text = text.replace("${" + name + "}", "")
        return _Stream(_frame(1, text) + (_frame(2, err) if err else b""))


def _home_spec(home=None):
    spec = copy.deepcopy(SPEC)
    if home is not None:
        spec["compose_config"]["interpolation_env"] = {"HOME": home}
    return spec


def _home_api(running_source=f"{OPERATOR_HOME}/.codex"):
    volumes = compose_project()["services"]["frontend-prod"]["volumes"] + [
        {"type": "bind", "source": "${HOME}/.codex", "target": HOME_BIND, "bind": {"create_host_path": True}}]
    api = _ComposeShell(compose=compose_project(volumes=volumes))
    api.by_name(NAME)["HostConfig"]["Binds"].append(f"{running_source}:{HOME_BIND}:rw")
    return api


def test_renderer_env_is_clean_with_only_declared_values():
    assert renderer_env(_home_spec()["compose_config"], REF) == [
        f"PATH={RENDER_PATH}", f"BAWTHUB_FRONTEND_IMAGE={REF}"]
    assert renderer_env(_home_spec(OPERATOR_HOME)["compose_config"], REF) == [
        f"PATH={RENDER_PATH}", f"BAWTHUB_FRONTEND_IMAGE={REF}", f"HOME={OPERATOR_HOME}"]


def test_regression_the_image_home_was_rendered_into_home_binds():
    """The old ``entrypoint=["docker"]`` path, emulated: ${HOME} became /root."""
    api = _home_api()
    api.compose = json.loads(json.dumps(api.compose))
    old = _ComposeShell.attach_stream
    container = {"Config": {"Cmd": ["compose", "config"], "Entrypoint": ["docker"],
                            "Env": [f"BAWTHUB_FRONTEND_IMAGE={REF}"]}}
    api.containers["old"] = container
    out, _ = demux_streams(old(api, "old").read(1 << 20))
    assert "/root/.codex" in out


def test_undeclared_home_refuses_instead_of_rendering_the_renderer_home():
    api = _home_api()
    with pytest.raises(ComposeError) as caught:
        ComposeRenderer(api, _home_spec(), tag="t1").render(compose_source(SPEC, api.by_name(NAME)), REF)
    assert caught.value.code == "unset_compose_variable" and caught.value.names == ["HOME"]
    assert not api.renderers_left()


@pytest.mark.parametrize("compose_config", [
    {"interpolation_env": {"HOME": "home/op"}},
    {"interpolation_env": {"LD_PRELOAD": f"{PROJECT_DIR}/payload.so"}},  # Codex r1 Blocker
    {"image_variable": "LD_PRELOAD"},
    {"image_variable": "DOCKER_HOST"},
    {"image_variable": "HOME"},
], ids=["relative-home", "ld-preload", "image-var-ld", "image-var-docker", "image-var-home"])
def test_bogus_or_execution_control_env_refuses_before_any_container(compose_config):
    api = _home_api()
    spec = _home_spec()
    spec["compose_config"].update(compose_config)
    with pytest.raises(ValueError):
        ComposeRenderer(api, spec, tag="t1").render(compose_source(SPEC, api.by_name(NAME)), REF)
    assert not api.render_bodies


def test_declared_home_renders_the_operator_sources():
    api = _home_api()
    result = evaluate(_home_spec(OPERATOR_HOME), api.by_name(NAME), IMAGE,
                      ComposeRenderer(api, _home_spec(OPERATOR_HOME), tag="t1"), REF)
    assert f"{OPERATOR_HOME}/.codex->{HOME_BIND}:rw" in result["mounts"]
    assert not [n for n in result["drift"] if n.startswith("mount:")]


def test_a_moved_data_mount_source_refuses_with_target_names_only():
    """Wrong-but-declared HOME (/root) against a container bound to the operator's."""
    api = _home_api()
    with pytest.raises(ComposeError) as caught:
        evaluate(_home_spec("/root"), api.by_name(NAME), IMAGE, ComposeRenderer(api, _home_spec("/root"), tag="t1"), REF)
    assert caught.value.code == "mount_source_changed" and caught.value.names == [HOME_BIND]
    assert "/root" not in str(caught.value).replace(HOME_BIND, "")


def test_mount_source_changes_ignores_read_only_flips_and_added_or_removed_mounts():
    want = {"/a": {"type": "bind", "source": "/x", "read_only": True},
            "/b": {"type": "bind", "source": "/y", "read_only": False}}
    have = {"/a": {"type": "bind", "source": "/x", "read_only": False},
            "/c": {"type": "volume", "source": "v", "read_only": False}}
    assert mount_source_changes(want, have) == []
    have["/a"] = {"type": "volume", "source": "/x", "read_only": True}
    assert mount_source_changes(want, have) == ["/a"]
