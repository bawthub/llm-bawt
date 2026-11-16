"""TASK-2386: bootstrap secrets reach a Compose target from the root-only
volume through the sealed renderer, never through argv, the catalog or ``.env``."""

from __future__ import annotations

import copy
import json

import pytest

from llm_bawt.ops.compose_config import evaluate
from llm_bawt.ops.compose_render import (
    COMPOSE_SECRET_NAMES, COMPOSE_SECRETS_DIR, COMPOSE_SECRETS_VOLUME, ComposeError, ComposeRenderer,
    compose_secret_args, compose_source, renderer_argv, validate_compose_secrets,
)
from llm_bawt.ops.config_reconcile import validate_compose_config, validate_target_fields
from tests.test_ops_image_deploy import (
    CONFIG_HASH, IMAGE_VARIABLE, NAME, PROJECT_DIR, REPO, SPEC, FakeDocker, _frame, _Stream, compose_project,
)

REF = REPO + "@sha256:" + "a" * 64
SERVICE = SPEC["compose_service"]
# Synthetic values exercising Compose env-file and JSON edge cases: `$`,
# quotes, `#`, spaces and `=`. Short on purpose (a 7-character password too).
SECRETS = {
    "POSTGRES_PASSWORD": "p@$s w#rd='x\"y'",
    "LLM_BAWT_POSTGRES_PASSWORD": "seven77",
    "LLM_BAWT_SECRET_KEY": "k3y=with=equals$$and#hash",
    "DATABASE_URL": "postgresql://u:pw$1@db:5432/x",
}


def _spec(names=("POSTGRES_PASSWORD",)):
    spec = copy.deepcopy(SPEC)
    spec["compose_config"]["compose_secrets"] = list(names)
    return spec


class _SecretsDocker(FakeDocker):
    """Emulates Compose env-file loading inside the renderer: the project
    ``.env`` plus each ``--env-file`` named on argv, read from the secrets
    volume only when that volume is actually mounted. Bare ``NAME`` entries
    (``environment: [NAME]``) resolve from those files and are silently
    dropped when unset, as real Compose does; ``${NAME}`` interpolates."""

    def __init__(self, *, volume=None, project_env=None, **kwargs):
        super().__init__(**kwargs)
        self.volume = dict(SECRETS if volume is None else volume)
        self.project_env = dict(project_env or {})

    def attach_stream(self, container_id):
        container = self.containers[container_id]
        argv = container["Config"]["Cmd"]
        if "--hash" in argv:
            return _Stream(_frame(1, f"{argv[-1]} {CONFIG_HASH}\n"))
        mounted = f"{COMPOSE_SECRETS_VOLUME}:{COMPOSE_SECRETS_DIR}:ro" in container["HostConfig"]["Binds"]
        env_files = [argv[i + 1] for i, a in enumerate(argv) if a == "--env-file"]
        lookup = dict(self.project_env) if not env_files or f"{PROJECT_DIR}/.env" in env_files else {}
        for path in env_files:
            name = path.rsplit("/", 1)[-1].removesuffix(".env")
            if path.startswith(COMPOSE_SECRETS_DIR + "/"):
                if not mounted or name not in self.volume:
                    self.render_status = 1  # Compose exits non-zero on a missing env file
                    return _Stream(_frame(2, "env file not found\n"))
                lookup[name] = self.volume[name]
        project = copy.deepcopy(self.compose)
        service = project["services"][SERVICE]
        env = {}
        for key, value in service["environment"].items():
            if value is None:  # bare passthrough
                if key in lookup:
                    env[key] = lookup[key]
            else:
                env[key] = value
        service["environment"] = env
        text = json.dumps(project)
        entry = dict(e.partition("=")[::2] for e in container["Config"]["Entrypoint"] if "=" in e)
        text = text.replace(IMAGE_VARIABLE, entry.get(IMAGE_VARIABLE.strip("${}"), ""))
        for name, value in lookup.items():
            text = text.replace("${" + name + "}", json.dumps(value)[1:-1])
        # Compose re-escapes every literal `$` in its output.
        text = text.replace("$", "$$").replace("$${", "${")
        return _Stream(_frame(1, text))


def _api(names=("POSTGRES_PASSWORD",), extra_env=None, **kwargs):
    environment = {"NODE_ENV": "production", **{n: None for n in names}, **(extra_env or {})}
    return _SecretsDocker(compose=compose_project(environment=environment), **kwargs)


def _render(spec, api):
    return ComposeRenderer(api, spec, tag="t1").render(compose_source(spec, api.by_name(NAME)), REF)


# ── argv and mounts carry paths only ─────────────────────────────────────
def test_argv_names_env_files_by_path_and_project_env_first():
    spec = _spec(["POSTGRES_PASSWORD", "DATABASE_URL"])
    source = {"project_dir": PROJECT_DIR, "compose_files": [f"{PROJECT_DIR}/docker-compose.yml"]}
    argv = renderer_argv(spec, source, hash_only=False)
    assert argv[:7] == ["compose", "--env-file", f"{PROJECT_DIR}/.env",
                        "--env-file", f"{COMPOSE_SECRETS_DIR}/DATABASE_URL.env",
                        "--env-file", f"{COMPOSE_SECRETS_DIR}/POSTGRES_PASSWORD.env"]
    assert renderer_argv(SPEC, source, hash_only=False)[:2] == ["compose", "-p"]  # undeclared: unchanged


def test_render_injects_values_but_never_into_the_renderer_container():
    names = sorted(COMPOSE_SECRET_NAMES)
    api = _api(names)
    result = _render(_spec(names), api)
    env = result["config"]["services"][SERVICE]["environment"]
    assert {n: env[n] for n in names} == SECRETS
    assert len(api.render_bodies) == 2  # config and --hash: both load the same files
    for body in api.render_bodies:
        assert f"{COMPOSE_SECRETS_VOLUME}:{COMPOSE_SECRETS_DIR}:ro" in body["HostConfig"]["Binds"]
        blob = json.dumps({k: body[k] for k in ("Entrypoint", "Cmd", "Env", "Labels")})
        assert not any(value in blob for value in SECRETS.values())


def test_undeclared_target_mounts_nothing():
    api = _SecretsDocker()
    _render(SPEC, api)
    assert all(len(b["HostConfig"]["Binds"]) == 1 for b in api.render_bodies)


def test_evaluate_carries_the_secret_in_private_desired_env_only():
    spec = _spec()
    api = _api()
    container = api.by_name(NAME)
    out = evaluate(spec, container, api.images[container["Image"]], ComposeRenderer(api, spec, tag="t1"), REF)
    assert out["desired"]["env"]["POSTGRES_PASSWORD"] == SECRETS["POSTGRES_PASSWORD"]
    public = json.dumps({k: v for k, v in out.items() if k != "desired"})
    assert SECRETS["POSTGRES_PASSWORD"] not in public and "POSTGRES_PASSWORD" not in out["display"]


# ── input boundary ──────────────────────────────────────────────────────
def test_undeclared_name_is_never_loaded_even_if_the_compose_file_asks():
    api = _api(["POSTGRES_PASSWORD", "LLM_BAWT_SECRET_KEY"])
    env = _render(_spec(["POSTGRES_PASSWORD"]), api)["config"]["services"][SERVICE]["environment"]
    assert "LLM_BAWT_SECRET_KEY" not in env


def test_project_env_value_never_stands_in_for_a_declared_secret():
    """The per-name file comes after ``.env`` and wins."""
    api = _api(project_env={"POSTGRES_PASSWORD": "stale-dotenv-value"})
    env = _render(_spec(), api)["config"]["services"][SERVICE]["environment"]
    assert env["POSTGRES_PASSWORD"] == SECRETS["POSTGRES_PASSWORD"]


# ── fail closed ─────────────────────────────────────────────────────────
def test_absent_secret_file_fails_closed():
    api = _api(volume={})
    with pytest.raises(ComposeError) as caught:
        _render(_spec(), api)
    assert caught.value.code.startswith("renderer_") and not api.renderers_left()


def test_declared_secret_the_compose_file_drops_is_refused():
    api = _SecretsDocker()  # compose file has no POSTGRES_PASSWORD entry
    with pytest.raises(ComposeError) as caught:
        _render(_spec(), api)
    assert caught.value.code == "compose_secret_missing" and caught.value.names == ["POSTGRES_PASSWORD"]


@pytest.mark.parametrize("where", ["env", "label", "command"])
def test_secret_aliased_under_another_name_is_refused_without_echoing_it(where):
    alias = "${POSTGRES_PASSWORD}"
    extra = {"extra_env": {"PGPASS_COPY": alias}} if where == "env" else {}
    api = _api(**extra)
    if where == "label":
        api.compose["services"][SERVICE]["labels"] = {"x": alias}
    if where == "command":
        api.compose["services"][SERVICE]["command"] = ["run", alias]
    with pytest.raises(ComposeError) as caught:
        _render(_spec(), api)
    assert caught.value.code == "compose_secret_aliased" and caught.value.names == ["POSTGRES_PASSWORD"]
    assert SECRETS["POSTGRES_PASSWORD"] not in str(caught.value)


def test_two_declared_names_may_share_one_value():
    api = _api(["POSTGRES_PASSWORD", "LLM_BAWT_POSTGRES_PASSWORD"],
               volume={"POSTGRES_PASSWORD": "same-value", "LLM_BAWT_POSTGRES_PASSWORD": "same-value"})
    _render(_spec(["POSTGRES_PASSWORD", "LLM_BAWT_POSTGRES_PASSWORD"]), api)


# ── catalog validation ──────────────────────────────────────────────────
@pytest.mark.parametrize("value", [[], "POSTGRES_PASSWORD", ["POSTGRES_PASSWORD", "POSTGRES_PASSWORD"],
                                   ["LD_PRELOAD"], ["COMPOSE_FILE"], ["HF_TOKEN"], [None]],
                         ids=["empty", "string", "duplicate", "loader", "compose-control", "not-bootstrap",
                              "non-string"])
def test_compose_secrets_must_be_known_bootstrap_names(value):
    config = {**SPEC["compose_config"], "compose_secrets": value}
    with pytest.raises(ValueError, match="compose_secrets"):
        validate_compose_config(config)


def test_compose_secrets_cannot_overlap_the_image_variable_or_interpolation_env():
    config = {**SPEC["compose_config"], "image_variable": "POSTGRES_PASSWORD",
              "compose_secrets": ["POSTGRES_PASSWORD"]}
    with pytest.raises(ValueError, match="overlaps"):
        validate_compose_secrets(config)


def test_valid_declaration_and_args_are_value_free():
    config = {**SPEC["compose_config"], "compose_secrets": ["DATABASE_URL"]}
    assert validate_compose_config(config) is config
    assert compose_secret_args(config, PROJECT_DIR) == [
        "--env-file", f"{PROJECT_DIR}/.env", "--env-file", f"{COMPOSE_SECRETS_DIR}/DATABASE_URL.env"]


def test_display_values_may_not_name_a_compose_secret():
    spec = {"display_values": {"POSTGRES_PASSWORD": ["guess1"]}, "stat_paths": [],
            "stop_grace_seconds": 10, "health_timeout_seconds": 60}
    with pytest.raises(ValueError, match="compose secret"):
        validate_target_fields(spec, {})
