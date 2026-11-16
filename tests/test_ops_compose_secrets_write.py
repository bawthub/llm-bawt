"""TASK-2386: the root-only action that fills the Compose secrets volume."""

from __future__ import annotations

import json
import os
import stat

import pytest

from llm_bawt.ops import compose_render, compose_secrets
from llm_bawt.ops.compose_secrets import (
    ACTION, SecretsError, collect, helper_body, helper_purge, helper_write, parse_env, project_dir, run_worker,
    validate_spec,
)
from llm_bawt.ops.executor import spec_capability_labels

KEY_FILE = "/home/op/.config/llm-bawt/secret.key"
WRITE = {"action": ACTION, "mode": "write", "compose_project": "llm-bawt",
         "sources": {"POSTGRES_PASSWORD": "project_env", "LLM_BAWT_POSTGRES_PASSWORD": "project_env",
                     "LLM_BAWT_SECRET_KEY": "key_file"},
         "key_file": KEY_FILE}
IMAGE = "sha256:" + "4" * 64
# Synthetic values only.
PW = "Short7!"
KEY = "A" * 43 + "="


def _files(tmp_path, env_text=None, key=KEY):
    env = tmp_path / "project.env"
    env.write_text(env_text if env_text is not None else
                   f"# comment\nPOSTGRES_PASSWORD={PW}\nexport LLM_BAWT_POSTGRES_PASSWORD='{PW} $x #y'\nOTHER=1\n")
    keyf = tmp_path / "secret.key"
    keyf.write_text(key)
    out, rec = tmp_path / "out", tmp_path / "recovery"
    out.mkdir()
    rec.mkdir()
    return {"project_env": str(env), "key_file": str(keyf), "out_dir": str(out), "recovery_dir": str(rec)}


# ── spec ──────────────────────────────────────────────────────────────────
def test_spec_matches_the_renderer_constants():
    assert compose_secrets.SECRET_NAMES == compose_render.COMPOSE_SECRET_NAMES
    assert compose_secrets.SECRETS_VOLUME == compose_render.COMPOSE_SECRETS_VOLUME


@pytest.mark.parametrize("change", [
    {"mode": "copy"}, {"compose_project": "Bad/Project"}, {"sources": {}},
    {"sources": {"HF_TOKEN": "project_env"}}, {"sources": {"POSTGRES_PASSWORD": "shell"}},
    {"key_file": "/home/op/.ssh/id_rsa"}, {"key_file": "/home/op/../x/.config/llm-bawt/secret.key"},
    {"extra": 1},
], ids=["mode", "project", "no-sources", "unknown-name", "unknown-kind", "key-path", "key-dotdot", "extra"])
def test_spec_refuses(change):
    with pytest.raises(ValueError):
        validate_spec({**WRITE, **change})


def test_key_file_iff_a_key_file_source():
    spec = {**WRITE, "sources": {"POSTGRES_PASSWORD": "project_env"}}
    with pytest.raises(ValueError, match="key_file"):
        validate_spec(spec)
    del spec["key_file"]
    assert validate_spec(spec) is spec
    assert validate_spec({"action": ACTION, "mode": "purge_recovery"})
    with pytest.raises(ValueError):
        validate_spec({"action": ACTION, "mode": "purge_recovery", "compose_project": "x"})


def test_worker_capabilities_cover_the_action_and_the_render_field():
    assert compose_secrets.WORKER_CAPABILITY_LABEL in spec_capability_labels(WRITE)
    spec = {"action": "reconcile_config", "compose_config": {"compose_secrets": ["POSTGRES_PASSWORD"]}}
    assert compose_render.COMPOSE_SECRETS_RENDER_LABEL in spec_capability_labels(spec)
    assert compose_render.COMPOSE_SECRETS_RENDER_LABEL not in spec_capability_labels(
        {"action": "reconcile_config", "compose_config": {}})


def test_dockerfile_ships_the_module_and_both_labels():
    text = open(os.path.join(os.path.dirname(__file__), "..", "docker", "Dockerfile.ops-worker")).read()
    assert "src/llm_bawt/ops/compose_secrets.py" in text
    for label in (compose_secrets.WORKER_CAPABILITY_LABEL, compose_render.COMPOSE_SECRETS_RENDER_LABEL):
        assert f'LABEL {label}="1"' in text


# ── parser ────────────────────────────────────────────────────────────────
def test_parser_reads_only_unambiguous_values():
    parsed = parse_env("A=plain\nexport B='lit $x #y \"q\"'\nC=has space\nD=\"dq\"\nE=a#b\nF=$X\nG=''\n")
    assert parsed["A"] == "plain" and parsed["B"] == 'lit $x #y "q"'
    assert all(parsed[n] is None for n in "CDEFG")


# ── helper ────────────────────────────────────────────────────────────────
def test_write_keeps_recovery_first_then_publishes_0400_files(tmp_path):
    paths = _files(tmp_path)
    result = helper_write(WRITE, **paths, stamp="20261116T000000Z")
    assert result == {"recovery": "llm-bawt-20261116T000000Z", "written": [
        {"name": "LLM_BAWT_POSTGRES_PASSWORD", "length": len(f"{PW} $x #y")},
        {"name": "LLM_BAWT_SECRET_KEY", "length": 44}, {"name": "POSTGRES_PASSWORD", "length": len(PW)}]}
    rec = os.path.join(paths["recovery_dir"], result["recovery"])
    assert open(os.path.join(rec, "project.env")).read() == open(paths["project_env"]).read()
    assert open(os.path.join(rec, "secret.key")).read() == KEY
    for entry in result["written"]:
        path = os.path.join(paths["out_dir"], entry["name"] + ".env")
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o400
        assert parse_env(open(path).read()) == {entry["name"]: collect(WRITE["sources"], project_env=paths[
            "project_env"], key_file=paths["key_file"])[entry["name"]]}
    assert sorted(os.listdir(paths["out_dir"])) == sorted(e["name"] + ".env" for e in result["written"])
    assert PW not in json.dumps(result) and KEY not in json.dumps(result)


def test_rewrite_replaces_atomically(tmp_path):
    paths = _files(tmp_path)
    helper_write(WRITE, **paths, stamp="1")
    helper_write(WRITE, **paths, stamp="2")
    assert sorted(os.listdir(paths["recovery_dir"])) == ["llm-bawt-1", "llm-bawt-2"]
    assert not [n for n in os.listdir(paths["out_dir"]) if n.startswith(".")]


@pytest.mark.parametrize("env_text,code", [
    ("OTHER=1\n", "source_missing"), ("POSTGRES_PASSWORD=\n", "source_missing"),
    ("POSTGRES_PASSWORD=a b\nLLM_BAWT_POSTGRES_PASSWORD=x\n", "source_unsupported"),
])
def test_unusable_source_writes_nothing(tmp_path, env_text, code):
    paths = _files(tmp_path, env_text)
    with pytest.raises(SecretsError) as caught:
        helper_write(WRITE, **paths)
    assert caught.value.code == code and "a b" not in str(caught.value)
    assert os.listdir(paths["out_dir"]) == [] and os.listdir(paths["recovery_dir"]) == []


def test_purge_removes_every_recovery_copy(tmp_path):
    paths = _files(tmp_path)
    helper_write(WRITE, **paths, stamp="1")
    assert helper_purge(recovery_dir=paths["recovery_dir"]) == {"purged": 1}
    assert os.listdir(paths["recovery_dir"]) == []


# ── worker side ───────────────────────────────────────────────────────────
def test_helper_body_is_sealed_and_binds_only_named_sources():
    body = helper_body(WRITE, IMAGE, "job1", "/srv/llm-bawt")
    host = body["HostConfig"]
    assert host["NetworkMode"] == "none" and host["ReadonlyRootfs"] and host["CapDrop"] == ["ALL"]
    assert host["CapAdd"] == ["DAC_READ_SEARCH"] and host["Binds"] == []
    mounts = {m["Target"]: m for m in host["Mounts"]}
    assert mounts["/in/project.env"] == {"Type": "bind", "Source": "/srv/llm-bawt/.env",
                                         "Target": "/in/project.env", "ReadOnly": True}
    assert mounts["/in/secret.key"]["Source"] == KEY_FILE and mounts["/in/secret.key"]["ReadOnly"]
    assert mounts["/out"]["Source"] == "llm-bawt-compose-secrets"
    assert mounts["/recovery"]["Source"] == "llm-bawt-env-recovery"
    purge = helper_body({"action": ACTION, "mode": "purge_recovery"}, IMAGE, "job1", None)["HostConfig"]
    assert [m["Target"] for m in purge["Mounts"]] == ["/recovery"] and purge["CapAdd"] == []


class _Api:
    def __init__(self, out, status=0, rows=None):
        self.out, self.status, self.calls, self.side_effect_started = out, status, [], False
        self.rows = rows if rows is not None else [
            {"Labels": {"com.docker.compose.project.working_dir": "/srv/llm-bawt"}}]

    def request(self, method, path, *, body=None, **_):
        self.calls.append((method, path, body))
        if path.startswith("/containers/json"):
            return self.rows
        if path.startswith("/containers/create"):
            return {"Id": "c" * 64}
        if path.endswith("/wait"):
            return {"StatusCode": self.status}
        return None

    def attach_stream(self, _cid):
        from tests.test_ops_image_deploy import _frame, _Stream
        return _Stream(_frame(1, self.out) + _frame(2, "stderr is never echoed"))


def _snapshot(spec=WRITE):
    return {"spec": spec, "execution": {"worker_image": IMAGE, "timeout_seconds": 60}}


def test_run_worker_records_names_and_lengths_and_removes_the_helper():
    out = json.dumps({"written": [{"name": "POSTGRES_PASSWORD", "length": 7}], "recovery": "llm-bawt-1"})
    api, receipt = _Api(out), {}
    assert run_worker(api, _snapshot(), "job1", receipt).startswith("compose secrets write")
    assert receipt["compose_secrets"]["written"][0] == {"name": "POSTGRES_PASSWORD", "length": 7}
    assert receipt["compose_secrets_project_dir"] == "/srv/llm-bawt"
    assert api.calls[-1][0] == "DELETE"


def test_run_worker_reports_a_helper_refusal_by_code():
    api, receipt = _Api(json.dumps({"error": "source_missing", "names": ["POSTGRES_PASSWORD"]}), status=1), {}
    with pytest.raises(SecretsError) as caught:
        run_worker(api, _snapshot(), "job1", receipt)
    assert caught.value.code == "source_missing" and api.calls[-1][0] == "DELETE"


@pytest.mark.parametrize("rows", [[], [{"Labels": {"com.docker.compose.project.working_dir": "/a"}},
                                       {"Labels": {"com.docker.compose.project.working_dir": "/b"}}]])
def test_project_dir_must_be_unanimous(rows):
    with pytest.raises(SecretsError, match="project_dir_unresolved"):
        project_dir(_Api("", rows=rows), "llm-bawt")


def test_run_worker_refuses_args_and_a_mutable_image():
    with pytest.raises(ValueError):
        run_worker(_Api("{}"), {**_snapshot(), "resolved_args": {"x": 1}}, "job1", {})
    with pytest.raises(ValueError):
        run_worker(_Api("{}"), {"spec": WRITE, "execution": {"worker_image": "worker:latest"}}, "job1", {})
