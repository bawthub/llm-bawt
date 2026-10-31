"""Startup and teardown failures must not leave private Docker resources behind."""

from contextlib import contextmanager
import json
import socket as socketlib
import subprocess
from types import SimpleNamespace

import pytest

from tests import private_postgres as pg


def test_docker_launch_failure_removes_fixture_socket(monkeypatch, tmp_path):
    monkeypatch.delenv("OPS_GATE_SANDBOX", raising=False)
    monkeypatch.delenv("OPS_GATE_PG_BINDING", raising=False)
    monkeypatch.setattr(pg.shutil, "which", lambda name: "/usr/bin/docker")
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[1] == "run":
            raise subprocess.CalledProcessError(125, command)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(pg.subprocess, "run", fake_run)
    factory = SimpleNamespace(mktemp=lambda name: tmp_path)
    fixture = pg.private_pg.__wrapped__(factory)
    with pytest.raises(subprocess.CalledProcessError):
        next(fixture)
    assert [command[1] for command in commands] == ["run", "inspect"]
    assert not (tmp_path / "socket").exists()
    assert commands[-1][-1].startswith("llm-bawt-pgtest-")


def test_interrupted_launch_stops_only_matching_owned_container(monkeypatch, tmp_path):
    monkeypatch.delenv("OPS_GATE_SANDBOX", raising=False)
    monkeypatch.delenv("OPS_GATE_PG_BINDING", raising=False)
    monkeypatch.setattr(pg.shutil, "which", lambda name: "/usr/bin/docker")
    commands = []
    token = None

    def fake_run(command, **kwargs):
        nonlocal token
        commands.append(command)
        if command[1] == "run":
            token = next(arg.split("=", 1)[1] for arg in command
                         if arg.startswith("llm-bawt.pg-acceptance.owner="))
            raise subprocess.CalledProcessError(125, command)
        if command[1] == "inspect":
            return SimpleNamespace(returncode=0, stdout=token + "\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pg.subprocess, "run", fake_run)
    fixture = pg.private_pg.__wrapped__(SimpleNamespace(mktemp=lambda name: tmp_path))
    with pytest.raises(subprocess.CalledProcessError):
        next(fixture)
    assert [command[1] for command in commands] == ["run", "inspect", "stop"]
    assert not (tmp_path / "socket").exists()


def test_isolated_gate_requires_private_pg_binding(monkeypatch, tmp_path):
    monkeypatch.setenv("OPS_GATE_SANDBOX", "1")
    monkeypatch.delenv("OPS_GATE_PG_BINDING", raising=False)
    monkeypatch.setattr(pg.shutil, "which", lambda _name: pytest.fail("Docker was probed"))
    fixture = pg.private_pg.__wrapped__(SimpleNamespace(mktemp=lambda _name: tmp_path))
    with pytest.raises(RuntimeError, match="private PostgreSQL binding"):
        next(fixture)


@pytest.mark.parametrize("marker,binding", [("0", "pg.json"), (None, "pg.json")])
def test_unpaired_or_invalid_gate_binding_refuses(monkeypatch, tmp_path, marker, binding):
    if marker is None:
        monkeypatch.delenv("OPS_GATE_SANDBOX", raising=False)
    else:
        monkeypatch.setenv("OPS_GATE_SANDBOX", marker)
    monkeypatch.setenv("OPS_GATE_PG_BINDING", str(tmp_path / binding))
    monkeypatch.setattr(pg.shutil, "which", lambda _name: pytest.fail("Docker was probed"))
    fixture = pg.private_pg.__wrapped__(SimpleNamespace(mktemp=lambda _name: tmp_path))
    with pytest.raises(RuntimeError, match="private PostgreSQL binding"):
        next(fixture)


def test_isolated_gate_binds_only_its_checked_out_private_socket(monkeypatch, tmp_path):
    socket = tmp_path / "sidecar-socket"
    socket.mkdir()
    receipt = tmp_path / "pg.json"
    sha = "a" * 40
    receipt.write_text(json.dumps({
        "schema": "ops.gate.pg/v1", "job_id": "b" * 32,
        "repo": "llm-bawt", "sha": sha, "mode": "unix",
        "jobs": {"private-pg": {
            "image": pg.IMAGE, "socket_dir": str(socket), "port": pg.PORT,
            "user": "test_user", "database": "llm_bawt_test",
        }},
    }))
    monkeypatch.setenv("OPS_GATE_SANDBOX", "1")
    monkeypatch.setenv("OPS_GATE_PG_BINDING", str(receipt))
    monkeypatch.setattr(pg.shutil, "which", lambda _name: pytest.fail("Docker was probed"))
    monkeypatch.setattr(pg.subprocess, "run", lambda argv, **_kwargs: (
        SimpleNamespace(returncode=0, stdout=sha + "\n")
        if argv[0] == "git" else pytest.fail("Docker was run")))
    calls = []

    @contextmanager
    def private_socket(path, data_dir, port):
        calls.append((path, data_dir, port))
        yield

    class Engine:
        def connect(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def exec_driver_sql(self, _sql):
            return SimpleNamespace(scalar_one=lambda: "postgres")

        def dispose(self):
            pass

    monkeypatch.setattr(pg, "private_cluster_socket", private_socket)
    monkeypatch.setattr(pg, "create_engine", lambda _url: Engine())
    fixture = pg.private_pg.__wrapped__(SimpleNamespace(mktemp=lambda _name: pytest.fail("temporary socket created")))
    with socketlib.socket(socketlib.AF_UNIX, socketlib.SOCK_STREAM) as server:
        server.bind(str(socket / f".s.PGSQL.{pg.PORT}"))
        assert next(fixture) == socket.resolve()
        with pytest.raises(StopIteration):
            next(fixture)
    assert calls == [(socket.resolve(), None, pg.PORT)]


def test_isolated_gate_missing_live_sidecar_socket_fails_not_skips(monkeypatch, tmp_path):
    descriptor = tmp_path / "pg.json"
    descriptor.write_text(json.dumps({
        "schema": "ops.gate.pg/v1", "job_id": "b" * 32,
        "repo": "llm-bawt", "sha": "a" * 40, "mode": "unix",
        "jobs": {"private-pg": {
            "image": pg.IMAGE, "socket_dir": str(tmp_path), "port": pg.PORT,
            "user": "test_user", "database": "llm_bawt_test",
        }},
    }))
    monkeypatch.setenv("OPS_GATE_SANDBOX", "1")
    monkeypatch.setenv("OPS_GATE_PG_BINDING", str(descriptor))
    monkeypatch.setattr(pg.shutil, "which", lambda _name: pytest.fail("Docker was probed"))
    monkeypatch.setattr(pg.subprocess, "run", lambda *_a, **_kw: SimpleNamespace(returncode=0, stdout="a" * 40))
    fixture = pg.private_pg.__wrapped__(SimpleNamespace(mktemp=lambda _name: pytest.fail("temporary socket created")))
    with pytest.raises(RuntimeError, match="private PostgreSQL cluster socket is not running"):
        next(fixture)


def test_isolated_gate_rejects_wrong_commit_and_extra_keys(monkeypatch, tmp_path):
    descriptor = tmp_path / "pg.json"
    descriptor.write_text(json.dumps({
        "schema": "ops.gate.pg/v1", "job_id": "b" * 32,
        "repo": "llm-bawt", "sha": "a" * 40, "mode": "unix",
        "jobs": {"private-pg": {
            "image": pg.IMAGE, "socket_dir": str(tmp_path), "port": pg.PORT,
            "user": "test_user", "database": "llm_bawt_test", "unexpected": "extra",
        }},
    }))
    monkeypatch.setenv("OPS_GATE_SANDBOX", "1")
    monkeypatch.setenv("OPS_GATE_PG_BINDING", str(descriptor))
    monkeypatch.setattr(pg.shutil, "which", lambda _name: pytest.fail("Docker was probed"))
    fixture = pg.private_pg.__wrapped__(SimpleNamespace(mktemp=lambda _name: tmp_path))
    with pytest.raises(RuntimeError, match="private PostgreSQL binding"):
        next(fixture)
    data = json.loads(descriptor.read_text())
    del data["jobs"]["private-pg"]["unexpected"]
    descriptor.write_text(json.dumps(data))
    monkeypatch.setattr(pg.subprocess, "run", lambda *_a, **_kw: SimpleNamespace(returncode=0, stdout="c" * 40))
    fixture = pg.private_pg.__wrapped__(SimpleNamespace(mktemp=lambda _name: tmp_path))
    with pytest.raises(RuntimeError, match="private PostgreSQL binding.*commit"):
        next(fixture)
