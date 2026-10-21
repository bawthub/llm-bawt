"""Startup and teardown failures must not leave private Docker resources behind."""

import subprocess
from types import SimpleNamespace

import pytest

from tests import private_postgres as pg


def test_docker_launch_failure_removes_fixture_socket(monkeypatch, tmp_path):
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
