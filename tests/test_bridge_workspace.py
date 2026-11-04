"""TASK-837: safe shared workspace writes, independent of any live tenant."""

import asyncio
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_bridge.health_http import health_app
from agent_bridge.workspace import Workspace, WorkspaceError, add_workspace_routes, clone_input


@pytest.mark.parametrize("url", [
    "file:///tmp/repo", "/tmp/repo", "../repo", "--upload-pack=evil",
    "ext::sh -c evil", "http://github.com/o/r", "https://user:secret@github.com/o/r",
    "https://secret@github.com/o/r", "https://github.com/o/r?token=secret",
    "https://github.com/o/r#secret", "https://github.com\\@evil.test/o/r",
    "ssh://root@host/repo", "git@host:../repo", "https://github.com/o/../r",
    "https://host:bad/repo",
    "https://host/repo\n", "https://host", "https://host/",
])
def test_reject_invalid_url_without_echo(url):
    with pytest.raises(WorkspaceError) as error:
        clone_input({"url": url})
    assert error.value.status == 400
    assert url not in str(error.value)


@pytest.mark.parametrize("name", ["../r", "/tmp/r", ".git", "-r", "a/b", "a\\b", "..", "", "x" * 129, "a."])
def test_invalid_folder(name):
    with pytest.raises(WorkspaceError):
        clone_input({"url": "https://example.test/team/repo.git", "name": name})


@pytest.mark.parametrize("branch", ["--upload-pack=bad", "/main", "a..b", "a.lock", ".main", "a/.b", "a//b",
                                  "a@{b", "a b", "a?", "a\\b", "a\nb", "", "@"])
def test_invalid_branch(branch):
    with pytest.raises(WorkspaceError):
        clone_input({"url": "https://example.test/team/repo.git", "branch": branch})


def test_normalize_and_derive():
    item = clone_input({"url": "git@github.com:team/repo.git", "branch": "feature/task-837"})
    assert (item.url, item.name, item.github, item.branch) == (
        "https://github.com/team/repo.git", "repo", True, "feature/task-837",
    )
    item = clone_input({"url": "ssh://git@git.example.test/team/repo.git", "name": "my.repo"})
    assert item.url.startswith("ssh://") and not item.github and item.name == "my.repo"


@pytest.mark.asyncio
async def test_existing_paths_and_symlinks_untouched(tmp_path):
    workspace = Workspace(tmp_path)
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    (outside / "keep").write_text("untouched")
    (tmp_path / "repo").symlink_to(outside)
    with pytest.raises(WorkspaceError) as error:
        await workspace.clone(clone_input({"url": "https://example.test/repo.git"}))
    assert error.value.status == 409
    assert (outside / "keep").read_text() == "untouched"
    assert workspace._active == 0
    assert workspace.listing() == {"entries": []}


@pytest.mark.asyncio
async def test_real_git_clone_clean_and_visible(tmp_path, monkeypatch):
    """Real Git checkout, with only the network transport replaced locally."""
    source = tmp_path / "source"
    source.mkdir()
    def git(*args, cwd=source):
        return subprocess.check_output(["git", *args], cwd=cwd)
    git("init", "-b", "main")
    (source / "hello.txt").write_text("hello\n")
    git("add", ".")
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.test", "commit", "-m", "fixture")
    git("branch", "feature/test")
    root = tmp_path / "workspace"
    root.mkdir()
    original = asyncio.create_subprocess_exec
    commands = []
    async def local_transport(*args, **kwargs):
        commands.append((args, kwargs))
        assert args[-2] == "https://example.test/repo.git"
        assert kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"
        assert kwargs["start_new_session"] is True
        args = [args[0], "-c", "protocol.file.allow=always", *args[1:-2], str(source), args[-1]]
        return await original(*args, **kwargs)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", local_transport)
    workspace = Workspace(root)
    result = await workspace.clone(clone_input({"url": "https://example.test/repo.git", "branch": "feature/test"}))
    assert result == {"name": "repo", "path": "repo", "branch": "feature/test"}
    assert (root / "repo" / "hello.txt").read_text() == "hello\n"
    assert git("status", "--porcelain", cwd=root / "repo") == b""
    assert git("branch", "--show-current", cwd=root / "repo").strip() == b"feature/test"
    assert workspace.listing() == {"entries": [{"name": "repo", "path": "repo", "isGitRepo": True}]}
    assert len(commands) == 1
    assert not list(root.glob(".clone-*"))


@pytest.mark.asyncio
async def test_failure_cleanup_static_diagnostic(tmp_path, monkeypatch):
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=SimpleNamespace(
        wait=AsyncMock(return_value=128), returncode=128,
    )))
    workspace = Workspace(tmp_path)
    with pytest.raises(WorkspaceError) as error:
        await workspace.clone(clone_input({"url": "https://example.test/repo.git"}))
    assert error.value.status == 502
    assert list(tmp_path.iterdir()) == []
    assert workspace._active == 0


@pytest.mark.asyncio
async def test_concurrent_duplicate_and_cancellation_cleanup(tmp_path, monkeypatch):
    entered, released = asyncio.Event(), asyncio.Event()
    proc = SimpleNamespace(pid=12345, returncode=None)
    async def wait():
        await released.wait()
        proc.returncode = -9
    proc.wait = wait
    async def spawn(*args, **kwargs):
        entered.set()
        return proc
    killed = []
    def kill(pid, sig):
        killed.append(pid)
        released.set()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr("agent_bridge.workspace.os.killpg", kill)
    workspace = Workspace(tmp_path)
    item = clone_input({"url": "https://example.test/repo.git"})
    task = asyncio.create_task(workspace.clone(item))
    await entered.wait()
    with pytest.raises(WorkspaceError) as error:
        await workspace.clone(item)
    assert error.value.status == 409
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert killed == [12345]
    assert list(tmp_path.iterdir()) == []
    assert workspace._active == 0


@pytest.mark.asyncio
async def test_timeout_kills_group_and_cleans(tmp_path, monkeypatch):
    released = asyncio.Event()
    proc = SimpleNamespace(pid=12345, returncode=None)
    async def wait():
        await released.wait()
        proc.returncode = -9
    proc.wait = wait
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=proc))
    monkeypatch.setattr("agent_bridge.workspace.os.killpg", lambda *_: released.set())
    # Zero deadline exercises wait_for's timeout path without wall-clock sleeps.
    with pytest.raises(WorkspaceError) as error:
        await Workspace(tmp_path, timeout=0).clone(clone_input({"url": "https://example.test/repo.git"}))
    assert error.value.status == 504
    assert list(tmp_path.iterdir()) == []


def test_http_validation_and_token_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("BRIDGE_CLAUDE_TOKEN_SECRET", "fixture-only")
    monkeypatch.delenv("BRIDGE_BROKER_TOKEN_FILE", raising=False)
    app = FastAPI()
    add_workspace_routes(app, tmp_path)
    with TestClient(app) as client:
        assert client.get("/workspace").status_code == 401
        headers = {"X-Bridge-Token": "fixture-only"}
        assert client.get("/workspace", headers=headers).json() == {"entries": []}
        response = client.post("/workspace/clone", json={"url": "https://user:secret@host/repo"}, headers=headers)
        assert response.status_code == 400 and "secret" not in response.text
        assert client.post("/workspace/clone", content="x" * 4097, headers=headers).status_code == 400


def test_bridge_accepts_broker_file_and_legacy_during_migration(tmp_path, monkeypatch):
    broker = tmp_path / "broker"
    broker.write_text("derived-fixture")
    monkeypatch.setenv("BRIDGE_BROKER_TOKEN_FILE", str(broker))
    monkeypatch.setenv("BRIDGE_CLAUDE_TOKEN_SECRET", "legacy-fixture")
    app = FastAPI()
    add_workspace_routes(app, tmp_path)
    with TestClient(app) as client:
        for value in ["derived-fixture", "legacy-fixture"]:
            assert client.get("/workspace", headers={"X-Bridge-Token": value}).status_code == 200
        assert client.get("/workspace", headers={"X-Bridge-Token": "bad"}).status_code == 401


def test_health_and_models_preserved(monkeypatch):
    monkeypatch.delenv("BRIDGE_CLAUDE_TOKEN_SECRET", raising=False)
    publisher = SimpleNamespace(connected=True)
    models = AsyncMock(return_value=[{"id": "fixture"}])
    with TestClient(health_app(publisher, models)) as client:
        assert client.get("/health").json() == {"redis": True}
        assert client.get("/models").json() == {"models": [{"id": "fixture"}]}
        publisher.connected = False
        assert client.get("/health").status_code == 503
        models.side_effect = RuntimeError("secret")
        response = client.get("/models")
        assert response.status_code == 503 and "secret" not in response.text
