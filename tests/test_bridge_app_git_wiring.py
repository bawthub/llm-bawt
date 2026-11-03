"""App git/gh wiring must not relax the ops runtime guard (TASK-1987)."""
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import yaml

from llm_bawt.ops.source_state import _classify_env, _path_list

ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = (
    "/home/bridge/.local/bin:/app/.venv/bin:/usr/local/sbin:/usr/local/bin:"
    "/usr/sbin:/usr/bin:/sbin:/bin:/opt/agent-skills/bin"
)
MOUNTS = {
    "/app/src": False,
    "/opt/agent-skills/bin": True,
    "/home/bridge/.config/bawthub-git": True,
}


@pytest.fixture
def dispatch():
    return runpy.run_path(str(ROOT / "docker/gh-app-dispatch"))


class Executed(Exception):
    def __init__(self, target, args, env):
        self.target, self.args_passed, self.env = target, args, env


@pytest.mark.parametrize(("rc", "value", "app"), [(1, "", False), (0, "false\n", False), (0, "true\n", True)])
def test_dispatch_native_or_app_without_changing_argv(dispatch, monkeypatch, rc, value, app):
    calls = []

    def config(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=rc, stdout=value)

    def execute(target, args, env):
        raise Executed(target, args, env)

    monkeypatch.setattr(dispatch["subprocess"], "run", config)
    monkeypatch.setattr(dispatch["os"], "execve", execute)
    monkeypatch.setenv("BAWTHUB_GH_BIN", "/unexpected/gh")
    monkeypatch.setenv("BAWTHUB_GIT_CREDENTIAL_HELPER", "/unexpected/helper")
    with pytest.raises(Executed) as raised:
        dispatch["main"](["gh", "api", "repos/owner/repo", "--jq", ".full_name"])
    call = raised.value
    target = dispatch["HELPER"] if app else dispatch["GH"]
    assert call.target == target
    assert call.args_passed == [target, "api", "repos/owner/repo", "--jq", ".full_name"]
    assert calls == [["/usr/bin/git", "config", "--type=bool", "--get", "bawthub.requireTurnIdentity"]]
    if app:
        assert call.env["BAWTHUB_GH_BIN"] == "/usr/bin/gh"
        assert call.env["BAWTHUB_GIT_CREDENTIAL_HELPER"] == dispatch["CREDENTIAL_HELPER"]
    # Per-child dispatch must never mutate the parent process configuration.
    assert dispatch["os"].environ["BAWTHUB_GH_BIN"] == "/unexpected/gh"


@pytest.mark.parametrize("rc", [2, 128])
def test_invalid_config_refuses_without_native_fallback(dispatch, monkeypatch, capsys, rc):
    monkeypatch.setattr(dispatch["subprocess"], "run", lambda *a, **k: SimpleNamespace(returncode=rc, stdout=""))
    monkeypatch.setattr(dispatch["os"], "execve", lambda *a: pytest.fail("must not execute any gh"))
    assert dispatch["main"](["gh", "api", "repos/owner/repo"]) == 1
    assert capsys.readouterr().err == "gh: account Git configuration could not be read\n"


def test_missing_app_helper_refuses_without_native_fallback(dispatch, monkeypatch, capsys):
    monkeypatch.setattr(dispatch["subprocess"], "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="true\n"))
    calls = []

    def execute(target, *args):
        calls.append(target)
        raise FileNotFoundError("private detail must not be logged")

    monkeypatch.setattr(dispatch["os"], "execve", execute)
    assert dispatch["main"](["gh", "api", "repos/owner/repo"]) == 1
    assert calls == [dispatch["HELPER"]]
    assert capsys.readouterr().err == "gh: configured GitHub command is unavailable\n"


@pytest.mark.parametrize("name", ["codex-bridge", "claude-code-bridge"])
def test_bridge_overlay_keeps_image_path_and_no_git_env(name):
    service = yaml.safe_load((ROOT / "docker-compose.dev.yml").read_text())["services"][name]
    env = service.get("environment", {})
    assert not any(key.startswith("GIT_") for key in env)
    assert "PATH" not in env
    assert any(v.endswith(":/home/bridge/.config/bawthub-git:ro") for v in service["volumes"])
    assert _classify_env({"PATH": IMAGE_PATH, **env}, MOUNTS, set()) == []


def test_single_file_ro_gh_mount_is_not_a_guard_workaround():
    assert _path_list(IMAGE_PATH, MOUNTS)
    assert not _path_list(IMAGE_PATH, {**MOUNTS, "/usr/local/bin/gh": True})


def test_image_bakes_optional_config_and_dispatcher():
    dockerfile = (ROOT / "Dockerfile.agent-bridge").read_text()
    assert "COPY docker/gh-app-dispatch /usr/local/bin/gh" in dockerfile
    assert "git config --system --add include.path /home/bridge/.config/bawthub-git/gitconfig" in dockerfile
    assert "ENV GIT_CONFIG" not in dockerfile
