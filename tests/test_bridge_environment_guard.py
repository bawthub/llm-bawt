"""TASK-1523: refuse shared-environment writers before executing any tool."""
from __future__ import annotations

import io
import json
import sys

import pytest

from claude_code_bridge.environment_guard import environment_gate
from claude_code_bridge.skill_hooks import main


@pytest.fixture
def workspace(tmp_path):
    shared = tmp_path / "llm-bawt"
    isolated = tmp_path / ".worktrees" / ".wt-llm-bawt-test-TASK-1"
    for root in (shared, isolated):
        root.mkdir(parents=True)
        (root / "pyproject.toml").write_text('[project]\nname = "llm-bawt"\n')
        (root / ".venv" / "bin").mkdir(parents=True)
    # A venv's interpreter commonly resolves outside the venv. Invocation via
    # this symlink still installs into the venv, not the interpreter's home.
    (shared / ".venv" / "bin" / "python").symlink_to(sys.executable)
    (shared / ".venv" / "bin" / "pip").symlink_to(sys.executable)
    return shared, isolated, {
        "BAWT_AGENT_BRIDGE": "1",
        "BRIDGE_WORKSPACE_ROOT": str(tmp_path),
        "HOME": str(tmp_path.parent),
        "PATH": str(shared / ".venv" / "bin"),
    }


def decide(workspace, command, *, isolated=False, extra=None):
    shared, task, env = workspace
    cwd = task if isolated else shared
    command = command.format(shared=shared, task=task)
    return environment_gate({"cwd": str(cwd), "tool_input": {"command": command}},
                            {**env, **(extra or {})})


@pytest.mark.parametrize("command", [
    "uv run pytest tests/test_x.py",
    "uv run --no-sync pytest tests/test_x.py",
    "uv sync",
    "/usr/local/bin/uv sync --frozen",
    "uv add pytest",
    "uv remove pytest",
    "uv venv",
    "uv pip install pytest",
    "uv pip sync requirements.txt",
    "uv pip uninstall pytest",
    ".venv/bin/python -m pip install pytest",
    ".venv/bin/pip install pytest",
    "pip install pytest",
    "python -m pip uninstall pytest",
    "uv --directory {shared} sync",
    "uv --project={shared} run python -V",
    "uv run --project {shared} python -V",
    "uv run --directory {shared} python -V",
    "cd {shared}/src && uv sync",
    "(cd {shared} && uv sync)",
    "bash -lc 'cd {shared} && uv run pytest tests/test_x.py'",
    "env UV_PROJECT_ENVIRONMENT={shared}/.venv uv sync",
    "UV_PROJECT_ENVIRONMENT={shared}/.venv uv sync",
    "export UV_PROJECT_ENVIRONMENT={shared}/.venv; uv sync",
    "command uv sync",
    "sudo -u bridge uv sync",
    'echo "$(uv sync)"',
    'echo "`uv sync`"',
    "echo $(uv sync)",
])
def test_shared_writers_are_denied(workspace, command):
    (workspace[0] / "src").mkdir(exist_ok=True)
    result = decide(workspace, command)
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "deny"
    assert "host-owned .venv" in output["permissionDecisionReason"]
    assert "task-local" in output["permissionDecisionReason"]


@pytest.mark.parametrize("command", [
    "uv sync",
    "uv run pytest tests/test_x.py",
    "uv pip install pytest",
    "uv venv",
    ".venv/bin/python -m pip install pytest",
    ".venv/bin/pip install pytest",
])
def test_task_local_environments_stay_allowed(workspace, command):
    assert decide(workspace, command, isolated=True) is None


@pytest.mark.parametrize("command", [
    "pytest tests/test_x.py",
    "python -m pytest tests/test_x.py",
    "ruff check src/example.py",
    "uv --version",
    "uv lock",
    "uv pip list",
    "uv pip show pytest",
    "uv pip show install",
    "pip show install",
    "uv add --no-sync pytest",
    "uv run --project {task} pytest tests/test_x.py",
    "uv run --directory {task} pytest tests/test_x.py",
    "uv --directory {task} sync",
    "uv --project {task} sync",
    "uv venv {task}/other-env",
    "UV_PROJECT_ENVIRONMENT={task}/.venv uv sync",
    "uv pip install --python {task}/.venv/bin/python pytest",
    "uv pip install --target {task}/packages pytest",
    "uv pip install --system pytest",
    "uv pip list; echo 'uv sync'",
    "printf '%s' 'uv run pytest tests/test_x.py'",
    "grep 'uv sync' docs/TESTING.md",
    "ssh test-host 'cd /repo && uv sync'",
])
def test_reads_and_explicit_isolation_are_not_blanket_banned(workspace, command):
    assert decide(workspace, command) is None


@pytest.mark.parametrize("command", [
    "uv sync --project {shared}",
    "cd {shared} && uv sync",
    "uv pip install --python {shared}/.venv/bin/python pytest",
    "python -m pip --python {shared}/.venv/bin/python install pytest",
    "uv pip install --target {shared}/.venv/lib pytest",
    "uv pip install --prefix {shared}/.venv pytest",
    "uv venv {shared}/.venv",
    "UV_PROJECT_ENVIRONMENT={shared}/.venv uv run pytest tests/test_x.py",
    "source {shared}/.venv/bin/activate; pip install pytest",
    "(cd {task} && true); uv sync --project {shared}",
    "uv -C{shared} sync",
    "python -m uv sync --project {shared}",
    "cd {shared}; PATH=.venv/bin:$PATH pip install pytest",
])
def test_task_cwd_does_not_hide_a_shared_target(workspace, command):
    assert decide(workspace, command, isolated=True) is not None


@pytest.mark.parametrize("command, variable", [
    ("uv sync", "UV_PROJECT_ENVIRONMENT"),
    ("uv pip install pytest", "VIRTUAL_ENV"),
    ("uv pip install pytest", "PIP_TARGET"),
    ("uv pip install pytest", "PIP_PREFIX"),
    ("uv run --active pytest tests/test_x.py", "VIRTUAL_ENV"),
])
def test_inherited_environment_targets_are_checked(workspace, command, variable):
    assert decide(workspace, command, isolated=True,
                  extra={variable: str(workspace[0] / ".venv")}) is not None


def test_symlinked_task_environment_is_not_isolated(workspace):
    shared, task, env = workspace
    alias = task / "shared-env"
    alias.symlink_to(shared / ".venv", target_is_directory=True)
    assert decide(workspace, "uv sync", isolated=True,
                  extra={"UV_PROJECT_ENVIRONMENT": str(alias)}) is not None
    assert decide(workspace, f"{alias}/bin/python -m pip install pytest", isolated=True) is not None


def test_uv_run_program_options_cannot_hide_a_shared_default(workspace):
    assert decide(workspace, "uv run python app.py --directory {task}") is not None
    assert decide(workspace, "uv run -- python app.py --project {task}") is not None


def test_subshell_cwd_does_not_leak_to_the_next_command(workspace):
    assert decide(workspace, "(cd {task} && true); uv sync") is not None
    assert decide(workspace, "(cd {shared} && true); uv sync", isolated=True) is None


def test_relative_uv_environment_is_relative_to_project_not_subdirectory(workspace):
    (workspace[0] / "src").mkdir()
    assert decide(workspace, "cd src && uv sync",
                  extra={"UV_PROJECT_ENVIRONMENT": ".venv"}) is not None


@pytest.mark.parametrize("command", [
    "env -i uv sync",
    "env -u UV_PROJECT_ENVIRONMENT uv sync",
    "unset UV_PROJECT_ENVIRONMENT; uv sync",
])
def test_clearing_an_isolation_override_rechecks_the_shared_default(workspace, command):
    assert decide(workspace, command, extra={
        "UV_PROJECT_ENVIRONMENT": str(workspace[1] / ".venv")}) is not None


@pytest.mark.parametrize("prefix", [
    "timeout 120",
    "/usr/bin/timeout 5m",
    "timeout --signal TERM --kill-after 5s 120",
    "timeout -s KILL -k 5s 120",
    "timeout --signal=TERM --kill-after=5s 120",
    "timeout --foreground -- 120",
    "env timeout 120 bash -c",
])
def test_timeout_wrappers_cannot_start_shared_uv(workspace, prefix):
    assert decide(workspace, f"{prefix} 'uv run pytest tests/test_x.py'"
                  if prefix.endswith("-c") else f"{prefix} uv run pytest tests/test_x.py") is not None
    assert decide(workspace, f"{prefix} 'uv run pytest tests/test_x.py'"
                  if prefix.endswith("-c") else f"{prefix} uv run pytest tests/test_x.py",
                  isolated=True) is None


def test_pushd_popd_and_subshell_directory_stack(workspace):
    assert decide(workspace, "pushd {shared}; uv sync", isolated=True) is not None
    assert decide(workspace, "pushd {shared}; popd; uv sync", isolated=True) is None
    assert decide(workspace, "pushd {task}; popd; uv sync") is not None
    assert decide(workspace, "(pushd {shared}); uv sync", isolated=True) is None


def test_does_not_apply_the_bridge_rule_on_the_host(workspace):
    assert decide(workspace, "uv sync", extra={"BAWT_AGENT_BRIDGE": "0"}) is None
    assert decide(workspace, "uv sync", extra={"BAWT_AGENT_BRIDGE": "0",
                                              "BRIDGE_BROKER_TOKEN_FILE": "/broker"}) is not None


def test_cli_emits_pretool_deny_without_starting_the_command(workspace, monkeypatch):
    import subprocess

    def forbidden(*args, **kwargs):
        pytest.fail("The admission hook must not start uv, pip or a shell.")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    shared, _, env = workspace
    data = {"cwd": str(shared), "tool_input": {"command": "uv sync"}}
    out = io.StringIO()
    assert main(["environment-gate"], stdin=io.StringIO(json.dumps(data)),
                env=env, stdout=out) == 0
    output = json.loads(out.getvalue())["hookSpecificOutput"]
    assert output["hookEventName"] == "PreToolUse"
    assert output["permissionDecision"] == "deny"
