"""TASK-2302: an inherited GIT_DIR/GIT_WORK_TREE cannot retarget a test's git."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from tests import git_env_guard

ROOT = Path(__file__).resolve().parents[1]
CLEAN = {k: v for k, v in os.environ.items() if k not in git_env_guard.SCRUBBED_ENV}


def git(cwd: Path, *args: str, env: dict[str, str] | None = None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True,
                          env=env or CLEAN).stdout.strip()


@pytest.fixture
def shared_with_worktree(tmp_path):
    """A shared checkout plus one linked worktree, like ~/dev/llm-bawt and a task tree."""
    shared = tmp_path / "dev" / "llm-bawt"
    shared.mkdir(parents=True)
    git(shared, "init", "-q", "-b", "master")
    git(shared, "-c", "user.name=T", "-c", "user.email=t@example.com",
        "commit", "-q", "--allow-empty", "-m", "base")
    tree = tmp_path / "dev" / ".worktrees" / ".wt-llm-bawt-bot-TASK-1"
    git(shared, "worktree", "add", "-q", "--detach", str(tree))
    return shared, tree, git(tree, "rev-parse", "--absolute-git-dir")


def init_scratch(tmp_path: Path, env: dict[str, str]) -> None:
    """The offending call from tests/test_general_use_contract.py setUp."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    subprocess.run(["git", "-C", str(scratch), "init", "-q"], check=True, env=env)


def test_scrub_removes_only_repo_locating_variables():
    env = {"GIT_DIR": "/x", "GIT_WORK_TREE": "/y", "GIT_INDEX_FILE": "/z",
           "GIT_EDITOR": "true", "GIT_AUTHOR_NAME": "a", "PATH": "/bin"}
    assert git_env_guard.scrub(env) == ["GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"]
    assert env == {"GIT_EDITOR": "true", "GIT_AUTHOR_NAME": "a", "PATH": "/bin"}


@pytest.mark.real_process(reason="reproduces the incident with real git init in temp repos")
def test_incident_reproduces_without_scrub_and_not_with_it(tmp_path, shared_with_worktree):
    shared, tree, admin = shared_with_worktree
    inherited = {**CLEAN, "GIT_DIR": admin, "GIT_WORK_TREE": str(tree)}

    init_scratch(tmp_path, dict(inherited))
    assert git(shared, "config", "--get", "core.worktree") == str(tree)  # the 2026-11-14 write

    git(shared, "config", "--unset", "core.worktree")
    (tmp_path / "scratch").rename(tmp_path / "scratch-before")
    scrubbed = dict(inherited)
    git_env_guard.scrub(scrubbed)
    init_scratch(tmp_path, scrubbed)
    unset = subprocess.run(["git", "config", "--get", "core.worktree"], cwd=shared, env=CLEAN)
    assert unset.returncode == 1
    assert (tmp_path / "scratch" / ".git").is_dir()


def test_root_conftest_installs_the_guard_before_tests_import():
    text = (ROOT / "tests" / "conftest.py").read_text()
    assert "git_env_guard.install()" in text
    assert not set(git_env_guard.SCRUBBED_ENV) & os.environ.keys()


@pytest.mark.real_process(reason="resolves a worktree .git link with real git")
def test_link_hint_names_a_relative_link_that_resolves(tmp_path, shared_with_worktree):
    shared, tree, admin = shared_with_worktree
    assert git_env_guard.link_hint(tree) is None

    # Simulate a tree created under another mount: its absolute link is dangling here.
    (tree / ".git").write_text(f"gitdir: /elsewhere/dev/llm-bawt/.git/worktrees/{Path(admin).name}\n")
    hint = git_env_guard.link_hint(tree)
    assert hint and "does not exist here" in hint
    relative = hint.rsplit("printf 'gitdir: ", 1)[1].split("\\n'", 1)[0]
    assert relative == f"../../llm-bawt/.git/worktrees/{Path(admin).name}"
    (tree / ".git").write_text(f"gitdir: {relative}\n")
    assert git_env_guard.link_hint(tree) is None
