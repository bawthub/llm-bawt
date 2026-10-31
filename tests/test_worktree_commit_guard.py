"""TASK-1275: direct commits stay in the owner's task worktree."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


HOOK = Path(__file__).resolve().parents[1] / ".githooks" / "pre-commit"
POST_CHECKOUT = HOOK.with_name("post-checkout")


def git(where: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args], cwd=where, text=True, capture_output=True,
        env={k: v for k, v in os.environ.items() if k not in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE")},
    )
    if check and result.returncode:
        raise AssertionError(f"git {' '.join(args)}: {result.stderr}")
    return result


@pytest.fixture
def repo(tmp_path: Path) -> tuple[Path, Path]:
    shared = tmp_path / "service"
    git(tmp_path, "init", "-q", "-b", "master", str(shared))
    git(shared, "config", "user.name", "Test")
    git(shared, "config", "user.email", "test@example.test")
    git(shared, "config", "branch.canonical", "master")
    git(shared, "commit", "--allow-empty", "-qm", "base")
    hooks = shared / ".githooks"
    hooks.mkdir()
    shutil.copy2(HOOK, hooks / "pre-commit")
    git(shared, "add", ".githooks/pre-commit")
    git(shared, "commit", "-qm", "seed hook before activation")
    git(shared, "config", "core.hooksPath", ".githooks")
    linked = tmp_path / ".wt-service-al"
    git(shared, "worktree", "add", "-q", "-b", "al/TASK-1275-hooks", str(linked))
    return shared, linked


def test_shared_checkout_refuses_direct_commit_even_on_canonical(repo: tuple[Path, Path]) -> None:
    shared, _ = repo
    original = git(shared, "rev-parse", "HEAD").stdout.strip()
    result = git(shared, "commit", "--allow-empty", "-m", "TASK-1275 direct", check=False)
    assert result.returncode != 0
    assert "shared canonical checkout" in result.stderr
    assert git(shared, "rev-parse", "HEAD").stdout.strip() == original


def test_missing_canonical_config_refuses_with_diagnostic(repo: tuple[Path, Path]) -> None:
    shared, _ = repo
    git(shared, "config", "--unset", "branch.canonical")
    denied = git(shared, "commit", "--allow-empty", "-m", "TASK-1275 missing config", check=False)
    assert denied.returncode != 0
    assert "branch.canonical is not configured" in denied.stderr


def test_owner_task_worktree_commits_and_other_owner_is_refused(repo: tuple[Path, Path]) -> None:
    shared, linked = repo
    git(linked, "commit", "--allow-empty", "-qm", "TASK-1275 owner")
    original = git(linked, "rev-parse", "HEAD").stdout.strip()
    git(linked, "switch", "-q", "-c", "byte/TASK-1276-other")
    result = git(linked, "commit", "--allow-empty", "-m", "TASK-1276 wrong owner", check=False)
    assert result.returncode != 0
    assert "own task branch" in result.stderr
    assert git(linked, "rev-parse", "HEAD").stdout.strip() == original
    assert git(shared, "branch", "--show-current").stdout.strip() == "master"


def test_linked_worktree_refuses_canonical_and_detached_commits(repo: tuple[Path, Path]) -> None:
    shared, linked = repo
    git(linked, "switch", "-q", "--detach")
    detached = git(linked, "commit", "--allow-empty", "-m", "TASK-1275 detached", check=False)
    assert detached.returncode != 0
    assert "own task branch" in detached.stderr
    # This command runs only inside the disposable fixture: Git normally blocks
    # the same branch being checked out in two worktrees.
    git(linked, "switch", "-q", "--ignore-other-worktrees", "master")
    canonical = git(linked, "commit", "--allow-empty", "-m", "TASK-1275 canonical", check=False)
    assert canonical.returncode != 0
    assert "own task branch" in canonical.stderr
    assert git(shared, "branch", "--show-current").stdout.strip() == "master"


def test_task_rebase_conflict_allows_commit_on_detached_head(repo: tuple[Path, Path]) -> None:
    _, linked = repo
    git(linked, "switch", "-q", "--detach")
    git_dir = Path(git(linked, "rev-parse", "--absolute-git-dir").stdout.strip())
    rebase = git_dir / "rebase-merge"
    rebase.mkdir()
    (rebase / "head-name").write_text("refs/heads/al/TASK-1275-hooks\n")
    git(linked, "commit", "--allow-empty", "-qm", "TASK-1275 resolved rebase")
    (rebase / "head-name").write_text("refs/heads/master\n")
    denied = git(linked, "commit", "--allow-empty", "-m", "TASK-1275 wrong rebase", check=False)
    assert denied.returncode != 0
    assert "own task branch" in denied.stderr


def test_fast_forward_only_in_shared_checkout_does_not_run_commit_hook(repo: tuple[Path, Path]) -> None:
    shared, linked = repo
    git(linked, "commit", "--allow-empty", "-qm", "TASK-1275 reviewed")
    tip = git(linked, "rev-parse", "HEAD").stdout.strip()
    git(shared, "merge", "--ff-only", tip)
    assert git(shared, "rev-parse", "HEAD").stdout.strip() == tip


def test_unmanaged_worktree_cannot_claim_task_branch(repo: tuple[Path, Path]) -> None:
    shared, _ = repo
    other = shared.parent / "manual-copy"
    git(shared, "worktree", "add", "-q", "-b", "al/TASK-1276-other", str(other))
    result = git(other, "commit", "--allow-empty", "-m", "TASK-1276 other", check=False)
    assert result.returncode != 0
    assert "own task branch" in result.stderr


def test_post_checkout_warns_only_when_shared_checkout_leaves_canonical(repo: tuple[Path, Path]) -> None:
    shared, linked = repo
    checkout_args = [str(POST_CHECKOUT), "old", "new", "1"]
    linked_result = subprocess.run(checkout_args, cwd=linked, text=True, capture_output=True)
    assert linked_result.returncode == 0 and not linked_result.stderr
    git(shared, "switch", "-q", "-c", "temporary")
    shared_result = subprocess.run(checkout_args, cwd=shared, text=True, capture_output=True)
    assert shared_result.returncode == 0
    assert "Shared checkout switched" in shared_result.stderr
