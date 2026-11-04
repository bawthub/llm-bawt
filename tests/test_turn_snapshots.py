"""TASK-1218: per-turn Git snapshots attribute command-tool edits."""

from __future__ import annotations

import base64
import os
import subprocess
import time
from pathlib import Path

import pytest

from agent_bridge import turn_snapshots
from agent_bridge.turn_snapshots import (
    PROVENANCE,
    REF_PREFIX,
    MAX_BYTES_PER_SIDE,
    TurnSnapshotObserver,
    file_identity,
    referenced_paths,
)

ENV = {
    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x",
    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@x",
}


def git(cwd: Path, *args: str, env: dict | None = None) -> str:
    return subprocess.run(
        ("git", "-C", str(cwd), *args), check=True, capture_output=True, text=True,
        env={**os.environ, **ENV, **(env or {})},
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "dev" / "proj"
    root.mkdir(parents=True)
    git(root, "init", "-q", "-b", "main")
    (root / "a.txt").write_text("one\ntwo\n")
    (root / "b.txt").write_text("bee\n")
    (root / ".gitignore").write_text("ignored/\n")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "init")
    return root


def bash(observer: TurnSnapshotObserver, request: str, command: str, cwd: Path, bot: str = "nova") -> None:
    observer.observe_tool(request_id=request, bot_id=bot, tool_input={"command": command}, cwd=str(cwd))


def by_path(files: list[dict]) -> dict[str, dict]:
    return {Path(f["path"]).name: f for f in files}


def test_sed_after_relative_cd_is_observed(repo: Path) -> None:
    obs = TurnSnapshotObserver()
    dev = repo.parent
    bash(obs, "r1", "cd proj && sed -i s/one/ONE/ a.txt", dev)
    (repo / "a.txt").write_text("ONE\ntwo\n")
    files, receipts = obs.finish("r1")
    assert [f["change_kind"] for f in files] == ["modified"]
    f = files[0]
    assert f["provenance"] == PROVENANCE
    assert f["path"] == str(repo / "a.txt") and f["absolute_path"] == f["path"]
    assert base64.b64decode(f["before_b64"]) == b"one\ntwo\n"
    assert base64.b64decode(f["after_b64"]) == b"ONE\ntwo\n"
    assert (f["additions"], f["deletions"]) == (1, 1)
    assert receipts[0]["status"] == "observed"


def test_alias_mount_path_is_one_tree(repo: Path, tmp_path: Path) -> None:
    alias = tmp_path / "alias-dev"
    alias.symlink_to(repo.parent, target_is_directory=True)
    obs = TurnSnapshotObserver()
    bash(obs, "r1", f"ssh host 'sed -i s/bee/BEE/ {alias}/proj/b.txt'", tmp_path)
    bash(obs, "r1", f"cat {repo}/b.txt", tmp_path)
    (repo / "b.txt").write_text("BEE\n")
    files, receipts = obs.finish("r1")
    assert len([r for r in receipts if "status" in r]) == 1
    assert [Path(f["path"]).name for f in files] == ["b.txt"]


def test_heredoc_untracked_file_and_formatter_multi_file(repo: Path) -> None:
    obs = TurnSnapshotObserver()
    bash(obs, "r1", "cat > new.py <<'EOF'\nx=1\nEOF\nruff format .", repo)
    (repo / "new.py").write_text("x = 1\n")
    (repo / "a.txt").write_text("one\ntwo\nthree\n")
    (repo / "b.txt").unlink()
    (repo / "ignored").mkdir()
    (repo / "ignored" / "cache.bin").write_bytes(b"x")
    files, _ = obs.finish("r1")
    kinds = {name: f["change_kind"] for name, f in by_path(files).items()}
    assert kinds == {"new.py": "added", "a.txt": "modified", "b.txt": "deleted"}


def test_commit_during_turn_is_kept(repo: Path) -> None:
    obs = TurnSnapshotObserver()
    bash(obs, "r1", "sed -i s/two/TWO/ a.txt && git commit -am x", repo)
    (repo / "a.txt").write_text("one\nTWO\n")
    git(repo, "commit", "-qam", "turn work", env={"GIT_AUTHOR_NAME": "nova", "GIT_AUTHOR_EMAIL": "nova@bots.x"})
    files, receipts = obs.finish("r1")
    assert [Path(f["path"]).name for f in files] == ["a.txt"]
    assert files[0]["git_head_sha"] == git(repo, "rev-parse", "HEAD")


def test_pulled_in_older_commits_are_not_attributed(repo: Path, tmp_path: Path) -> None:
    other = tmp_path / "other"
    git(tmp_path, "clone", "-q", str(repo), str(other))
    (other / "b.txt").write_text("theirs\n")
    old = "2020-01-01T00:00:00Z"
    git(other, "commit", "-qam", "theirs", env={"GIT_COMMITTER_DATE": old, "GIT_AUTHOR_DATE": old})
    obs = TurnSnapshotObserver()
    bash(obs, "r1", "bawt-wt sync proj TASK-1", repo)
    git(repo, "pull", "-q", "--ff-only", str(other), "main")
    files, receipts = obs.finish("r1")
    assert files == []
    assert receipts[0]["status"] == "ineligible_head_moved"


def test_rebase_onto_other_work_is_ineligible(repo: Path) -> None:
    old = {"GIT_COMMITTER_DATE": "2020-01-01T00:00:00Z", "GIT_AUTHOR_DATE": "2020-01-01T00:00:00Z"}
    (repo / "a.txt").write_text("someone else's main work\n")
    git(repo, "commit", "-qam", "main", env=old)
    git(repo, "checkout", "-q", "-b", "side", "HEAD~1")
    (repo / "b.txt").write_text("side\n")
    git(repo, "commit", "-qam", "side", env=old)
    obs = TurnSnapshotObserver()
    bash(obs, "r1", "bawt-wt sync proj TASK-1", repo)
    git(repo, "rebase", "-q", "main")
    _, receipts = obs.finish("r1")
    assert receipts[0]["status"] == "ineligible_head_moved"


def test_worktree_created_mid_turn_has_no_false_edits(repo: Path) -> None:
    wt = repo.parent / ".wt-proj-nova-TASK-1"
    obs = TurnSnapshotObserver()
    bash(obs, "r1", f"git worktree add -q {wt} && cd {wt} && sed -i s/one/1/ a.txt", repo)
    git(repo, "worktree", "add", "-q", "-b", "task", str(wt))
    (wt / "a.txt").write_text("1\ntwo\n")
    files, receipts = obs.finish("r1")
    trees = {r["tree"]: r for r in receipts if "status" in r}
    assert trees[str(repo)]["status"] == "unchanged"
    assert [f["path"] for f in files] == [str(wt / "a.txt")]


def test_refs_survive_worktree_removal(repo: Path) -> None:
    wt = repo.parent / ".wt-proj-nova-TASK-2"
    git(repo, "worktree", "add", "-q", "-b", "t2", str(wt))
    obs = TurnSnapshotObserver()
    bash(obs, "r9", "sed -i s/bee/b/ b.txt", wt)
    (wt / "b.txt").write_text("b\n")
    obs.finish("r9")
    git(repo, "worktree", "remove", "--force", str(wt))
    refs = git(repo, "for-each-ref", "--format=%(refname)", REF_PREFIX).splitlines()
    start = next(r for r in refs if r.endswith("/start"))
    end = start[: -len("start")] + "end"
    assert "b.txt" in git(repo, "diff", "--name-only", start, end)


def test_overlapping_turns_on_one_tree_are_both_ineligible(repo: Path) -> None:
    a, b = TurnSnapshotObserver(), TurnSnapshotObserver()
    bash(a, "ra", "sed -i s/one/A/ a.txt", repo, bot="al")
    bash(b, "rb", "sed -i s/bee/B/ b.txt", repo, bot="byte")
    (repo / "a.txt").write_text("A\ntwo\n")
    (repo / "b.txt").write_text("B\n")
    files_b, receipts_b = b.finish("rb")
    files_a, receipts_a = a.finish("ra")
    assert files_a == files_b == []
    assert receipts_a[0]["status"] == receipts_b[0]["status"] == "ineligible_shared_tree_busy"
    # The registry is clean afterwards: a later solo turn is eligible again.
    c = TurnSnapshotObserver()
    bash(c, "rc", "sed -i s/two/2/ a.txt", repo)
    (repo / "a.txt").write_text("A\n2\n")
    files_c, _ = c.finish("rc")
    assert [Path(f["path"]).name for f in files_c] == ["a.txt"]


def test_large_file_changed_past_prefix_is_detected(repo: Path) -> None:
    big = repo / "big.txt"
    big.write_bytes(b"x" * (MAX_BYTES_PER_SIDE + 10))
    git(repo, "add", "big.txt")
    git(repo, "commit", "-qm", "big")
    obs = TurnSnapshotObserver()
    bash(obs, "r1", "printf y >> big.txt", repo)
    with big.open("ab") as handle:
        handle.write(b"y")
    files, _ = obs.finish("r1")
    (f,) = files
    assert f["truncated"] is True and f["additions"] is None
    assert f["git_before_blob"] != f["git_after_blob"]


def test_real_index_and_head_are_untouched(repo: Path) -> None:
    (repo / "a.txt").write_text("staged\n")
    git(repo, "add", "a.txt")
    (repo / "b.txt").write_text("unstaged\n")
    (repo / "u.txt").write_text("untracked\n")
    index = (repo / ".git" / "index").read_bytes()
    head = (repo / ".git" / "HEAD").read_bytes()
    branch = git(repo, "rev-parse", "main")
    obs = TurnSnapshotObserver()
    bash(obs, "r1", "true", repo)
    (repo / "u.txt").write_text("changed\n")
    files, _ = obs.finish("r1")
    assert [Path(f["path"]).name for f in files] == ["u.txt"]
    assert (repo / ".git" / "index").read_bytes() == index
    assert (repo / ".git" / "HEAD").read_bytes() == head
    assert git(repo, "rev-parse", "main") == branch
    assert git(repo, "status", "--porcelain") == "M  a.txt\n M b.txt\n?? u.txt"


def test_edit_tool_covered_paths_are_skipped(repo: Path) -> None:
    obs = TurnSnapshotObserver()
    obs.observe_tool(request_id="r1", bot_id="nova", tool_input={"file_path": str(repo / "a.txt")}, cwd=None)
    (repo / "a.txt").write_text("via edit\n")
    (repo / "b.txt").write_text("via sed\n")
    files, _ = obs.finish("r1", covered=[file_identity(repo / "a.txt")])
    assert [Path(f["path"]).name for f in files] == ["b.txt"]


def test_paths_outside_repos_and_discard() -> None:
    obs = TurnSnapshotObserver()
    bash(obs, "r1", "echo hi > /tmp/x", Path("/"))
    obs.discard("r1")
    assert obs.finish("r1") == ([], [])


def test_referenced_paths_parses_command_text() -> None:
    found = referenced_paths({"command": "cd sub && sed -i x /abs/f.py; ssh h 'cat /home/u/dev/r/a'"}, "/base")
    assert {str(p) for p in found} >= {"/base", "/base/sub", "/abs/f.py", "/home/u/dev/r/a"}


def test_stale_registry_marker_is_ignored(repo: Path) -> None:
    registry = repo / ".git" / "bawt-turns"
    registry.mkdir()
    stale = registry / "dead.json"
    stale.write_text("{}")
    old = time.time() - 7 * 3600
    os.utime(stale, (old, old))
    obs = TurnSnapshotObserver()
    bash(obs, "r1", "sed", repo)
    (repo / "a.txt").write_text("z\n")
    files, _ = obs.finish("r1")
    assert len(files) == 1 and not stale.exists()


def test_fresh_commits_by_another_author_are_not_attributed(repo: Path, tmp_path: Path) -> None:
    """Byte SF1: an integrator's rebased publish has fresh dates but isn't ours."""
    other = tmp_path / "other"
    git(tmp_path, "clone", "-q", str(repo), str(other))
    obs = TurnSnapshotObserver()
    bash(obs, "r1", "cat a.txt", repo)
    (other / "b.txt").write_text("integrated\n")
    git(other, "commit", "-qam", "publish", env={"GIT_AUTHOR_NAME": "gavel", "GIT_AUTHOR_EMAIL": "gavel@bots.x"})
    git(repo, "pull", "-q", "--ff-only", str(other), "main")
    files, receipts = obs.finish("r1")
    assert files == [] and receipts[0]["status"] == "ineligible_head_moved"


def test_refs_kept_only_for_observed_trees(repo: Path, tmp_path: Path) -> None:
    """Byte SF2: read-only references leave no refs behind."""
    quiet = tmp_path / "quiet"
    git(tmp_path, "clone", "-q", str(repo), str(quiet))
    obs = TurnSnapshotObserver()
    bash(obs, "r1", f"cat {quiet}/a.txt && sed -i s/one/1/ a.txt", repo)
    (repo / "a.txt").write_text("1\ntwo\n")
    obs.finish("r1")
    assert git(quiet, "for-each-ref", REF_PREFIX) == ""
    assert len(git(repo, "for-each-ref", REF_PREFIX).splitlines()) == 2
    obs2 = TurnSnapshotObserver()
    bash(obs2, "r2", "true", repo)
    obs2.discard("r2")
    assert len(git(repo, "for-each-ref", REF_PREFIX).splitlines()) == 2


def test_content_budget_makes_later_rows_metadata_only(repo: Path, monkeypatch) -> None:
    """Byte SF3: a repo-wide formatter can't ship unbounded diff bytes."""
    monkeypatch.setattr(turn_snapshots, "MAX_CONTENT_BYTES_PER_TURN", 20)
    obs = TurnSnapshotObserver()
    bash(obs, "r1", "ruff format .", repo)
    (repo / "a.txt").write_text("one\ntwo!\n")
    (repo / "b.txt").write_text("bee!\n")
    files, _ = obs.finish("r1")
    rows = by_path(files)
    assert rows["a.txt"]["after_b64"] is not None
    assert rows["b.txt"]["after_b64"] is None and rows["b.txt"]["truncated"] is True


def test_too_many_changed_paths_skips_tree(repo: Path, monkeypatch) -> None:
    monkeypatch.setattr(turn_snapshots, "MAX_STATUS_PATHS", 2)
    obs = TurnSnapshotObserver()
    bash(obs, "r1", "codegen", repo)
    for n in range(3):
        (repo / f"gen{n}.txt").write_text("x\n")
    files, receipts = obs.finish("r1")
    assert files == [] and receipts[0]["status"] == "skipped_too_many_paths"


def test_marker_from_previous_bridge_process_is_dead(repo: Path) -> None:
    import json
    import socket

    registry = repo / ".git" / "bawt-turns"
    registry.mkdir()
    (registry / "crashed.json").write_text(json.dumps({
        "turn_id": "crashed", "host": socket.gethostname(), "process": "old", "registered_at": 0,
    }))
    obs = TurnSnapshotObserver()
    bash(obs, "r1", "sed", repo)
    (repo / "a.txt").write_text("z\n")
    files, _ = obs.finish("r1")
    assert len(files) == 1 and not (registry / "crashed.json").exists()
