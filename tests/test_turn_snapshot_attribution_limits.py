"""TASK-2276: late discovery and omitted content must not falsify attribution."""

from pathlib import Path

import pytest

from agent_bridge import turn_snapshots
from agent_bridge.turn_snapshots import TurnSnapshotObserver
from tests.test_turn_snapshots import bash, by_path, git
from tests.test_turn_snapshots import repo as repo


def test_new_worktree_second_tool_keeps_creation_baseline(repo: Path) -> None:
    wt = repo.parent / "new-worktree"
    git(repo, "commit", "--amend", "--no-edit", "--date=2020-01-01T00:00:00Z",
        env={"GIT_COMMITTER_DATE": "2020-01-01T00:00:00Z"})
    obs = TurnSnapshotObserver()
    bash(obs, "new-tree", f"git worktree add {wt} && touch {wt}/a.txt", repo)
    git(repo, "worktree", "add", "-q", "-b", "new-tree", str(wt))
    (wt / "a.txt").write_text("first tool edit\n")
    bash(obs, "new-tree", "touch b.txt", wt)
    (wt / "b.txt").write_text("second tool edit\n")
    files, _ = obs.finish("new-tree")
    assert set(by_path(files)) == {"a.txt", "b.txt"}


def test_untracked_growth_past_cap_is_not_deletion(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(turn_snapshots, "MAX_UNTRACKED_BYTES", 32)
    path = repo / "untracked.txt"
    path.write_text("small\n")
    obs = TurnSnapshotObserver()
    bash(obs, "grow", "generate untracked.txt", repo)
    path.write_text("x" * 33)
    files, receipts = obs.finish("grow")
    assert path.exists()
    assert not any(row["change_kind"] == "deleted" for row in files)
    # Excluded content must be explicitly identified, not silently unchanged.
    assert any(row.get("excluded_paths") == ["untracked.txt"] for row in receipts)
    assert receipts[0]["status"] == "excluded_only"
    assert receipts[0]["excluded_count"] == 1


@pytest.mark.parametrize("after", ["small", "large", "deleted"])
def test_oversized_start_never_manufactures_diff(repo: Path, monkeypatch, after) -> None:
    monkeypatch.setattr(turn_snapshots, "MAX_UNTRACKED_BYTES", 32)
    path = repo / "excluded.txt"
    path.write_text("x" * 33)
    obs = TurnSnapshotObserver()
    bash(obs, "limit", "generate excluded.txt", repo)
    if after == "deleted":
        path.unlink()
    else:
        path.write_text("small" if after == "small" else "y" * 34)
    files, receipts = obs.finish("limit")
    assert files == []
    assert receipts[0]["status"] == "excluded_only"
    assert receipts[0]["excluded_paths"] == ["excluded.txt"]
    assert git(repo, "for-each-ref", "--format=%(refname)", "refs/bawt/turns/limit") == ""


def test_excluded_content_keeps_other_rows_and_refs(repo: Path, monkeypatch) -> None:
    monkeypatch.setattr(turn_snapshots, "MAX_UNTRACKED_BYTES", 32)
    (repo / "excluded.txt").write_text("x" * 33)
    obs = TurnSnapshotObserver()
    bash(obs, "partial", "generate files", repo)
    (repo / "a.txt").write_text("actual edit\n")
    files, receipts = obs.finish("partial")
    assert set(by_path(files)) == {"a.txt"}
    assert receipts[0]["status"] == "observed_partial"
    assert git(repo, "for-each-ref", "--format=%(refname)", "refs/bawt/turns/partial")


def test_tracked_content_ignores_untracked_cap(repo: Path, monkeypatch) -> None:
    monkeypatch.setattr(turn_snapshots, "MAX_UNTRACKED_BYTES", 32)
    obs = TurnSnapshotObserver()
    bash(obs, "tracked", "generate a.txt", repo)
    (repo / "a.txt").write_text("x" * 33)
    files, receipts = obs.finish("tracked")
    assert files[0]["change_kind"] == "modified"
    assert receipts[0]["status"] == "observed"
    assert "excluded_paths" not in receipts[0]


def test_small_untracked_real_deletion_still_reported(repo: Path) -> None:
    path = repo / "small.txt"
    path.write_text("small")
    obs = TurnSnapshotObserver()
    bash(obs, "delete", "rm small.txt", repo)
    path.unlink()
    files, receipts = obs.finish("delete")
    assert files[0]["change_kind"] == "deleted"
    assert receipts[0]["status"] == "observed"


@pytest.mark.parametrize("failure", ["state", "inode"])
def test_pending_file_does_not_rebase_existing_dirty_tree(repo: Path, monkeypatch, failure) -> None:
    (repo / "a.txt").write_text("pre-existing dirty content\n")
    obs = TurnSnapshotObserver()
    with monkeypatch.context() as patch:
        if failure == "state":
            patch.setattr(obs, "_new_state", lambda root: None)
        else:
            patch.setattr(turn_snapshots, "_inode_key", lambda root: None)
        bash(obs, "existing", f"touch {repo}/missing.txt", repo.parent)
    bash(obs, "existing", "touch b.txt", repo)
    (repo / "b.txt").write_text("our edit\n")
    files, _ = obs.finish("existing")
    assert set(by_path(files)) == {"b.txt"}


def test_new_worktree_overlap_still_refuses_attribution(repo: Path) -> None:
    wt = repo.parent / "shared-new"
    a, b = TurnSnapshotObserver(), TurnSnapshotObserver()
    bash(a, "creator", f"git worktree add {wt}", repo)
    git(repo, "worktree", "add", "-q", "-b", "shared-new", str(wt))
    (wt / "a.txt").write_text("first edit\n")
    bash(b, "other", "touch b.txt", wt)
    bash(a, "creator", "touch a.txt", wt)
    (wt / "b.txt").write_text("other edit\n")
    files_a, receipts_a = a.finish("creator")
    files_b, receipts_b = b.finish("other")
    assert files_a == files_b == []
    assert any(r.get("status") == "ineligible_shared_tree_busy" for r in receipts_a)
    assert receipts_b[0]["status"] == "ineligible_shared_tree_busy"


def test_declined_creation_baseline_falls_back_to_snapshot(repo: Path, monkeypatch) -> None:
    wt = repo.parent / "no-reflog"
    obs = TurnSnapshotObserver()
    bash(obs, "fallback", f"git worktree add {wt}", repo)
    git(repo, "worktree", "add", "-q", "-b", "no-reflog", str(wt))
    (wt / "a.txt").write_text("cannot prove baseline\n")
    monkeypatch.setattr(obs, "_creation_commit", lambda root, since: None)
    bash(obs, "fallback", "touch b.txt", wt)
    (wt / "b.txt").write_text("known edit\n")
    files, _ = obs.finish("fallback")
    assert set(by_path(files)) == {"b.txt"}


def test_finish_without_creation_evidence_reports_missing_baseline(repo: Path, monkeypatch) -> None:
    wt = repo.parent / "no-finish-baseline"
    obs = TurnSnapshotObserver()
    bash(obs, "finish-fallback", f"git worktree add {wt}", repo)
    git(repo, "worktree", "add", "-q", "-b", "no-finish-baseline", str(wt))
    (wt / "a.txt").write_text("unknown baseline\n")
    monkeypatch.setattr(obs, "_creation_commit", lambda root, since: None)
    files, receipts = obs.finish("finish-fallback")
    assert files == []
    assert next(r for r in receipts if r["tree"] == str(wt))["status"] == "no_start_snapshot"


def test_old_reflog_is_not_creation_evidence(repo: Path) -> None:
    obs = TurnSnapshotObserver()
    assert obs._creation_commit(repo, turn_snapshots.time.time() + 60) is None


def test_excluded_receipt_path_list_is_capped(repo: Path, monkeypatch) -> None:
    monkeypatch.setattr(turn_snapshots, "MAX_UNTRACKED_BYTES", 1)
    monkeypatch.setattr(turn_snapshots, "MAX_FILES_PER_TREE", 2)
    for index in range(3):
        (repo / f"excluded-{index}").write_text("large")
    obs = TurnSnapshotObserver()
    bash(obs, "capped", "generate files", repo)
    files, receipts = obs.finish("capped")
    assert files == []
    assert receipts[0]["excluded_count"] == 3
    assert len(receipts[0]["excluded_paths"]) == 2


def test_excluded_paths_count_toward_snapshot_cap(repo: Path, monkeypatch) -> None:
    monkeypatch.setattr(turn_snapshots, "MAX_UNTRACKED_BYTES", 1)
    monkeypatch.setattr(turn_snapshots, "MAX_STATUS_PATHS", 1)
    obs = TurnSnapshotObserver()
    bash(obs, "paths", "generate files", repo)
    (repo / "one").write_text("large")
    (repo / "two").write_text("large")
    files, receipts = obs.finish("paths")
    assert files == []
    assert receipts[0]["status"] == "skipped_too_many_paths"
