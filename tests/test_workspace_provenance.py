from datetime import datetime, timezone
from types import SimpleNamespace

from llm_bawt.service.changed_files_store import FileContent, TurnChangedFile
from llm_bawt.service.schemas_workspace_provenance import WorkspaceProvenanceRequest
from llm_bawt.service.workspace_provenance import (
    CurrentWorkspaceFile,
    LineChange,
    TurnFileSnapshot,
    attribute_workspace,
    sha256_text,
)


def current(before: str, after: str) -> CurrentWorkspaceFile:
    return CurrentWorkspaceFile(
        repo_id="repo-1",
        repo_key="bawthub",
        repo_aliases=("bawthub", "/repos/bawthub"),
        path="src/example.ts",
        staged=False,
        status="M",
        old_text=before,
        new_text=after,
    )


def snapshot(
    turn_id: str,
    before: str,
    after: str,
    *,
    created_at: datetime,
    path: str = "src/example.ts",
) -> TurnFileSnapshot:
    return TurnFileSnapshot(
        turn_id=turn_id,
        trigger_message_id=f"msg-{turn_id}",
        bot_id="caid",
        user_id="nick",
        created_at=created_at,
        repo_key="workspace",
        repo_label="bawthub",
        path=path,
        old_path=None,
        change_kind="modified",
        before_sha256=sha256_text(before),
        after_sha256=sha256_text(after),
        before_text=before,
        after_text=after,
        binary=False,
        truncated=False,
    )


def test_exact_transition_is_owned() -> None:
    before = "one\ntwo\n"
    after = "one\nchanged\n"
    result = attribute_workspace([
        current(before, after),
    ], [
        snapshot("turn-a", before, after, created_at=datetime(2026, 8, 4, tzinfo=timezone.utc)),
    ])[0]

    assert result.ownership == "owned"
    assert result.unmatched_changes == 0
    assert [candidate.confidence for candidate in result.candidates] == ["exact"]
    assert result.candidates[0].exact_transition is True


def test_disjoint_turn_changes_are_shared() -> None:
    original = "alpha\nbeta\ngamma\ndelta\n"
    after_first = "ALPHA\nbeta\ngamma\ndelta\n"
    final = "ALPHA\nbeta\ngamma\nDELTA\n"
    result = attribute_workspace([
        current(original, final),
    ], [
        snapshot("turn-a", original, after_first, created_at=datetime(2026, 8, 4, 1, tzinfo=timezone.utc)),
        snapshot("turn-b", after_first, final, created_at=datetime(2026, 8, 4, 2, tzinfo=timezone.utc)),
    ])[0]

    assert result.ownership == "shared"
    assert result.unmatched_changes == 0
    assert {candidate.turn_id for candidate in result.candidates} == {"turn-a", "turn-b"}
    assert all(candidate.dirty_hunks for candidate in result.candidates)


def test_path_only_candidate_does_not_claim_ownership() -> None:
    result = attribute_workspace([
        current("before\n", "manual\n"),
    ], [
        snapshot(
            "turn-a",
            "different\n",
            "unrelated\n",
            created_at=datetime(2026, 8, 4, tzinfo=timezone.utc),
        ),
    ])[0]

    assert result.ownership == "unattributed"
    assert result.candidates[0].confidence == "probable"
    assert result.unmatched_changes > 0


def test_supplied_hunk_ids_are_preserved() -> None:
    before = "alpha\nbeta\n"
    after = "ALPHA\nbeta\n"
    live = current(before, after)
    live = CurrentWorkspaceFile(**{
        **live.__dict__,
        "changes": (
            LineChange("-", "alpha\n", 7, 1),
            LineChange("+", "ALPHA\n", 7, 1),
        ),
    })
    result = attribute_workspace([live], [
        snapshot("turn-a", before, after, created_at=datetime(2026, 8, 4, tzinfo=timezone.utc)),
    ])[0]

    assert result.candidates[0].dirty_hunks == [7]


def test_same_path_in_different_repo_does_not_match() -> None:
    candidate = snapshot(
        "turn-a",
        "before\n",
        "after\n",
        created_at=datetime(2026, 8, 4, tzinfo=timezone.utc),
    )
    candidate = TurnFileSnapshot(**{
        **candidate.__dict__,
        "repo_label": "llm-bawt",
        "repo_key": "llm-bawt",
    })

    result = attribute_workspace([current("before\n", "after\n")], [candidate])[0]

    assert result.ownership == "unattributed"
    assert result.candidates == ()


def test_route_reads_only_matching_metadata_before_any_blob_or_turn_lookup(monkeypatch) -> None:
    from llm_bawt.service.routes import workspace_provenance as route

    now = datetime(2026, 10, 24, tzinfo=timezone.utc)
    rows = [TurnChangedFile(id=i + 1, turn_id=f"irrelevant-{i}", bot_id="agent",
                            user_id="nick", created_at=now, repo_key="workspace",
                            repo_label="bawthub", path=f"src/unrelated-{i}.ts")
            for i in range(1400)]
    for index in range(7):
        rows.append(TurnChangedFile(
            id=1401 + index, turn_id=f"match-{index}", bot_id="agent", user_id="nick",
            created_at=now, repo_key="workspace", repo_label="bawthub",
            path=(f"/home/nick/dev/bawthub/src/match-{index}.ts" if index % 2
                  else f"src/match-{index}.ts"),
            old_path=f"src/old-{index}.ts" if index == 0 else None,
            change_kind="renamed" if index == 0 else "modified",
            before_sha256=sha256_text("before\n"), after_sha256=sha256_text("after\n"),
        ))
    rows.append(TurnChangedFile(id=1500, turn_id="other-repo", bot_id="agent",
                                user_id="nick", created_at=now, repo_key="llm-bawt",
                                repo_label="llm-bawt", path="src/match-0.ts"))
    fetched = []

    class Store:
        def __init__(self, engine):
            pass

        def recent_rows(self, **kwargs):
            assert kwargs["limit"] == 2000
            return rows, False

        def content_for_rows(self, selected):
            fetched.extend(selected)
            return {row.id: (FileContent("before\n", False, False),
                             FileContent("after\n", False, False)) for row in selected}

    class SessionStub:
        def __init__(self, engine):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def exec(self, statement):
            return SimpleNamespace(all=lambda: [])

    monkeypatch.setattr(route, "ChangedFilesStore", Store)
    monkeypatch.setattr(route, "get_turn_log_store", lambda: SimpleNamespace(engine=object()))
    monkeypatch.setattr(route, "Session", SessionStub)
    payload = WorkspaceProvenanceRequest(
        user_id="nick", status_fingerprint="stable", files=[{
            "repo_id": "bawthub", "repo_key": "bawthub",
            "repo_aliases": ["bawthub", "/home/nick/dev/bawthub"],
            "path": f"src/match-{index}.ts", "staged": index == 0,
            "status": "R" if index == 0 else "M", "old_text": "before\n", "new_text": "after\n",
        } for index in range(7)],
    )

    result = route.workspace_provenance(payload)

    assert len(fetched) == 7  # 1,401 unrelated snapshots never trigger blob reads
    assert {row.turn_id for row in fetched} == {f"match-{i}" for i in range(7)}
    assert len(result.files) == 7
    assert all(file.ownership == "owned" and file.candidates[0].confidence == "exact"
               for file in result.files)
    assert result.files[0].staged is True
    assert result.truncated is False
