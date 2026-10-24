"""Tests for per-turn changed-file persistence (TASK-661).

Exercises the store's save/read roundtrip against both the object-store blob
path (a temp FS backend) and the bounded Postgres fallback, plus the canonical
serializer shape used by both the live event and the read API.
"""

from __future__ import annotations

import pytest
from sqlalchemy import inspect, text
from sqlmodel import SQLModel, create_engine

from llm_bawt.service import changed_files_store as cfs
from llm_bawt.service.changed_files_store import (
    MAX_BYTES_PER_SIDE,
    MAX_FILES_PER_TURN,
    ChangedFileInput,
    ChangedFilesStore,
    TurnChangedFile,
    TurnDiffPayloadRecord,
    build_turn_summary,
    reset_diff_blob_backend,
)


@pytest.fixture
def store(tmp_path, monkeypatch):
    # Point the diff blob backend at a temp FS dir and clear the cache so the
    # store writes/reads content-addressed blobs on disk (the prod path).
    monkeypatch.setenv("LLM_BAWT_DIFFBLOBS_FS_ROOT", str(tmp_path / "diffs"))
    monkeypatch.setenv("LLM_BAWT_STORAGE_BACKEND", "fs")
    reset_diff_blob_backend()
    from sqlalchemy.pool import StaticPool
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(
        engine,
        tables=[
            TurnChangedFile.__table__,
            TurnDiffPayloadRecord.__table__,
        ],
    )
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE sessions (
                id VARCHAR PRIMARY KEY,
                bot_id VARCHAR NOT NULL,
                user_id VARCHAR,
                status VARCHAR NOT NULL,
                started_at DATETIME NOT NULL
            )
        """))
        conn.execute(text("""
            CREATE TABLE messages (
                bot_id VARCHAR NOT NULL,
                id VARCHAR NOT NULL,
                session_id VARCHAR,
                PRIMARY KEY (bot_id, id)
            )
        """))
    from llm_bawt.service import changed_file_commits
    monkeypatch.setattr(changed_file_commits, "reconcile_files", lambda files: None)
    s = ChangedFilesStore(engine)
    yield s
    reset_diff_blob_backend()


def _mod(path, before, after, **kw):
    return ChangedFileInput(
        repo_key="repoA", repo_label="llm-bawt", path=path,
        change_kind="modified",
        before=before.encode() if before is not None else None,
        after=after.encode() if after is not None else None,
        **kw,
    )


def test_ensure_schema_removes_legacy_overlap_columns():
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE turn_changed_files (
                id INTEGER PRIMARY KEY,
                created_at DATETIME NOT NULL,
                turn_id VARCHAR(128) NOT NULL,
                bot_id VARCHAR,
                user_id VARCHAR,
                trigger_message_id VARCHAR,
                repo_key VARCHAR(128) NOT NULL,
                repo_label VARCHAR(255),
                path TEXT NOT NULL,
                old_path TEXT,
                change_kind VARCHAR(16) NOT NULL,
                additions INTEGER,
                deletions INTEGER,
                binary BOOLEAN NOT NULL,
                truncated BOOLEAN NOT NULL,
                overlapping BOOLEAN NOT NULL,
                manifest_truncated BOOLEAN NOT NULL,
                before_sha256 VARCHAR(64),
                before_bytes BIGINT,
                before_blob_key VARCHAR(128),
                after_sha256 VARCHAR(64),
                after_bytes BIGINT,
                after_blob_key VARCHAR(128),
                content_type VARCHAR(128),
                source_tool_call_ids TEXT
            )
        """))

    legacy = ChangedFilesStore(engine)
    legacy.ensure_schema()

    columns = {column["name"] for column in inspect(engine).get_columns("turn_changed_files")}
    assert "overlapping" not in columns
    assert "manifest_truncated" not in columns
    assert legacy.save_turn_files(
        turn_id="turn-legacy", bot_id="b", user_id="u", trigger_message_id="m",
        files=[_mod("a.py", "old", "new")],
    ) == 1


def test_save_and_read_roundtrip(store):
    n = store.save_turn_files(
        turn_id="turn-1", bot_id="snark", user_id="nick",
        trigger_message_id="msg-1",
        files=[_mod("a.py", "old\n", "new\n", additions=1, deletions=1)],
    )
    assert n == 1
    resolved = store.get_file_content(turn_id="turn-1", repo_key="repoA", path="a.py")
    assert resolved is not None
    before, after = resolved
    assert before.text == "old\n"
    assert after.text == "new\n"
    assert not before.binary and not after.binary


def test_summary_shape_and_totals(store):
    store.save_turn_files(
        turn_id="turn-2", bot_id="snark", user_id="nick", trigger_message_id="msg-2",
        files=[
            _mod("a.py", "x", "y", additions=3, deletions=2),
            ChangedFileInput(repo_key="repoA", repo_label="llm-bawt", path="new.py",
                             change_kind="added", before=None, after=b"hi", additions=1, deletions=0),
        ],
    )
    summary = store.summary_for_turn("turn-2")
    assert summary["total_files"] == 2
    assert summary["total_additions"] == 4
    assert summary["total_deletions"] == 2
    files = {f["path"]: f for f in summary["files"]}
    assert files["a.py"]["turn_id"] == "turn-2"
    assert files["a.py"]["has_content"] is True
    assert files["new.py"]["change_kind"] == "added"
    # Added file: before side absent, after present -> still has content.
    assert files["new.py"]["has_content"] is True


def test_added_and_deleted_have_empty_absent_side(store):
    store.save_turn_files(
        turn_id="turn-3", bot_id="b", user_id="u", trigger_message_id="m",
        files=[
            ChangedFileInput(repo_key="r", path="added.py", change_kind="added",
                             before=None, after=b"content"),
            ChangedFileInput(repo_key="r", path="gone.py", change_kind="deleted",
                             before=b"was here", after=None),
        ],
    )
    add = store.get_file_content(turn_id="turn-3", repo_key="r", path="added.py")
    assert add[0].text == "" and add[1].text == "content"
    delete = store.get_file_content(turn_id="turn-3", repo_key="r", path="gone.py")
    assert delete[0].text == "was here" and delete[1].text == ""


def test_binary_stores_no_content(store):
    store.save_turn_files(
        turn_id="turn-4", bot_id="b", user_id="u", trigger_message_id="m",
        files=[ChangedFileInput(repo_key="r", path="img.png", change_kind="modified",
                                binary=True, before=b"\x00\x01", after=b"\x00\x02")],
    )
    summary = store.summary_for_turn("turn-4")
    f = summary["files"][0]
    assert f["binary"] is True
    assert f["has_content"] is False
    before, after = store.get_file_content(turn_id="turn-4", repo_key="r", path="img.png")
    assert before.binary and after.binary
    assert before.text == "" and after.text == ""


def test_upsert_is_idempotent(store):
    for after in ("v1", "v2"):
        store.save_turn_files(
            turn_id="turn-5", bot_id="b", user_id="u", trigger_message_id="m",
            files=[_mod("a.py", "base", after)],
        )
    summary = store.summary_for_turn("turn-5")
    assert summary["total_files"] == 1  # not duplicated
    _, after = store.get_file_content(turn_id="turn-5", repo_key="repoA", path="a.py")
    assert after.text == "v2"  # last write wins


def test_truncation_flag_and_cap(store):
    big = "a" * (MAX_BYTES_PER_SIDE + 100)
    store.save_turn_files(
        turn_id="turn-6", bot_id="b", user_id="u", trigger_message_id="m",
        files=[_mod("big.txt", "", big)],
    )
    f = store.summary_for_turn("turn-6")["files"][0]
    assert f["truncated"] is True
    _, after = store.get_file_content(turn_id="turn-6", repo_key="repoA", path="big.txt")
    assert len(after.text) == MAX_BYTES_PER_SIDE


def test_file_count_cap(store):
    files = [_mod(f"f{i}.py", "a", "b") for i in range(MAX_FILES_PER_TURN + 20)]
    n = store.save_turn_files(
        turn_id="turn-7", bot_id="b", user_id="u", trigger_message_id="m", files=files,
    )
    assert n == MAX_FILES_PER_TURN


def test_repeated_file_upsert_keeps_first_before_latest_after(store):
    store.save_turn_files(
        turn_id="turn-repeat", bot_id="b", user_id="u", trigger_message_id="m",
        files=[_mod("a.py", "v1", "v2", source_tool_call_ids=["tool-1"])],
    )
    store.save_turn_files(
        turn_id="turn-repeat", bot_id="b", user_id="u", trigger_message_id="m",
        files=[_mod("a.py", "v1", "v3", source_tool_call_ids=["tool-2"])],
    )
    before, after = store.get_file_content(
        turn_id="turn-repeat", repo_key="repoA", path="a.py"
    )
    assert before.text == "v1"
    assert after.text == "v3"


def test_summaries_for_triggers_keys_by_message(store):
    store.save_turn_files(
        turn_id="turn-8", bot_id="b", user_id="u", trigger_message_id="msg-8",
        files=[_mod("a.py", "x", "y")],
    )
    out = store.summaries_for_triggers(["msg-8", "nope"])
    assert set(out.keys()) == {"msg-8"}
    assert out["msg-8"]["turn_id"] == "turn-8"


def _conversation(store, *, session_id, bot_id="b", user_id="u", status="active", started_at="2026-08-04 10:00:00"):
    with store.engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO sessions (id, bot_id, user_id, status, started_at) "
            "VALUES (:id, :bot_id, :user_id, :status, :started_at)"
        ), {
            "id": session_id,
            "bot_id": bot_id,
            "user_id": user_id,
            "status": status,
            "started_at": started_at,
        })


def _trigger(store, *, message_id, session_id, bot_id="b"):
    with store.engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO messages (bot_id, id, session_id) "
            "VALUES (:bot_id, :id, :session_id)"
        ), {"bot_id": bot_id, "id": message_id, "session_id": session_id})


def test_uncommitted_summary_is_session_scoped_and_deduplicates_paths(store):
    _conversation(store, session_id="session-a")
    _conversation(store, session_id="session-b", status="completed", started_at="2026-08-03 10:00:00")
    _trigger(store, message_id="msg-a1", session_id="session-a")
    _trigger(store, message_id="msg-a2", session_id="session-a")
    _trigger(store, message_id="msg-b1", session_id="session-b")
    store.save_turn_files(
        turn_id="turn-a1", bot_id="b", user_id="u", trigger_message_id="msg-a1",
        files=[_mod("same.py", "v1", "v2", additions=2, deletions=1)],
    )
    store.save_turn_files(
        turn_id="turn-a2", bot_id="b", user_id="u", trigger_message_id="msg-a2",
        files=[_mod("same.py", "v2", "v3", additions=3, deletions=2), _mod("new.py", "", "x", additions=1, deletions=0)],
    )
    store.save_turn_files(
        turn_id="turn-b1", bot_id="b", user_id="u", trigger_message_id="msg-b1",
        files=[_mod("other.py", "x", "y", additions=9, deletions=9)],
    )

    resolved, summary = store.uncommitted_summary_for_session(
        session_id="session-a", bot_id="b", user_id="u"
    )

    assert resolved == "session-a"
    assert summary["turn_ids"] == ["turn-a1", "turn-a2"]
    assert summary["total_files"] == 2
    same = next(file for file in summary["files"] if file["path"] == "same.py")
    assert same["turn_id"] == "turn-a2"
    assert same["additions"] == 5
    assert same["deletions"] == 3


def test_uncommitted_summaries_partition_current_from_other_turns(store):
    _conversation(store, session_id="session-a")
    _trigger(store, message_id="msg-a1", session_id="session-a")
    _trigger(store, message_id="msg-a2", session_id="session-a")
    store.save_turn_files(
        turn_id="turn-a1", bot_id="b", user_id="u", trigger_message_id="msg-a1",
        files=[
            _mod("shared.py", "v1", "v2", additions=2, deletions=1),
            _mod("other-only.py", "", "x", additions=1, deletions=0),
        ],
    )
    store.save_turn_files(
        turn_id="turn-a2", bot_id="b", user_id="u", trigger_message_id="msg-a2",
        files=[
            _mod("shared.py", "v2", "v3", additions=3, deletions=2),
            _mod("current-only.py", "", "x", additions=4, deletions=0),
        ],
    )

    resolved, summary, other_summary = store.uncommitted_summaries_for_session(
        session_id="session-a", anchor_turn_id="turn-a2", bot_id="b", user_id="u"
    )

    assert resolved == "session-a"
    assert summary["turn_ids"] == ["turn-a1", "turn-a2"]
    assert {file["path"] for file in summary["files"]} == {
        "shared.py", "other-only.py", "current-only.py",
    }
    assert other_summary["turn_ids"] == ["turn-a1"]
    assert {file["path"] for file in other_summary["files"]} == {
        "shared.py", "other-only.py",
    }
    shared = next(file for file in other_summary["files"] if file["path"] == "shared.py")
    assert shared["turn_id"] == "turn-a1"
    assert shared["additions"] == 2
    assert shared["deletions"] == 1


def test_uncommitted_summary_anchor_survives_active_session_rotation(store):
    _conversation(store, session_id="session-old", status="completed", started_at="2026-08-04 10:00:00")
    _conversation(store, session_id="session-new", status="active", started_at="2026-08-05 10:00:00")
    _trigger(store, message_id="msg-old", session_id="session-old")
    store.save_turn_files(
        turn_id="turn-old", bot_id="b", user_id="u", trigger_message_id="msg-old",
        files=[_mod("old.py", "a", "b")],
    )

    resolved, summary = store.uncommitted_summary_for_session(
        session_id=None, anchor_turn_id="turn-old", bot_id="b", user_id="u"
    )

    assert resolved == "session-old"
    assert summary["turn_ids"] == ["turn-old"]
    assert [file["path"] for file in summary["files"]] == ["old.py"]


def test_commit_requested_advances_only_selected_turns_in_active_session(store):
    _conversation(store, session_id="session-active")
    _trigger(store, message_id="msg-1", session_id="session-active")
    _trigger(store, message_id="msg-2", session_id="session-active")
    store.save_turn_files(
        turn_id="turn-1", bot_id="b", user_id="u", trigger_message_id="msg-1",
        files=[_mod("one.py", "a", "b")],
    )
    store.save_turn_files(
        turn_id="turn-2", bot_id="b", user_id="u", trigger_message_id="msg-2",
        files=[_mod("two.py", "a", "b")],
    )

    resolved, marked = store.mark_commit_requested(
        session_id=None, bot_id="b", user_id="u", turn_ids=["turn-1"]
    )
    _, remaining = store.uncommitted_summary_for_session(
        session_id=None, bot_id="b", user_id="u"
    )
    committed = store.summary_for_turn("turn-1")

    assert resolved == "session-active"
    assert marked == 1
    # A request is not a successful commit; failed/requested scope stays visible.
    assert remaining["turn_ids"] == ["turn-1", "turn-2"]
    assert [file["path"] for file in remaining["files"]] == ["one.py", "two.py"]
    assert committed["commit_requested"] is True


def test_commit_requested_uses_selected_turn_after_active_session_rotation(store):
    _conversation(store, session_id="session-old", status="completed", started_at="2026-08-04 10:00:00")
    _conversation(store, session_id="session-new", status="active", started_at="2026-08-05 10:00:00")
    _trigger(store, message_id="msg-old", session_id="session-old")
    store.save_turn_files(
        turn_id="turn-old", bot_id="b", user_id="u", trigger_message_id="msg-old",
        files=[_mod("old.py", "a", "b")],
    )

    resolved, marked = store.mark_commit_requested(
        session_id=None, bot_id="b", user_id="u", turn_ids=["turn-old"]
    )

    assert resolved == "session-old"
    assert marked == 1
    assert store.summary_for_turn("turn-old")["commit_requested"] is True


def test_commit_requested_rejects_turn_from_other_session(store):
    _conversation(store, session_id="session-a")
    _conversation(store, session_id="session-b", status="completed")
    _trigger(store, message_id="msg-b", session_id="session-b")
    store.save_turn_files(
        turn_id="turn-b", bot_id="b", user_id="u", trigger_message_id="msg-b",
        files=[_mod("other.py", "a", "b")],
    )

    _, marked = store.mark_commit_requested(
        session_id="session-a", bot_id="b", user_id="u", turn_ids=["turn-b"]
    )

    assert marked == 0
    assert "commit_requested" not in store.summary_for_turn("turn-b")


def test_commit_evidence_reconciles_two_repositories_across_turns(store, monkeypatch):
    from llm_bawt.service import changed_file_commits

    _conversation(store, session_id="incident")
    for turn, repo, count in (("older-ui", "bawthub", 5), ("current-backend", "llm-bawt", 6)):
        _trigger(store, message_id=turn, session_id="incident")
        store.save_turn_files(
            turn_id=turn, bot_id="b", user_id="u", trigger_message_id=turn,
            files=[ChangedFileInput(repo_key="workspace", repo_label=repo,
                   path=f"~/dev/{repo}/{index}.py", before=b"old", after=b"new")
                   for index in range(count)],
        )

    def proof(files):
        for file in files:
            file["commit_state"] = "committed"
            file["commit_evidence"] = {"commit_hash": "4ded2da" if file["repo_label"] == "bawthub" else "e507a58"}

    monkeypatch.setattr(changed_file_commits, "reconcile_files", proof)
    _, scope, other = store.uncommitted_summaries_for_session(
        session_id="incident", anchor_turn_id="current-backend", bot_id="b", user_id="u",
    )
    assert scope["total_files"] == other["total_files"] == 0
    assert len(scope["history_files"]) == 11
    assert len(other["history_files"]) == 5
    assert {f["commit_evidence"]["commit_hash"] for f in scope["history_files"]} == {"4ded2da", "e507a58"}
    before, after = store.get_file_content(turn_id="older-ui", repo_key="workspace", path="~/dev/bawthub/0.py")
    assert (before.text, after.text) == ("old", "new")
    history = store.summaries_for_triggers(["older-ui", "current-backend"])
    assert all(f["commit_state"] == "committed" for s in history.values() for f in s["files"])


def test_latest_snapshot_controls_action_scope_not_older_unverified_bytes(store, monkeypatch):
    from llm_bawt.service import changed_file_commits

    _conversation(store, session_id="s")
    for turn, after in (("old", "v1"), ("new", "v2")):
        _trigger(store, message_id=turn, session_id="s")
        store.save_turn_files(turn_id=turn, bot_id="b", user_id="u", trigger_message_id=turn,
                              files=[_mod("same.py", "base", after)])

    def proof(files):
        for file in files:
            file["commit_state"] = "committed" if file["turn_id"] == "new" else "unknown"

    monkeypatch.setattr(changed_file_commits, "reconcile_files", proof)
    _, scope = store.uncommitted_summary_for_session(session_id="s", bot_id="b", user_id="u")
    assert scope["total_files"] == 0
    assert len(scope["history_files"]) == 2
    assert scope["history_files"][0]["commit_state"] == "unknown"


def test_postgres_fallback_when_no_object_store(store, monkeypatch):
    # Force "no object store": content must round-trip via turn_diff_payloads.
    monkeypatch.setattr(cfs, "_get_diff_blob_backend", lambda: None)
    store.save_turn_files(
        turn_id="turn-9", bot_id="b", user_id="u", trigger_message_id="m",
        files=[_mod("a.py", "before-text", "after-text")],
    )
    before, after = store.get_file_content(turn_id="turn-9", repo_key="repoA", path="a.py")
    assert before.text == "before-text"
    assert after.text == "after-text"


def test_content_for_rows_deduplicates_blob_reads_but_keeps_row_flags(store, monkeypatch):
    calls = []

    class Backend:
        def get(self, key):
            calls.append(key)
            return {"old": b"before", "new": b"after"}[key]

    monkeypatch.setattr(cfs, "_get_diff_blob_backend", lambda: Backend())
    rows = [TurnChangedFile(id=index, before_blob_key="old", before_sha256="old",
                            after_blob_key="new", after_sha256="new",
                            binary=False, truncated=index == 2) for index in (1, 2, 3)]
    content = store.content_for_rows(rows)

    assert [side.text for side in content[1]] == ["before", "after"]
    assert content[2][0].truncated is True and content[1][0].truncated is False
    assert content[3][0].truncated is False
    assert calls.count("old") == calls.count("new") == 2  # one per distinct flag


def test_missing_file_returns_none(store):
    assert store.get_file_content(turn_id="nope", repo_key="r", path="x") is None


def test_empty_summary_shape():
    summary = build_turn_summary("turn-x", [])
    assert summary == {
        "turn_id": "turn-x", "files": [],
        "total_files": 0, "total_additions": 0, "total_deletions": 0,
    }
