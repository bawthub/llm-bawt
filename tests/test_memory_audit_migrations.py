"""TASK-2248 migration safety on disposable PostgreSQL, never live data."""
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, event, text

from llm_bawt.memory import embeddings, migrations_memory as migration
from llm_bawt.memory import migrations_session_backfill as legacy
from tests.private_postgres import (
    private_module_db as private_module_db,
    private_pg as private_pg,
    private_url,
)


@pytest.fixture
def backend(private_pg, private_module_db):  # noqa: F811
    engine = create_engine(private_url(private_pg, private_module_db))
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text("""
            CREATE TABLE synthetic_memories (
                id TEXT PRIMARY KEY, content TEXT, intent TEXT, stakes TEXT,
                emotional_charge FLOAT, recurrence_keywords JSONB,
                meaning_embedding vector(3), meaning_updated_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        """))
        conn.execute(text("CREATE TABLE synthetic_messages (session_id TEXT, timestamp FLOAT)"))
        conn.execute(text("INSERT INTO synthetic_messages VALUES ('history', 1000)"))
    yield SimpleNamespace(engine=engine, _memories_table_name="synthetic_memories",
                          embedding_model="synthetic")
    with engine.begin() as conn:
        for table in ("synthetic_memories", "synthetic_messages", "sessions", "messages"):
            conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
    engine.dispose()


def add(backend, ident, content="fallback", intent=None, emotion=None, keywords=None):
    import json
    with backend.engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO synthetic_memories
                (id, content, intent, emotional_charge, recurrence_keywords)
            VALUES (:id, :content, :intent, :emotion, CAST(:keywords AS jsonb))
        """), {"id": ident, "content": content, "intent": intent, "emotion": emotion,
               "keywords": json.dumps(keywords) if keywords is not None else None})


def test_failed_empty_and_throwing_rows_do_not_starve_later_rows(backend, monkeypatch):
    add(backend, "a", content="", intent="")
    add(backend, "b", intent="failed")
    add(backend, "c", intent="throw")
    add(backend, "d", intent="success")
    calls = []

    def embed(body, _model):
        calls.append(body)
        assert len(calls) <= 3, "migration retried an already examined row"
        if body == "throw":
            raise RuntimeError("synthetic outage")
        return None if body == "failed" else [0.1, 0.2, 0.3]

    monkeypatch.setattr(embeddings, "generate_embedding", embed)
    assert migration.backfill_meaning_embeddings(backend, batch_size=1) == {
        "total": 4, "updated": 1, "failed": 2, "skipped": 1}
    assert calls == ["failed", "throw", "success"]


def test_canonical_emotion_and_content_fallback_with_sql_null_keywords(backend, monkeypatch):
    add(backend, "a", intent="")
    add(backend, "b", emotion=0.7)
    add(backend, "c", intent="intent", emotion=0.5, keywords=["repeat"])
    calls = []

    def embed(body, _model):
        calls.append(body)
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(embeddings, "generate_embedding", embed)
    assert migration.backfill_meaning_embeddings(backend)["updated"] == 3
    assert calls == ["fallback", "emotional", "intent | emotional | repeat"]


def test_failed_database_write_rolls_back_before_next_row(backend, monkeypatch):
    add(backend, "a", intent="first")
    add(backend, "b", intent="second")
    monkeypatch.setattr(embeddings, "generate_embedding", lambda *_: [0.1, 0.2, 0.3])
    failed = []

    def reject_first(_conn, _cursor, statement, _params, _context, _many):
        if statement.lstrip().startswith("UPDATE") and not failed:
            failed.append(True)
            # A real PG statement failure leaves the transaction aborted.
            _cursor.execute("SELECT 1 / 0")

    event.listen(backend.engine, "before_cursor_execute", reject_first)
    result = migration.backfill_meaning_embeddings(backend, batch_size=2)
    assert result["updated"] == result["failed"] == 1


def test_embedding_pass_ends_at_initial_upper_id(backend, monkeypatch):
    add(backend, "a", intent="first")
    calls = []

    def embed(body, _model):
        calls.append(body)
        add(backend, "z", intent="next pass")
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(embeddings, "generate_embedding", embed)
    assert migration.backfill_meaning_embeddings(backend, batch_size=1)["updated"] == 1
    assert calls == ["first"]


@pytest.mark.parametrize("change", [
    "content='edited'", "intent='edited'", "stakes='edited'",
    "emotional_charge=0.8", "recurrence_keywords='[\"edited\"]'::jsonb",
    "meaning_embedding='[0.9,0.9,0.9]'",
])
def test_concurrent_edit_or_vector_is_not_overwritten(backend, monkeypatch, change):
    add(backend, "a", intent="original")

    def embed(_body, _model):
        with backend.engine.begin() as conn:
            conn.execute(text(f"UPDATE synthetic_memories SET {change} WHERE id='a'"))
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(embeddings, "generate_embedding", embed)
    result = migration.backfill_meaning_embeddings(backend)
    assert result["updated"] == 0 and result["skipped"] == 1
    with backend.engine.connect() as conn:
        vector = conn.execute(text(
            "SELECT meaning_embedding::text FROM synthetic_memories WHERE id='a'")).scalar()
    assert vector == ("[0.9,0.9,0.9]" if change.startswith("meaning_embedding") else None)


@pytest.mark.parametrize("existing", [False, True])
def test_session_preview_is_read_only_including_bootstrap(backend, existing):
    if existing:
        with backend.engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE sessions (id VARCHAR(36) PRIMARY KEY, bot_id TEXT,
                    status TEXT, ended_at TIMESTAMP)
            """))
            conn.execute(text("INSERT INTO sessions VALUES ('old', 'synthetic', 'completed', NOW())"))
    statements, commits = [], []
    event.listen(backend.engine, "before_cursor_execute",
                 lambda _c, _cur, statement, _p, _ctx, _many: statements.append(statement.strip()))
    event.listen(backend.engine, "commit", lambda _c: commits.append(True))
    result = migration.backfill_sessions(backend, dry_run=True)
    assert result["distinct_sessions"] == 1 and result["inserted"] == 0
    assert not commits
    assert all(s.upper().startswith("SELECT") for s in statements)
    with backend.engine.connect() as conn:
        if existing:
            assert conn.execute(text("SELECT status FROM sessions")).scalar() == "completed"
        else:
            assert conn.execute(text("SELECT to_regclass('sessions')")).scalar() is None


def test_executing_session_backfill_still_bootstraps_and_is_idempotent(backend):
    assert migration.backfill_sessions(backend)["inserted"] == 1
    assert migration.backfill_sessions(backend)["inserted"] == 0
    with backend.engine.connect() as conn:
        row = conn.execute(text("SELECT status, archived_at, ended_at FROM sessions")).one()
    assert row.status == "archived" and row.archived_at == row.ended_at


def test_legacy_ids_keep_short_names_and_separate_long_prefixes():
    assert legacy._legacy_id("synthetic") == "legacy-synthetic"
    prefix = "synthetic-" + "x" * 35
    first, second = legacy._legacy_id(prefix + "a"), legacy._legacy_id(prefix + "b")
    assert first != second and len(first) == len(second) == 36
    assert first == legacy._legacy_id(prefix + "a")


@pytest.mark.parametrize("conflict", [None, "bot", "user", "status", "metadata"])
def test_legacy_conflicts_validate_owner_and_marker_before_linking(backend, conflict):
    with backend.engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE sessions (id VARCHAR(36) PRIMARY KEY, bot_id TEXT, user_id TEXT,
                started_at TIMESTAMP, ended_at TIMESTAMP, status TEXT, session_metadata JSONB)
        """))
        conn.execute(text("CREATE TABLE messages (bot_id TEXT, session_id TEXT, timestamp FLOAT)"))
        conn.execute(text("INSERT INTO messages VALUES ('synthetic', NULL, 1000)"))
        sid = legacy._upsert_legacy_session(conn, "synthetic", "owner")
    if conflict:
        change = {"bot": "bot_id='other'", "user": "user_id='other'",
                  "status": "status='active'", "metadata": "session_metadata='{}'::jsonb"}[conflict]
        with backend.engine.begin() as conn:
            conn.execute(text(f"UPDATE sessions SET {change}"))
        with pytest.raises(ValueError, match="does not match"):
            legacy.run_execute(backend.engine, "owner")
        with backend.engine.connect() as conn:
            assert conn.execute(text("SELECT session_id FROM messages")).scalar() is None
    else:
        assert legacy.run_execute(backend.engine, "owner")["ok"]
        assert legacy.run_execute(backend.engine, "owner")["ok"]
        with backend.engine.connect() as conn:
            assert conn.execute(text("SELECT session_id FROM messages")).scalar() == sid
