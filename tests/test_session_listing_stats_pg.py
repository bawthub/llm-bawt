"""Cross-bot session metadata from private partitioned PostgreSQL, never live data."""

from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text

from llm_bawt.memory.postgresql_short_term import PostgreSQLShortTermManager
from tests.private_postgres import private_url

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def manager(private_pg, private_module_db):
    engine = create_engine(private_url(private_pg, private_module_db))
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY, bot_id TEXT, user_id TEXT, status TEXT,
                started_at TIMESTAMP, ended_at TIMESTAMP, archived_at TIMESTAMP,
                session_metadata JSONB DEFAULT '{}'
            )
        """))
        conn.execute(text("""
            CREATE TABLE messages (
                bot_id TEXT, session_id TEXT, role TEXT, content TEXT, timestamp FLOAT8
            ) PARTITION BY LIST (bot_id)
        """))
        for key in ("synthetica", "syntheticb", "syntheticdenied"):
            conn.execute(text(f"CREATE TABLE messages_{key} PARTITION OF messages FOR VALUES IN ('{key}')"))
        for ident, bot, user, started, status in (
            ("a", "synthetic-a", "test-user", "2026-02-01", "archived"),
            ("b", "synthetic-b", "test-user", "2026-01-01", "archived"),
            ("empty", "synthetic-b", "test-user", "2026-01-01", "archived"),
            ("other-user", "synthetic-b", "other-user", "2026-01-01", "archived"),
            ("denied", "synthetic-denied", "test-user", "2026-01-01", "archived"),
            ("deleted", "synthetic-b", "test-user", "2026-01-01", "deleted"),
        ):
            conn.execute(text("""
                INSERT INTO sessions (id, bot_id, user_id, started_at, status)
                VALUES (:id, :bot, :user, :started, :status)
            """), {"id": ident, "bot": bot, "user": user, "started": started, "status": status})
        for bot, session, role, body, stamp in (
            ("synthetica", "a", "user", "A first", 100),
            ("synthetica", "a", "assistant", "A reply", 110),
            ("syntheticb", "b", "user", "B first", 200),
            ("syntheticb", "b", "assistant", "B reply", 220),
            ("syntheticb", "b", "user", "B latest searchable", 230),
            ("syntheticb", "b", "summary", "B summary", 240),
            # Wrong-partition rows must not contaminate A's metadata or query match.
            ("syntheticdenied", "a", "user", "DENIED-COLLISION", 1000),
            ("syntheticdenied", "denied", "assistant", "DENIED-ONLY", 2000),
            ("syntheticb", "other-user", "user", "OTHER-USER-ONLY", 3000),
            ("syntheticb", "deleted", "user", "DELETED-ONLY", 4000),
        ):
            conn.execute(text("INSERT INTO messages VALUES (:bot, :session, :role, :body, :stamp)"),
                         {"bot": bot, "session": session, "role": role, "body": body, "stamp": stamp})
    result = object.__new__(PostgreSQLShortTermManager)
    result.bot_id = "synthetic-a"
    result._backend = SimpleNamespace(engine=engine, _messages_table_name="messages_synthetica")
    yield result
    engine.dispose()


def listing(manager, **kwargs):
    return manager.list_sessions(
        user_id="test-user", allowed_bot_ids=frozenset({"synthetic-a", "synthetic-b"}), **kwargs)


def test_cross_bot_counts_excerpts_activity_and_order_use_all_eligible_partitions(manager):
    rows = listing(manager, bot_id="")
    assert [row["id"] for row in rows] == ["empty", "b", "a"]
    by_id = {row["id"]: row for row in rows}
    a, b = by_id["a"], by_id["b"]
    assert a["message_count"] == 2 and a["turn_count"] == 1
    assert a["first_user_message"] == a["last_user_message"] == "A first"
    assert b["message_count"] == 3 and b["turn_count"] == 2
    assert b["first_user_message"] == "B first" and b["last_user_message"] == "B latest searchable"
    assert b["last_assistant_message"] == "B reply" and b["summary"] == "B summary"
    assert b["first_message_at"] == "1970-01-01T00:03:20+00:00"
    assert b["last_activity_at"] == "1970-01-01T00:03:50+00:00"
    assert by_id["empty"]["message_count"] == 0 and by_id["empty"]["last_user_message"] is None
    assert "DENIED" not in str(rows) and "OTHER-USER" not in str(rows) and "DELETED" not in str(rows)


def test_cross_bot_query_filters_before_limit_and_does_not_match_wrong_partition(manager):
    rows = listing(manager, bot_id="", query="searchable", limit=1)
    assert [row["id"] for row in rows] == ["b"]
    assert rows[0]["message_count"] == 3 and rows[0]["last_assistant_message"] == "B reply"
    assert listing(manager, bot_id="", query="DENIED-COLLISION") == []
    assert manager.list_sessions(bot_id="", user_id="test-user", allowed_bot_ids=frozenset()) == []


def test_default_and_named_bot_remain_scoped(manager):
    default = listing(manager)
    named = listing(manager, bot_id="synthetic-a")
    assert default == named and [row["id"] for row in default] == ["a"]
    assert default[0]["message_count"] == 2 and default[0]["last_assistant_message"] == "A reply"
