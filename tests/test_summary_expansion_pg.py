"""Stored summary provenance and per-call ownership on disposable PostgreSQL."""

import json
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
        conn.execute(text("CREATE TABLE sessions (id TEXT PRIMARY KEY, bot_id TEXT, user_id TEXT)"))
        conn.execute(text("""
            CREATE TABLE messages (
                id TEXT PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, timestamp FLOAT8,
                summary_metadata JSONB, author_entity_type TEXT, author_entity_id TEXT
            )
        """))
        conn.execute(text("""
            INSERT INTO sessions VALUES
              ('a', 'synthetic-bot', 'owner'), ('b', 'synthetic-bot', 'other'),
              ('c', 'synthetic-bot', 'owner'), ('foreign-bot', 'another-bot', 'owner')
        """))
        for ident, session, role, body, stamp, author_type, author_id in (
            ("early", "a", "user", "Canonical early original", 1000, "user", "owner"),
            ("reply", "a", "assistant", "Canonical reply", 10000, "bot", "synthetic-bot"),
            ("same-session-unlisted", "a", "user", "Not in this summary", 10001, "user", "owner"),
            ("foreign", "b", "user", "OTHER-USER-SECRET", 10001, "user", "other"),
            ("second", "c", "user", "Second owned session", 1100, "bot", "peer"),
            ("wrong-bot", "foreign-bot", "user", "WRONG-BOT-SECRET", 10002, "user", "owner"),
        ):
            conn.execute(text("""
                INSERT INTO messages
                    (id, session_id, role, content, timestamp, author_entity_type, author_entity_id)
                VALUES (:id, :session, :role, :body, :stamp, :kind, :author)
            """), {"id": ident, "session": session, "role": role, "body": body, "stamp": stamp,
                   "kind": author_type, "author": author_id})
    result = object.__new__(PostgreSQLShortTermManager)
    result.bot_id = "synthetic-bot"
    result.user_id = None  # A cached manager does not retain the calling user.
    result._backend = SimpleNamespace(engine=engine, _messages_table_name="messages")
    yield result
    engine.dispose()


def summary(manager, meta, session="a"):
    with manager._backend.engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO messages (id, session_id, role, content, timestamp, summary_metadata)
            VALUES ('summary', :session, 'summary', 'Summary', 10000, CAST(:meta AS jsonb))
            ON CONFLICT (id) DO UPDATE SET session_id=:session, summary_metadata=CAST(:meta AS jsonb)
        """), {"session": session, "meta": json.dumps(meta)})


def ids(manager, user="owner"):
    return [row.db_id for row in manager.get_messages_for_summary("summary", user_id=user)]


def test_exact_sources_include_old_originals_not_overlapping_or_unlisted_messages(manager):
    summary(manager, {"message_ids": ["reply", "early", "foreign", "wrong-bot"],
                      "session_start": 9999, "session_end": 10001})
    rows = manager.get_messages_for_summary("summary", user_id="owner")
    assert [row.db_id for row in rows] == ["early", "reply"]
    assert rows[0].session_id == "a" and rows[0].author_entity_id == "owner"
    assert rows[1].author_entity_type == "bot" and rows[1].author_entity_id == "synthetic-bot"
    assert ids(manager, user="other") == []


def test_foreign_summary_is_denied_before_expanding_even_owned_references(manager):
    summary(manager, {"message_ids": ["early", "foreign"], "source_session_ids": ["a"]}, session="b")
    assert ids(manager) == []
    assert ids(manager, user="other") == ["foreign"]
    assert manager.user_id is None


def test_multi_session_summary_requires_complete_same_owner_provenance(manager):
    summary(manager, {"message_ids": ["early", "second"], "source_session_ids": ["a", "c"]}, session=None)
    rows = manager.get_messages_for_summary("summary", user_id="owner")
    assert [row.db_id for row in rows] == ["early", "second"]
    assert rows[1].author_entity_type == "bot" and rows[1].author_entity_id == "peer"
    summary(manager, {"message_ids": ["early", "foreign"], "source_session_ids": ["a", "b"]}, session=None)
    assert ids(manager) == []
    # Unbound operator compatibility is separate from a personal caller.
    assert ids(manager, user=None) == ["early", "foreign"]


@pytest.mark.parametrize("meta", [
    {"session_start": 1000, "session_end": 10000},
    {"start_timestamp": 1000, "end_timestamp": 10000},
])
def test_explicit_legacy_range_stays_in_owning_session(manager, meta):
    summary(manager, meta)
    assert ids(manager) == ["early", "reply"]


@pytest.mark.parametrize("meta,session", [
    ({}, "a"), ({"message_ids": []}, "a"), ({"message_ids": "early"}, "a"),
    ({"message_ids": [None], "session_start": 0, "session_end": 20000}, "a"),
    ({"message_ids": ["early"]}, None),
    ({"message_ids": ["early"], "source_session_ids": "a"}, None),
    ({"session_start": 0, "session_end": 20000}, None),
    ({"session_start": 20000, "session_end": 0}, "a"),
    ({"session_start": "nan", "session_end": 20000}, "a"),
    ({"session_start": 0, "session_end": "inf"}, "a"),
    ({"session_start": False, "session_end": 20000}, "a"),
    ({"message_ids": ["wrong-bot"]}, "foreign-bot"),
    (["not", "metadata"], "a"),
])
def test_unverifiable_or_malformed_provenance_never_falls_back_to_time_guess(manager, meta, session):
    summary(manager, meta, session)
    assert ids(manager) == []
