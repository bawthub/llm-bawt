"""TASK-1323 F1 on real PostgreSQL: identifier queries match exactly, in scope.

Private PostgreSQL, isolated schema. Exercises the FtsQuery SQL through both
production callers: the scoped cross-bot searcher and per-bot text search.
"""
from __future__ import annotations

import uuid

import pytest
from sqlalchemy import create_engine, text

from llm_bawt.bot_access import BotPolicyResolver
from llm_bawt.mcp_server.message_search import CrossBotMessageSearcher, OperatorMessageScope
from llm_bawt.memory.postgresql_messages import PostgreSQLMessageMixin
from llm_bawt.utils import db
from tests.pg_isolation import require_test_db_config

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]

# id -> (partition, content)
MESSAGES = {
    "t1281": ("pub", "Picked up TASK-1281 slice 2b today"),
    "t12810": ("pub", "TASK-12810 is a different ticket"),
    "plain_task": ("pub", "this task is about the stop button"),
    "snake": ("pub", "it calls messages_search_all under the hood"),
    "snake_words": ("pub", "messages were searched all day"),
    "sha": ("pub", "landed as 54386cc0ffee on master"),
    "path": ("pub", "edited src/llm_bawt/memory/postgresql.py again"),
    "camel": ("pub", "the searchAll helper"),
    "quoted": ("pub", "please press the stop button now"),
    "date": ("pub", "deadline is 2026-10-24 for this"),
    "private": ("priv", "TASK-1281 in the protected partition"),
}


@pytest.fixture
def engine(private_app_pg):
    url = db.build_postgres_url(require_test_db_config())
    root = create_engine(url)
    schema = "ftsquery_" + uuid.uuid4().hex
    with root.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA {schema}"))
    eng = create_engine(url, connect_args={"options": f"-csearch_path={schema},public"})
    with eng.begin() as conn:
        conn.execute(text("""CREATE TABLE messages (
            id text, bot_id text, session_id text, role text, content text,
            timestamp double precision, author_entity_type text, author_entity_id text)"""))
        conn.execute(text("CREATE TABLE bot_profiles (slug text, include_in_global_search boolean, "
                          "name text, color text, avatar text, avatar_render text)"))
        conn.execute(text("INSERT INTO bot_profiles (slug, include_in_global_search) VALUES "
                          "('pub', true), ('priv', false)"))
        conn.execute(text("CREATE TABLE entity_profiles (entity_type text, entity_id text, "
                          "display_name text, color text, avatar text, avatar_render text)"))
        conn.execute(text("CREATE TABLE sessions (id text, bot_id text, user_id text, status text)"))
        for ts, (mid, (partition, content)) in enumerate(MESSAGES.items(), start=1):
            conn.execute(text("INSERT INTO messages VALUES (:id, :bot, 's', 'user', :c, :ts, NULL, NULL)"),
                         {"id": mid, "bot": partition, "c": content, "ts": float(ts)})
    try:
        yield eng
    finally:
        eng.dispose()
        with root.begin() as conn:
            conn.execute(text(f"DROP SCHEMA {schema} CASCADE"))
        root.dispose()


def scoped(engine, query, partitions=frozenset({"pub"})):
    searcher = CrossBotMessageSearcher(engine, lambda rows, _source: rows)
    rows = searcher.search_fts(query, partitions=partitions, n_results=50, role_filter=None,
                               sort_by="relevance", since=None, until=None)
    return {row["id"] for row in rows}


@pytest.mark.parametrize(("query", "expected"), [
    ("TASK-1281", {"t1281"}),
    ("messages_search_all", {"snake"}),
    ("54386cc", {"sha"}),
    ("src/llm_bawt/memory/postgresql.py", {"path"}),
    ("searchAll", {"camel"}),
    ("2026-10-24", {"date"}),
    ('"press the stop"', {"quoted"}),
    ("stop button", {"plain_task", "quoted"}),
    ("TASK-1281 54386cc", {"t1281", "sha"}),
])
def test_identifier_queries_match_exactly(engine, query, expected):
    assert scoped(engine, query) == expected


def test_partitions_are_the_only_scope(engine):
    assert scoped(engine, "TASK-1281", frozenset({"pub", "priv"})) == {"t1281", "private"}
    assert scoped(engine, "TASK-1281", frozenset({"priv"})) == {"private"}
    assert scoped(engine, "TASK-1281", frozenset()) == set()


def test_operator_fts_uses_the_same_parser(engine):
    access = BotPolicyResolver(engine).resolve_sync(None)
    scope = OperatorMessageScope.from_access(access)
    searcher = CrossBotMessageSearcher(engine, lambda rows, _source: rows)
    rows = searcher.search_operator("fts", "TASK-1281", scope, n_results=50, role_filter=None,
                                    sort_by="relevance", since=None, until=None)
    assert {row["id"] for row in rows} == {"t1281"}


def test_per_bot_text_search_runs_the_parsed_query(engine):
    backend = PostgreSQLMessageMixin()
    backend.engine = engine
    backend._messages_table_name = "messages"
    rows = backend.search_messages_by_text("TASK-1281", n_results=10, exclude_recent_seconds=0,
                                           role_filter=None)
    # Raw per-bot table: no partition filter here, so both rows match.
    assert {row["id"] for row in rows} == {"t1281", "private"}
    assert all(row["relevance"] > 0 for row in rows)
