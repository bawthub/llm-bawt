"""Real SQL proof in a private temporary PostgreSQL cluster, never live data.

Run: pytest -m integration tests/test_session_search_pg.py
Requires locally installed PostgreSQL binaries; no credentials or shared tables.
"""
import json

import pytest
from sqlalchemy import create_engine, text

from llm_bawt.memory.session_search import SessionSearch
from tests.private_postgres import private_url

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def repository(private_pg):
    """Own schema in a test-owned, networkless PostgreSQL container."""
    engine = create_engine(private_url(private_pg, "postgres"))
    try:
        with engine.connect() as conn:
            assert conn.exec_driver_sql("SHOW unix_socket_directories").scalar_one() == "/var/run/postgresql"
        yield from _populate_private_repository(engine)
    finally:
        engine.dispose()


def _populate_private_repository(engine):
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE sessions (id text, bot_id text, user_id text, status text, started_at timestamp, ended_at timestamp, session_metadata jsonb)"))
        rows = [
            ("one", "sample-a", "alice", "active", "2026-01-01", {"title": "Pocket", "agent_session_key_models": {"test": "test-model"}, "agent_session_keys": {"test": "secret-sdk-key"}, "reset_predecessor_ids": ["old"]}),
            ("two", "sample-b", "alice", "archived", "2026-02-01", {"title": "Pocket experiments", "provider": "test-provider"}),
            ("three", "sample-a", "bob", "active", "2026-03-01", {"title": "Pocket private"}),
            ("four", "sample-a", "alice", "deleted", "2026-04-01", {"title": "Pocket deleted"}),
            ("five", "sample-a", None, "archived", "2026-05-01", {"title": "Pocket unowned"}),
            ("six", "sample-a", "alice", "completed", "2026-06-01", {"title": "100%_literal", "agent_session_key_models": ["malformed"], "reset_predecessor_ids": "malformed"}),
            ("seven", "sample-a", "alice", "archived", "2026-07-01", {"title": {"bad": "shape"}, "agent_session_key_models": {"bad": {"nested": "secret"}, "ok": "another-model"}}),
        ]
        for id_, bot, user, status, started, meta in rows:
            conn.execute(text("INSERT INTO sessions VALUES (:id,:bot,:user,:status,:started,NULL,CAST(:meta AS jsonb))"), {"id": id_, "bot": bot, "user": user, "status": status, "started": started, "meta": json.dumps(meta)})
    yield SessionSearch(engine)


def test_owned_ranked_search_and_metadata_projection(repository):
    page = repository.search(user_id="alice", query="pocket")
    assert [row["id"] for row in page["sessions"]] == ["one", "two"]
    assert page["total_count"] == 2
    first = page["sessions"][0]
    assert first["models"] == "test-model" and first["has_predecessor"]
    assert "session_metadata" not in first
    assert repository.search(user_id="alice", query="secret-sdk-key")["total_count"] == 0
    assert repository.search(user_id="alice", query="test-model")["total_count"] == 1
    assert repository.search(user_id="alice", query="test-provider")["total_count"] == 1


def test_paging_has_real_total_even_past_end(repository):
    assert repository.search(user_id="alice", query="pocket", limit=1)["total_count"] == 2
    assert repository.search(user_id="alice", query="pocket", limit=1, offset=1)["sessions"][0]["id"] == "two"
    assert repository.search(user_id="alice", query="pocket", offset=30) == {"sessions": [], "total_count": 2}


def test_archive_alias_bot_time_sort_and_empty_query(repository):
    assert repository.search(user_id="alice", status="active")["total_count"] == 1
    archived = repository.search(user_id="alice", status="archived", sort_by="oldest")
    assert [row["id"] for row in archived["sessions"]] == ["two", "six", "seven"]
    assert all(row["status"] == "archived" for row in archived["sessions"])
    assert repository.search(user_id="alice", bot_id="sample-b")["total_count"] == 1
    assert repository.search(user_id="alice", since=1780272000)["total_count"] == 2
    assert repository.search(user_id="alice", sort_by="recent")["sessions"][0]["id"] == "seven"
    assert repository.search(user_id="alice", query="sample-a")["total_count"] == 3


def test_literal_queries_and_malformed_metadata(repository):
    assert repository.search(user_id="alice", query="%_")["sessions"][0]["id"] == "six"
    assert repository.search(user_id="alice", query="' OR TRUE --")["total_count"] == 0
    assert repository.search(user_id="alice", query="nested")["total_count"] == 0
    hit = repository.search(user_id="alice", query="another-model")["sessions"][0]
    assert hit["title"] == "" and hit["models"] == "another-model"
    assert repository.search(user_id="unknown")["total_count"] == 0
