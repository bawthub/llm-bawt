"""Stored-summary proofs on fixture-owned, networkless PostgreSQL, never live."""

from contextlib import contextmanager
from datetime import date
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, text

from llm_bawt.service.routes import history_timeline
from llm_bawt.service.routes.history_scope import local_day_bounds
from llm_bawt.service.routes.history_timeline_store import timeline_rows
from llm_bawt.service.routes.history_timeline_summaries import (
    MAX_SUMMARY_SOURCES,
    stored_summary_rows,
    summary_availability,
)
from tests.private_postgres import private_url

pytestmark = pytest.mark.integration
START, END = local_day_bounds(date(2026, 11, 1), date(2026, 11, 1), "America/New_York")


@pytest.fixture(scope="module")
def engine(private_pg, private_module_db):
    engine = create_engine(private_url(private_pg, private_module_db))
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE sessions (id text PRIMARY KEY, bot_id text, user_id text, "
                          "status text DEFAULT 'active', session_metadata jsonb DEFAULT '{}')"))
        conn.execute(text("CREATE TABLE messages (id text, bot_id text, session_id text, role text, "
                          "content text, timestamp float8, created_at timestamp, summary_metadata jsonb, "
                          "author_entity_type text, author_entity_id text, PRIMARY KEY(bot_id, id))"))
        conn.execute(text("CREATE VIEW messages_test AS SELECT * FROM messages WHERE bot_id='synthetic'"))
        conn.execute(text("CREATE TABLE inter_bot_deliveries (id text, user_message_id text, "
                          "target_bot_id text, author_entity_type text, author_entity_id text)"))
        conn.execute(text("CREATE TABLE prompt_occurrences (job_id text, delivery_id text, session_id text)"))
        conn.execute(text("CREATE TABLE prompt_schedules (job_id text, owner_user_id text, creator_bot_id text)"))
    yield engine
    engine.dispose()


@pytest.fixture
def db(engine, monkeypatch):
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM messages"))
        conn.execute(text("DELETE FROM sessions"))
        conn.execute(text("INSERT INTO sessions (id,bot_id,user_id) VALUES "
                          "('own','synthetic','viewer'), ('sibling','synthetic','viewer'), "
                          "('foreign','synthetic','other'), ('null','synthetic',NULL), "
                          "('wrong-bot','other','viewer')"))
    monkeypatch.setattr(history_timeline, "_engine_and_table", lambda _bot: (engine, "messages_test"))
    # This read path must not create a summarizer, even when nothing is stored.
    from llm_bawt.memory.summarization import HistorySummarizer
    def forbidden(*_a, **_k):
        pytest.fail("Read-only timeline instantiated a summarizer")
    monkeypatch.setattr(HistorySummarizer, "__init__", forbidden)
    return engine


def source(db, id="m", session="own", ts=START + 7200, role="assistant", bot="synthetic"):
    with db.begin() as conn:
        conn.execute(text("INSERT INTO messages (id,bot_id,session_id,role,content,timestamp) "
                          "VALUES (:id,:bot,:session,:role,'source',:ts)"),
                     dict(id=id, bot=bot, session=session, role=role, ts=ts))


def summary(db, id="sum", session="own", meta=None, content="stored prose"):
    if meta is None:
        meta = metadata()
    with db.begin() as conn:
        # Deliberately stale/future clocks: neither defines source coverage/order.
        conn.execute(text("INSERT INTO messages "
            "(id,bot_id,session_id,role,content,timestamp,created_at,summary_metadata) "
            "VALUES (:id,'synthetic',:session,'summary',:content,-1000,'2099-01-01',CAST(:meta AS jsonb))"),
            dict(id=id, session=session, content=content, meta=json.dumps(meta)))


def metadata(**changes):
    return {"message_ids": ["m"], "message_count": 1, "source_session_ids": ["own"],
            "session_start": START + 7200, "session_end": START + 7200, **changes}


def load(db, session=None, start=START, end=END, has_day=True):
    with db.connect() as conn:
        return stored_summary_rows(conn, "messages_test", "synthetic", "viewer",
                                   session, start, end, has_day=has_day)


@contextmanager
def captured(db):
    calls = []
    def capture(_conn, _cursor, sql, params, _context, _many):
        calls.append((sql, params))
    event.listen(db, "before_cursor_execute", capture)
    try:
        yield calls
    finally:
        event.remove(db, "before_cursor_execute", capture)


@contextmanager
def client():
    app = FastAPI()
    app.include_router(history_timeline.router)
    with TestClient(app, raise_server_exceptions=False) as http:
        yield http


def test_day_only_assistant_sources_and_opt_in_are_independent_of_prompts(db):
    source(db)
    summary(db)
    source(db, "sibling-message", "sibling", role="user")
    summary(db, "sibling-summary", "sibling", metadata(
        message_ids=["sibling-message"], source_session_ids=["sibling"]))
    with client() as http:
        params = dict(bot_id="synthetic", user_id="viewer", date="2026-11-01",
                      tz="America/New_York", limit=1)
        with captured(db) as calls:
            old = http.get("/v1/history/timeline/prompts", params=params)
        assert old.status_code == 200
        assert len(calls) == 1
        assert old.json()["summaries"] == [] and not old.json()["summaries_truncated"]
        with captured(db) as calls:
            response = http.get("/v1/history/timeline/prompts", params={**params, "include_summaries": True})
        assert response.status_code == 200, response.text
        assert len(calls) == 2
        assert [p["id"] for p in response.json()["prompts"]] == ["sibling-message"]
        assert {s["id"] for s in response.json()["summaries"]} == {"sum", "sibling-summary"}
        # Both eligible excerpts survive even if the user-prompt page is full.
        source(db, "later-prompt", "sibling", START + 7300, role="user")
        paged = http.get("/v1/history/timeline/prompts", params={**params, "include_summaries": True}).json()
        assert paged["truncated"] and len(paged["summaries"]) == 2
        selected = http.get("/v1/history/timeline/prompts", params={
            **params, "session_id": "own", "include_summaries": True}).json()
        assert selected["prompts"] == [] and [s["id"] for s in selected["summaries"]] == ["sum"]


@pytest.mark.parametrize("changes", [
    {"source_session_ids": []}, {"source_session_ids": ["own", "foreign"]},
    {"source_session_ids": ["sibling"]}, {"source_session_ids": "own"},
    {"source_session_ids": [None]}, {"source_session_ids": {"0": "own"}},
    {"message_ids": []}, {"message_ids": "m"}, {"message_ids": {}},
    {"message_ids": [None]}, {"message_ids": [1]}, {"message_ids": [["m"]]},
    {"message_ids": ["m", "m"], "message_count": 2},
    {"message_ids": ["m", "deleted"], "message_count": 2},
    {"message_ids": ["x" * 129]}, {"message_ids": [""]},
    {"message_count": 0}, {"message_count": None}, {"message_count": "1"},
    {"message_count": True}, {"message_count": 1.5}, {"message_count": 2},
    {"message_ids": ["m"] * (MAX_SUMMARY_SOURCES + 1), "message_count": MAX_SUMMARY_SOURCES + 1},
])
def test_malformed_or_unresolvable_provenance_is_unavailable(db, changes):
    source(db)
    summary(db, meta=metadata(**changes))
    assert load(db) == []
    with db.connect() as conn:
        availability = summary_availability(conn, "messages_test", "synthetic", "viewer",
                                            [{"day": "2026-11-01", "session_id": "own"}])
    assert availability[("2026-11-01", "own")]["summary_count"] == 0


@pytest.mark.parametrize("bad_metadata", [[], "malformed", 42, False, {}, {"message_ids": ["m"]}])
def test_malformed_metadata_roots(db, bad_metadata):
    source(db)
    summary(db, meta=bad_metadata)
    assert load(db) == []


@pytest.mark.parametrize("foreign_session", ["foreign", "null", "sibling", None])
def test_mixed_sources_cannot_hide_behind_singleton_provenance(db, foreign_session):
    source(db)
    source(db, "mixed", foreign_session)
    summary(db, meta=metadata(message_ids=["m", "mixed"], message_count=2))
    assert load(db) == []


def test_source_validation_accepts_exact_cap(db):
    ids = [f"source-{i}" for i in range(MAX_SUMMARY_SOURCES)]
    with db.begin() as conn:
        conn.execute(text("INSERT INTO messages (id,bot_id,session_id,role,content,timestamp) "
                          "VALUES (:id,'synthetic','own','assistant','source',:ts)"),
                     [{"id": id_, "ts": START + 7200} for id_ in ids])
    summary(db, meta=metadata(message_ids=ids, message_count=len(ids)), content="x" * 1200)
    row, = load(db)
    assert row["source_message_count"] == MAX_SUMMARY_SOURCES
    assert len(row["excerpt"]) == 1200 and not row["truncated"]


@pytest.mark.parametrize("session,bot,role", [
    ("foreign", "synthetic", "user"), ("null", "synthetic", "user"),
    ("wrong-bot", "synthetic", "user"), ("sibling", "synthetic", "assistant"),
    (None, "synthetic", "user"), ("missing-session", "synthetic", "user"),
    ("own", "other", "user"), ("own", "synthetic", "system"),
    ("own", "synthetic", "summary"),
])
def test_source_ownership_not_summary_stamp_controls_exposure(db, session, bot, role):
    source(db, "m", session, role=role, bot=bot)
    source(db, "active")  # owned activity must not authorize foreign prose
    summary(db)
    assert load(db, "own") == []


def test_deleted_source_and_deleted_owner_are_unavailable(db):
    source(db)
    source(db, "active")
    summary(db)
    assert len(load(db)) == 1
    with db.begin() as conn:
        conn.execute(text("DELETE FROM messages WHERE id='m'"))
    assert load(db) == []
    source(db)
    with db.begin() as conn:
        conn.execute(text("DELETE FROM sessions WHERE id='own'"))
    assert load(db) == []


@pytest.mark.parametrize("changes,coverage", [
    ({}, "day"),
    ({"session_start": START, "session_end": END - 0.001}, "day"),
    ({"session_start": START - 1}, "conversation"),
    ({"session_end": END}, "conversation"),  # recorded end is inclusive
    ({"session_start": START - 86400, "session_end": END + 86400}, "conversation"),
    ({"session_start": None}, "conversation"),
    ({"session_end": "not a timestamp"}, "conversation"),
    ({"session_start": []}, "conversation"),
    ({"session_end": 1e100}, "conversation"),
    ({"session_end": START - 1}, "conversation"),
])
def test_recorded_extents_never_generation_clocks_define_coverage(db, changes, coverage):
    source(db)
    summary(db, meta=metadata(**changes))
    row, = load(db, "own")
    assert row["coverage"] == coverage
    assert row["source_message_count"] == 1 and row["excerpt"] == "stored prose"
    if "session_start" not in changes:
        assert row["source_start_utc"] == START + 7200
    assert load(db, "own", has_day=False)[0]["coverage"] == "conversation"


def test_missing_dates_are_explicit_conversation_context(db):
    source(db)
    meta = metadata()
    del meta["session_start"], meta["session_end"]
    summary(db, meta=meta)
    row, = load(db, "own")
    assert row["coverage"] == "conversation"
    assert row["source_start_utc"] is None and row["source_end_utc"] is None


def test_dst_and_utc_clock_intersection_in_lazy_route(db):
    assert END - START == 25 * 3600
    source(db)
    summary(db)
    with client() as http:
        params = dict(bot_id="synthetic", user_id="viewer", date="2026-11-01",
                      session_id="own", tz="America/New_York", include_summaries=True)
        for lower, upper, expected in [
            (START, END, "day"),
            (START + 7200, START + 7201, "day"),
            (START + 7100, START + 7200, None),  # activity at exclusive end
            (END, END + 3600, None),  # disjoint with selected day
        ]:
            response = http.get("/v1/history/timeline/prompts", params={
                **params, "from_utc": lower, "to_utc": upper})
            assert response.status_code == 200, response.text
            rows = response.json()["summaries"]
            assert ([r["coverage"] for r in rows] if expected else rows) == ([expected] if expected else [])
        # Active message is within the clock range, recorded extent is wider.
        summary(db, "wider", meta=metadata(session_start=START))
        response = http.get("/v1/history/timeline/prompts", params={
            **params, "from_utc": START + 7200, "to_utc": START + 7201}).json()
        assert {r["id"]: r["coverage"] for r in response["summaries"]} == {"sum": "day", "wider": "conversation"}


def test_excerpt_and_result_truncation_deterministic_and_sql_bounded(db):
    source(db)
    for i in reversed(range(10)):
        summary(db, f"s{i:02d}", content="é" * 1201)
    with client() as http, captured(db) as calls:
        response = http.get("/v1/history/timeline/prompts", params={
            "bot_id": "synthetic", "user_id": "viewer", "session_id": "own",
            "include_summaries": True})
    body = response.json()
    assert response.status_code == 200, response.text
    assert [s["id"] for s in body["summaries"]] == [f"s{i:02d}" for i in range(8)]
    assert body["summaries_truncated"]
    assert all(len(s["excerpt"]) == 1200 and s["truncated"] for s in body["summaries"])
    assert len(calls) == 2
    assert calls[-1][1]["summary_limit"] == 9 and calls[-1][1]["excerpt_chars"] == 1200


def test_index_counts_revision_and_constant_batch_query_count(db):
    source(db)
    summary(db)
    with captured(db) as calls:
        groups, _, _, _, _ = timeline_rows(db, "messages_test", "synthetic", "viewer",
                                          None, "America/New_York", START, END)
    assert len(calls) == 4
    assert groups[0]["summary_count"] == 1
    first_revision = groups[0]["summary_revision"]
    assert "excerpt" not in groups[0] and "content" not in groups[0]
    with db.begin() as conn:
        conn.execute(text("UPDATE messages SET content='changed' WHERE id='sum'"))
    with db.connect() as conn, captured(db) as calls:
        availability = summary_availability(conn, "messages_test", "synthetic", "viewer",
            [{"day": f"key-{i}", "session_id": "own"} for i in range(2000)])
    assert len(calls) == 1 and len(availability) == 2000
    assert availability[("key-0", "own")]["summary_revision"] != first_revision
    assert all(a["summary_count"] == 1 for a in availability.values())
    with captured(db) as calls:
        result = timeline_rows(db, "messages_test", "synthetic", "viewer",
                               None, "America/New_York", START, END, overview_only=True)
    assert len(calls) == 1 and result[0] == []
    source(db, "previous", ts=START - 1)
    with captured(db) as calls:
        timeline_rows(db, "messages_test", "synthetic", "viewer", None,
                      "America/New_York", START, END, recent=True)
    assert len(calls) == 6


def test_summary_fetch_error_is_not_empty_success(db, monkeypatch):
    source(db)
    def broken(*_a, **_k):
        raise RuntimeError("database read failed")
    monkeypatch.setattr(history_timeline, "stored_summary_rows", broken)
    with client() as http:
        params = dict(bot_id="synthetic", user_id="viewer", session_id="own")
        assert http.get("/v1/history/timeline/prompts", params=params).status_code == 200
        assert http.get("/v1/history/timeline/prompts", params={
            **params, "include_summaries": True}).status_code == 500


def test_summary_availability_error_fails_index(db, monkeypatch):
    from llm_bawt.service.routes import history_timeline_store
    source(db)
    def broken(*_a, **_k):
        raise RuntimeError("summary availability read failed")
    monkeypatch.setattr(history_timeline_store, "summary_availability", broken)
    with client() as http:
        assert http.get("/v1/history/timeline", params={
            "bot_id": "synthetic", "user_id": "viewer", "mode": "custom",
            "from_date": "2026-11-01", "to_date": "2026-11-01",
        }).status_code == 500


def test_absent_summaries_are_unavailable_without_generation(db):
    source(db)
    assert load(db) == []
