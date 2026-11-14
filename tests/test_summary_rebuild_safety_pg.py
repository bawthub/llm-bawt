"""Summary rebuild never loses unrelated or unreplaced summaries (TASK-2257).

Fixture-owned, networkless PostgreSQL, never live. Replacement is scoped to
explicit source IDs; purge runs only after successful rebuilds and restores
source eligibility for summaries it deletes without a replacement.
"""

from __future__ import annotations

import json
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text

from llm_bawt.memory import summarization
from llm_bawt.memory.summarization import HistorySummarizer, Session
from tests.private_postgres import private_url

pytestmark = pytest.mark.integration

TABLE = "messages_rebuild"


@pytest.fixture(scope="module")
def engine(private_pg, private_module_db):
    engine = create_engine(private_url(private_pg, private_module_db))
    with engine.begin() as conn:
        conn.execute(text(
            f"CREATE TABLE {TABLE} (id text PRIMARY KEY, role text, content text, "
            "timestamp float8, summarized boolean DEFAULT FALSE, summary_metadata jsonb)"
        ))
    yield engine
    engine.dispose()


@pytest.fixture
def db(engine):
    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {TABLE}"))
    return engine


def _source(db, mid, ts, summarized=True):
    with db.begin() as conn:
        conn.execute(text(
            f"INSERT INTO {TABLE} (id, role, content, timestamp, summarized) "
            "VALUES (:id, 'user', 'source', :ts, :s)"
        ), {"id": mid, "ts": ts, "s": summarized})


def _summary(db, sid, ts, message_ids=None, **meta):
    if message_ids is not None:
        meta["message_ids"] = message_ids
    with db.begin() as conn:
        conn.execute(text(
            f"INSERT INTO {TABLE} (id, role, content, timestamp, summary_metadata) "
            "VALUES (:id, 'summary', 'old summary', :ts, CAST(:meta AS jsonb))"
        ), {"id": sid, "ts": ts, "meta": json.dumps(meta)})


def _summaries(db) -> set[str]:
    with db.connect() as conn:
        return {r.id for r in conn.execute(text(f"SELECT id FROM {TABLE} WHERE role='summary'"))}


def _summarized(db) -> dict[str, bool]:
    with db.connect() as conn:
        return {r.id: r.summarized for r in conn.execute(
            text(f"SELECT id, summarized FROM {TABLE} WHERE role <> 'summary'"))}


def _summarizer(db) -> HistorySummarizer:
    s = object.__new__(HistorySummarizer)
    s._backend = SimpleNamespace(engine=db, _messages_table_name=TABLE)
    s._summarize_batch_fn = None
    s.session_gap_seconds = 3600
    s.max_context_tokens = 0
    s.protected_recent_turns = 0
    s.min_messages_per_session = 1
    return s


def _session(ids, start, end):
    return Session(start_timestamp=start, end_timestamp=end, messages=[], message_ids=ids)


def test_replacement_skips_other_thread_summary_with_disjoint_sources(db):
    _summary(db, "summary-B", 150, ["b1", "b2"], source_session_ids=["thread-B"],
             session_start=140, session_end=160)
    _summary(db, "summary-A-old", 150, ["a1"], session_start=100, session_end=200)
    summarizer = _summarizer(db)

    with db.connect() as conn:
        ids = summarizer._find_replaceable_summary_ids(conn, _session(["a1", "a2"], 100, 200))

    assert ids == ["summary-A-old"]


def test_replacement_skips_adjacent_chunk_inside_one_second_window(db):
    _summary(db, "chunk-1", 100, ["c1", "c2"], session_start=90, session_end=100)
    summarizer = _summarizer(db)

    with db.connect() as conn:
        ids = summarizer._find_replaceable_summary_ids(conn, _session(["c3", "c4"], 100.5, 110))

    assert ids == []


def test_legacy_summary_without_sources_still_uses_time_fallback(db):
    _summary(db, "legacy", 150, session_start=100, session_end=200)
    summarizer = _summarizer(db)

    with db.connect() as conn:
        ids = summarizer._find_replaceable_summary_ids(conn, _session(["x"], 100, 200))

    assert ids == ["legacy"]


@pytest.fixture
def rebuild_rig(db, monkeypatch):
    _source(db, "m1", 100)
    _source(db, "m2", 110)
    _summary(db, "old", 110, ["m1", "m2"])
    summarizer = _summarizer(db)
    monkeypatch.setattr(summarizer, "_get_all_messages", lambda: [])
    monkeypatch.setattr(summarization, "prioritize_summarizable_sessions", lambda s: s)
    return summarizer


def test_zero_eligible_purge_keeps_old_summary(db, rebuild_rig, monkeypatch):
    monkeypatch.setattr(summarization, "find_budget_overflow_sessions", lambda *_a, **_k: [])

    result = rebuild_rig.rebuild_recent_sessions(purge_existing=True)

    assert result["summaries_purged"] == 0
    assert _summaries(db) == {"old"}
    assert _summarized(db) == {"m1": True, "m2": True}


def test_failed_generation_purge_keeps_old_summary(db, rebuild_rig, monkeypatch):
    monkeypatch.setattr(summarization, "find_budget_overflow_sessions",
                        lambda *_a, **_k: [_session(["m1", "m2"], 100, 110)])
    monkeypatch.setattr(rebuild_rig, "summarize_session",
                        lambda *_a, **_k: {"success": False, "error": "provider unavailable"})

    result = rebuild_rig.rebuild_recent_sessions(
        purge_existing=True, use_heuristic_fallback=False)

    assert result["sessions_summarized"] == 0
    assert result["summaries_purged"] == 0
    assert _summaries(db) == {"old"}
    assert _summarized(db) == {"m1": True, "m2": True}


def test_partial_rebuild_purge_restores_unreplaced_sources(db, rebuild_rig, monkeypatch):
    _source(db, "m3", 300)
    _source(db, "m4", 310)
    _summary(db, "old-unreplaced", 310, ["m3", "m4"])
    sessions = [_session(["m1", "m2"], 100, 110), _session(["m3", "m4"], 300, 310)]
    monkeypatch.setattr(summarization, "find_budget_overflow_sessions", lambda *_a, **_k: sessions)

    def summarize(chunk, **_kwargs):
        if chunk.message_ids[0] == "m3":
            return {"success": False, "error": "provider unavailable"}
        new_id = f"new-{uuid.uuid4().hex[:8]}"
        _summary(db, new_id, 110, list(chunk.message_ids))
        with db.begin() as conn:
            conn.execute(text(f"DELETE FROM {TABLE} WHERE id='old'"))
        return {"success": True, "summary_id": new_id, "replaced_existing": 1}

    monkeypatch.setattr(rebuild_rig, "summarize_session", summarize)

    result = rebuild_rig.rebuild_recent_sessions(purge_existing=True, session_limit=0)

    summaries = _summaries(db)
    assert "old-unreplaced" not in summaries and len(summaries) == 1
    assert result["summaries_purged"] == 1
    # The deleted, unreplaced summary's sources are eligible again; the
    # replaced session's sources stay summarized.
    assert _summarized(db) == {"m1": True, "m2": True, "m3": False, "m4": False}


def test_purge_keeps_legacy_summary_without_sources(db, rebuild_rig):
    _summary(db, "legacy", 200, session_start=190, session_end=210)

    purged = rebuild_rig._purge_historical_summaries(keep_ids={"old"})

    assert purged == 0
    assert _summaries(db) == {"old", "legacy"}
