"""TASK-980 — timeline index + indexed/session-scoped /v1/history/around.

Hermetic by default (in-memory SQLite + pure helpers). The ``integration``
class runs READ-ONLY checks against live PostgreSQL: the tz/DST bucketing
SQL on literal epochs, and window parity against a full-scan reference on a
real bot partition. It never writes rows or creates partitions.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from llm_bawt.media import assets
from llm_bawt.service.routes import history_pages, history_timeline
from llm_bawt.service.routes.history_timeline import (
    LOCAL_DATE_SQL,
    LOCAL_DAY_END_SQL,
    LOCAL_DAY_START_SQL,
    build_timeline,
    resolve_session_label,
    snippet,
    validate_tz,
)

ET = ZoneInfo("America/New_York")


def _reference_window(rows, anchor_id, before, after, session_id=None):
    """The pre-TASK-980 algorithm: full sorted scan + index slice."""
    visible = [
        r for r in rows
        if r["role"] not in ("system", "summary")
        and (session_id is None or r["session_id"] == session_id)
    ]
    visible.sort(key=lambda r: (r["timestamp"], r["id"]))
    idx = next(i for i, r in enumerate(visible) if r["id"] == anchor_id)
    start, end = max(0, idx - before), min(len(visible), idx + after + 1)
    return [r["id"] for r in visible[start:end]], start > 0, end < len(visible)


# ── hermetic: indexed window ──────────────────────────────────────────────


@pytest.fixture
def window_db(monkeypatch):
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    rows = []
    # 12 messages, two threads interleaved, a tied timestamp pair, plus
    # system/summary rows that must be invisible.
    for i in range(12):
        rows.append({
            "id": f"m{i:02d}",
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"msg {i}",
            "timestamp": 100.0 + (i if i != 7 else 6),  # m06/m07 tie
            "session_id": "sA" if i < 6 else "sB",
        })
    rows.append({"id": "sys", "role": "system", "content": "x", "timestamp": 103.5, "session_id": "sA"})
    rows.append({"id": "sum", "role": "summary", "content": "x", "timestamp": 108.5, "session_id": "sB"})
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE messages_p_bot (id TEXT, role TEXT, content TEXT, timestamp REAL, "
            "session_id TEXT, author_entity_type TEXT, author_entity_id TEXT)"
        ))
        for r in rows:
            conn.execute(text(
                "INSERT INTO messages_p_bot (id, role, content, timestamp, session_id) "
                "VALUES (:id, :role, :content, :timestamp, :session_id)"
            ), r)
    monkeypatch.setattr(assets, "_build_engine", lambda config: engine)
    monkeypatch.setattr(history_pages, "_hydrate_window_authors", lambda rows, bot_id: rows)
    service = SimpleNamespace(config=SimpleNamespace())
    yield service, rows
    engine.dispose()


@pytest.mark.parametrize("anchor", ["m00", "m03", "m06", "m07", "m11"])
@pytest.mark.parametrize("before,after", [(0, 0), (2, 1), (3, 3), (50, 50)])
@pytest.mark.parametrize("session_id", [None, "sA", "sB"])
def test_window_matches_full_scan_reference(window_db, anchor, before, after, session_id):
    service, rows = window_db
    in_scope = session_id is None or next(r for r in rows if r["id"] == anchor)["session_id"] == session_id
    if not in_scope:
        with pytest.raises(history_pages._WindowAnchorNotFound):
            history_pages._load_window_via_sql(service, "bot", anchor, before, after, session_id)
        return
    page, has_older, has_newer = history_pages._load_window_via_sql(
        service, "bot", anchor, before, after, session_id
    )
    assert ([m["id"] for m in page], has_older, has_newer) == _reference_window(
        rows, anchor, before, after, session_id
    )


def test_window_excludes_system_and_summary_anchors(window_db):
    service, _ = window_db
    for hidden in ("sys", "sum", "nope"):
        with pytest.raises(history_pages._WindowAnchorNotFound):
            history_pages._load_window_via_sql(service, "bot", hidden, 5, 5)


def test_around_route_session_scope_and_404(window_db, monkeypatch):
    service, _ = window_db
    monkeypatch.setattr(history_pages, "get_service", lambda: service)
    for name in [
        "_hydrate_attachments_for_page", "_hydrate_reasoning_for_page",
        "_hydrate_reply_links_for_page", "_hydrate_interrupt_anchors_for_page",
    ]:
        monkeypatch.setattr(history_pages, name, lambda *a: {})
    monkeypatch.setattr(history_pages, "hydrate_scheduler_for_page", lambda *a: {})
    app = FastAPI()
    app.include_router(history_pages.read_router)
    with TestClient(app) as client:
        body = client.get("/v1/history/around?bot_id=bot&message_id=m08&before=5&after=5&session_id=sB").json()
        assert [m["id"] for m in body["messages"]] == [f"m{i:02d}" for i in range(6, 12)]
        assert body["has_older"] is False and body["has_more"] is False
        assert body["has_newer"] is False and body["anchor_id"] == "m08"
        assert body["oldest_timestamp"] == 106.0 and body["newest_timestamp"] == 111.0
        assert client.get("/v1/history/around?bot_id=bot&message_id=m02&session_id=sB").status_code == 404


# ── hermetic: pure timeline logic ─────────────────────────────────────────


def test_session_label_precedence():
    assert resolve_session_label("  Fix the rail ", "hello") == ("Fix the rail", "title")
    assert resolve_session_label(None, "real  prompt\nhere") == ("real prompt here", "first_prompt")
    assert resolve_session_label("", "  /new") == (None, "none")
    assert resolve_session_label(None, None) == (None, "none")


def test_snippet_caps_and_collapses():
    assert snippet("a\n\n b") == "a b"
    out = snippet("x" * 500)
    assert len(out) == 120 and out.endswith("…")


def test_validate_tz():
    assert validate_tz("America/New_York") == "America/New_York"
    with pytest.raises(HTTPException) as exc:
        validate_tz("Mars/Olympus")
    assert exc.value.status_code == 400


def test_build_timeline_folds_groups():
    groups = [
        # day 1: session s1 spans into day 2; an orphan (no-session) row.
        {"day": "2026-09-01", "session_id": "s1", "n": 4, "prompts": 2,
         "first_ts": 10.0, "last_ts": 20.0, "first_id": "a"},
        {"day": "2026-09-01", "session_id": None, "n": 1, "prompts": 1,
         "first_ts": 5.0, "last_ts": 5.0, "first_id": "orphan"},
        {"day": "2026-09-02", "session_id": "s2", "n": 2, "prompts": 1,
         "first_ts": 40.0, "last_ts": 45.0, "first_id": "c"},
        {"day": "2026-09-02", "session_id": "s1", "n": 3, "prompts": 1,
         "first_ts": 30.0, "last_ts": 35.0, "first_id": "b"},
    ]
    meta = {"s1": {"title": "Rail work", "status": "archived"}, "s2": {"title": None, "status": "active"}}
    days, sessions, version = build_timeline(groups, meta, {"s2": "why is it slow"})

    assert [d.date for d in days] == ["2026-09-01", "2026-09-02"]
    d1, d2 = days
    assert (d1.first_message_id, d1.first_ts, d1.message_count, d1.user_prompt_count) == ("orphan", 5.0, 5, 3)
    assert d1.session_ids == ["s1"]
    assert d2.session_ids == ["s1", "s2"]  # ordered by first activity that day
    assert (d2.first_message_id, d2.last_ts) == ("b", 45.0)
    # Per-day conversation segments: the rail's blips and their jump targets.
    assert [(s.session_id, s.first_message_id, s.first_ts, s.last_ts, s.message_count, s.user_prompt_count)
            for s in d2.segments] == [("s1", "b", 30.0, 35.0, 3, 1), ("s2", "c", 40.0, 45.0, 2, 1)]
    assert [s.first_message_id for s in d1.segments] == ["a"]  # orphan rows get no segment

    assert [s.id for s in sessions] == ["s1", "s2"]
    s1, s2 = sessions
    assert (s1.first_message_id, s1.last_ts, s1.message_count) == ("a", 35.0, 7)
    assert (s1.title, s1.label_source, s1.status) == ("Rail work", "title", "archived")
    assert (s2.title, s2.label_source) == ("why is it slow", "first_prompt")
    assert version == "10:45.0"


def test_build_timeline_empty():
    assert build_timeline([], {}, {}) == ([], [], "0:0.0")


def test_prompts_route_validates_inputs(monkeypatch):
    app = FastAPI()
    app.include_router(history_timeline.router)
    with TestClient(app) as client:
        assert client.get("/v1/history/timeline/prompts?bot_id=b").status_code == 400
        assert client.get("/v1/history/timeline/prompts?bot_id=b&date=2026-13-40").status_code == 400
        assert client.get("/v1/history/timeline/prompts?bot_id=b&date=2026-01-01&tz=Nope/Zone").status_code == 400
        assert client.get("/v1/history/timeline?bot_id=b&tz=Nope/Zone").status_code == 400


# ── integration: live PostgreSQL, read-only ───────────────────────────────


def _et_epoch(*parts, fold=0):
    return datetime(*parts, tzinfo=ET, fold=fold).timestamp()


@pytest.mark.integration
class TestLivePostgres:
    @pytest.fixture(scope="class")
    def engine(self):
        from llm_bawt.utils.config import Config

        engine = assets._build_engine(Config())
        if engine is None:
            pytest.skip("no database")
        return engine

    @pytest.mark.parametrize("epoch,expected", [
        (_et_epoch(2026, 9, 28, 23, 59, 59), "2026-09-28"),   # ET late evening = next day UTC
        (_et_epoch(2026, 9, 29, 0, 0, 0), "2026-09-29"),
        (_et_epoch(2026, 3, 8, 1, 59, 59), "2026-03-08"),     # spring-forward day
        (_et_epoch(2026, 3, 8, 3, 0, 0), "2026-03-08"),
        (_et_epoch(2026, 3, 8, 23, 59, 59), "2026-03-08"),
        (_et_epoch(2026, 11, 1, 1, 30, fold=0), "2026-11-01"),  # fall-back, 1st 01:30 (EDT)
        (_et_epoch(2026, 11, 1, 1, 30, fold=1), "2026-11-01"),  # 2nd 01:30 (EST)
        (_et_epoch(2026, 11, 1, 23, 59, 59), "2026-11-01"),
        (_et_epoch(2026, 11, 2, 0, 0, 0), "2026-11-02"),
    ])
    def test_local_date_bucketing(self, engine, epoch, expected):
        with engine.connect() as conn:
            got = conn.execute(
                text(f"SELECT {LOCAL_DATE_SQL} FROM (SELECT CAST(:ts AS double precision) AS timestamp) t"),
                {"ts": epoch, "tz": "America/New_York"},
            ).scalar()
        assert str(got) == expected

    @pytest.mark.parametrize("day,hours", [("2026-03-08", 23), ("2026-11-01", 25), ("2026-09-29", 24)])
    def test_day_range_matches_bucket_and_dst_length(self, engine, day, hours):
        with engine.connect() as conn:
            start, end = conn.execute(
                text(f"SELECT {LOCAL_DAY_START_SQL}, {LOCAL_DAY_END_SQL}"),
                {"day": day, "tz": "America/New_York"},
            ).one()
        assert float(start) == _et_epoch(*map(int, day.split("-")))
        assert float(end) - float(start) == hours * 3600

    @pytest.mark.parametrize("bot", ["snark", "mira"])
    def test_window_parity_with_full_scan(self, engine, bot, monkeypatch):
        from llm_bawt.memory.postgresql import MESSAGES_PARENT, partition_name

        table = partition_name(MESSAGES_PARENT, bot)
        with engine.connect() as conn:
            try:
                all_rows = [dict(r) for r in conn.execute(text(
                    f"SELECT id, role, timestamp, session_id FROM {table} "
                    "WHERE role NOT IN ('system', 'summary') ORDER BY timestamp, id"
                )).mappings().all()]
            except Exception:
                pytest.skip(f"no partition for {bot}")
        if len(all_rows) < 50:
            pytest.skip("not enough history")
        monkeypatch.setattr(assets, "_build_engine", lambda config: engine)
        monkeypatch.setattr(history_pages, "_hydrate_window_authors", lambda rows, bot_id: rows)
        service = SimpleNamespace(config=None)
        picks = [0, 1, len(all_rows) // 3, len(all_rows) // 2, len(all_rows) - 2, len(all_rows) - 1]
        for idx in picks:
            anchor = all_rows[idx]
            for session_id in (None, anchor["session_id"]):
                page, has_older, has_newer = history_pages._load_window_via_sql(
                    service, bot, anchor["id"], 30, 10, session_id
                )
                scoped = [r for r in all_rows if session_id is None or r["session_id"] == session_id]
                pos = next(i for i, r in enumerate(scoped) if r["id"] == anchor["id"])
                start, end = max(0, pos - 30), min(len(scoped), pos + 11)
                assert [m["id"] for m in page] == [r["id"] for r in scoped[start:end]]
                assert (has_older, has_newer) == (start > 0, end < len(scoped))
