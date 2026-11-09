"""TASK-980 — timeline index + indexed/session-scoped /v1/history/around.

Hermetic by default (in-memory SQLite + pure helpers). The ``integration``
class runs READ-ONLY checks against live PostgreSQL: the tz/DST bucketing
SQL on literal epochs, and window parity against a full-scan reference on a
real bot partition. It never writes rows or creates partitions.
"""

from __future__ import annotations

from datetime import date, datetime
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
        conn.execute(text("CREATE TABLE sessions (id TEXT, bot_id TEXT, user_id TEXT)"))
        conn.execute(text("INSERT INTO sessions (id, bot_id, user_id) VALUES ('sA', 'bot', 'nick'), ('sB', 'bot', 'other')"))
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


def test_history_pages_keep_timeline_owner_after_jump(window_db, monkeypatch):
    service, rows = window_db
    service.get_memory_client = lambda _bot: SimpleNamespace(get_messages=lambda **_kw: rows)
    monkeypatch.setattr(history_pages, "get_service", lambda: service)
    for name in [
        "_hydrate_attachments_for_page", "_hydrate_reasoning_for_page",
        "_hydrate_turn_meta_for_page", "_hydrate_interrupt_anchors_for_page",
    ]:
        monkeypatch.setattr(history_pages, name, lambda *a: {})
    monkeypatch.setattr(history_pages, "hydrate_scheduler_for_page", lambda *a: {})
    app = FastAPI()
    app.include_router(history_pages.read_router)
    with TestClient(app) as client:
        query = "bot_id=bot&user_id=nick"
        around = client.get(f"/v1/history/around?{query}&message_id=m03&before=2&after=2")
        assert around.status_code == 200
        assert [m["id"] for m in around.json()["messages"]] == ["m01", "m02", "m03", "m04", "m05"]
        assert client.get(f"/v1/history/around?{query}&message_id=m08").status_code == 404
        for cursor in ("after=102", "before=105"):
            page = client.get(f"/v1/history?{query}&{cursor}&limit=20")
            assert page.status_code == 200
            assert all(m["id"] in {f"m{i:02d}" for i in range(6)} for m in page.json()["messages"])
        assert client.get("/v1/history?bot_id=bot&user_id=other&after=105").json()["messages"][0]["id"] == "m06"
        # The legacy unscoped read remains available, but cannot be used to
        # continue an explicitly owned timeline window.
        assert any(m["id"] == "m08" for m in client.get("/v1/history?bot_id=bot").json()["messages"])


def test_around_route_session_scope_and_404(window_db, monkeypatch):
    service, _ = window_db
    monkeypatch.setattr(history_pages, "get_service", lambda: service)
    for name in [
        "_hydrate_attachments_for_page", "_hydrate_reasoning_for_page",
        "_hydrate_turn_meta_for_page", "_hydrate_interrupt_anchors_for_page",
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
    first = {"s2": {"first_message_text": "why is it slow\nmore detail",
                    "initiated_by": {"kind": "user", "id": "nick"}}}
    days, sessions, version = build_timeline(groups, meta, first)

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
    assert (s2.title, s2.label_source) == ("why is it slow more detail", "first_prompt")
    assert s2.first_message_text == "why is it slow\nmore detail"
    assert s2.initiated_by == {"kind": "user", "id": "nick"}
    assert s1.first_message_text is None and s1.initiated_by is None
    assert version == "10:45.0"


def test_build_timeline_empty():
    assert build_timeline([], {}, {}) == ([], [], "0:0.0")


def test_prompts_route_validates_inputs(monkeypatch):
    app = FastAPI()
    app.include_router(history_timeline.router)
    with TestClient(app) as client:
        assert client.get("/v1/history/timeline/prompts?bot_id=b").status_code == 422
        assert client.get("/v1/history/timeline?bot_id=b").status_code == 422
        assert client.get("/v1/history/timeline/prompts?bot_id=b&user_id=u").status_code == 400
        assert client.get("/v1/history/timeline/prompts?bot_id=b&user_id=u&date=2026-13-40").status_code == 400
        assert client.get("/v1/history/timeline/prompts?bot_id=b&user_id=u&date=2026-01-01&tz=Nope/Zone").status_code == 400
        assert client.get("/v1/history/timeline?bot_id=b&user_id=u&tz=Nope/Zone").status_code == 400
        assert client.get("/v1/history/timeline?bot_id=b&user_id=u&mode=custom").status_code == 400


def test_default_timeline_is_exact_seven_local_days(monkeypatch):
    from datetime import timedelta

    calls = []
    def rows(_bot, _user, _session, _tz, start, end, recent, overview_only, _cursor):
        calls.append((start, end, recent, overview_only))
        return [], {}, {}, [], start

    monkeypatch.setattr(history_timeline, "_load_timeline_rows", rows)
    app = FastAPI()
    app.include_router(history_timeline.router)
    with TestClient(app) as client:
        response = client.get("/v1/history/timeline?bot_id=snark&user_id=nick&tz=America/New_York")
        assert response.status_code == 200
        coverage = response.json()["coverage"]
        assert (date.fromisoformat(coverage["to_date"]) - date.fromisoformat(coverage["from_date"])) == timedelta(days=6)
        assert calls[0][2:] == (False, False)  # never extend the seven-day default
        wider_start = (datetime.now(ET).date() - timedelta(days=29)).isoformat()
        wider = client.get(f"/v1/history/timeline?bot_id=snark&user_id=nick&mode=recent&from_date={wider_start}")
        assert wider.status_code == 200
        assert calls[1][2:] == (True, False)  # deliberate wider preset retains fallback


def test_scoped_timeline_dst_and_range_bounds():
    from llm_bawt.service.routes.history_scope import local_day_bounds, owned_message_scope

    spring = local_day_bounds(date(2026, 3, 8), date(2026, 3, 8), "America/New_York")
    autumn = local_day_bounds(date(2026, 11, 1), date(2026, 11, 1), "America/New_York")
    assert spring[1] - spring[0] == 23 * 3600
    assert autumn[1] - autumn[0] == 25 * 3600
    assert spring[0] == _et_epoch(2026, 3, 8)
    assert "owner.user_id = :user_id" in owned_message_scope()
    assert "owner.bot_id = :bot_id" in owned_message_scope()
    with pytest.raises(HTTPException):
        local_day_bounds(date(2026, 1, 1), date(2026, 4, 1), "America/New_York")


def test_timeline_repository_constant_sql_and_owner_predicates():
    from llm_bawt.service.routes.history_timeline_store import timeline_rows

    class Result:
        def __init__(self, values):
            self.values = values
        def mappings(self):
            return self
        def all(self):
            return list(self.values)
        def __iter__(self):
            return iter(self.values)

    class Connection:
        def __init__(self):
            self.calls = []
        def execute(self, statement, params):
            sql = str(statement)
            self.calls.append((sql, params))
            if "WITH groups" in sql:
                return Result([{"day": date(2026, 10, 6),
                    "session_id": "own" if i == 0 else f"own-{i}", "first_id": f"id-{i}",
                    "n": 2, "prompts": 1, "first_ts": 20., "last_ts": 30., "title": "Hi",
                    "status": "active", "continued": False} for i in range(count)])
            if "AS month" in sql:
                return Result([{"month": date(2026, 10, 1), "messages": 2}])
            if "AS k(id, bot_id, user_id)" in sql:
                return Result([])
            if "AS summary_count" in sql:
                return Result([{"day": date(2026, 10, 6), "session_id": "own",
                                "summary_count": 2, "summary_revision": "revision"}])
            raise AssertionError("Unexpected query")
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False

    class Engine:
        def __init__(self):
            self.conn = Connection()
        def connect(self):
            return self.conn

    for count in (10, 1000):
        engine = Engine()
        result = timeline_rows(engine, "messages_p_snark", "snark", "nick", None,
                               "America/New_York", 1., 100.)
        assert len(result[0]) == count
        assert len(engine.conn.calls) == 4
        assert result[0][0]["summary_count"] == 2
        assert result[0][0]["summary_revision"] == "revision"
        owned = [(sql, p) for sql, p in engine.conn.calls
                 if "AS k(id, bot_id, user_id)" not in sql and "AS summary_count" not in sql]
        assert len(owned) == 2
        assert all("owner.user_id = :user_id" in sql for sql, _ in owned)
        assert all(params["user_id"] == "nick" for _, params in owned)
        # TASK-2061: the shared first-message read joins sessions on the owner key.
        (first_sql, first_params), = [c for c in engine.conn.calls
                                    if "AS k(id, bot_id, user_id)" in c[0]]
        assert "s.user_id IS NOT DISTINCT FROM k.user_id" in first_sql
        assert len(first_params["ids"]) == count and "own" in first_params["ids"]
        assert first_params["users"] == ["nick"] * count
        assert first_params["bots"] == ["snark"] * count
        group_sql, group_params = engine.conn.calls[0]
        assert "m.timestamp >= :start AND m.timestamp < :end" in group_sql
        assert (group_params["start"], group_params["end"]) == (1., 100.)


def test_timeline_summary_schema_defaults_and_cap():
    from pydantic import ValidationError
    from llm_bawt.service.schemas_history_timeline import TimelinePromptsResponse, TimelineStoredSummary

    response = TimelinePromptsResponse(bot_id="b", user_id="u", date=None,
        session_id="s", tz="UTC", prompts=[], truncated=False)
    assert response.summaries == [] and response.summaries_truncated is False
    fields = dict(id="summary", session_id="s", excerpt="x" * 1200,
                  coverage="conversation", truncated=True)
    stored = TimelineStoredSummary(**fields)
    assert stored.source_start_utc is None and stored.source_message_count is None
    with pytest.raises(ValidationError):
        TimelineStoredSummary(**{**fields, "excerpt": "x" * 1201})


def test_summary_counts_fold_without_prose():
    groups = [{"day": "2026-09-01", "session_id": sid, "n": 1, "prompts": 0,
               "first_ts": ts, "last_ts": ts, "first_id": sid, "summary_count": n}
              for sid, ts, n in [("one", 1., 2), ("two", 2., 3)]]
    days, _, _ = build_timeline(groups, {}, {})
    assert days[0].summary_count == 5
    assert "excerpt" not in days[0].model_dump()


def test_index_version_includes_summary_revision(monkeypatch):
    group = {"day": "2026-09-01", "session_id": "s", "n": 1, "prompts": 0,
             "first_ts": 1., "last_ts": 1., "first_id": "m", "continued": False,
             "summary_count": 1, "summary_revision": "first"}
    monkeypatch.setattr(history_timeline, "_load_timeline_rows",
                        lambda *a: ([group], {}, {}, [], a[4]))
    app = FastAPI()
    app.include_router(history_timeline.router)
    with TestClient(app) as client:
        first = client.get("/v1/history/timeline?bot_id=b&user_id=u").json()
        group["summary_revision"] = "edited"
        second = client.get("/v1/history/timeline?bot_id=b&user_id=u").json()
    assert first["version"] != second["version"]
    assert first["days"][0]["summary_count"] == 1


def test_timeline_anchors_reject_oversize_without_sql():
    from llm_bawt.service.routes.history_timeline_store import anchor_rows
    with pytest.raises(HTTPException) as error:
        anchor_rows(None, "messages_p_test", "bot", "user", None, ["x"] * 2001)
    assert error.value.status_code == 413


@pytest.mark.parametrize("from_time,to_time,expected", [
    (None, None, (0, 24 * 3600)),
    ("09:15", None, (9 * 3600 + 15 * 60, 24 * 3600)),
    (None, "17:30", (0, 17 * 3600 + 30 * 60)),
    ("09:15", "17:30", (9 * 3600 + 15 * 60, 17 * 3600 + 30 * 60)),
])
def test_custom_clock_bounds_defaults(from_time, to_time, expected):
    from llm_bawt.service.routes.history_scope import local_clock_bounds, local_day_bounds

    day = date(2026, 9, 1)
    midnight, _ = local_day_bounds(day, day, "America/New_York")
    start, end = local_clock_bounds(day, day, "America/New_York", from_time, to_time)
    assert (start - midnight, end - midnight) == expected


@pytest.mark.parametrize("from_time,to_time", [
    ("9:00", None), ("24:00", None), ("09:60", None),
    ("09:00:00", None), (" 09:00", None), ("０９:00", None),
    ("", None), (None, ""), (None, "noon"),
    ("12:00", "12:00"), ("18:00", "09:00"), (None, "00:00"),
])
def test_custom_clock_bounds_reject_invalid(from_time, to_time):
    from llm_bawt.service.routes.history_scope import local_clock_bounds

    with pytest.raises(HTTPException) as exc:
        local_clock_bounds(date(2026, 9, 1), date(2026, 9, 1),
                           "America/New_York", from_time, to_time)
    assert exc.value.status_code == 400


def test_custom_clock_bounds_dst_and_date_limit():
    from datetime import timedelta
    from llm_bawt.service.routes.history_scope import local_clock_bounds

    spring = date(2026, 3, 8)
    for start, end in [("02:30", None), (None, "02:30")]:
        with pytest.raises(HTTPException) as exc:
            local_clock_bounds(spring, spring, "America/New_York", start, end)
        assert exc.value.status_code == 400
    assert local_clock_bounds(spring, spring, "America/New_York", "01:30", "03:30") == (
        _et_epoch(2026, 3, 8, 1, 30), _et_epoch(2026, 3, 8, 3, 30))
    autumn = date(2026, 11, 1)
    # Even equal wall clocks delimit one real hour: earlier start, later end.
    start, end = local_clock_bounds(autumn, autumn, "America/New_York", "01:30", "01:30")
    assert (start, end) == (_et_epoch(2026, 11, 1, 1, 30, fold=0),
                            _et_epoch(2026, 11, 1, 1, 30, fold=1))
    assert end - start == 3600
    first = date(2026, 1, 1)
    assert local_clock_bounds(first, first + timedelta(days=89), "UTC", "12:00", "13:00")
    for last in (first - timedelta(days=1), first + timedelta(days=90)):
        with pytest.raises(HTTPException):
            local_clock_bounds(first, last, "UTC", "12:00", "13:00")


def test_custom_clocks_route_passes_exact_index_bounds(monkeypatch):
    from llm_bawt.service.routes.history_scope import local_clock_bounds

    calls = []

    def rows(*args):
        calls.append(args)
        return [], {}, {}, [], args[4]

    monkeypatch.setattr(history_timeline, "_load_timeline_rows", rows)
    app = FastAPI()
    app.include_router(history_timeline.router)
    with TestClient(app) as client:
        response = client.get("/v1/history/timeline", params={
            "bot_id": "synthetic", "user_id": "viewer", "session_id": "own",
            "mode": "custom", "tz": "America/New_York",
            "from_date": "2026-09-01", "to_date": "2026-09-02",
            "from_time": "09:15", "to_time": "17:30",
        })
    assert response.status_code == 200
    start, end = local_clock_bounds(date(2026, 9, 1), date(2026, 9, 2),
                                    "America/New_York", "09:15", "17:30")
    assert calls == [("synthetic", "viewer", "own", "America/New_York",
                      start, end, False, False, None)]
    assert (response.json()["from_utc"], response.json()["to_utc"]) == (start, end)
    assert response.json()["coverage"]["from_date"] == "2026-09-01"


@pytest.mark.parametrize("extra", [
    {"mode": "recent", "from_time": "09:00"},
    {"mode": "all", "to_time": "17:00"},
    {"from_time": ""},
    {"mode": "custom", "from_time": "09:00"},
    {"mode": "custom", "from_date": "2026-03-08", "to_date": "2026-03-08", "from_time": "02:30"},
    {"mode": "custom", "from_date": "2026-09-01", "to_date": "2026-09-01", "to_time": "00:00"},
    {"mode": "custom", "from_date": "2026-01-01", "to_date": "2026-04-01", "from_time": "12:00"},
])
def test_timeline_invalid_clocks_never_query(monkeypatch, extra):
    def unexpected(*args):
        pytest.fail("Invalid clock range reached SQL")

    monkeypatch.setattr(history_timeline, "_load_timeline_rows", unexpected)
    app = FastAPI()
    app.include_router(history_timeline.router)
    with TestClient(app) as client:
        response = client.get("/v1/history/timeline", params={
            "bot_id": "synthetic", "user_id": "viewer", "tz": "America/New_York", **extra})
    assert response.status_code == 400


@pytest.mark.parametrize("bounds,status", [
    ({"from_utc": "1"}, 400), ({"to_utc": "2"}, 400),
    ({"from_utc": "1", "to_utc": "1"}, 400),
    ({"from_utc": "2", "to_utc": "1"}, 400),
    ({"from_utc": "nan", "to_utc": "2"}, 400),
    ({"from_utc": "1", "to_utc": "inf"}, 400),
    ({"from_utc": "-inf", "to_utc": "2"}, 400),
    ({"from_utc": "1", "to_utc": "1e999"}, 400),
    ({"from_utc": "bad", "to_utc": "2"}, 422),
])
def test_prompt_utc_bounds_validation_never_queries(monkeypatch, bounds, status):
    def unexpected(*args):
        pytest.fail("Invalid UTC pair reached SQL")

    monkeypatch.setattr(history_timeline, "_engine_and_table", unexpected)
    app = FastAPI()
    app.include_router(history_timeline.router)
    with TestClient(app) as client:
        response = client.get("/v1/history/timeline/prompts", params={
            "bot_id": "synthetic", "user_id": "viewer", "session_id": "own", **bounds})
        assert response.status_code == status
        # A pair is not a substitute for owned day/thread selection.
        assert client.get("/v1/history/timeline/prompts", params={
            "bot_id": "synthetic", "user_id": "viewer", "from_utc": 1, "to_utc": 2,
        }).status_code == 400


@pytest.fixture
def prompt_bounds_db(monkeypatch):
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    midnight = _et_epoch(2026, 9, 1)
    rows = [
        ("before-day", "own", "user", midnight - 1),
        ("before", "own", "user", midnight + 8 * 3600),
        ("start", "own", "user", midnight + 9 * 3600),
        ("inside", "own", "user", midnight + 12 * 3600),
        ("end", "own", "user", midnight + 17 * 3600),
        ("after-day", "own", "user", midnight + 25 * 3600),
        ("assistant", "own", "assistant", midnight + 10 * 3600),
        ("other-thread", "sibling", "user", midnight + 10 * 3600),
        ("foreign-user", "foreign-user", "user", midnight + 10 * 3600),
        ("foreign-bot", "foreign-bot", "user", midnight + 10 * 3600),
        ("null-owner", "null-owner", "user", midnight + 10 * 3600),
        ("orphan", "missing", "user", midnight + 10 * 3600),
    ]
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE sessions (id TEXT, bot_id TEXT, user_id TEXT)"))
        conn.execute(text("INSERT INTO sessions VALUES "
                          "('own', 'synthetic', 'viewer'), ('sibling', 'synthetic', 'viewer'), "
                          "('foreign-user', 'synthetic', 'other'), ('foreign-bot', 'other', 'viewer'), "
                          "('null-owner', 'synthetic', NULL)"))
        conn.execute(text("CREATE TABLE messages_test "
                          "(id TEXT, session_id TEXT, role TEXT, timestamp REAL, content TEXT)"))
        conn.execute(text("INSERT INTO messages_test VALUES (:id, :session, :role, :ts, :id)"),
                     [{"id": id_, "session": sid, "role": role, "ts": ts} for id_, sid, role, ts in rows])
    monkeypatch.setattr(history_timeline, "_engine_and_table", lambda _bot: (engine, "messages_test"))
    app = FastAPI()
    app.include_router(history_timeline.router)
    with TestClient(app) as client:
        yield client, midnight
    engine.dispose()


def test_prompt_utc_pair_intersects_day_thread_and_ownership(prompt_bounds_db):
    client, midnight = prompt_bounds_db
    params = {"bot_id": "synthetic", "user_id": "viewer", "tz": "America/New_York",
              "date": "2026-09-01", "session_id": "own"}

    def ids(**extra):
        response = client.get("/v1/history/timeline/prompts", params={**params, **extra})
        assert response.status_code == 200
        return [p["id"] for p in response.json()["prompts"]]

    assert ids() == ["before", "start", "inside", "end"]  # old whole-day behavior
    assert ids(from_utc=midnight + 9 * 3600, to_utc=midnight + 17 * 3600) == ["start", "inside"]
    assert ids(from_utc=midnight - 10, to_utc=midnight + 26 * 3600) == ["before", "start", "inside", "end"]
    assert ids(from_utc=midnight - 10, to_utc=midnight) == []  # valid disjoint intersection
    assert ids(date=None, from_utc=midnight + 9 * 3600,
               to_utc=midnight + 17 * 3600) == ["start", "inside"]
    assert ids(session_id=None, from_utc=midnight + 9 * 3600,
               to_utc=midnight + 17 * 3600) == ["start", "other-thread", "inside"]
    response = client.get("/v1/history/timeline/prompts", params={
        **params, "from_utc": midnight + 9 * 3600, "to_utc": midnight + 17 * 3600, "limit": 1})
    assert response.status_code == 200
    assert [p["id"] for p in response.json()["prompts"]] == ["start"]
    assert response.json()["truncated"] is True


# ── integration: live PostgreSQL, read-only ───────────────────────────────


def _et_epoch(*parts, fold=0):
    return datetime(*parts, tzinfo=ET, fold=fold).timestamp()


@pytest.mark.integration
@pytest.mark.external
class TestLivePostgres:
    @pytest.fixture(scope="class")
    def engine(self):
        from tests.pg_isolation import require_test_db_config

        engine = assets._build_engine(require_test_db_config())
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

    @pytest.mark.parametrize("bot", ["snark"])
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
