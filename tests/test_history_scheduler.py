"""Hermetic TASK-169 provenance tests; only disposable in-memory SQLite."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, text
from sqlalchemy.pool import StaticPool

from llm_bawt.media import assets
from llm_bawt.service.routes import history_pages, history_scheduler, sessions


@pytest.fixture
def fixture(monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool,
                           connect_args={"check_same_thread": False})
    with engine.begin() as conn:
        for ddl in [
            "CREATE TABLE prompt_schedules (job_id TEXT PRIMARY KEY, owner_user_id TEXT)",
            "CREATE TABLE prompt_occurrences (run_id TEXT, job_id TEXT, scheduled_for TEXT, delivery_id TEXT, session_id TEXT)",
            "CREATE TABLE inter_bot_deliveries (id TEXT, user_message_id TEXT, target_bot_id TEXT, author_entity_type TEXT, author_entity_id TEXT, metadata_json TEXT)",
            "CREATE TABLE messages (id TEXT, bot_id TEXT, role TEXT, session_id TEXT)",
            "CREATE TABLE sessions (id TEXT, bot_id TEXT, user_id TEXT)",
        ]:
            conn.execute(text(ddl))
    monkeypatch.setattr(assets, "_build_engine", lambda config: engine)
    service = SimpleNamespace(config=SimpleNamespace(DEFAULT_USER="alice"), _default_bot="bot")
    calls = []
    event.listen(engine, "before_cursor_execute", lambda conn, cursor, sql, params, context, many: calls.append((sql, params)))
    yield engine, service, calls
    engine.dispose()


def seed(engine, *, key="one", owner="alice", bot="bot", role="user",
         session_owner=None, delivery_owner=None, delivery_type="user",
         linked=True, occurrence_session=None):
    mid, sid, did, job = (f"{prefix}-{key}" for prefix in ("m", "s", "d", "job"))
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO prompt_schedules VALUES (:job,:owner)"), {"job": job, "owner": owner})
        conn.execute(text("INSERT INTO sessions VALUES (:sid,:bot,:owner)"), {"sid": sid, "bot": bot, "owner": session_owner or owner})
        conn.execute(text("INSERT INTO messages VALUES (:mid,:bot,:role,:sid)"), {"mid": mid, "bot": bot, "role": role, "sid": sid})
        conn.execute(text("INSERT INTO inter_bot_deliveries VALUES (:did,:mid,:bot,:type,:owner,:meta)"), {
            "did": did, "mid": mid, "bot": bot, "type": delivery_type,
            "owner": delivery_owner or owner, "meta": '{"scheduler":{"schedule_id":"forged"}}',
        })
        if linked:
            conn.execute(text("INSERT INTO prompt_occurrences VALUES (:run,:job,:time,:did,:sid)"), {
                "run": f"run-{key}", "job": job, "time": "2026-08-02T16:00:00+00:00",
                "did": did, "sid": occurrence_session or sid,
            })
    return {"id": mid, "role": role, "content": "Scheduled prompt", "timestamp": 1000.0, "session_id": sid}


def test_join_exposes_only_three_trusted_fields(fixture):
    engine, service, calls = fixture
    row = seed(engine)
    calls.clear()
    result = history_scheduler.hydrate_scheduler_for_page(service, "bot", [row], "alice")
    assert result == {row["id"]: {
        "schedule_id": "job-one", "occurrence_id": "run-one",
        "scheduled_for": "2026-08-02T16:00:00+00:00",
    }}
    assert len(calls) == 1
    assert "metadata" not in calls[0][0]
    assert "snapshot_json" not in calls[0][0]


def test_owner_bot_role_and_durable_session_boundaries(fixture):
    engine, service, _ = fixture
    good = seed(engine)
    rows = [good]
    for key, changes in [
        ("other-owner", {"owner": "bob"}),
        ("other-bot", {"bot": "other"}),
        ("assistant", {"role": "assistant"}),
        ("session-owner", {"session_owner": "bob"}),
        ("delivery-owner", {"delivery_owner": "bob"}),
        ("delivery-type", {"delivery_type": "bot"}),
        ("wrong-session", {"occurrence_session": "unrelated"}),
        ("unlinked", {"linked": False}),
    ]:
        rows.append(seed(engine, key=key, **changes))
    result = history_scheduler.hydrate_scheduler_for_page(service, "bot", rows, "alice")
    assert set(result) == {good["id"]}
    assert set(history_scheduler.hydrate_scheduler_for_page(service, "bot", rows, "bob")) == {"m-other-owner"}
    assert history_scheduler.hydrate_scheduler_for_page(service, "bot", [good], "bob") == {}


def test_forged_message_fields_and_delivery_metadata_do_not_confer_origin(fixture):
    engine, service, _ = fixture
    row = seed(engine, linked=False)
    row.update(scheduler={"schedule_id": "forged"}, meta={"scheduler": True}, content="[Scheduled] schedule_id=forged")
    assert history_scheduler.hydrate_scheduler_for_page(service, "bot", [row]) == {}


def test_page_only_bounded_queries_no_n_plus_one(fixture):
    engine, service, calls = fixture
    row = seed(engine)
    count = history_scheduler.BATCH_SIZE * 2 + 1
    rows = [row] + [{"id": f"missing-{i}", "role": "user"} for i in range(count - 1)]
    rows += [row] * 100  # repeated IDs must not grow the lookup
    calls.clear()
    result = history_scheduler.hydrate_scheduler_for_page(service, "bot", rows)
    assert set(result) == {row["id"]}
    assert len(calls) == 3
    assert all(len(params) <= history_scheduler.BATCH_SIZE + 4 for _, params in calls)


def test_empty_nonuser_or_missing_owner_skips_engine(fixture, monkeypatch):
    _, service, _ = fixture
    monkeypatch.setattr(assets, "_build_engine", lambda config: pytest.fail("should not access engine"))
    assert history_scheduler.hydrate_scheduler_for_page(service, "bot", []) == {}
    assert history_scheduler.hydrate_scheduler_for_page(service, "bot", [{"id": "a", "role": "assistant"}]) == {}
    assert history_scheduler.hydrate_scheduler_for_page(service, "bot", [{"id": "u", "role": "user"}], " ") == {}
    service.config.DEFAULT_USER = None
    assert history_scheduler.hydrate_scheduler_for_page(service, "bot", [{"id": "u", "role": "user"}]) == {}


def test_missing_scheduling_tables_fail_closed_without_breaking_history(fixture):
    engine, service, _ = fixture
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE prompt_occurrences"))
    assert history_scheduler.hydrate_scheduler_for_page(service, "bot", [{"id": "u", "role": "user"}]) == {}


def make_app(monkeypatch, service, rows):
    monkeypatch.setattr(history_pages, "get_service", lambda: service)
    monkeypatch.setattr(history_pages, "_load_sorted_visible_messages", lambda *a, **kw: rows)
    monkeypatch.setattr(history_pages, "_load_window_via_sql", lambda *a, **kw: (rows, False, False))
    for name in ["_hydrate_attachments_for_page", "_hydrate_reasoning_for_page", "_hydrate_reply_links_for_page", "_hydrate_interrupt_anchors_for_page"]:
        monkeypatch.setattr(history_pages, name, lambda *a: {})
    app = FastAPI()
    app.include_router(history_pages.read_router)
    app.include_router(sessions.router)
    return app


@pytest.mark.parametrize("path", ["/v1/history", "/v1/history/around?message_id=m-one"])
def test_history_routes_enrich_without_changing_generic_timeline(fixture, monkeypatch, path):
    engine, service, _ = fixture
    row = seed(engine)
    other = seed(engine, key="other", owner="bob")
    other["timestamp"] += 1
    app = make_app(monkeypatch, service, [row, other])
    with TestClient(app) as client:
        for owner, expected in [("alice", "job-one"), ("bob", "job-other")]:
            separator = "&" if "?" in path else "?"
            payload = client.get(f"{path}{separator}bot_id=bot&user_id={owner}").json()
            assert len(payload["messages"]) == 2  # deliberately NOT a history ACL rewrite
            origins = [m["scheduler"] for m in payload["messages"] if m["scheduler"]]
            assert [o["schedule_id"] for o in origins] == [expected]
            assert payload["messages"][0]["content"] == row["content"]


def test_history_enrichment_only_queries_the_selected_page(fixture, monkeypatch):
    engine, service, calls = fixture
    first = seed(engine)
    second = seed(engine, key="two")
    second["timestamp"] += 1
    app = make_app(monkeypatch, service, [first, second])
    calls.clear()
    with TestClient(app) as client:
        result = client.get("/v1/history?bot_id=bot&user_id=alice&limit=1").json()
    assert len(result["messages"]) == 1
    assert result["messages"][0]["scheduler"]["schedule_id"] == "job-two"
    assert "m-one" not in calls[0][1]
    assert "m-two" in calls[0][1]


def test_session_route_owner_check_precedes_enrichment(fixture, monkeypatch):
    engine, service, calls = fixture
    row = seed(engine)
    app = make_app(monkeypatch, service, [row])
    monkeypatch.setattr(sessions, "get_service", lambda: service)
    monkeypatch.setattr(sessions, "get_effective_bot_id", lambda bot: bot)
    storage = SimpleNamespace(
        get_session=AsyncMock(return_value={"id": "s-one", "bot_id": "bot", "user_id": "alice", "status": "active"}),
        get_messages=AsyncMock(return_value=[row]),
    )
    monkeypatch.setattr(sessions, "get_storage", lambda: storage)
    calls.clear()
    with TestClient(app) as client:
        denied = client.get("/v1/sessions/s-one/messages?bot_id=bot&user=bob")
        assert denied.status_code == 404
        assert not calls
        storage.get_messages.assert_not_called()
        accepted = client.get("/v1/sessions/s-one/messages?bot_id=bot&user=alice")
        assert accepted.status_code == 200
        assert accepted.json()["messages"][0]["scheduler"]["occurrence_id"] == "run-one"
    assert len(calls) == 1
