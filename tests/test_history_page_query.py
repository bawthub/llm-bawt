"""Page SQL parity and bounded hydration, without a live DB or MCP server."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, event, text
from sqlalchemy.pool import StaticPool

from llm_bawt.media import assets
from llm_bawt.service.routes import history_pages
from llm_bawt.service.routes.history_page_query import load_history_page


@pytest.fixture
def page_db(monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool)
    rows = [
        dict(id=f"m{i:02}", role="user" if i % 2 else "assistant",
             content=f"row {i}", timestamp=100.0 + i // 2,
             session_id="a" if i % 3 else "b", summarized=i < 4)
        for i in range(20)
    ]
    rows += [
        dict(id="hidden", role="system", content="", timestamp=120.0, session_id="a", summarized=False),
        dict(id="summary", role="summary", content="", timestamp=121.0, session_id="a", summarized=False),
        dict(id="orphan", role="user", content="", timestamp=122.0, session_id="missing", summarized=False),
    ]
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE messages_p_bot (id TEXT, role TEXT, content TEXT, timestamp REAL, "
            "session_id TEXT, summarized BOOLEAN, author_entity_type TEXT, author_entity_id TEXT)"
        ))
        conn.execute(text("CREATE TABLE sessions (id TEXT, bot_id TEXT, user_id TEXT)"))
        conn.execute(text("INSERT INTO sessions VALUES ('a','bot','owner'),('b','bot','other')"))
        conn.execute(text(
            "INSERT INTO messages_p_bot (id,role,content,timestamp,session_id,summarized) "
            "VALUES (:id,:role,:content,:timestamp,:session_id,:summarized)"
        ), rows)
    hydrated = []
    statements = []
    event.listen(engine, "before_cursor_execute",
                 lambda conn, cursor, sql, params, context, many: statements.append((sql, params)))
    monkeypatch.setattr(assets, "_build_engine", lambda config: engine)
    monkeypatch.setattr(history_pages, "_hydrate_window_authors",
                        lambda page, bot: hydrated.append(page) or page)
    yield SimpleNamespace(config=None), rows, statements, hydrated
    engine.dispose()


@pytest.mark.parametrize("session", [None, "a", "b", "missing"])
@pytest.mark.parametrize("user", [None, "owner", "other"])
@pytest.mark.parametrize("before,after", [(None, None), ("105", None), (None, "105"), ("m13", None), (None, "m12"), (None, " ")])
def test_page_matches_legacy_visibility_and_pagination(page_db, session, user, before, after):
    service, rows, statements, hydrated = page_db
    visible = [
        row for row in rows if row["role"] not in ("system", "summary")
        and (row["session_id"] == session if session else not row["summarized"])
        and (user is None or row["session_id"] == {"owner": "a", "other": "b"}[user])
    ]
    visible.sort(key=lambda row: (row["timestamp"], row["id"]))
    try:
        cursor = history_pages._resolve_cursor(after or before, visible) if after or before else None
    except HTTPException:
        with pytest.raises(HTTPException) as error:
            load_history_page(service, "bot", 3, before, after, session, user)
        assert error.value.status_code == 400
        return
    forward = bool(after and cursor is not None)
    candidates = [r for r in visible if cursor is None or cursor.includes(r, after=forward)]
    expected = candidates[:3] if forward else candidates[-3:]
    page, older, newer = load_history_page(service, "bot", 3, before, after, session, user)
    assert [r["id"] for r in page] == [r["id"] for r in expected]
    assert (older, newer) == (len(candidates) > 3 and not forward, len(candidates) > 3 and forward)
    assert len(hydrated) == 1 and len(hydrated[0]) <= 3
    assert len(statements) <= 2  # One scoped cursor lookup at most, then one page.
    assert "LIMIT" in statements[-1][0]
    assert statements[-1][1][-1] == 4  # Boundary flag costs one extra row, not a count.


@pytest.mark.parametrize("cursor", ["hidden", "summary", "m00", "nonexistent"])
def test_continuous_cursor_cannot_resolve_hidden_or_summarized_row(page_db, cursor):
    service, _, _, _ = page_db
    with pytest.raises(HTTPException) as error:
        load_history_page(service, "bot", 10, cursor, None, None, "owner")
    assert error.value.status_code == 400


@pytest.mark.parametrize("cursor", ["105", "1970-01-01T00:01:45Z", " 105 "])
def test_timestamp_formats_preserve_strict_time_boundary(page_db, cursor):
    service, _, _, _ = page_db
    page, _, _ = load_history_page(service, "bot", 50, None, cursor, None, None)
    assert page and all(row["timestamp"] > 105 for row in page)


def test_route_uses_bounded_page_and_existing_response_hydrator(page_db, monkeypatch):
    service, _, _, hydrated = page_db
    monkeypatch.setattr(history_pages, "get_service", lambda: service)
    monkeypatch.setattr(history_pages, "_load_sorted_visible_messages",
                        lambda *a, **kw: pytest.fail("Unbounded legacy loader must not run"))
    calls = []
    monkeypatch.setattr(history_pages, "_build_history_response",
                        lambda *a, **kw: calls.append((a, kw)) or "response")
    assert history_pages.get_history("bot", 3, "owner", None, None, None) == "response"
    args, kwargs = calls[0]
    assert len(args[3]) == 3 and len(hydrated[0]) == 3
    assert kwargs == dict(candidate_count=None, has_older=True, has_newer=False, origin_user_id="owner")


@pytest.mark.parametrize("limit", [0, -1])
def test_unlimited_legacy_contract_falls_back_without_query(page_db, limit):
    service, _, statements, _ = page_db
    assert load_history_page(service, "bot", limit, None, None, None, None) is None
    assert statements == []


def test_unavailable_engine_falls_back(monkeypatch):
    monkeypatch.setattr(assets, "_build_engine", lambda config: None)
    assert load_history_page(SimpleNamespace(config=None), "bot", 10, None, None, None, None) is None
