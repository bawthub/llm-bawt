"""Hermetic HTTP contract and query construction for session discovery."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_bawt.memory.session_search import SessionSearch
from llm_bawt.service.routes import session_search


@pytest.fixture
def client(monkeypatch):
    search = Mock(return_value={"sessions": [], "total_count": 0})
    monkeypatch.setattr(SessionSearch, "search", search)
    monkeypatch.setattr(session_search, "get_storage", lambda: SimpleNamespace(engine=object()))
    app = FastAPI()
    app.include_router(session_search.router)
    return TestClient(app), search


def test_scope_and_paging_are_forwarded(client):
    http, search = client
    response = http.get("/v1/sessions/search", params={"user": "alice", "query": "Pocket", "bot_id": "sample", "status": "archived", "limit": 6, "offset": 12, "since": 100, "sort_by": "oldest"})
    assert response.status_code == 200
    search.assert_called_once_with(user_id="alice", query="Pocket", bot_id="sample", status="archived", since=100.0, sort_by="oldest", limit=6, offset=12)


@pytest.mark.parametrize("params", [
    {}, {"user": ""}, {"user": "alice", "limit": 201},
    {"user": "alice", "offset": -1}, {"user": "alice", "status": "deleted"},
    {"user": "alice", "sort_by": "anything"}, {"user": "alice", "since": "nan"},
    {"user": "alice", "query": "x" * 301},
])
def test_invalid_queries_do_not_touch_storage(client, params):
    http, search = client
    assert http.get("/v1/sessions/search", params=params).status_code == 422
    search.assert_not_called()


def test_registry_timestamps_are_serialized_with_explicit_utc(client):
    http, search = client
    search.return_value = {"sessions": [{"id": "one", "bot_id": "sample", "user_id": "alice", "title": "Test", "status": "active", "started_at": "2026-10-01T01:00:00", "ended_at": None}], "total_count": 1}
    response = http.get("/v1/sessions/search?user=alice")
    assert response.status_code == 200
    assert response.json()["sessions"][0]["started_at"] == "2026-10-01T01:00:00Z"


def test_empty_user_is_not_defaulted(client):
    http, search = client
    assert http.get("/v1/sessions/search?user=%20").status_code == 400
    search.assert_not_called()


def test_failure_is_not_reported_as_zero_results(client):
    http, search = client
    search.side_effect = RuntimeError("private database details")
    response = http.get("/v1/sessions/search?user=alice")
    assert response.status_code == 503
    assert "private" not in response.text


def test_search_precedes_dynamic_session_id():
    from llm_bawt.service.routes.sessions import router
    paths = [route.path for route in router.routes]
    assert paths.index("/v1/sessions/search") < paths.index("/v1/sessions/{session_id}")


def test_repository_rejects_unowned_or_unbounded_requests():
    engine = Mock()
    repository = SessionSearch(engine)
    for kwargs in [{"user_id": ""}, {"user_id": "alice", "limit": 201}, {"user_id": "alice", "offset": -1}, {"user_id": "alice", "sort_by": "DROP TABLE sessions"}]:
        with pytest.raises(ValueError):
            repository.search(**kwargs)
    engine.connect.assert_not_called()
