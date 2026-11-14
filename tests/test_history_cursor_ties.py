"""Message-ID cursors must retain peers sharing a boundary timestamp."""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_bawt.service.routes import history_pages, history_search


@pytest.fixture
def history_client(monkeypatch):
    # Deliberately unsorted, with hidden rows sharing visible timestamps.
    rows = [
        {"id": "c", "role": "assistant", "content": "match c", "timestamp": 101.0},
        {"id": "hidden", "role": "summary", "content": "match hidden", "timestamp": 100.0},
        {"id": "b", "role": "user", "content": "match b", "timestamp": 100.0},
        {"id": "a", "role": "user", "content": "match a", "timestamp": 100.0},
    ]
    def get_messages(**kwargs):
        session_id = kwargs.get("session_id")
        return [row for row in rows if not session_id or row.get("session_id") == session_id]

    client = SimpleNamespace(get_messages=get_messages)
    service = SimpleNamespace(_default_bot="sample", get_memory_client=lambda bot: client)
    for module in (history_pages, history_search):
        monkeypatch.setattr(module, "get_service", lambda: service)
        for name in (
            "_hydrate_attachments_for_page",
            "_hydrate_reasoning_for_page",
            "_hydrate_turn_meta_for_page",
        ):
            monkeypatch.setattr(module, name, lambda *args: {})
    for name in (
        "_hydrate_interrupt_anchors_for_page",
        "hydrate_scheduler_for_page",
        "hydrate_approver_notices_for_page",
        "hydrate_resumptions_for_page",
    ):
        monkeypatch.setattr(history_pages, name, lambda *args: {})
    app = FastAPI()
    app.include_router(history_pages.read_router)
    app.include_router(history_search.router)
    with TestClient(app) as transport:
        yield transport, rows


@pytest.mark.parametrize("direction", ["before", "after"])
def test_history_id_cursor_visits_timestamp_peers_once(history_client, direction):
    client, _ = history_client
    params = {"limit": 1}
    if direction == "after":
        # Start before every visible row using a legacy timestamp.
        params["after"] = "99"
    visited = []
    flags = []
    for _ in range(4):
        response = client.get("/v1/history", params=params)
        assert response.status_code == 200, response.text
        body = response.json()
        messages = body["messages"]
        if not messages:
            assert not body["has_older"] and not body["has_newer"]
            break
        assert len(messages) == 1
        visited.append(messages[0]["id"])
        flags.append(body["has_older" if direction == "before" else "has_newer"])
        params[direction] = messages[0]["id"]
    assert visited == (["c", "b", "a"] if direction == "before" else ["a", "b", "c"])
    assert flags == [True, True, False]


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_search_id_cursor_visits_timestamp_peers_once(history_client, method):
    client, _ = history_client
    params = {"query": "match", "limit": 1}
    visited = []
    for expected_more in (True, True, False):
        response = client.request(method, "/v1/history/search", params=params)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["total_count"] == 3
        assert len(body["messages"]) == 1
        assert body["has_older"] == body["has_more"] == expected_more
        mid = body["messages"][0]["id"]
        visited.append(mid)
        params["before"] = mid
    assert visited == ["c", "b", "a"]


@pytest.mark.parametrize("method,path", [
    ("GET", "/v1/history"),
    ("GET", "/v1/history/search"),
    ("POST", "/v1/history/search"),
])
@pytest.mark.parametrize("before,expected", [
    ("100", []),
    ("1970-01-01T00:01:40Z", []),
    ("101", ["a", "b"]),
    ("1970-01-01T00:01:41+00:00", ["a", "b"]),
    ("a", []),
    ("b", ["a"]),
])
def test_backward_boundaries_preserve_legacy_timestamps(history_client, method, path, before, expected):
    client, _ = history_client
    response = client.request(method, path, params={"query": "match", "limit": 0, "before": before})
    assert response.status_code == 200, response.text
    body = response.json()
    assert [row["id"] for row in body["messages"]] == expected
    assert body["has_older"] is False and body["has_more"] is False


@pytest.mark.parametrize("after,expected", [
    ("100", ["c"]),
    ("1970-01-01T00:01:40Z", ["c"]),
    ("a", ["b", "c"]),
    ("b", ["c"]),
    ("c", []),
])
def test_forward_boundaries_preserve_legacy_timestamps(history_client, after, expected):
    client, _ = history_client
    response = client.get("/v1/history", params={"limit": 0, "after": after})
    assert response.status_code == 200, response.text
    body = response.json()
    assert [row["id"] for row in body["messages"]] == expected
    assert not body["has_newer"] and not body["has_older"]


@pytest.mark.parametrize("method,path", [
    ("GET", "/v1/history"),
    ("GET", "/v1/history/search"),
    ("POST", "/v1/history/search"),
])
@pytest.mark.parametrize("before", ["hidden", "missing"])
def test_hidden_or_unknown_id_cursor_still_rejected(history_client, method, path, before):
    client, _ = history_client
    response = client.request(method, path, params={"query": "match", "before": before})
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid cursor"


@pytest.mark.parametrize("method,path,direction", [
    ("GET", "/v1/history", "before"),
    ("GET", "/v1/history", "after"),
    ("GET", "/v1/history/search", "before"),
    ("POST", "/v1/history/search", "before"),
])
@pytest.mark.parametrize("limit", [1, 2, 4])
def test_entire_timestamp_group_can_span_multiple_pages(history_client, method, path, direction, limit):
    client, rows = history_client
    rows[:] = [
        {"id": f"msg-{index:02}", "role": "user", "content": "match", "timestamp": 100.0}
        for index in reversed(range(9))
    ]
    expected = sorted(row["id"] for row in rows)
    params = {"query": "match", "limit": limit}
    if direction == "after":
        params["after"] = "99"
    visited = []
    for _ in range(10):
        response = client.request(method, path, params=params)
        assert response.status_code == 200, response.text
        body = response.json()
        ids = [row["id"] for row in body["messages"]]
        assert ids
        visited = visited + ids if direction == "after" else ids + visited
        has_more = body["has_newer"] if direction == "after" else body["has_older"]
        params[direction] = ids[-1] if direction == "after" else ids[0]
        if not has_more:
            break
    assert visited == expected
    assert len(set(visited)) == len(visited)


def test_message_id_resolution_stays_within_session_scope(history_client):
    client, rows = history_client
    rows[0]["session_id"] = "foreign"
    for row in rows[1:]:
        row["session_id"] = "own"
    response = client.get("/v1/history", params={"session_id": "own", "before": "b"})
    assert response.status_code == 200
    assert [row["id"] for row in response.json()["messages"]] == ["a"]
    response = client.get("/v1/history", params={"session_id": "own", "before": "c"})
    assert response.status_code == 400


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_search_cursor_stays_within_query_and_time_bounds(history_client, method):
    client, _ = history_client
    response = client.request(method, "/v1/history/search",
                              params={"query": "match", "until": 100, "before": "b"})
    assert response.status_code == 200
    assert [row["id"] for row in response.json()["messages"]] == ["a"]
    for params in (
        {"query": "match", "until": 100, "before": "c"},
        {"query": "match a", "before": "b"},
    ):
        response = client.request(method, "/v1/history/search", params=params)
        assert response.status_code == 400
