"""tasks_board MCP tool (TASK-1460): a thin forwarder to BawtHub's
GET /api/tasks/board.

BawtHub lays the board out with the board page's own lane rules; the tool only
forwards the board URL's params and returns the reply. These tests stub the
endpoint at the HTTP layer, so the real task_api client builds the request.
"""

import asyncio
from typing import Any

import httpx
import pytest

from llm_bawt.mcp_server import task_api, task_tools

BOARD = {
    "view": "kanban",
    "updated": "7d",
    "q": "",
    "filters": {"project": "all", "bot": "alpha", "priority": "all"},
    "rows": "active",
    "now": "2026-10-25T16:00:00.000Z",
    "lanes": [
        {
            "id": "todo", "label": "To do", "windowed": False, "shown": 2, "older": 0,
            "rows": [{
                "shortId": "TASK-1", "title": "First", "status": "QUEUED", "priority": "HIGH",
                "agentBotId": "alpha", "focusRank": 1, "archivedAt": None,
                "updatedAt": "2026-10-25T15:00:00.000Z", "blockedReason": None,
            }],
        },
        {"id": "backlog", "label": "Backlog", "in": "todo", "windowed": True, "shown": 0, "older": 4, "rows": []},
    ],
}


def _echo(request: httpx.Request) -> dict:
    """BOARD as a BawtHub that applies every param it was sent."""
    applied = {key: request.url.params[key] for key in ("view", "updated", "rows") if key in request.url.params}
    return {**BOARD, **applied}


def _stub(monkeypatch: pytest.MonkeyPatch, status: int = 200, body: Any = _echo) -> list[httpx.Request]:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status, json=body(request) if callable(body) else body)

    client = httpx.AsyncClient(base_url="http://bawthub.test", transport=httpx.MockTransport(handle))
    monkeypatch.setattr(task_api, "_client", client)
    return requests


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_forwards_the_board_params_and_returns_the_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = _stub(monkeypatch)

    result = _run(task_tools.tasks_board(
        view="status", updated="24h", bot="alpha", q="board", project="none", priority="HIGH", limit=5,
    ))

    assert result == {**BOARD, "view": "status", "updated": "24h", "rows": "all"}
    [request] = requests
    assert request.method == "GET"
    assert request.url.path == "/api/tasks/board"
    assert dict(request.url.params) == {
        "view": "status", "updated": "24h", "bot": "alpha", "q": "board",
        "project": "none", "priority": "HIGH", "limit": "5", "rows": "all",
    }


def test_defaults_send_the_row_limit_and_active_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    # TASK-1555: "what's going on" pays for the open columns' cards, not the
    # backlog, Done and Archived drawers; their counts still come back.
    requests = _stub(monkeypatch)

    assert _run(task_tools.tasks_board()) == BOARD
    assert dict(requests[0].url.params) == {"limit": "20", "rows": "active"}


@pytest.mark.parametrize("kwargs,sent", [
    ({"q": "popup"}, "all"),
    ({"q": "popup", "rows": "active"}, "active"),
    ({"rows": "all"}, "all"),
])
def test_rows_follow_q_unless_given(monkeypatch: pytest.MonkeyPatch, kwargs: dict, sent: str) -> None:
    requests = _stub(monkeypatch)

    _run(task_tools.tasks_board(**kwargs))

    assert requests[0].url.params["rows"] == sent


@pytest.mark.parametrize("key,asked,used", [("updated", "30d", "7d"), ("view", "status", "kanban")])
def test_an_arg_the_board_did_not_apply_is_an_error(
    monkeypatch: pytest.MonkeyPatch, key: str, asked: str, used: str,
) -> None:
    # The board falls back to its default for a value it doesn't know; the
    # tool never passes that off as the board that was asked for.
    _stub(monkeypatch, body={**BOARD, key: used})

    result = _run(task_tools.tasks_board(**{key: asked}))

    assert result == {"error": (
        f"the board did not apply {key}={asked!r} (it used {used!r}); that board is not returned.")}


def test_a_board_without_the_rows_filter_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    old = {key: value for key, value in BOARD.items() if key != "rows"}
    _stub(monkeypatch, body=old)

    result = _run(task_tools.tasks_board())

    assert result == {**old, "note": "this BawtHub build ignores rows=active; every lane's cards are included."}


def test_the_schema_rejects_an_unknown_window(monkeypatch: pytest.MonkeyPatch) -> None:
    # "3d" once came back as the 7d board with no word of the swap.
    from mcp.server.fastmcp.exceptions import ToolError

    from llm_bawt.mcp_server.server import mcp

    requests = _stub(monkeypatch)
    tool = mcp._tool_manager.get_tool("tasks_board")
    with pytest.raises(ToolError, match="Input should be '1h', '6h', '24h', '48h', '7d', '30d' or 'all'"):
        _run(tool.run({"updated": "3d"}))
    assert requests == []


@pytest.mark.parametrize("limit,sent", [(500, "50"), (0, "1"), (-3, "1")])
def test_limit_is_clamped_to_1_50(monkeypatch: pytest.MonkeyPatch, limit: int, sent: str) -> None:
    requests = _stub(monkeypatch)

    _run(task_tools.tasks_board(limit=limit))

    assert requests[0].url.params["limit"] == sent


def test_endpoint_error_is_returned_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, status=400, body={"error": "limit must be a positive integer"})

    assert _run(task_tools.tasks_board()) == {"error": "limit must be a positive integer", "status": 400}


def _stub_web_page(monkeypatch: pytest.MonkeyPatch) -> None:
    # A BawtHub build without the route answers 200 with its web app page.
    client = httpx.AsyncClient(base_url="http://bawthub.test", transport=httpx.MockTransport(
        lambda request: httpx.Response(200, text="<!doctype html><html></html>",
                                       headers={"content-type": "text/html; charset=utf-8"})))
    monkeypatch.setattr(task_api, "_client", client)


def test_a_web_page_reply_names_the_missing_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_web_page(monkeypatch)

    assert _run(task_tools.tasks_board()) == {"error": (
        "BawtHub answered GET /api/tasks/board with text/html, not JSON; "
        "this BawtHub build may predate the endpoint.")}


def test_every_task_api_call_rejects_a_web_page_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_web_page(monkeypatch)

    with pytest.raises(task_api.NotJsonResponse, match="PATCH /api/tasks/tasks/x with text/html"):
        _run(task_api.api_patch("/tasks/x", json={}))
