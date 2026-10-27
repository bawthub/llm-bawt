"""TASK-1380: tasks_update focus controls and tasks_list(focused=...) over BawtHub's API.

BawtHub owns ranks, state rules and the activity record; the tool maps chat args
to one PATCH, keeps the server's 409 focus order, and never reports success
from a BawtHub that predates focus (it would silently ignore the field/filter).
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from llm_bawt.mcp_server import task_tools


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _task(**extra: Any) -> dict[str, Any]:
    return {
        "id": "task-id",
        "shortId": "TASK-7",
        "title": "Board focus",
        "status": "QUEUED",
        "agentBotId": "caid",
        "focusRank": 1,
        "focusedAt": "2026-10-24T17:00:00Z",
        **extra,
    }


def _capture(monkeypatch: pytest.MonkeyPatch, response: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    async def fake_patch(path: str, json: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
        calls.append({"path": path, "json": json, "headers": headers})
        return response if response is not None else _task()

    async def no_get(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("focus never looks up the owner: any actor may set it")

    monkeypatch.setattr(task_tools, "_api_patch", fake_patch)
    monkeypatch.setattr(task_tools, "_api_get", no_get)
    return calls


@pytest.mark.parametrize(
    ("args", "focus_body"),
    [
        ({"focus": "on"}, {"on": True}),
        ({"focus": "off"}, {"on": False}),
        ({"focus_position": "top"}, {"on": True, "placement": "top"}),
        ({"focus": "on", "focus_position": "bottom"}, {"on": True, "placement": "bottom"}),
        (
            {"focus_position": "after", "focus_anchor": "TASK-42"},
            {"on": True, "placement": "after", "anchorId": "TASK-42"},
        ),
        # Request-shape rules the server can see stay the server's (400).
        ({"focus_position": "before"}, {"on": True, "placement": "before"}),
    ],
)
def test_focus_args_become_one_patch(monkeypatch: pytest.MonkeyPatch, args: dict, focus_body: dict) -> None:
    calls = _capture(monkeypatch)

    result = _run(task_tools.update_task("TASK-7", bot_id="al", **args))

    assert calls == [{"path": "/tasks/TASK-7", "json": {"focus": focus_body}, "headers": {"X-Agent-Bot-Id": "al"}}]
    assert result["ok"] is True
    assert result["updated"] == ["focus"]


def test_status_and_focus_travel_in_the_same_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture(monkeypatch, _task(status="IN_PROGRESS"))

    _run(task_tools.update_task("TASK-7", status="IN_PROGRESS", focus_position="top", bot_id="al"))

    assert len(calls) == 1
    assert calls[0]["json"] == {"status": "IN_PROGRESS", "focus": {"on": True, "placement": "top"}}


@pytest.mark.parametrize(
    "args",
    [
        {"focus": "off", "focus_position": "top"},
        {"focus": "off", "focus_anchor": "TASK-42"},
        {"focus_anchor": "TASK-42"},
        {"focus_position": "top", "focus_anchor": "TASK-42"},
    ],
)
def test_contradictory_focus_args_send_nothing(monkeypatch: pytest.MonkeyPatch, args: dict) -> None:
    calls = _capture(monkeypatch)

    result = _run(task_tools.update_task("TASK-7", status="IN_PROGRESS", bot_id="al", **args))

    assert calls == []
    assert set(result) == {"error"}


def _status_error(status: int, body: dict[str, Any]) -> httpx.HTTPStatusError:
    request = httpx.Request("PATCH", "http://echo.test/api/tasks/tasks/TASK-7")
    return httpx.HTTPStatusError("boom", request=request, response=httpx.Response(status, json=body, request=request))


def _raise_on_patch(monkeypatch: pytest.MonkeyPatch, error: httpx.HTTPStatusError) -> None:
    async def fake_patch(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise error

    monkeypatch.setattr(task_tools, "_api_patch", fake_patch)


def test_focus_conflict_keeps_the_servers_current_order(monkeypatch: pytest.MonkeyPatch) -> None:
    _raise_on_patch(monkeypatch, _status_error(409, {"error": "Anchor TASK-42 is not focused", "focused": ["a", "b"]}))

    result = _run(task_tools.update_task("TASK-7", focus_position="before", focus_anchor="TASK-42", bot_id="al"))

    assert result == {"error": "Anchor TASK-42 is not focused", "status": 409, "focused": ["a", "b"]}


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (409, {"error": "Task changed; retry"}),
        (400, {"error": "anchor is the task itself", "focused": ["a"]}),
    ],
)
def test_other_patch_errors_pass_through_unchanged(
    monkeypatch: pytest.MonkeyPatch, status: int, body: dict[str, Any]
) -> None:
    _raise_on_patch(monkeypatch, _status_error(status, body))

    result = _run(task_tools.update_task("TASK-7", focus_position="top", bot_id="al"))

    assert result == {"error": body["error"], "status": status}


def test_focus_on_a_server_without_focus_is_not_reported_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    legacy = {key: value for key, value in _task(status="IN_PROGRESS").items() if not key.startswith("focus")}
    calls = _capture(monkeypatch, legacy)

    result = _run(task_tools.update_task("TASK-7", status="IN_PROGRESS", focus="on", bot_id="al"))

    assert len(calls) == 1
    assert "ok" not in result
    assert "focus was not changed" in result["error"]
    assert result["updated"] == ["status"]
    assert result["task"]["status"] == "IN_PROGRESS"

    # Without focus args the same legacy response is an ordinary success.
    plain = _run(task_tools.update_task("TASK-7", status="IN_PROGRESS", bot_id="al"))
    assert plain["ok"] is True


def test_compact_rows_carry_rank_only_when_focused() -> None:
    focused = task_tools._compact_task(_task(focusRank=2))
    unfocused = task_tools._compact_task(_task(focusRank=None, focusedAt=None))

    assert (focused["focusRank"], focused["focusedAt"]) == (2, "2026-10-24T17:00:00Z")
    assert "focusRank" not in unfocused and "focusedAt" not in unfocused


_ABSENT = object()


def _capture_list(
    monkeypatch: pytest.MonkeyPatch, tasks: list[dict[str, Any]], focus_filter: Any = _ABSENT
) -> list[dict[str, str]]:
    """Fake GET /tasks; focus_filter is BawtHub's echoed ``focusFilter`` (absent = pre-focus server)."""
    seen: list[dict[str, str]] = []

    async def fake_get(path: str, params: dict | None = None) -> dict[str, Any]:
        assert path == "/tasks"
        seen.append(dict(params or {}))
        payload: dict[str, Any] = {"tasks": tasks, "total": len(tasks)}
        if focus_filter is not _ABSENT:
            payload["focusFilter"] = focus_filter
        return payload

    monkeypatch.setattr(task_tools, "_api_get", fake_get)
    return seen


def _legacy_row(**extra: Any) -> dict[str, Any]:
    return {key: value for key, value in _task(**extra).items() if not key.startswith("focus")}


def test_focused_list_asks_for_rank_order_and_keeps_it(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_task(id="b", shortId="TASK-2", focusRank=1), _task(id="a", shortId="TASK-1", focusRank=2)]
    seen = _capture_list(monkeypatch, rows, focus_filter=True)

    result = _run(task_tools.list_tasks(focused=True))

    assert seen == [{"limit": "10", "sort": "focus", "focused": "true"}]
    assert [(t["shortId"], t["focusRank"]) for t in result["tasks"]] == [("TASK-2", 1), ("TASK-1", 2)]


def test_unfocused_list_is_the_backlog_by_recency(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture_list(monkeypatch, [_task(focusRank=None, focusedAt=None)], focus_filter=False)

    result = _run(task_tools.list_tasks(focused=False, status="QUEUED"))

    assert seen == [{"limit": "10", "sort": "updated", "status": "QUEUED", "focused": "false"}]
    assert "focusRank" not in result["tasks"][0]


@pytest.mark.parametrize("focused", [True, False])
def test_an_empty_list_from_a_focus_server_is_truly_empty(monkeypatch: pytest.MonkeyPatch, focused: bool) -> None:
    _capture_list(monkeypatch, [], focus_filter=focused)

    assert _run(task_tools.list_tasks(focused=focused, status="BLOCKED")) == {
        "tasks": [],
        "total": 0,
        "focusFilter": focused,
        "more": False,
    }


@pytest.mark.parametrize(
    ("focused", "rows", "focus_filter"),
    [
        # Pre-focus BawtHub: no focusFilter echo, whatever the rows look like.
        (True, [], _ABSENT),
        (False, [], _ABSENT),
        (True, [_task(focusRank=1)], _ABSENT),
        (False, [_legacy_row()], _ABSENT),
        # The echo must be exactly the boolean that was sent.
        (True, [_task(focusRank=1)], None),
        (True, [_task(focusRank=1)], "true"),
        (False, [_task(focusRank=None)], True),
        # Rows must agree with the filter and carry the focusRank key.
        (True, [_task(focusRank=1), _task(focusRank=None)], True),
        (False, [_task(focusRank=3)], False),
        (False, [_legacy_row()], False),
    ],
)
def test_a_focus_list_bawthub_did_not_confirm_is_an_error(
    monkeypatch: pytest.MonkeyPatch, focused: bool, rows: list[dict[str, Any]], focus_filter: Any
) -> None:
    _capture_list(monkeypatch, rows, focus_filter=focus_filter)

    result = _run(task_tools.list_tasks(focused=focused))

    assert set(result) == {"error"}
    assert "does not support focus" in result["error"]
    assert f"focused={str(focused).lower()}" in result["error"]


def test_unfiltered_list_needs_no_focus_support(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture_list(monkeypatch, [_legacy_row()])

    result = _run(task_tools.list_tasks())

    assert seen == [{"limit": "10", "sort": "updated"}]
    assert result["tasks"][0]["shortId"] == "TASK-7"
