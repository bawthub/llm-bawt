"""TASK-1380: tasks_update(archived=...) and tasks_list(archived=...) over BawtHub's API.

BawtHub owns which tasks may be archived (a finished resulting status), clears
the archive when a task reopens, and records who changed it. The tool sends one
PATCH and never reports success from a BawtHub that predates archive: such a
server silently ignores the field and filter.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from llm_bawt.mcp_server import task_tools

ARCHIVED_AT = "2026-10-24T18:00:00Z"


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _task(**extra: Any) -> dict[str, Any]:
    return {
        "id": "task-id",
        "shortId": "TASK-7",
        "title": "Ship the board",
        "status": "COMPLETED",
        "agentBotId": "caid",
        "focusRank": None,
        "focusedAt": None,
        "archivedAt": ARCHIVED_AT,
        **extra,
    }


def _without(row: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in keys}


def _capture(monkeypatch: pytest.MonkeyPatch, response: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    async def fake_patch(path: str, json: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
        calls.append({"path": path, "json": json, "headers": headers})
        return response if response is not None else _task()

    async def no_get(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("archive never looks up the owner: any actor may set it")

    monkeypatch.setattr(task_tools, "_api_patch", fake_patch)
    monkeypatch.setattr(task_tools, "_api_get", no_get)
    return calls


@pytest.mark.parametrize(
    ("archived", "response"),
    [(True, _task()), (False, _task(archivedAt=None))],
)
def test_archive_is_one_patch_field(monkeypatch: pytest.MonkeyPatch, archived: bool, response: dict) -> None:
    calls = _capture(monkeypatch, response)

    result = _run(task_tools.update_task("TASK-7", archived=archived, bot_id="al"))

    assert calls == [{"path": "/tasks/TASK-7", "json": {"archived": archived}, "headers": {"X-Agent-Bot-Id": "al"}}]
    assert result["ok"] is True
    assert result["updated"] == ["archived"]


def test_status_and_archive_travel_in_the_same_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture(monkeypatch)

    result = _run(task_tools.update_task("TASK-7", status="COMPLETED", archived=True, bot_id="al"))

    assert [call["json"] for call in calls] == [{"status": "COMPLETED", "archived": True}]
    assert result["ok"] is True
    assert result["task"]["archivedAt"] == ARCHIVED_AT


def _status_error(status: int, body: dict[str, Any]) -> httpx.HTTPStatusError:
    request = httpx.Request("PATCH", "http://echo.test/api/tasks/tasks/TASK-7")
    return httpx.HTTPStatusError("boom", request=request, response=httpx.Response(status, json=body, request=request))


def test_archiving_an_open_task_returns_the_servers_400(monkeypatch: pytest.MonkeyPatch) -> None:
    error = _status_error(400, {"error": "Only finished tasks can be archived"})

    async def fake_patch(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise error

    monkeypatch.setattr(task_tools, "_api_patch", fake_patch)

    result = _run(task_tools.update_task("TASK-7", status="QUEUED", archived=True, bot_id="al"))

    assert result == {"error": "Only finished tasks can be archived", "status": 400}


@pytest.mark.parametrize("archived", [True, False])
def test_archive_on_a_server_without_archive_is_not_reported_ok(
    monkeypatch: pytest.MonkeyPatch, archived: bool
) -> None:
    calls = _capture(monkeypatch, _without(_task(), "archivedAt"))

    result = _run(task_tools.update_task("TASK-7", status="COMPLETED", archived=archived, bot_id="al"))

    assert len(calls) == 1
    assert "ok" not in result
    assert "does not support archive" in result["error"]
    assert "archive was not changed" in result["error"]
    assert result["updated"] == ["status"]


def test_each_unsupported_state_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    # Pre-focus, pre-archive BawtHub: both fields ignored, status applied.
    _capture(monkeypatch, _without(_task(status="QUEUED"), "archivedAt", "focusRank", "focusedAt"))

    result = _run(task_tools.update_task("TASK-7", status="QUEUED", focus="off", archived=False, bot_id="al"))

    assert "ok" not in result
    assert "focus was not changed" in result["error"]
    assert "archive was not changed" in result["error"]
    assert result["updated"] == ["status"]


def test_a_focus_only_server_still_confirms_focus(monkeypatch: pytest.MonkeyPatch) -> None:
    # Focus shipped, archive not yet: only the archive part is unconfirmed.
    _capture(monkeypatch, _without(_task(status="QUEUED", focusRank=1, focusedAt=ARCHIVED_AT), "archivedAt"))

    result = _run(task_tools.update_task("TASK-7", focus="on", archived=False, bot_id="al"))

    assert "archive was not changed" in result["error"]
    assert "focus was not changed" not in result["error"]
    assert result["updated"] == ["focus"]


def test_compact_rows_carry_archived_at_only_when_archived() -> None:
    archived = task_tools._compact_task(_task())
    active = task_tools._compact_task(_task(status="QUEUED", archivedAt=None))

    assert archived["archivedAt"] == ARCHIVED_AT
    assert "archivedAt" not in active


_ABSENT = object()


def _capture_list(
    monkeypatch: pytest.MonkeyPatch,
    tasks: list[dict[str, Any]],
    archive_filter: Any = _ABSENT,
    focus_filter: Any = _ABSENT,
) -> list[dict[str, str]]:
    """Fake GET /tasks; a filter echo left _ABSENT is a server predating that state."""
    seen: list[dict[str, str]] = []

    async def fake_get(path: str, params: dict | None = None) -> dict[str, Any]:
        assert path == "/tasks"
        seen.append(dict(params or {}))
        payload: dict[str, Any] = {"tasks": tasks, "total": len(tasks)}
        if archive_filter is not _ABSENT:
            payload["archiveFilter"] = archive_filter
        if focus_filter is not _ABSENT:
            payload["focusFilter"] = focus_filter
        return payload

    monkeypatch.setattr(task_tools, "_api_get", fake_get)
    return seen


@pytest.mark.parametrize(
    ("archived", "row"),
    [(True, _task()), (False, _task(archivedAt=None))],
)
def test_archived_filter_is_sent_and_confirmed(monkeypatch: pytest.MonkeyPatch, archived: bool, row: dict) -> None:
    seen = _capture_list(monkeypatch, [row], archive_filter=archived)

    result = _run(task_tools.list_tasks(archived=archived, status="COMPLETED"))

    assert seen == [
        {"limit": "20", "sort": "updated", "status": "COMPLETED", "archived": str(archived).lower()}
    ]
    assert result["tasks"][0]["shortId"] == "TASK-7"
    assert ("archivedAt" in result["tasks"][0]) is archived


@pytest.mark.parametrize("archived", [True, False])
def test_an_empty_list_from_an_archive_server_is_truly_empty(
    monkeypatch: pytest.MonkeyPatch, archived: bool
) -> None:
    _capture_list(monkeypatch, [], archive_filter=archived)

    assert _run(task_tools.list_tasks(archived=archived)) == {"tasks": [], "total": 0, "archiveFilter": archived}


@pytest.mark.parametrize(
    ("archived", "rows", "archive_filter"),
    [
        # Pre-archive BawtHub: no archiveFilter echo, whatever the rows look like.
        (True, [], _ABSENT),
        (False, [], _ABSENT),
        (True, [_task()], _ABSENT),
        (False, [_without(_task(), "archivedAt")], _ABSENT),
        # The echo must be exactly the boolean that was sent.
        (True, [_task()], None),
        (True, [_task()], "true"),
        (False, [_task(archivedAt=None)], True),
        # Rows must agree with the filter and carry the archivedAt key.
        (True, [_task(), _task(archivedAt=None)], True),
        (False, [_task()], False),
        (False, [_without(_task(), "archivedAt")], False),
    ],
)
def test_an_archive_list_bawthub_did_not_confirm_is_an_error(
    monkeypatch: pytest.MonkeyPatch, archived: bool, rows: list[dict[str, Any]], archive_filter: Any
) -> None:
    _capture_list(monkeypatch, rows, archive_filter=archive_filter)

    result = _run(task_tools.list_tasks(archived=archived))

    assert set(result) == {"error"}
    assert "does not support archive" in result["error"]
    assert f"archived={str(archived).lower()}" in result["error"]


def test_both_filters_must_be_confirmed(monkeypatch: pytest.MonkeyPatch) -> None:
    row = _task(status="QUEUED", focusRank=1, focusedAt=ARCHIVED_AT, archivedAt=None)
    seen = _capture_list(monkeypatch, [row], focus_filter=True, archive_filter=False)

    result = _run(task_tools.list_tasks(focused=True, archived=False))

    assert seen == [{"limit": "20", "sort": "focus", "focused": "true", "archived": "false"}]
    assert result["tasks"][0]["focusRank"] == 1

    # Focus shipped, archive not yet: the archive half is unconfirmed.
    _capture_list(monkeypatch, [row], focus_filter=True)
    result = _run(task_tools.list_tasks(focused=True, archived=False))
    assert set(result) == {"error"}
    assert "does not support archive" in result["error"]


def test_default_list_needs_no_archive_support(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture_list(monkeypatch, [_without(_task(), "archivedAt")])

    result = _run(task_tools.list_tasks())

    assert seen == [{"limit": "20", "sort": "updated"}]
    assert result["tasks"][0]["shortId"] == "TASK-7"
