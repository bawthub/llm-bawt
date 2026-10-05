import asyncio
import json
from typing import Any

import pytest

from llm_bawt.mcp_server import task_tools


GIANT_TEXT = "x" * 20_000


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _full_task() -> dict[str, Any]:
    return {
        "id": "task-id",
        "shortId": "TASK-266",
        "title": "Fix garage task context blowup",
        "description": GIANT_TEXT,
        "status": "REVIEW",
        "priority": "HIGH",
        "planned": True,
        "project": {
            "id": "project-id",
            "name": "Garage",
            "description": GIANT_TEXT,
            "agentBotId": "snark",
            "_count": {"tasks": 7},
        },
        "agentBotId": "snark",
        "createdAt": "2026-06-13T20:00:00Z",
        "updatedAt": "2026-06-13T21:00:00Z",
        "url": "https://example.test/tasks/TASK-266",
        "response": "done" * 1_000,
        "steps": [
            {"id": "step-1", "title": "Read logs", "status": "COMPLETED"},
            {"id": "step-2", "title": "Patch MCP", "status": "RUNNING"},
            {"id": "step-3", "title": "Verify", "status": "COMPLETED"},
        ],
        "dependencies": [{"id": "dep-1"}],
        "dependents": [{"id": "child-1"}, {"id": "child-2"}],
    }


def test_tasks_list_returns_compact_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_get(path: str, params: dict | None = None) -> dict[str, Any]:
        assert path == "/tasks"
        assert params == {"limit": "20", "sort": "updated", "q": "garage"}
        return {"tasks": [_full_task()], "total": 1}

    monkeypatch.setattr(task_tools, "_api_get", fake_get)

    result = _run(task_tools.list_tasks(q="garage"))

    assert result["total"] == 1
    task = result["tasks"][0]
    assert task == {
        "id": "task-id",
        "shortId": "TASK-266",
        "title": "Fix garage task context blowup",
        "status": "REVIEW",
        "priority": "HIGH",
        "planned": True,
        "project": {
            "id": "project-id",
            "name": "Garage",
            "agentBotId": "snark",
            "taskCount": 7,
        },
        "agentBotId": "snark",
        "createdAt": "2026-06-13T20:00:00Z",
        "updatedAt": "2026-06-13T21:00:00Z",
        "url": "https://example.test/tasks/TASK-266",
        "descriptionChars": 20_000,
        "responseChars": 4_000,
        "stepCount": 3,
        "stepStatusCounts": {"COMPLETED": 2, "RUNNING": 1},
        "dependencyCount": 1,
        "dependentCount": 2,
    }
    assert "description" not in task
    assert "response" not in task
    assert "steps" not in task
    assert len(json.dumps(result)) < 1_000


def test_tasks_update_returns_compact_task(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_patch(
        path: str,
        json: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        calls.append({"path": path, "json": json, "headers": headers})
        return _full_task()

    monkeypatch.setattr(task_tools, "_api_patch", fake_patch)

    async def unowned(_path: str, params: dict | None = None) -> dict[str, Any]:
        return {"agentBotId": None}

    monkeypatch.setattr(task_tools, "_api_get", unowned)

    result = _run(
        task_tools.update_task(
            "TASK-266",
            status="REVIEW",
            response=GIANT_TEXT,
            bot_id="snark",
        )
    )

    # Moving to REVIEW auto-assigns the caller (bot_id) as owner, so agentBotId
    # is added to the PATCH body without the caller passing it explicitly.
    assert calls == [
        {
            "path": "/tasks/TASK-266",
            "json": {"status": "REVIEW", "response": GIANT_TEXT, "agentBotId": "snark"},
            "headers": {"X-Agent-Bot-Id": "snark"},
        }
    ]
    assert result["ok"] is True
    assert result["updated"] == ["agentBotId", "response", "status"]
    assert result["task"]["shortId"] == "TASK-266"
    assert result["task"]["descriptionChars"] == 20_000
    assert "description" not in result["task"]
    assert "response" not in result["task"]
    assert "steps" not in result["task"]
    assert len(json.dumps(result)) < 1_100


def test_projects_list_returns_compact_projects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_get(path: str, params: dict | None = None) -> list[dict[str, Any]]:
        assert path == "/projects"
        assert params is None
        return [
            {
                "id": "project-id",
                "name": "Garage",
                "description": GIANT_TEXT,
                "color": "#94a3b8",
                "icon": "wrench",
                "agentBotId": "snark",
                "_count": {"tasks": 12},
                "tasks": [_full_task()],
                "createdAt": "2026-06-13T20:00:00Z",
                "updatedAt": "2026-06-13T21:00:00Z",
            }
        ]

    monkeypatch.setattr(task_tools, "_api_get", fake_get)

    result = _run(task_tools.list_projects())

    assert result == [
        {
            "id": "project-id",
            "name": "Garage",
            "descriptionChars": 20_000,
            "color": "#94a3b8",
            "icon": "wrench",
            "agentBotId": "snark",
            "taskCount": 12,
            "createdAt": "2026-06-13T20:00:00Z",
            "updatedAt": "2026-06-13T21:00:00Z",
        }
    ]
    assert len(json.dumps(result)) < 300


# --- Issue 2: REVIEW auto-assigns the owner (reviewTransitionMissingBot guard) ---


def _capture_patch(
    monkeypatch: pytest.MonkeyPatch, *, current_owner: str | None = None,
) -> list[dict[str, Any]]:
    """Patch _api_patch to record bodies and return a minimal valid task."""
    calls: list[dict[str, Any]] = []

    async def fake_get(_path: str, params: dict | None = None) -> dict[str, Any]:
        return {"agentBotId": current_owner}

    monkeypatch.setattr(task_tools, "_api_get", fake_get)

    async def fake_patch(
        path: str,
        json: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        calls.append({"path": path, "json": json, "headers": headers})
        return _full_task()

    monkeypatch.setattr(task_tools, "_api_patch", fake_patch)
    return calls


def test_review_autofills_owner_from_bot_id(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture_patch(monkeypatch)

    result = _run(task_tools.update_task("TASK-1", status="REVIEW", bot_id="byte"))

    # The guard requires an owner; bot_id supplies it without the caller knowing.
    assert calls[0]["json"] == {"status": "REVIEW", "agentBotId": "byte"}
    assert result["updated"] == ["agentBotId", "status"]


def test_review_actor_bot_id_never_reassigns_owned_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TASK-1015: TASK-1013 became Al's because Caid passed bot_id="al"."""
    calls = _capture_patch(monkeypatch, current_owner="caid")

    result = _run(task_tools.update_task("TASK-1013", status="REVIEW", bot_id="al"))

    assert calls[0]["json"] == {"status": "REVIEW"}
    assert calls[0]["headers"] == {"X-Agent-Bot-Id": "al"}  # still attributed
    assert result["updated"] == ["status"]


def test_review_owner_lookup_failure_does_not_guess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture_patch(monkeypatch)

    async def broken_get(_path: str, params: dict | None = None) -> dict[str, Any]:
        raise task_tools.httpx.ConnectError("down")

    monkeypatch.setattr(task_tools, "_api_get", broken_get)
    _run(task_tools.update_task("TASK-1", status="REVIEW", bot_id="byte"))
    # Fail closed: the server's own REVIEW guard reports a missing owner.
    assert calls[0]["json"] == {"status": "REVIEW"}


def test_review_explicit_agent_bot_id_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture_patch(monkeypatch)

    _run(
        task_tools.update_task(
            "TASK-1", status="REVIEW", agent_bot_id="vex", bot_id="byte"
        )
    )

    # An explicit assignment is never overridden by the bot_id default.
    assert calls[0]["json"] == {"status": "REVIEW", "agentBotId": "vex"}


def test_non_review_status_does_not_autofill_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture_patch(monkeypatch)

    _run(task_tools.update_task("TASK-1", status="IN_PROGRESS", bot_id="byte"))

    # Only REVIEW is guarded; other transitions must not silently reassign.
    assert calls[0]["json"] == {"status": "IN_PROGRESS"}


def test_review_without_bot_id_does_not_autofill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture_patch(monkeypatch)

    _run(task_tools.update_task("TASK-1", status="REVIEW"))

    # Nothing to fill from; the server guard will then return its own 400,
    # which _http_error surfaces verbatim (see error-body tests below).
    assert calls[0]["json"] == {"status": "REVIEW"}


# --- Issue 1: HTTP errors surface the server's JSON message, not str(e) ---


def _http_status_error(status: int, body: Any, *, json_body: bool = True) -> "task_tools.httpx.HTTPStatusError":
    import httpx

    request = httpx.Request("PATCH", "http://echo.test/api/tasks/tasks/TASK-1")
    if json_body:
        response = httpx.Response(status, json=body, request=request)
    else:
        response = httpx.Response(status, text=body, request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


def test_http_error_surfaces_server_error_field() -> None:
    err = _http_status_error(
        400, {"error": "Cannot move task to REVIEW without an assigned bot."}
    )

    assert task_tools._http_error(err) == {
        "error": "Cannot move task to REVIEW without an assigned bot.",
        "status": 400,
    }


def test_http_error_surfaces_detail_field() -> None:
    err = _http_status_error(404, {"detail": "Task not found"})

    assert task_tools._http_error(err) == {"error": "Task not found", "status": 404}


def test_http_error_falls_back_to_str_when_body_not_json() -> None:
    err = _http_status_error(500, "<html>Internal Server Error</html>", json_body=False)

    result = task_tools._http_error(err)
    assert result["status"] == 500
    # No JSON error field -> fall back to the exception's own string.
    assert result["error"] == str(err)


def test_update_forwards_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture_patch(monkeypatch)

    result = _run(
        task_tools.update_task(
            "TASK-1", status="IN_PROGRESS", outcome="Tasks now show what changed."
        )
    )

    # TASK-1010: outcome is a first-class task field, forwarded verbatim.
    assert calls[0]["json"] == {
        "status": "IN_PROGRESS",
        "outcome": "Tasks now show what changed.",
    }
    assert result["updated"] == ["outcome", "status"]


def test_create_forwards_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_post(path: str, **kwargs: Any) -> dict[str, Any]:
        calls.append({"path": path, **kwargs})
        return _full_task()

    monkeypatch.setattr(task_tools, "_api_post", fake_post)

    _run(task_tools.create_task("New thing", outcome="Users can do X."))

    assert calls[0]["path"] == "/tasks"
    assert calls[0]["json"]["outcome"] == "Users can do X."
