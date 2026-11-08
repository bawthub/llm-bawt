import asyncio
import json
from typing import Any

import pytest
from mcp.server.fastmcp import FastMCP

from llm_bawt.mcp_server import task_tools
from llm_bawt.mcp_server.server import mcp


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
        # BawtHub's real field names (TASK-1240).
        "dependsOn": [{"id": "dep-1"}],
        "dependedOnBy": [{"id": "child-1"}, {"id": "child-2"}],
    }


def test_tasks_list_returns_lean_rows_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # TASK-1555: the default row is what a bot needs to pick the one task to open.
    async def fake_get(path: str, params: dict | None = None) -> dict[str, Any]:
        assert params == {"limit": "10", "sort": "updated", "q": "garage"}
        return {"tasks": [_full_task()], "total": 1}

    monkeypatch.setattr(task_tools, "_api_get", fake_get)

    result = _run(task_tools.list_tasks(q="garage"))

    assert result == {
        "tasks": [{
            "shortId": "TASK-266",
            "title": "Fix garage task context blowup",
            "status": "REVIEW",
            "priority": "HIGH",
            "agentBotId": "snark",
            "updatedAt": "2026-06-13T21:00:00Z",
            "project": "Garage",
        }],
        "total": 1,
        "more": False,
    }


@pytest.mark.parametrize("limit,sent", [(500, "25"), (25, "25"), (0, "1"), (-3, "1")])
def test_tasks_list_limit_is_clamped_to_1_25(monkeypatch: pytest.MonkeyPatch, limit: int, sent: str) -> None:
    seen: list[str] = []

    async def fake_get(path: str, params: dict | None = None) -> dict[str, Any]:
        seen.append(params["limit"])
        return {"tasks": [], "total": 0}

    monkeypatch.setattr(task_tools, "_api_get", fake_get)

    _run(task_tools.list_tasks(limit=limit))

    assert seen == [sent]


def test_tasks_list_says_when_there_is_more(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_get(path: str, params: dict | None = None) -> dict[str, Any]:
        return {"tasks": [_full_task()] * 10, "total": 182}

    monkeypatch.setattr(task_tools, "_api_get", fake_get)

    result = _run(task_tools.list_tasks())

    assert (len(result["tasks"]), result["total"], result["more"]) == (10, 182, True)


def _realistic_task(n: int) -> dict[str, Any]:
    """A GET /tasks row at the size BawtHub sends: long text, steps, links."""
    task = _full_task()
    task.update({
        "id": f"cmvq{n:021d}",
        "shortId": f"TASK-{1000 + n}",
        "title": "Bots burn 5-16k tokens per lookup: cheap-lookup rules exist only in one profile",
        "steps": [{"id": f"s{i}", "title": "Step", "status": "COMPLETED"} for i in range(8)],
    })
    return task


def test_the_default_list_fits_a_lookup_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    # ~2k tokens at ~4 chars a token, as compact JSON (catalog.compact_json_text).
    async def fake_get(path: str, params: dict | None = None) -> dict[str, Any]:
        return {"tasks": [_realistic_task(n) for n in range(int(params["limit"]))], "total": 400}

    monkeypatch.setattr(task_tools, "_api_get", fake_get)

    result = _run(task_tools.list_tasks())

    assert len(result["tasks"]) == 10
    assert len(json.dumps(result, separators=(",", ":"))) < 8_000


def test_tasks_list_detail_returns_compact_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_get(path: str, params: dict | None = None) -> dict[str, Any]:
        assert path == "/tasks"
        assert params == {"limit": "10", "sort": "updated", "q": "garage"}
        return {"tasks": [_full_task()], "total": 1}

    monkeypatch.setattr(task_tools, "_api_get", fake_get)

    result = _run(task_tools.list_tasks(q="garage", detail=True))

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


@pytest.mark.parametrize("operation", ["create", "update"])
@pytest.mark.parametrize("length", [401, 600])
def test_long_outcome_is_rejected_before_any_side_effect(monkeypatch, operation, length):
    async def unexpected(*args, **kwargs):
        pytest.fail("Invalid outcome must not read, write, resolve images or associate a task")

    for name in ("_api_get", "_api_post", "_api_patch", "resolve_image_refs",
                 "linked_turn_images", "associate_current_task"):
        monkeypatch.setattr(task_tools, name, unexpected)
    function = task_tools.create_task if operation == "create" else task_tools.update_task
    result = _run(function("TASK-1", outcome="x" * length, associate_current_turn=True))
    assert "400" in result["error"]
    assert "response" in result["error"]
    assert "check_links" in result["error"]


@pytest.mark.parametrize("name,key", [("tasks_create", "title"), ("tasks_update", "task_id")])
def test_long_outcome_error_reaches_mcp_caller(monkeypatch, name, key):
    async def unexpected(*args, **kwargs):
        pytest.fail("Rejected outcome must not reach BawtHub")

    monkeypatch.setattr(task_tools, "_api_post", unexpected)
    monkeypatch.setattr(task_tools, "_api_patch", unexpected)
    content = _run(FastMCP.call_tool(
        mcp, name, {key: "TASK-1", "outcome": "x" * 600},
    ))
    structured = json.loads(content[0].text)
    assert "400" in structured["error"]
    assert "response" in structured["error"]
    assert "check_links" in structured["error"]


@pytest.mark.parametrize("operation", ["create", "update"])
@pytest.mark.parametrize("outcome", [None, "", "x" * 400, "🐕" * 400])
def test_outcome_limit_preserves_optional_and_boundary_values(monkeypatch, operation, outcome):
    calls = []

    async def fake_write(path, **kwargs):
        calls.append(kwargs["json"])
        return _full_task()

    monkeypatch.setattr(task_tools, "_api_post", fake_write)
    monkeypatch.setattr(task_tools, "_api_patch", fake_write)
    if operation == "create":
        _run(task_tools.create_task("New thing", outcome=outcome))
    else:
        _run(task_tools.update_task("TASK-1", outcome=outcome, response=GIANT_TEXT))
        assert calls[0]["response"] == GIANT_TEXT
    if outcome is None:
        assert "outcome" not in calls[0]
    else:
        assert calls[0]["outcome"] == outcome


def test_update_forwards_blocked_reason_and_compact_shows_it(monkeypatch: pytest.MonkeyPatch) -> None:
    # TASK-1368: blocked_reason passes through as blockedReason; BawtHub owns
    # validation (required entering BLOCKED, cleared on exit).
    calls = _capture_patch(monkeypatch)
    blocked = {**_full_task(), "status": "BLOCKED", "blockedReason": "Approve the migration."}

    async def fake_patch(path: str, json: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
        calls.append({"path": path, "json": json, "headers": headers})
        return blocked

    monkeypatch.setattr(task_tools, "_api_patch", fake_patch)
    result = _run(task_tools.update_task("TASK-1", status="BLOCKED", blocked_reason="Approve the migration."))

    assert calls[0]["json"] == {"status": "BLOCKED", "blockedReason": "Approve the migration."}
    assert result["updated"] == ["blockedReason", "status"]
    assert result["task"]["blockedReason"] == "Approve the migration."
    # Non-BLOCKED compact rows stay unchanged.
    assert "blockedReason" not in task_tools._compact_task({**_full_task(), "blockedReason": None})


def test_update_forwards_waiting_and_compact_shows_it(monkeypatch: pytest.MonkeyPatch) -> None:
    # TASK-2115: waiting_on/waiting_reason pass through; BawtHub validates.
    calls = _capture_patch(monkeypatch)
    waiting = {**_full_task(), "status": "WAITING", "waitingOn": "RESTART",
               "waitingReason": "App restart.", "waitingSince": "2026-11-06T22:30:00.000Z"}

    async def fake_patch(path: str, json: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
        calls.append({"path": path, "json": json, "headers": headers})
        return waiting

    monkeypatch.setattr(task_tools, "_api_patch", fake_patch)
    result = _run(task_tools.update_task(
        "TASK-1", status="WAITING", waiting_on="RESTART", waiting_reason="App restart."))

    assert calls[0]["json"] == {"status": "WAITING", "waitingOn": "RESTART", "waitingReason": "App restart."}
    assert result["task"]["waitingOn"] == "RESTART"
    assert result["task"]["waitingSince"] == "2026-11-06T22:30:00.000Z"
    assert "waitingOn" not in task_tools._compact_task({**_full_task(), "waitingOn": None})


def test_create_forwards_blocked_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_post(path: str, **kwargs: Any) -> dict[str, Any]:
        calls.append({"path": path, **kwargs})
        return _full_task()

    monkeypatch.setattr(task_tools, "_api_post", fake_post)
    _run(task_tools.create_task("Ship it", status="BLOCKED", blocked_reason="Say ship."))
    _run(task_tools.create_task("Plain"))

    assert calls[0]["json"]["status"] == "BLOCKED"
    assert calls[0]["json"]["blockedReason"] == "Say ship."
    assert "blockedReason" not in calls[1]["json"]


def test_check_links_pass_through_and_compact_shows_them_in_review(monkeypatch: pytest.MonkeyPatch) -> None:
    # TASK-1422: check_links passes through as checkLinks; BawtHub validates
    # paths and requires them for a bot's move to REVIEW.
    links = [{"label": "Tasks board", "path": "/tasks/board", "steps": ["Open a REVIEW card"]}]
    review = {**_full_task(), "status": "REVIEW", "agentBotId": "nova", "checkLinks": links}
    patches: list[dict[str, Any]] = []

    async def fake_patch(path: str, json: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
        patches.append(json)
        return review

    monkeypatch.setattr(task_tools, "_api_patch", fake_patch)
    result = _run(task_tools.update_task("TASK-1", check_links=links))

    assert patches[0] == {"checkLinks": links}
    assert result["task"]["checkLinks"] == links
    assert "checkLinks" not in task_tools._compact_task({**_full_task(), "status": "IN_PROGRESS", "checkLinks": links})
    assert "check_links" in (task_tools.update_task.__doc__ or "")
    assert "{check_links_doc}" not in (task_tools.update_task.__doc__ or "")


def _capture_post(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    async def fake_post(
        path: str, json: dict[str, Any], headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        calls.append({"path": path, "json": json, "headers": headers})
        return {"shortId": "TASK-1", "agentBotId": json.get("agentBotId")}

    monkeypatch.setattr(task_tools, "_api_post", fake_post)
    return calls


def test_create_sends_agent_bot_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """TASK-1657: tasks_create had no agent_bot_id, so the owner was dropped."""
    calls = _capture_post(monkeypatch)

    result = _run(task_tools.create_task("t", agent_bot_id="vex", bot_id="byte"))

    assert calls[0]["path"] == "/tasks"
    assert calls[0]["json"]["agentBotId"] == "vex"
    assert calls[0]["headers"] == {"X-Agent-Bot-Id": "byte"}
    assert result["agentBotId"] == "vex"


@pytest.mark.parametrize("value", [None, ""])
def test_create_without_agent_bot_id_leaves_unassigned(
    monkeypatch: pytest.MonkeyPatch, value: str | None,
) -> None:
    # bot_id is attribution only; it never assigns a new task.
    calls = _capture_post(monkeypatch)

    _run(task_tools.create_task("t", agent_bot_id=value, bot_id="byte"))

    assert "agentBotId" not in calls[0]["json"]


def test_create_tool_schema_exposes_agent_bot_id() -> None:
    from llm_bawt.mcp_server.server import mcp

    tools = {tool.name: tool for tool in asyncio.run(mcp.list_tools())}
    assert "agent_bot_id" in tools["tasks_create"].inputSchema["properties"]


def test_compact_task_counts_dependencies_by_real_and_legacy_names() -> None:
    # TASK-1240: BawtHub sends dependsOn/dependedOnBy; older shapes still count.
    from llm_bawt.mcp_server.task_api import compact_task

    real = compact_task({"dependsOn": [{"id": "a"}] * 5, "dependedOnBy": [{"id": "b"}]})
    assert (real["dependencyCount"], real["dependentCount"]) == (5, 1)

    legacy = compact_task({"dependencies": [{"id": "a"}], "dependents": [{"id": "b"}] * 2})
    assert (legacy["dependencyCount"], legacy["dependentCount"]) == (1, 2)


def _dependency_task() -> dict[str, Any]:
    task = _full_task()
    task["dependsOn"] = [
        {"id": "dep-1", "shortId": "TASK-1", "title": "Prereq", "status": "QUEUED",
         "description": GIANT_TEXT},
    ]
    return task


def test_add_dependency_returns_compact_receipt(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_post(path: str, json: Any = None, headers: Any = None) -> dict:
        assert path == "/tasks/task-id/dependencies"
        assert json == {"depId": "dep-1"}
        return _dependency_task()

    monkeypatch.setattr(task_tools, "_api_post", fake_post)
    result = _run(task_tools.add_task_dependency("task-id", "dep-1", bot_id="al"))
    assert result["shortId"] == "TASK-266"
    assert result["dependsOn"] == [
        {"id": "dep-1", "shortId": "TASK-1", "title": "Prereq", "status": "QUEUED"}
    ]
    assert result["dependentCount"] == 2
    assert "steps" not in result and "description" not in result
    assert len(json.dumps(result)) < 2_000


def test_remove_dependency_returns_compact_receipt(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResp:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            task = _dependency_task()
            task["dependsOn"] = []
            return task

    class FakeClient:
        async def request(self, method: str, url: str, **kwargs: Any) -> FakeResp:
            assert method == "DELETE"
            return FakeResp()

    monkeypatch.setattr(task_tools, "_get_client", lambda: FakeClient())
    monkeypatch.setattr(task_tools, "_json_body", lambda resp: resp.json())
    result = _run(task_tools.remove_task_dependency("task-id", "dep-1", bot_id="al"))
    assert result["dependsOn"] == []
    assert result["dependencyCount"] == 0
    assert len(json.dumps(result)) < 2_000


def test_dependency_error_passes_through() -> None:
    assert task_tools._dependency_receipt({"error": "cycle"}) == {"error": "cycle"}
