from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from mcp.server.mcpserver import MCPServer as FastMCP
from mcp.server.mcpserver.exceptions import ToolError

from llm_bawt.mcp_server import project_tools, step_tools
from llm_bawt.mcp_server.server import mcp

STATUS_ERROR_TAIL = "must be one of: PENDING, RUNNING, COMPLETED, FAILED, SKIPPED"


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def mcp_call(name: str, arguments: dict[str, Any]) -> Any:
    """Dispatch through FastMCP's own argument and output validation.

    Calling the tool function directly skips both, which is how TASK-874's
    tests missed the hidden error. The base dispatch keeps the approval
    policy store (a DB dependency) out of these unit tests.
    """
    result = run(FastMCP.call_tool(mcp, name, arguments))
    return result.content, result.structured_content


def bawthub_error(status_code: int, message: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://bawthub.test/api/tasks/tasks/x/steps")
    response = httpx.Response(status_code, json={"error": message}, request=request)
    return httpx.HTTPStatusError("bawthub rejected", request=request, response=response)


def forbid_http(monkeypatch: pytest.MonkeyPatch, module: Any = step_tools) -> None:
    async def unexpected_request(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("invalid step input must not reach BawtHub")

    for name in ("_api_get", "_api_post", "_api_put", "_api_patch"):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, unexpected_request)


@pytest.mark.parametrize("tool_name", ["add_steps", "set_steps"])
def test_step_list_tools_reject_invalid_type_before_http(
    monkeypatch: pytest.MonkeyPatch,
    tool_name: str,
) -> None:
    async def unexpected_request(*args: Any, **kwargs: Any) -> dict:
        raise AssertionError("invalid step types must not reach BawtHub")

    monkeypatch.setattr(step_tools, "_api_post", unexpected_request)
    monkeypatch.setattr(step_tools, "_api_put", unexpected_request)

    result = run(getattr(step_tools, tool_name)(
        "TASK-872",
        [{"title": "Deploy", "type": "DEPLOY"}],  # type: ignore[list-item]
    ))

    assert result == {
        "error": (
            "steps[0].type must be one of: PLAN, READ_FILE, EDIT_FILE, "
            "CREATE_FILE, DELETE_FILE, RUN_COMMAND, SEARCH, ASK_USER, REVIEW"
        )
    }


@pytest.mark.parametrize(
    ("tool_name", "transport_name", "expected_path"),
    [
        ("add_steps", "_api_post", "/tasks/TASK-872/steps"),
        ("set_steps", "_api_put", "/tasks/TASK-872/steps"),
    ],
)
def test_step_list_tools_forward_valid_types(
    monkeypatch: pytest.MonkeyPatch,
    tool_name: str,
    transport_name: str,
    expected_path: str,
) -> None:
    calls: list[tuple[str, list[dict]]] = []

    async def fake_request(
        path: str,
        json: list[dict],
        headers: dict | None = None,
    ) -> list[dict]:
        calls.append((path, json))
        return json

    async def fake_get(path: str, params: dict | None = None) -> dict:
        return {"steps": [{"id": "s1", "status": "PENDING"}]}

    monkeypatch.setattr(step_tools, transport_name, fake_request)
    monkeypatch.setattr(step_tools, "_api_get", fake_get)
    steps = [{"title": "Run tests", "type": "RUN_COMMAND"}]

    receipt = run(getattr(step_tools, tool_name)("TASK-872", steps))
    assert receipt["steps"] == [{"id": None, "title": "Run tests", "status": None}]
    assert calls == [(expected_path, steps)]


# ── Empty steps_set guard (TASK-1343) ──────────────────────────────────────

def test_set_steps_refuses_empty_list_without_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    forbid_http(monkeypatch)

    result = run(step_tools.set_steps("TASK-1343", []))

    assert result == {"error": step_tools.EMPTY_STEPS_SET_ERROR}
    assert "clear=true" in result["error"] and "steps_update" in result["error"]


def test_set_steps_clears_when_explicitly_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, Any]] = []

    async def put(path: str, json: Any, headers: dict | None = None) -> list:
        calls.append((path, json))
        return []

    monkeypatch.setattr(step_tools, "_api_put", put)

    receipt = run(step_tools.set_steps("TASK-1343", [], clear=True))

    assert calls == [("/tasks/TASK-1343/steps", [])]
    assert receipt["steps"] == [] and "error" not in receipt


def test_steps_set_mcp_call_refuses_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
    forbid_http(monkeypatch)

    _content, structured = mcp_call("steps_set", {"task_id": "TASK-1343", "steps": []})

    assert structured == {"error": step_tools.EMPTY_STEPS_SET_ERROR}


# ── Step status (TASK-1313) ────────────────────────────────────────────────

@pytest.mark.parametrize("tool_name", ["add_steps", "set_steps"])
@pytest.mark.parametrize("status", ["IN_PROGRESS", "DONE", "BLOCKED", "running"])
def test_step_list_tools_reject_invalid_status_before_http(
    monkeypatch: pytest.MonkeyPatch,
    tool_name: str,
    status: str,
) -> None:
    forbid_http(monkeypatch)

    result = run(getattr(step_tools, tool_name)(
        "TASK-1313",
        [{"title": "ok"}, {"title": "Bad", "status": status}],  # type: ignore[list-item]
    ))

    assert result == {"error": f"steps[1].status {STATUS_ERROR_TAIL}"}


def test_steps_update_rejects_invalid_scalar_status_before_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbid_http(monkeypatch)

    result = run(step_tools.update_step(
        "TASK-1313", step_id="s1", status="IN_PROGRESS",  # type: ignore[arg-type]
    ))

    assert result == {"error": f"status {STATUS_ERROR_TAIL}"}


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        (
            [{"step_id": "s1", "status": "RUNNING"}, {"step_id": "s2", "status": "DONE"}],
            f"updates[1].status {STATUS_ERROR_TAIL}",
        ),
        (
            [{"stepId": "s1", "type": "DEPLOY"}],
            "updates[0].type must be one of: PLAN, READ_FILE, EDIT_FILE, "
            "CREATE_FILE, DELETE_FILE, RUN_COMMAND, SEARCH, ASK_USER, REVIEW",
        ),
        ([{"status": "COMPLETED"}], "updates[0] needs a step_id"),
    ],
)
def test_steps_update_batch_rejects_invalid_entries_before_http(
    monkeypatch: pytest.MonkeyPatch,
    updates: list[dict],
    expected: str,
) -> None:
    forbid_http(monkeypatch)

    result = run(step_tools.update_step("TASK-1313", updates=updates))

    assert result == {"error": expected}


def test_steps_update_batch_forwards_valid_statuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict]] = []

    async def fake_patch(path: str, json: dict, headers: dict | None = None) -> list:
        calls.append((path, json))
        return []

    monkeypatch.setattr(step_tools, "_api_patch", fake_patch)
    updates = [
        {"step_id": status.lower(), "status": status}
        for status in ("PENDING", "RUNNING", "COMPLETED", "FAILED", "SKIPPED")
    ]

    assert run(step_tools.update_step("TASK-1313", updates=updates)) == {
        "task_id": "TASK-1313", "steps": [], "stepStatusCounts": {},
    }
    assert calls == [(
        "/tasks/TASK-1313/steps",
        {"updates": [{"stepId": u["step_id"], "status": u["status"]} for u in updates]},
    )]


def test_published_schemas_carry_the_status_enum() -> None:
    tools = {tool.name: tool for tool in run(mcp.list_tools())}
    enum = ["PENDING", "RUNNING", "COMPLETED", "FAILED", "SKIPPED"]

    for name in ("steps_add", "steps_set", "tasks_create"):
        schema = tools[name].input_schema
        step_def = schema["$defs"]["TaskStepInput"]
        assert step_def["properties"]["status"]["enum"] == enum, name

    status = tools["steps_update"].input_schema["properties"]["status"]
    assert {"enum": enum, "type": "string"} in status["anyOf"]


# Through FastMCP: the bot must see the real error, never the output-model
# "Input should be a valid list" that used to replace it.

@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("steps_add", {"task_id": "TASK-1313", "steps": [{"title": "x", "status": "DONE"}]}),
        ("steps_set", {"task_id": "TASK-1313", "steps": [{"title": "x", "status": "BLOCKED"}]}),
        ("steps_update", {"task_id": "TASK-1313", "step_id": "s1", "status": "IN_PROGRESS"}),
    ],
)
def test_invalid_status_through_mcp_names_the_valid_statuses(
    monkeypatch: pytest.MonkeyPatch,
    tool: str,
    args: dict[str, Any],
) -> None:
    forbid_http(monkeypatch)

    with pytest.raises(ToolError) as caught:
        mcp_call(tool, args)

    message = str(caught.value)
    assert "Arguments" in message and "Output" not in message
    assert "valid list" not in message
    for status in ("PENDING", "RUNNING", "COMPLETED", "FAILED", "SKIPPED"):
        assert status in message


@pytest.mark.parametrize(
    ("tool", "transport", "args"),
    [
        ("steps_add", "_api_post", {"task_id": "TASK-1313", "steps": [{"title": "x"}]}),
        ("steps_set", "_api_put", {"task_id": "TASK-1313", "steps": [{"title": "x"}]}),
    ],
)
def test_bawthub_error_reaches_the_bot_through_mcp(
    monkeypatch: pytest.MonkeyPatch,
    tool: str,
    transport: str,
    args: dict[str, Any],
) -> None:
    async def rejected(*_args: Any, **_kwargs: Any) -> Any:
        raise bawthub_error(400, "steps[0].status must be one of: PENDING")

    monkeypatch.setattr(step_tools, transport, rejected)

    _content, structured = mcp_call(tool, args)

    assert structured == {"error": "steps[0].status must be one of: PENDING", "status": 400}


def test_step_list_tools_still_return_lists_through_mcp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def created(path: str, json: list[dict], headers: dict | None = None) -> list[dict]:
        return [{"id": "s1", **json[0]}]

    async def task(path: str, params: dict | None = None) -> dict:
        return {"steps": [{"id": "s0", "status": "COMPLETED"}, {"id": "s1", "status": "RUNNING"}]}

    monkeypatch.setattr(step_tools, "_api_post", created)
    monkeypatch.setattr(step_tools, "_api_get", task)

    _content, structured = mcp_call(
        "steps_add",
        {"task_id": "TASK-1313", "steps": [{"title": "x", "status": "RUNNING"}]},
    )

    assert structured == {
        "task_id": "TASK-1313",
        "steps": [{"id": "s1", "title": "x", "status": "RUNNING"}],
        "stepStatusCounts": {"COMPLETED": 1, "RUNNING": 1},
    }


# ── Compact write receipts (TASK-1494) ─────────────────────────────────────

LONG_OUTPUT = "x" * 4000


def _long_steps(n: int = 40) -> list[dict]:
    return [
        {"id": f"s{i}", "taskId": "t", "title": f"Step {i}", "type": "PLAN",
         "status": "COMPLETED" if i % 2 else "PENDING", "output": LONG_OUTPUT,
         "filePath": None, "orderIndex": i}
        for i in range(n)
    ]


@pytest.mark.parametrize("mode", ["single", "batch", "add", "set"])
def test_step_writes_on_a_task_with_long_outputs_return_a_small_receipt(
    monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    rows = _long_steps()

    async def patch(path: str, json: dict, headers: dict | None = None) -> Any:
        return rows if path.endswith("/steps") else rows[3]

    async def write_list(path: str, json: list, headers: dict | None = None) -> list:
        return rows if mode == "set" else rows[-2:]

    async def task(path: str, params: dict | None = None) -> dict:
        return {"id": "t", "steps": rows}

    monkeypatch.setattr(step_tools, "_api_patch", patch)
    monkeypatch.setattr(step_tools, "_api_post", write_list)
    monkeypatch.setattr(step_tools, "_api_put", write_list)
    monkeypatch.setattr(step_tools, "_api_get", task)
    new = [{"title": "n", "output": LONG_OUTPUT}]
    calls = {
        "single": ("steps_update", {"task_id": "TASK-1494", "step_id": "s3", "status": "COMPLETED"}),
        "batch": ("steps_update", {"task_id": "TASK-1494", "updates": [
            {"step_id": "s3", "status": "COMPLETED"}, {"step_id": "s4", "status": "COMPLETED"}]}),
        "add": ("steps_add", {"task_id": "TASK-1494", "steps": new}),
        "set": ("steps_set", {"task_id": "TASK-1494", "steps": new}),
    }
    name, args = calls[mode]

    content, structured = mcp_call(name, args)

    assert len(str(content)) + len(str(structured)) < 8000  # 40 x 4000-char outputs were echoed before
    assert LONG_OUTPUT not in str(structured)
    assert structured["stepStatusCounts"] == {"PENDING": 20, "COMPLETED": 20}
    expected_ids = {"single": ["s3"], "batch": ["s3", "s4"], "add": ["s38", "s39"],
                    "set": [f"s{i}" for i in range(40)]}[mode]
    assert [s["id"] for s in structured["steps"]] == expected_ids


def test_a_failed_count_read_still_returns_the_write_receipt(monkeypatch: pytest.MonkeyPatch) -> None:
    async def patch(path: str, json: dict, headers: dict | None = None) -> dict:
        return {"id": "s1", "title": "t", "status": "COMPLETED", "output": "o"}

    async def broken(path: str, params: dict | None = None) -> dict:
        raise httpx.ConnectError("down")

    monkeypatch.setattr(step_tools, "_api_patch", patch)
    monkeypatch.setattr(step_tools, "_api_get", broken)

    assert run(step_tools.update_step("TASK-1494", step_id="s1", status="COMPLETED")) == {
        "task_id": "TASK-1494", "steps": [{"id": "s1", "title": "t", "status": "COMPLETED"}],
    }


def test_projects_list_error_reaches_the_bot_through_mcp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def rejected(*_args: Any, **_kwargs: Any) -> Any:
        raise bawthub_error(503, "task API unavailable")

    monkeypatch.setattr(project_tools, "_api_get", rejected)

    _content, structured = mcp_call("projects_list", {})

    assert structured == {"result": {"error": "task API unavailable", "status": 503}}
