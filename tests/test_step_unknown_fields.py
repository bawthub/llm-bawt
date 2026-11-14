"""TASK-1663: unknown step fields are refused, not silently dropped.

Argument validation used to strip keys such as ``description`` from each step,
so steps_add wrote the step without them and returned a success receipt.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from llm_bawt.mcp_server import step_tools, task_tools
from llm_bawt.mcp_server.server import mcp


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def mcp_call(name: str, arguments: dict[str, Any]) -> Any:
    """Dispatch through FastMCP's own argument validation (see test_task_step_tools)."""
    return run(FastMCP.call_tool(mcp, name, arguments))


def forbid_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    async def unexpected(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a step with unknown fields must not reach BawtHub")

    for module in (step_tools, task_tools):
        for name in ("_api_get", "_api_post", "_api_put", "_api_patch"):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, unexpected)


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("steps_add", {"task_id": "TASK-1663", "steps": [{"title": "x", "description": "d"}]}),
        ("steps_set", {"task_id": "TASK-1663", "steps": [{"title": "x"}, {"title": "y", "description": "d"}]}),
        ("tasks_create", {"title": "t", "steps": [{"title": "x", "description": "d"}]}),
    ],
)
def test_unknown_step_field_through_mcp_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch,
    tool: str,
    args: dict[str, Any],
) -> None:
    forbid_writes(monkeypatch)

    with pytest.raises(ToolError) as caught:
        mcp_call(tool, args)

    message = str(caught.value)
    assert "description" in message
    assert "Extra inputs are not permitted" in message


@pytest.mark.parametrize("tool_name", ["add_steps", "set_steps"])
def test_unknown_step_fields_rejected_by_the_shared_validator(
    monkeypatch: pytest.MonkeyPatch,
    tool_name: str,
) -> None:
    forbid_writes(monkeypatch)

    result = run(getattr(step_tools, tool_name)(
        "TASK-1663",
        [{"title": "ok"}, {"title": "x", "description": "d", "notes": "n"}],  # type: ignore[list-item]
    ))

    assert result == {"error": (
        "steps[1] has unknown field(s): description, notes; allowed: title, type, "
        "status, output, file_path, filePath. Nothing was written."
    )}


def test_published_step_schema_forbids_extra_fields() -> None:
    tools = {tool.name: tool for tool in run(mcp.list_tools())}

    for name in ("steps_add", "steps_set", "tasks_create"):
        step_def = tools[name].inputSchema["$defs"]["TaskStepInput"]
        assert step_def.get("additionalProperties") is False, name


def test_every_known_step_field_still_reaches_bawthub(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[Any] = []

    async def capture(path: str, json: Any = None, headers: Any = None) -> Any:
        sent.append(json)
        return [{"id": "s1", "title": "x", "status": "COMPLETED"}]

    async def counts(path: str) -> Any:
        return {"steps": []}

    monkeypatch.setattr(step_tools, "_api_post", capture)
    monkeypatch.setattr(step_tools, "_api_get", counts)
    step = {"title": "x", "type": "EDIT_FILE", "status": "COMPLETED", "output": "o",
            "file_path": "a.py", "filePath": "a.py"}

    mcp_call("steps_add", {"task_id": "TASK-1663", "steps": [step]})

    # TASK-2238: the file_path alias is sent as the filePath BawtHub stores.
    expected = {key: value for key, value in step.items() if key != "file_path"}
    assert sent == [[expected]]
