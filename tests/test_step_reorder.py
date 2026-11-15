"""TASK-434: the reorder adapter uses the existing task-scoped endpoint."""
from __future__ import annotations

import asyncio
import copy
import json
from typing import Any

import httpx
import pytest
from mcp.server.mcpserver import MCPServer as FastMCP
from mcp.server.mcpserver.exceptions import ToolError

from llm_bawt.mcp_server import step_tools
from llm_bawt.mcp_server.server import mcp


def call(arguments: dict[str, Any]) -> Any:
    return asyncio.run(FastMCP.call_tool(mcp, "steps_reorder", arguments)).structured_content


@pytest.mark.parametrize("task_id", ["TASK-434", "task-cuid"])
def test_reorder_forwards_order_and_actor_with_compact_receipt(monkeypatch, task_id):
    rows = [
        {"id": "s2", "title": "Second", "status": "COMPLETED", "output": "x" * 20000},
        {"id": "s1", "title": "First", "status": "PENDING", "output": "retain me"},
    ]
    before = copy.deepcopy(rows)
    calls = []

    async def post(path, json, headers):
        calls.append((path, json, headers))
        return rows

    monkeypatch.setattr(step_tools, "_api_post", post)
    result = call({"task_id": task_id, "step_ids": ["s2", "s1"], "bot_id": "byte"})
    assert calls == [(f"/tasks/{task_id}/steps/reorder",
                      {"stepIds": ["s2", "s1"]}, {"X-Agent-Bot-Id": "byte"})]
    assert result == {
        "task_id": task_id,
        "steps": [{"id": "s2", "title": "Second", "status": "COMPLETED"},
                  {"id": "s1", "title": "First", "status": "PENDING"}],
        "stepStatusCounts": {"COMPLETED": 1, "PENDING": 1},
    }
    assert rows == before
    assert len(json.dumps(result)) < 500
    assert "retain me" not in json.dumps(result)


@pytest.mark.parametrize("step_ids", [[], [""], [" "], ["s1", "s1"]])
def test_invalid_ids_rejected_before_http(monkeypatch, step_ids):
    async def forbidden(*args, **kwargs):
        pytest.fail("invalid input reached the API")

    monkeypatch.setattr(step_tools, "_api_post", forbidden)
    assert "error" in call({"task_id": "TASK-434", "step_ids": step_ids})


@pytest.mark.parametrize("step_ids", ["s1", [123], [None], [{}]])
def test_mcp_rejects_wrong_id_types_before_http(monkeypatch, step_ids):
    async def forbidden(*args, **kwargs):
        pytest.fail("invalid input reached the API")

    monkeypatch.setattr(step_tools, "_api_post", forbidden)
    with pytest.raises(ToolError, match="Arguments"):
        call({"task_id": "TASK-434", "step_ids": step_ids})


@pytest.mark.parametrize("status", [400, 403, 404, 500])
def test_reorder_surfaces_api_error(monkeypatch, status):
    async def rejected(path, json, headers):
        request = httpx.Request("POST", f"http://bawthub.test/api/tasks{path}")
        response = httpx.Response(status, json={"error": "Reorder rejected"}, request=request)
        raise httpx.HTTPStatusError("rejected", request=request, response=response)

    monkeypatch.setattr(step_tools, "_api_post", rejected)
    assert call({"task_id": "TASK-434", "step_ids": ["s1"]}) == {
        "error": "Reorder rejected", "status": status,
    }


def test_reorder_leaves_task_membership_and_result_order_to_server(monkeypatch):
    """The adapter must not delete/replace steps or invent a requested order."""
    async def post(path, json, headers):
        assert path == "/tasks/TASK-434/steps/reorder"
        assert json == {"stepIds": ["foreign", "s2"]}
        assert headers == {}
        # Existing endpoint ignores foreign IDs and returns all target steps.
        return [
            {"id": "s1", "title": "Unlisted", "status": "PENDING", "output": "saved"},
            {"id": "s2", "title": "Listed", "status": "COMPLETED", "output": "saved too"},
        ]

    async def forbidden(*args, **kwargs):
        pytest.fail("reorder must use only the existing POST endpoint")

    monkeypatch.setattr(step_tools, "_api_post", post)
    for name in ("_api_get", "_api_put", "_api_patch", "_api_delete"):
        monkeypatch.setattr(step_tools, name, forbidden)
    result = call({"task_id": "TASK-434", "step_ids": ["foreign", "s2"]})
    assert [step["id"] for step in result["steps"]] == ["s1", "s2"]
    assert result["stepStatusCounts"] == {"PENDING": 1, "COMPLETED": 1}


def test_reorder_registered_schema_is_flat_and_typed():
    tools = {tool.name: tool for tool in asyncio.run(mcp.list_tools())}
    schema = tools["steps_reorder"].input_schema
    assert set(schema["required"]) == {"task_id", "step_ids"}
    assert schema["properties"]["step_ids"]["type"] == "array"
    assert schema["properties"]["step_ids"]["items"]["type"] == "string"
    assert "action" not in schema["properties"]
    assert "at_index" not in schema["properties"]
