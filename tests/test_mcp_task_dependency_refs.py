"""TASK-1126: dependency MCP tools accept both task cuids and TASK-N refs."""

import asyncio
from typing import Any

import httpx
import pytest

from llm_bawt.mcp_server import task_tools
from llm_bawt.mcp_server.catalog_contracts import tool_description


def run(coro: Any) -> Any:
    return asyncio.run(coro)


# BawtHub dependency writes reply with the full dependent task; the tools
# return a compact receipt of it (TASK-1607).
TASK_FULL = {
    "id": "task-cuid", "shortId": "TASK-12", "title": "Dependent", "status": "QUEUED",
    "description": "d" * 5_000, "steps": [{"id": "s1", "status": "COMPLETED", "output": "o" * 5_000}],
    "dependsOn": [{"id": "dep-cuid", "shortId": "TASK-34", "title": "Prereq", "status": "IN_PROGRESS"}],
    "dependedOnBy": [],
}
DEPS_RECEIPT = [{"id": "dep-cuid", "shortId": "TASK-34", "title": "Prereq", "status": "IN_PROGRESS"}]


def assert_receipt(result: dict) -> None:
    assert result["shortId"] == "TASK-12"
    assert result["dependsOn"] == DEPS_RECEIPT
    assert result["stepCount"] == 1
    assert "description" not in result and "steps" not in result


def task_lookup(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    looked_up: list[str] = []

    async def fake_get(path: str) -> dict[str, str]:
        looked_up.append(path)
        return {"/tasks/TASK-12": {"id": "task-cuid"}, "/tasks/TASK-34": {"id": "dep-cuid"}}[path]

    monkeypatch.setattr(task_tools, "_api_get", fake_get)
    return looked_up


def test_add_resolves_both_short_ids_before_writing(monkeypatch: pytest.MonkeyPatch) -> None:
    looked_up = task_lookup(monkeypatch)
    writes: list[tuple] = []

    async def fake_post(path: str, json: dict, headers: dict) -> dict:
        writes.append((path, json, headers))
        return TASK_FULL

    monkeypatch.setattr(task_tools, "_api_post", fake_post)
    assert_receipt(run(task_tools.add_task_dependency("TASK-12", "TASK-34", bot_id="nova")))
    assert looked_up == ["/tasks/TASK-12", "/tasks/TASK-34"]
    assert writes == [("/tasks/task-cuid/dependencies", {"depId": "dep-cuid"}, {"X-Agent-Bot-Id": "nova"})]


@pytest.mark.parametrize("dependent,prerequisite,lookups", [
    ("task-cuid", "TASK-34", ["/tasks/TASK-34"]),
    ("TASK-12", "dep-cuid", ["/tasks/TASK-12"]),
    ("task-cuid", "dep-cuid", []),
])
def test_add_keeps_cuid_inputs(
    monkeypatch: pytest.MonkeyPatch, dependent: str, prerequisite: str, lookups: list[str]
) -> None:
    looked_up = task_lookup(monkeypatch)

    async def fake_post(path: str, json: dict, headers: dict) -> dict:
        assert path == "/tasks/task-cuid/dependencies"
        assert json == {"depId": "dep-cuid"}
        return TASK_FULL

    monkeypatch.setattr(task_tools, "_api_post", fake_post)
    assert_receipt(run(task_tools.add_task_dependency(dependent, prerequisite)))
    assert looked_up == lookups


def test_remove_resolves_both_short_ids_in_query_and_body(monkeypatch: pytest.MonkeyPatch) -> None:
    looked_up = task_lookup(monkeypatch)
    calls: list[tuple] = []

    class FakeClient:
        async def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
            calls.append((method, url, kwargs))
            return httpx.Response(200, json=TASK_FULL, request=httpx.Request(method, "http://test" + url))

    monkeypatch.setattr(task_tools, "_get_client", lambda: FakeClient())
    assert_receipt(run(task_tools.remove_task_dependency("TASK-12", "TASK-34", bot_id="nova")))
    assert looked_up == ["/tasks/TASK-12", "/tasks/TASK-34"]
    assert calls == [("DELETE", "/api/tasks/tasks/task-cuid/dependencies", {
        "params": {"depId": "dep-cuid"}, "json": {"depId": "dep-cuid"},
        "headers": {"X-Agent-Bot-Id": "nova"},
    })]


@pytest.mark.parametrize("operation", ["add_task_dependency", "remove_task_dependency"])
def test_missing_short_id_fails_without_mutation(monkeypatch: pytest.MonkeyPatch, operation: str) -> None:
    request = httpx.Request("GET", "http://test/api/tasks/tasks/TASK-404")
    response = httpx.Response(404, json={"error": "Task not found"}, request=request)

    async def fake_get(path: str) -> dict:
        if path == "/tasks/TASK-404":
            raise httpx.HTTPStatusError("not found", request=request, response=response)
        return {"id": "task-cuid"}

    monkeypatch.setattr(task_tools, "_api_get", fake_get)
    async def unexpected_post(*args: Any, **kwargs: Any) -> dict:
        pytest.fail("dependency mutation must not run after failed lookup")

    monkeypatch.setattr(task_tools, "_api_post", unexpected_post)
    monkeypatch.setattr(task_tools, "_get_client", lambda: pytest.fail("DELETE must not run"))
    result = run(getattr(task_tools, operation)("TASK-12", "TASK-404"))
    assert result == {"error": "Task not found", "status": 404}


def test_malformed_task_lookup_is_not_used_as_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_get(path: str) -> dict:
        return {"shortId": "TASK-12"}

    monkeypatch.setattr(task_tools, "_api_get", fake_get)
    assert run(task_tools.add_task_dependency("TASK-12", "TASK-34")) == {
        "error": "Task lookup for TASK-12 returned no task ID"
    }


def test_tool_descriptions_name_cuids_not_uuids() -> None:
    for name in ("tasks_get", "tasks_get_context", "tasks_add_dependency", "tasks_remove_dependency"):
        text = tool_description(name)
        assert "cuid" in text
        assert "UUID" not in text
