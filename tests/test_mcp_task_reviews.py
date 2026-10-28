"""Task-owned review MCP proxy contracts (TASK-1147)."""

import asyncio

import httpx
import pytest
from mcp.server.fastmcp import FastMCP

from llm_bawt.mcp_server import task_api, task_tools
from llm_bawt.mcp_server.server import mcp


def run(coro):
    return asyncio.run(coro)


def test_reviews_list_forwards_bounded_pagination_and_returns_full_entries(monkeypatch):
    page = {
        "reviews": [{"id": "review-1", "body": "Keep this note", "authorActorId": "caid"}],
        "nextCursor": "opaque/cursor+==",
    }
    calls = []

    async def fake_get(path, params=None):
        calls.append((path, params))
        return page

    monkeypatch.setattr(task_tools, "_api_get", fake_get)
    assert run(task_tools.list_task_reviews("TASK-1144", cursor="previous", limit=500)) == page
    assert calls == [("/tasks/TASK-1144/reviews", {"limit": "50", "cursor": "previous"})]
    assert run(task_tools.list_task_reviews("task-cuid", limit=0)) == page
    assert calls[-1] == ("/tasks/task-cuid/reviews", {"limit": "1"})
    assert run(task_tools.list_task_reviews("TASK-1144")) == page
    assert calls[-1][1] == {"limit": "20"}


@pytest.mark.parametrize("stage", ["DESIGN", "IMPLEMENTATION"])
@pytest.mark.parametrize("verdict", ["APPROVE", "APPROVE_WITH_CHANGES", "REQUEST_CHANGES", "COMMENT"])
def test_reviews_add_forwards_typed_contract_and_bot_actor(monkeypatch, stage, verdict):
    calls = []

    async def fake_post(path, json, headers=None):
        calls.append((path, json, headers))
        return {"id": "review-1", "authorActorId": "snark", "body": json["body"]}

    monkeypatch.setattr(task_tools, "_api_post", fake_post)
    receipt = run(task_tools.add_task_review(
        "TASK-1144", stage, verdict, "Review text", "snark",
        target_revision="sha256:abc", idempotency_key="review:1144:r1",
    ))
    assert receipt == {"id": "review-1", "authorActorId": "snark", "body": "Review text"}
    assert calls == [("/tasks/TASK-1144/reviews", {
        "stage": stage,
        "verdict": verdict,
        "body": "Review text",
        "targetRevision": "sha256:abc",
        "idempotencyKey": "review:1144:r1",
        "provenance": "unsigned",
    }, {"X-Agent-Bot-Id": "snark"})]
    assert "author" not in calls[0][1]
    assert "agentBotId" not in calls[0][1]


def test_reviews_add_requires_actor_and_omits_optional_fields(monkeypatch):
    calls = []

    async def fake_post(path, json, headers=None):
        calls.append((path, json, headers))
        return {"id": "review-2"}

    monkeypatch.setattr(task_tools, "_api_post", fake_post)
    assert run(task_tools.add_task_review("TASK-1", "DESIGN", "APPROVE", "", "  ")) == {
        "error": "bot_id is required for review attribution"
    }
    assert calls == []
    assert run(task_tools.add_task_review("TASK-1", "DESIGN", "APPROVE", "", "al")) == {
        "id": "review-2"
    }
    assert calls == [("/tasks/TASK-1/reviews", {
        "stage": "DESIGN", "verdict": "APPROVE", "body": "", "provenance": "unsigned",
    }, {"X-Agent-Bot-Id": "al"})]


def test_review_proxy_uses_shared_http_client_and_preserves_api_errors(monkeypatch):
    requests = []

    def respond(request):
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(404, json={"error": "Task not found"})
        return httpx.Response(409, json={"error": "Review key already used"})

    client = httpx.AsyncClient(base_url="http://bawthub.test", transport=httpx.MockTransport(respond))
    monkeypatch.setattr(task_api, "_client", client)
    try:
        assert run(task_tools.list_task_reviews("TASK-99", cursor="page+two")) == {
            "error": "Task not found", "status": 404,
        }
        assert run(task_tools.add_task_review(
            "TASK-99", "IMPLEMENTATION", "COMMENT", "note", "al", idempotency_key="key"
        )) == {"error": "Review key already used", "status": 409}
        assert requests[0].url.path == "/api/tasks/tasks/TASK-99/reviews"
        assert requests[0].url.params["cursor"] == "page+two"
        assert requests[1].headers["X-Agent-Bot-Id"] == "al"
        assert requests[1].headers["Content-Type"] == "application/json"
    finally:
        run(client.aclose())


@pytest.mark.anyio
async def test_review_tool_discovery_requires_typed_stage_verdict_and_actor():
    tools = {tool.name: tool for tool in await FastMCP.list_tools(mcp)}
    assert {"tasks_reviews_list", "tasks_reviews_add"} <= tools.keys()
    add = tools["tasks_reviews_add"].inputSchema
    assert set(add["required"]) == {"task_id", "stage", "verdict", "body", "bot_id"}
    assert add["properties"]["stage"]["enum"] == ["DESIGN", "IMPLEMENTATION"]
    assert add["properties"]["verdict"]["enum"] == [
        "APPROVE", "APPROVE_WITH_CHANGES", "REQUEST_CHANGES", "COMMENT",
    ]
    assert "author" not in add["properties"]
    assert set(tools["tasks_reviews_list"].inputSchema["required"]) == {"task_id"}


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_reviews_add_forwards_findings_in_order_with_wire_keys(monkeypatch):
    """TASK-1615: findings keep their order, failure_scenario maps to BawtHub's key,
    and a call without findings sends no findings field at all."""
    calls = []

    async def fake_post(path, json, headers=None):
        calls.append(json)
        return {"id": "review-3"}

    monkeypatch.setattr(task_tools, "_api_post", fake_post)
    findings = [
        {"severity": "BLOCKER", "title": "Forged r2", "detail": "Public POST accepts it.",
         "file": "server/routes/task-reviews.ts", "line": 42,
         "failure_scenario": "Same actor posts r2 and clears r1."},
        {"severity": "NIT", "title": "Wording", "detail": "Tighten the label."},
    ]
    run(task_tools.add_task_review(
        "TASK-1615", "IMPLEMENTATION", "REQUEST_CHANGES", "Two findings", "loopy",
        findings=findings,
    ))
    assert calls[0]["findings"] == [
        {"severity": "BLOCKER", "title": "Forged r2", "detail": "Public POST accepts it.",
         "file": "server/routes/task-reviews.ts", "line": 42,
         "failureScenario": "Same actor posts r2 and clears r1."},
        {"severity": "NIT", "title": "Wording", "detail": "Tighten the label."},
    ]
    assert findings[0]["failure_scenario"], "the caller's list is not mutated"
    run(task_tools.add_task_review("TASK-1615", "DESIGN", "COMMENT", "note", "loopy"))
    assert "findings" not in calls[1]


@pytest.mark.anyio
async def test_review_tool_discovery_types_findings():
    tools = {tool.name: tool for tool in await FastMCP.list_tools(mcp)}
    add = tools["tasks_reviews_add"].inputSchema
    assert "findings" in add["properties"] and "findings" not in add["required"]
    item = next(iter(add["$defs"].values()))
    assert item["properties"]["severity"]["enum"] == [
        "BLOCKER", "SHOULD_FIX", "NIT", "QUESTION", "PRE_EXISTING",
    ]
    assert set(item["required"]) == {"severity", "title", "detail"}
