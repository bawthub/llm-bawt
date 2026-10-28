"""Task-owned review MCP proxy contracts (TASK-1147)."""

import asyncio

import httpx
import pytest
from mcp.server.fastmcp import FastMCP

from llm_bawt.mcp_server import task_api, task_review_tools
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

    monkeypatch.setattr(task_review_tools, "_api_get", fake_get)
    assert run(task_review_tools.list_task_reviews("TASK-1144", cursor="previous", limit=500)) == page
    assert calls == [("/tasks/TASK-1144/reviews", {"limit": "50", "cursor": "previous"})]
    assert run(task_review_tools.list_task_reviews("task-cuid", limit=0)) == page
    assert calls[-1] == ("/tasks/task-cuid/reviews", {"limit": "1"})
    assert run(task_review_tools.list_task_reviews("TASK-1144")) == page
    assert calls[-1][1] == {"limit": "20"}


@pytest.mark.parametrize("stage", ["DESIGN", "IMPLEMENTATION"])
@pytest.mark.parametrize("verdict", ["APPROVE", "APPROVE_WITH_CHANGES", "REQUEST_CHANGES", "COMMENT"])
def test_reviews_add_forwards_typed_contract_and_bot_actor(monkeypatch, stage, verdict):
    calls = []

    async def fake_post(path, json, headers=None):
        calls.append((path, json, headers))
        return {"id": "review-1", "authorActorId": "snark", "body": json["body"]}

    monkeypatch.setattr(task_review_tools, "_api_post", fake_post)
    receipt = run(task_review_tools.add_task_review(
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

    monkeypatch.setattr(task_review_tools, "_api_post", fake_post)
    assert run(task_review_tools.add_task_review("TASK-1", "DESIGN", "APPROVE", "", "  ")) == {
        "error": "bot_id is required for review attribution"
    }
    assert calls == []
    assert run(task_review_tools.add_task_review("TASK-1", "DESIGN", "APPROVE", "", "al")) == {
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
        assert run(task_review_tools.list_task_reviews("TASK-99", cursor="page+two")) == {
            "error": "Task not found", "status": 404,
        }
        assert run(task_review_tools.add_task_review(
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

    monkeypatch.setattr(task_review_tools, "_api_post", fake_post)
    findings = [
        {"severity": "BLOCKER", "title": "Forged r2", "detail": "Public POST accepts it.",
         "file": "server/routes/task-reviews.ts", "line": 42,
         "failure_scenario": "Same actor posts r2 and clears r1."},
        {"severity": "NIT", "title": "Wording", "detail": "Tighten the label."},
    ]
    run(task_review_tools.add_task_review(
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
    run(task_review_tools.add_task_review("TASK-1615", "DESIGN", "COMMENT", "note", "loopy"))
    assert "findings" not in calls[1]


@pytest.mark.anyio
async def test_review_tool_discovery_types_findings():
    tools = {tool.name: tool for tool in await FastMCP.list_tools(mcp)}
    add = tools["tasks_reviews_add"].inputSchema
    assert "findings" in add["properties"] and "findings" not in add["required"]
    item = add["$defs"]["ReviewFindingInput"]
    assert item["properties"]["severity"]["enum"] == [
        "BLOCKER", "SHOULD_FIX", "NIT", "QUESTION", "PRE_EXISTING",
    ]
    assert set(item["required"]) == {"severity", "title", "detail"}


# TASK-1615 slice 2: signed reviews use the internal route; old builds fall back.

@pytest.fixture
def signed(monkeypatch):
    """A signed reviewer turn, with provenance resolution stubbed out."""
    async def fake_get(path, params=None):
        return {"id": "task-cuid", "shortId": "TASK-1"}

    monkeypatch.setattr(task_review_tools, "signed_review_turn", lambda *args, **kwargs: object())
    monkeypatch.setattr(task_review_tools, "resolve_review_provenance",
                        lambda turn, refs: {"provenance": "signed_turn"})
    monkeypatch.setattr(task_review_tools, "_api_get", fake_get)


@pytest.fixture
def internal(monkeypatch):
    """BawtHub's internal listener, answering with the handler set per test."""
    state = {"requests": [], "respond": None}
    real_client = httpx.AsyncClient

    def handler(request):
        state["requests"].append(request)
        return state["respond"](request)

    monkeypatch.setattr(task_review_tools, "service_auth", lambda: None)
    monkeypatch.setattr(task_review_tools.httpx, "AsyncClient",
                        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs))
    return state


def test_internal_route_404_is_missing_only_when_it_is_not_json(internal):
    internal["respond"] = lambda request: httpx.Response(404, text="404 Not Found")
    with pytest.raises(task_review_tools.InternalReviewRouteMissing):
        run(task_review_tools._internal_post("/internal/tasks/TASK-1/reviews", {}))
    internal["respond"] = lambda request: httpx.Response(
        404, json={"error": "Review not found", "code": "review_not_found"})
    with pytest.raises(httpx.HTTPStatusError) as raised:
        run(task_review_tools._internal_post("/internal/tasks/TASK-1/reviews", {}))
    assert task_review_tools._internal_error(raised.value) == {
        "error": "Review not found", "status": 404, "code": "review_not_found"}


def test_signed_review_rule_errors_keep_code_and_details(internal, signed):
    internal["respond"] = lambda request: httpx.Response(409, json={
        "error": "Blocker still open", "code": "blocker_unresolved", "details": {"findingIds": ["f1"]}})
    assert run(task_review_tools.add_task_review(
        "TASK-1", "IMPLEMENTATION", "APPROVE", "ok", "loopy",
        idempotency_key="k", rechecks_review_id="r1", resolutions=[],
    )) == {"error": "Blocker still open", "status": 409, "code": "blocker_unresolved",
           "details": {"findingIds": ["f1"]}}
    [request] = internal["requests"]
    assert request.url.path == "/internal/tasks/TASK-1/reviews/r1/recheck"


def test_signed_review_falls_back_to_public_route_on_an_old_bawthub(monkeypatch, internal, signed):
    internal["respond"] = lambda request: httpx.Response(404, text="404 Not Found")
    public = []

    async def fake_post(path, json, headers=None):
        public.append((path, json, headers))
        return {"id": "review-old"}

    monkeypatch.setattr(task_review_tools, "_api_post", fake_post)
    assert run(task_review_tools.add_task_review(
        "TASK-1", "DESIGN", "APPROVE", "ok", "loopy")) == {"id": "review-old"}
    [request] = internal["requests"]
    assert request.url.path == "/internal/tasks/TASK-1/reviews"
    # The public route gets the review as before slice 2: no actor in the body.
    assert public == [("/tasks/TASK-1/reviews", {
        "stage": "DESIGN", "verdict": "APPROVE", "body": "ok", "provenance": "signed_turn",
    }, {"X-Agent-Bot-Id": "loopy"})]


def test_recheck_and_resolution_never_fall_back(monkeypatch, internal, signed):
    internal["respond"] = lambda request: httpx.Response(404, text="404 Not Found")

    async def fake_post(path, json, headers=None):
        raise AssertionError("no public fallback")

    monkeypatch.setattr(task_review_tools, "_api_post", fake_post)
    assert run(task_review_tools.add_task_review(
        "TASK-1", "IMPLEMENTATION", "APPROVE", "ok", "loopy",
        idempotency_key="k", rechecks_review_id="r1",
    )) == {"error": "This BawtHub build cannot record rechecks yet", "code": "recheck_unavailable"}
    assert run(task_review_tools.resolve_review_finding(
        "TASK-1", "r1", "f1", "WITHDRAWN", "loopy",
    )) == {"error": "This BawtHub build cannot resolve findings yet", "code": "resolve_unavailable"}


def test_recheck_arguments_are_checked_before_any_request(monkeypatch):
    def no_turn(*args, **kwargs):
        raise AssertionError("checked before the signed turn")

    monkeypatch.setattr(task_review_tools, "signed_review_turn", no_turn)
    assert run(task_review_tools.add_task_review(
        "TASK-1", "IMPLEMENTATION", "APPROVE", "ok", "loopy",
        resolutions=[{"finding_id": "f1", "resolution": "FIXED"}],
    )) == {"error": "resolutions need rechecks_review_id", "code": "invalid_review"}
    assert run(task_review_tools.add_task_review(
        "TASK-1", "IMPLEMENTATION", "APPROVE", "ok", "loopy", rechecks_review_id="r1",
    )) == {"error": "a recheck needs idempotency_key", "code": "invalid_review"}


@pytest.mark.anyio
async def test_review_tool_discovery_types_rechecks_and_resolution():
    tools = {tool.name: tool for tool in await FastMCP.list_tools(mcp)}
    add = tools["tasks_reviews_add"].inputSchema
    assert {"rechecks_review_id", "resolutions"} <= add["properties"].keys()
    assert not {"rechecks_review_id", "resolutions"} & set(add["required"])
    resolve = tools["tasks_reviews_resolve_finding"].inputSchema
    assert set(resolve["required"]) == {"task_id", "review_id", "finding_id", "resolution", "bot_id"}
    assert resolve["properties"]["resolution"]["enum"] == ["FIXED", "DEFERRED", "WITHDRAWN"]


def test_null_or_empty_optional_entry_fields_are_not_sent():
    """TASK-1615 follow-up: a strict-schema client sends every optional key.

    Loopy's FIXED resolution arrived with task="" and BawtHub refused it
    ("FIXED takes no task"). Null and "" mean absent for optional entry keys.
    """
    assert task_review_tools._resolution_payload(
        {"finding_id": "f1", "resolution": "FIXED", "revision": "abc1234", "task": "", "note": None}
    ) == {"findingId": "f1", "resolution": "FIXED", "revision": "abc1234"}
    assert task_review_tools._finding_payload(
        {"severity": "NIT", "title": "t", "detail": "d", "file": "", "line": None,
         "failure_scenario": None}
    ) == {"severity": "NIT", "title": "t", "detail": "d"}
    assert task_review_tools._resolution_payload(
        {"finding_id": "f1", "resolution": "DEFERRED", "task": "TASK-9", "note": "later"}
    ) == {"findingId": "f1", "resolution": "DEFERRED", "task": "TASK-9", "note": "later"}


@pytest.mark.anyio
async def test_review_tool_discovery_lets_optional_entry_fields_be_null():
    tools = {tool.name: tool for tool in await FastMCP.list_tools(mcp)}
    defs = tools["tasks_reviews_add"].inputSchema["$defs"]
    resolution = defs["FindingResolutionInput"]
    assert set(resolution["required"]) == {"finding_id", "resolution"}
    for key in ("revision", "task", "note"):
        assert {"type": "null"} in resolution["properties"][key]["anyOf"]
    finding = defs["ReviewFindingInput"]
    for key in ("file", "line", "failure_scenario"):
        assert {"type": "null"} in finding["properties"][key]["anyOf"]
