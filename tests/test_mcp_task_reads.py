"""Bound the serialized task-reader contract without touching live task data."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from llm_bawt.mcp_server import project_tools, task_tools
from llm_bawt.mcp_server.task_reads import MAX_BUDGET

GIANT = "a" * 170_281


def fixture():
    task = {
        "id": "task-id", "shortId": "TASK-1907", "title": "Bound task reads",
        "status": "BLOCKED", "priority": "HIGH", "agentBotId": "caid",
        "blockedReason": "Rook must obtain the publication receipt.",
        "description": GIANT, "response": "Current receipt. Next: inspect the exact published tip.",
        "outcome": "Bounded and complete task reads", "updatedAt": "2026-11-01T19:00:00Z",
        "project": {"id": "project-id", "name": "Chat"},
        "steps": [{"id": f"step-{i}", "title": f"Step {i}", "type": "REVIEW",
                   "status": "RUNNING" if i == 0 else "PENDING" if i == 1 else "COMPLETED",
                   "output": "Receipt"} for i in range(24)],
        "dependsOn": [{"id": "dependency", "shortId": "TASK-1", "status": "IN_PROGRESS"}],
        "checkLinks": [{"label": "DEV", "path": "/chat", "steps": ["Open chat"]}],
        "unknownFutureField": {"a/b": {"~key": "Preserved unknown field"}},
    }
    reviews = [{"id": f"r-{i}", "verdict": "APPROVE", "body": f"Review {i}",
                "actorId": "loopy", "stage": "IMPLEMENTATION", "targetRevision": "abc123", "signedVia": "INTERNAL_SIGNED"} for i in range(5)]
    project = {"id": "project-id", "name": "Chat", "description": "Shared project",
               "contextPrompt": "Inspect the exact revision", "tasks": [task]}
    return task, reviews, project


class Api:
    def __init__(self):
        self.task, self.reviews, self.project = fixture()
        self.calls = []
        self.fail_reviews = False
        self.fail_project = False

    async def __call__(self, path, params=None):
        self.calls.append((path, params))
        if path == "/tasks/TASK-1907":
            return self.task
        if path == "/tasks/TASK-1907/reviews":
            if self.fail_reviews:
                raise httpx.ConnectError("offline")
            assert params["limit"] == "3"
            start = int(params.get("cursor", "0"))
            end = start + 3
            return {"reviews": self.reviews[start:end], "total": len(self.reviews),
                    "nextCursor": str(end) if end < len(self.reviews) else None}
        if path == "/projects/project-id":
            if self.fail_project:
                raise httpx.ConnectError("offline")
            return self.project
        raise AssertionError(f"Unexpected API read {path}")


def run(coro):
    return asyncio.run(coro)


def wire_size(value):
    return len(json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode())


def install(monkeypatch, api):
    monkeypatch.setattr(task_tools, "_api_get", api)
    monkeypatch.setattr(project_tools, "_api_get", api)


@pytest.mark.parametrize("reader", [task_tools.get_task, project_tools.get_task_context])
def test_compact_defaults_preserve_current_owner_blocker_next_action_and_counts(monkeypatch, reader):
    api = Api()
    install(monkeypatch, api)
    result = run(reader("TASK-1907"))
    assert len(json.dumps(api.task)) > 170_281
    assert result["view"] == "summary"
    task = result["task"]
    assert (task["status"], task["agentBotId"], task["blockedReason"]) == (
        "BLOCKED", "caid", "Rook must obtain the publication receipt.")
    assert "Next: inspect" in task["response"]
    assert task["description"]["complete"] is False
    assert task["description"]["total_chars"] == len(GIANT)
    assert task["description"]["path"] == "/task/description"
    assert task["steps"]["total"] == 24
    assert {row["index"] for row in task["steps"]["items"]} == {0, 1, 23}
    assert task["reviews"]["shown"] == 3
    assert task["reviews"]["next_cursor"] == "3"
    assert wire_size(result) <= MAX_BUDGET
    assert not api.task.get("view"), "Never mutate the source"
    if reader is project_tools.get_task_context:
        assert task["project"]["contextPrompt"] == api.project["contextPrompt"]
        assert "tasks" not in task["project"]


def test_recent_review_summary_keeps_revision_provenance_and_open_blocker_counts(monkeypatch):
    api = Api()
    api.reviews[0]["findings"] = [
        {"severity": "BLOCKER", "resolution": None},
        {"severity": "BLOCKER", "resolution": "FIXED"},
        {"severity": "NIT", "resolution": None},
    ]
    install(monkeypatch, api)
    result = run(task_tools.get_task("TASK-1907"))
    reviews = result["task"]["reviews"]
    assert reviews["total"] == 5
    review = reviews["items"][0]
    assert review["actorId"] == "loopy"
    assert review["signedVia"] == "INTERNAL_SIGNED"
    assert review["targetRevision"] == "abc123"
    assert review["findingCount"] == 3
    assert review["openBlockers"] == 1


async def complete(reader, arguments):
    chunks = []
    previous = 0
    first = None
    for _ in range(1000):
        result = await reader(**arguments)
        assert "error" not in result, result
        assert wire_size(result) <= arguments.get("max_chars", MAX_BUDGET)
        if first is None:
            first = result
        assert result["offset"] == previous
        chunks.append(result["text"])
        previous += len(result["text"])
        if result["next_offset"] is None:
            assert previous == result["total_chars"]
            text = "".join(chunks)
            return (json.loads(text) if result["encoding"] == "json" else text), first, result
        assert result["next_offset"] == previous
        assert result["next"]["arguments"]["version"] == result["version"]
        assert result["next"]["tool"] in ("tasks_get", "tasks_get_context")
        arguments = result["next"]["arguments"]
    raise AssertionError("Paging did not terminate")


@pytest.mark.parametrize("reader", [task_tools.get_task, project_tools.get_task_context])
def test_complete_raw_task_is_lossless_including_unknown_fields(monkeypatch, reader):
    api = Api()
    install(monkeypatch, api)
    value, first, last = run(complete(reader, {"task_id": "TASK-1907", "path": "/task"}))
    assert value == api.task
    assert first["next_offset"] > 0
    assert "next" not in last


@pytest.mark.parametrize("field", ["description", "response", "blockedReason", "step-title", "step-output", "review", "project"])
def test_huge_single_line_fields_remain_completely_retrievable(monkeypatch, field):
    api = Api()
    # Quote/backslash/control and multi-byte unicode cannot evade the JSON budget.
    text = ('"\\\t漢😀' * 40_000) + "NEXT ACTION: obtain receipt"
    if field == "step-title":
        api.task["steps"][0]["title"] = text
        path = "/task/steps/0/title"
    elif field == "step-output":
        api.task["steps"][0]["output"] = text
        path = "/task/steps/0/output"
    elif field == "review":
        api.reviews[0]["body"] = text
        path = "/reviews/reviews/0/body"
    elif field == "project":
        api.project["contextPrompt"] = text
        path = "/project/contextPrompt"
    else:
        api.task[field] = text
        path = f"/task/{field}"
    install(monkeypatch, api)
    compact = run(project_tools.get_task_context("TASK-1907"))
    assert compact["view"] == "summary"
    assert wire_size(compact) <= MAX_BUDGET
    value, _, _ = run(complete(project_tools.get_task_context,
                              {"task_id": "TASK-1907", "path": path}))
    assert value == text


def test_all_review_pages_including_huge_first_review_are_lossless(monkeypatch):
    api = Api()
    api.reviews[0]["body"] = GIANT
    install(monkeypatch, api)
    seen = []
    arguments = {"task_id": "TASK-1907", "path": "/reviews"}
    for _ in range(3):
        page, _, last = run(complete(task_tools.get_task, arguments))
        seen.extend(page["reviews"])
        if "next_page" not in last:
            break
        arguments = last["next_page"]["arguments"]
        assert "version" not in arguments, "A new backend page has a new revision"
    assert seen == api.reviews


@pytest.mark.parametrize("changed", ["description", "steps"])
def test_changed_detail_is_refused_instead_of_splicing_revisions(monkeypatch, changed):
    api = Api()
    install(monkeypatch, api)
    first = run(task_tools.get_task("TASK-1907", path="/task"))
    if changed == "description":
        api.task["description"] += "changed"
    else:
        api.task["steps"].pop()
    result = run(task_tools.get_task(**first["next"]["arguments"]))
    assert result["error"] == "detail_changed"
    assert "text" not in result


def test_pointer_escaping_and_types(monkeypatch):
    api = Api()
    install(monkeypatch, api)
    value, _, _ = run(complete(task_tools.get_task, {"task_id": "TASK-1907",
                          "path": "/task/unknownFutureField/a~1b/~0key"}))
    assert value == "Preserved unknown field"
    for value in [None, False, 0, "", [], {}]:
        api.task["value"] = value
        read, _, _ = run(complete(task_tools.get_task, {"task_id": "TASK-1907", "path": "/task/value"}))
        assert read == value
        assert type(read) is type(value)


@pytest.mark.parametrize("arguments", [
    {"path": "/task/steps/-1/output"}, {"path": "/task/steps/01"},
    {"path": "/task/no~2escape"}, {"path": "/task/missing"},
    {"path": "description"}, {"path": "/other"}, {"offset": -1},
    {"path": "/task", "offset": 1}, {"offset": 1, "version": "x"},
    {"path": "/task", "cursor": "x"}, {"cursor": "x"},
    {"path": "/" + "x" * 501}, {"path": "/reviews", "cursor": "x" * 513},
])
def test_invalid_reads_fail_with_bounded_errors(monkeypatch, arguments):
    api = Api()
    install(monkeypatch, api)
    result = run(task_tools.get_task("TASK-1907", **arguments))
    assert "error" in result
    assert wire_size(result) < 1000


@pytest.mark.parametrize("budget", [1, 2000, 4000, 8000, 1_000_000])
def test_whole_serialized_page_budget_is_clamped(monkeypatch, budget):
    api = Api()
    api.task["description"] = '"\\😀' * 60_000
    install(monkeypatch, api)
    result = run(task_tools.get_task("TASK-1907", path="/task/description", max_chars=budget))
    assert "error" not in result
    assert wire_size(result) <= max(2000, min(budget, MAX_BUDGET))
    assert result["next_offset"] > 0


@pytest.mark.parametrize("budget", [2000, 4000, 8000])
def test_summary_budget_with_many_huge_fields_keeps_current_state(monkeypatch, budget):
    api = Api()
    text = '"\\\\😀' * 60_000
    for field in ("title", "description", "response", "outcome", "blockedReason"):
        api.task[field] = text
    for row in api.task["steps"]:
        row["title"] = row["output"] = text
    for row in api.reviews:
        row["body"] = text
    api.project["contextPrompt"] = api.project["description"] = text
    install(monkeypatch, api)
    result = run(project_tools.get_task_context("TASK-1907", max_chars=budget))
    assert result["view"] == "summary"
    assert result["task"]["agentBotId"] == "caid"
    assert result["task"]["status"] == "BLOCKED"
    assert wire_size(result) <= budget


def test_optional_sources_unavailable_do_not_hide_task_state(monkeypatch):
    api = Api()
    api.fail_reviews = api.fail_project = True
    install(monkeypatch, api)
    result = run(project_tools.get_task_context("TASK-1907"))
    assert result["task"]["agentBotId"] == "caid"
    assert result["task"]["reviews"]["unavailable"] is True
    for path in ("/reviews", "/project"):
        result = run(task_tools.get_task("TASK-1907", path=path))
        assert result["error"] == "task_read_unavailable"


def test_not_found_error_does_not_echo_huge_server_body(monkeypatch):
    async def missing(path, params=None):
        raise httpx.HTTPStatusError(GIANT, request=httpx.Request("GET", "http://local/task"),
                                   response=httpx.Response(404))
    install(monkeypatch, missing)
    assert run(task_tools.get_task("TASK-1907")) == {"error": "task_read_failed", "status": 404}
    assert run(project_tools.get_task_context("TASK-1907")) == {"error": "task_read_failed", "status": 404}


@pytest.mark.parametrize("reader", [task_tools.get_task, project_tools.get_task_context])
@pytest.mark.parametrize("path", [None, "/task/description"])
def test_real_fastmcp_serialized_result_fits_transport_budget(monkeypatch, reader, path):
    from mcp.server.fastmcp import FastMCP
    from mcp.types import CallToolResult
    api = Api()
    api.task["description"] = '"\\😀' * 60_000
    install(monkeypatch, api)

    async def exercise():
        server = FastMCP("bounded-task-test")
        server.add_tool(reader, name="read")
        result = await server.call_tool("read", {"task_id": "TASK-1907", "path": path})
        if isinstance(result, tuple):
            content, structured = result
        else:
            content, structured = result, None
        return CallToolResult(content=content, structuredContent=structured).model_dump_json()

    wire = run(exercise())
    # Includes FastMCP's text/structured duplication and escaping, not just value.
    assert len(wire.encode()) < 24_000


def test_wrappers_preserve_same_detail_and_catalog_inputs(monkeypatch):
    import inspect
    api = Api()
    install(monkeypatch, api)
    args = {"task_id": "TASK-1907", "path": "/task/steps/0/output"}
    assert run(task_tools.get_task(**args))["text"] == run(project_tools.get_task_context(**args))["text"]
    assert set(inspect.signature(task_tools.get_task).parameters) == set(inspect.signature(project_tools.get_task_context).parameters)
