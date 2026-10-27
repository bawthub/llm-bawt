"""Tests for the tasks_search_semantic MCP proxy tool (TASK-417).

The tool is a thin proxy over bawthub's POST /api/tasks/tasks/search/semantic
endpoint. These tests verify request shaping (path, body, limit clamp), the
empty-query guard, and pass-through of both semantic and keyword-fallback
payloads — without any network.
"""

import asyncio
from typing import Any

import httpx
import pytest

from llm_bawt.mcp_server import task_tools


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_semantic_search_posts_query_and_returns_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    payload = {
        "mode": "semantic",
        "results": [
            {
                "id": "task-id",
                "shortId": "TASK-266",
                "title": "login returns 500",
                "description": "auth path throws",
                "status": "REVIEW",
                "priority": "HIGH",
                "projectId": None,
                "score": 0.83,
            }
        ],
    }

    async def fake_post(path: str, json: dict[str, Any], headers: dict | None = None) -> dict:
        calls.append({"path": path, "json": json, "headers": headers})
        return payload

    monkeypatch.setattr(task_tools, "_api_post", fake_post)

    result = _run(task_tools.search_tasks_semantic("auth is broken"))

    assert calls == [
        {"path": "/tasks/search/semantic", "json": {"query": "auth is broken", "limit": 10}, "headers": None}
    ]
    # TASK-1555: the row's description becomes its length; tasks_get_context reads it.
    row = {key: value for key, value in payload["results"][0].items() if key != "description"}
    assert result == {"mode": "semantic", "results": [{**row, "descriptionChars": 16}]}
    assert result["mode"] == "semantic"
    assert result["results"][0]["shortId"] == "TASK-266"


def test_semantic_search_clamps_limit_to_25(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    async def fake_post(path: str, json: dict[str, Any], headers: dict | None = None) -> dict:
        seen.update(json)
        return {"mode": "semantic", "results": []}

    monkeypatch.setattr(task_tools, "_api_post", fake_post)

    _run(task_tools.search_tasks_semantic("anything", limit=5000))

    assert seen["limit"] == 25


def test_semantic_search_empty_query_short_circuits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    async def fake_post(path: str, json: dict[str, Any], headers: dict | None = None) -> dict:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(task_tools, "_api_post", fake_post)

    result = _run(task_tools.search_tasks_semantic("   "))

    assert result == {"error": "query is required"}
    assert called is False, "blank query must not hit the API"


def test_semantic_search_passes_through_keyword_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_post(path: str, json: dict[str, Any], headers: dict | None = None) -> dict:
        # Endpoint degrades to keyword scan when the embed service is down.
        return {"mode": "keyword", "results": [{"shortId": "TASK-9", "score": 0}]}

    monkeypatch.setattr(task_tools, "_api_post", fake_post)

    result = _run(task_tools.search_tasks_semantic("login"))

    assert result["mode"] == "keyword"
    assert result["results"][0]["score"] == 0


def test_semantic_search_surfaces_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_post(path: str, json: dict[str, Any], headers: dict | None = None) -> dict:
        request = httpx.Request("POST", "http://echo/api/tasks/tasks/search/semantic")
        response = httpx.Response(500, request=request)
        raise httpx.HTTPStatusError("boom", request=request, response=response)

    monkeypatch.setattr(task_tools, "_api_post", fake_post)

    result = _run(task_tools.search_tasks_semantic("login"))

    assert result["status"] == 500
    assert "error" in result
