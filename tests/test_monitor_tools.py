"""Monitor MCP tools over BawtHub's health API (TASK-1712)."""

from __future__ import annotations

import asyncio

import httpx

import llm_bawt.mcp_server.server as server  # noqa: F401  registers tools; import first
import llm_bawt.mcp_server.monitor_tools as tools

CHECK = {"id": "c1", "slug": "svc.a", "kind": "http", "config": {"timeoutMs": 2000, "wake": {"bot": "helper"}}}


def _client(monkeypatch, seen: list):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, request.url.query.decode(), request.content))
        if request.url.path.endswith("/checks"):
            return httpx.Response(200, json={"checks": [CHECK]})
        if request.url.path == "/api/system/health":
            return httpx.Response(200, json={"checks": [{"slug": "svc.a", "status": "down", "reason": "HTTP 503"}]})
        return httpx.Response(200, json={"check": {"id": "c1"}})

    client = httpx.AsyncClient(base_url="http://bawthub.test", transport=httpx.MockTransport(handler))
    monkeypatch.setattr(tools, "get_client", lambda: client)


def test_mute_merges_config_by_slug(monkeypatch):
    seen: list = []
    _client(monkeypatch, seen)
    asyncio.run(tools.monitors_mute("svc.a", minutes=30))
    method, path, _, body = seen[-1]
    assert (method, path) == ("PATCH", "/api/system/health/checks/c1")
    config = httpx.Response(200, content=body).json()["config"]
    assert config["timeoutMs"] == 2000 and config["wake"] == {"bot": "helper"}
    assert config["mutedUntil"].endswith("Z")


def test_list_joins_verdicts_and_test_runs_fresh(monkeypatch):
    seen: list = []
    _client(monkeypatch, seen)
    listed = asyncio.run(tools.monitors_list())
    assert listed["monitors"][0]["status"] == "down"
    assert asyncio.run(tools.monitors_test("c1"))["reason"] == "HTTP 503"
    assert seen[-1][2] == "fresh=1&probe=svc.a"


def test_unknown_monitor_is_an_error(monkeypatch):
    _client(monkeypatch, [])
    assert "error" in asyncio.run(tools.monitors_mute("nope"))
