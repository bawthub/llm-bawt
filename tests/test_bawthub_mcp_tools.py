"""Regular bots: allowlisted BawtHub MCP tools with deferred discovery (TASK-2599)."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llm_bawt.integrations.bawthub_mcp import client as mcp_client
from llm_bawt.integrations.bawthub_mcp.client import (
    BawtHubMcpClient, McpToolDefinition, parse_allowlist,
)
from llm_bawt.task_turn_context import TASK_TURN_CONTEXT_HEADER
from llm_bawt.tools.executor import ToolExecutor
from llm_bawt.tools.parser import ToolCall, strip_tool_result_tags, tool_result_is_error

CATALOG = [
    SimpleNamespace(name="memory_search", description="Search stored memories",
                    inputSchema={"type": "object", "properties": {
                        "query": {"type": "string", "description": "What to find"}},
                        "required": ["query"]}),
    SimpleNamespace(name="messages_find", description="Find past messages", inputSchema=None),
    SimpleNamespace(name="ops_run", description="Run a service operation", inputSchema=None),
]


@pytest.fixture
def client(monkeypatch):
    calls = []

    def bounded(coro, _timeout):
        return asyncio.run(coro)

    def fake(kind, value):
        async def run(self, headers, *args):
            calls.append((kind, value, headers, args))
            return value
        return run

    monkeypatch.setattr(mcp_client, "run_bounded", bounded)
    monkeypatch.setattr(BawtHubMcpClient, "_list_async", fake("list", CATALOG))
    result = SimpleNamespace(content=[SimpleNamespace(type="text", text='{"ok": true}')], isError=False)
    monkeypatch.setattr(BawtHubMcpClient, "_call_async", fake("call", result))
    out = BawtHubMcpClient("http://mcp.example:8001", parse_allowlist("memory_search, messages_find"))
    out.calls, out.result = calls, result
    return out


def _executor(client, capability="turn-token"):
    return ToolExecutor(user_id="fixture-owner", bot_id="fixture-bot",
                        bawthub_mcp=client, turn_capability=capability)


def test_allowlist_filters_the_server_catalog_and_search_ranks_by_name(client):
    assert [t.name for t in client.search_tools("turn-token", "")] == ["memory_search", "messages_find"]
    assert [t.name for t in client.search_tools("turn-token", "messages")] == ["messages_find"]
    assert client.search_tools("turn-token", "service operation") == []  # ops_run not allowed
    headers = client.calls[0][2]
    assert headers[TASK_TURN_CONTEXT_HEADER] == "turn-token"
    assert client._url == "http://mcp.example:8001/mcp"


def test_no_turn_identity_fails_closed_without_a_request(client):
    result = _executor(client, capability=None).execute(
        ToolCall(name="tool_search", arguments={"query": "memory"}, raw_text=""))
    assert tool_result_is_error(result) and "signed turn identity" in result
    assert client.calls == []


def test_tool_search_loads_schemas_and_renders_tool_references(client):
    executor = _executor(client)
    result = executor.execute(ToolCall(name="tool_search", arguments={"query": "memories"}, raw_text=""))
    assert not tool_result_is_error(result)
    assert json.loads(strip_tool_result_tags(result)) == [
        {"type": "tool_reference", "tool_name": "memory_search"}]
    (definition,) = executor.mcp_tool_definitions()
    assert definition.name == "memory_search"
    assert [(p.name, p.required) for p in definition.parameters] == [("query", True)]


def test_select_loads_exact_names_and_reports_no_match(client):
    executor = _executor(client)
    executor.execute(ToolCall(name="tool_search",
                              arguments={"query": "select:messages_find,ops_run"}, raw_text=""))
    assert list(executor.mcp_loaded) == ["messages_find"]
    empty = executor.execute(ToolCall(name="tool_search", arguments={"query": "weather"}, raw_text=""))
    assert strip_tool_result_tags(empty) == "No matching tools found."


def test_allowed_tool_calls_under_turn_identity_and_denied_tool_never_calls(client):
    executor = _executor(client)
    ok = executor.execute(ToolCall(name="memory_search", arguments={"query": "dogs", "n": None}, raw_text=""))
    assert strip_tool_result_tags(ok) == '{"ok": true}'
    kind, _, headers, args = client.calls[-1]
    assert kind == "call" and args == ("memory_search", {"query": "dogs"})
    assert headers[TASK_TURN_CONTEXT_HEADER] == "turn-token"
    before = len(client.calls)
    denied = executor.execute(ToolCall(name="ops_run", arguments={}, raw_text=""))
    assert tool_result_is_error(denied) and len(client.calls) == before


def test_server_error_and_unavailable_server_are_error_results(client, monkeypatch):
    executor = _executor(client)
    client.result.isError = True
    client.result.content[0].text = "approval required"
    held = executor.execute(ToolCall(name="memory_search", arguments={"query": "x"}, raw_text=""))
    assert tool_result_is_error(held) and "approval required" in held
    monkeypatch.setattr(mcp_client, "run_bounded", Mock(side_effect=OSError("refused")))
    down = executor.execute(ToolCall(name="memory_search", arguments={"query": "x"}, raw_text=""))
    assert tool_result_is_error(down) and "refused" in down


def test_opt_in_replaces_local_wrappers_with_tool_search():
    from llm_bawt.core.base import BaseLLMBawt

    bot = SimpleNamespace(search_client=None, news_client=None, web_fetch_client=None,
                          home_client=None, model_lifecycle=None, ha_native_client=None,
                          memory=object(), bawthub_mcp=None)
    names = {t.name for t in BaseLLMBawt._get_tool_definitions(bot)}
    assert {"memory", "history", "profile"} <= names and "tool_search" not in names
    bot.bawthub_mcp = object()
    names = {t.name for t in BaseLLMBawt._get_tool_definitions(bot)}
    assert "tool_search" in names and not names & {"memory", "history", "profile"}
    assert "time" in names  # Local tools without an MCP equivalent stay.

