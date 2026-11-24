"""Regular bots: allowlisted BawtHub MCP tools with deferred discovery (TASK-2599)."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock

import mcp.types as mcp_types
import pytest

from llm_bawt.integrations.bawthub_mcp import client as mcp_client
from llm_bawt.integrations.bawthub_mcp.client import (
    BawtHubMcpClient, McpToolDefinition, parse_allowlist,
)
from llm_bawt.task_turn_context import TASK_TURN_CONTEXT_HEADER
from llm_bawt.tools.executor import ToolExecutor
from llm_bawt.tools.parser import ToolCall, strip_tool_result_tags, tool_result_is_error

_EMPTY = {"type": "object", "properties": {}}
# Real SDK objects, so the client reads the field the installed SDK exposes.
CATALOG = [
    mcp_types.Tool(name="memory_search", description="Search stored memories",
                   input_schema={"type": "object", "properties": {
                       "query": {"type": "string", "description": "What to find"},
                       "bot_id": {"type": "string"}},
                       "required": ["query"]}),
    mcp_types.Tool(name="messages_find", description="Find past messages", input_schema=_EMPTY),
    mcp_types.Tool(name="ops_run", description="Run a service operation", input_schema=_EMPTY),
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
    # Only the signed turn: the :8001 gate refuses a turn plus a key.
    assert headers == {TASK_TURN_CONTEXT_HEADER: "turn-token"}
    assert client._url == "http://mcp.example:8001/mcp"


def test_no_turn_identity_fails_closed_without_a_request(client):
    result = _executor(client, capability=None).execute(
        ToolCall(name="tool_search", arguments={"query": "memory"}, raw_text=""))
    assert tool_result_is_error(result) and "signed turn identity" in result
    assert client.calls == []


def test_warm_cache_still_requires_turn_identity(client):
    client.search_tools("turn-token", "")  # warms the catalog
    result = _executor(client, capability=None).execute(
        ToolCall(name="tool_search", arguments={"query": "memory"}, raw_text=""))
    assert tool_result_is_error(result) and "signed turn identity" in result


def test_only_an_absent_allowlist_gets_the_default():
    from llm_bawt.core.initialization import OrchestratorInitializationMixin

    assert parse_allowlist(None) is None
    assert parse_allowlist("") == frozenset() and parse_allowlist([]) == frozenset()
    assert "profile" not in mcp_client.DEFAULT_ALLOWLIST  # can target any entity

    def init(allowlist):
        settings = {"bawthub_mcp_tools": "true", "bawthub_mcp_tool_allowlist": allowlist}
        bot = SimpleNamespace(bot=SimpleNamespace(uses_tools=True), model_definition={},
                              bot_id="fixture-bot",
                              _resolve_setting=lambda key, fallback=None: settings.get(key, fallback))
        OrchestratorInitializationMixin._init_bawthub_mcp(
            bot, SimpleNamespace(MCP_SERVER_URL="http://mcp.example:8001"))
        return bot.bawthub_mcp.allowlist

    assert init(None) == mcp_client.DEFAULT_ALLOWLIST
    assert init("") == frozenset()
    assert init("messages_find") == {"messages_find"}


def test_tool_search_loads_schemas_and_renders_tool_references(client):
    executor = _executor(client)
    result = executor.execute(ToolCall(name="tool_search", arguments={"query": "memories"}, raw_text=""))
    assert not tool_result_is_error(result)
    assert json.loads(strip_tool_result_tags(result)) == [
        {"type": "tool_reference", "tool_name": "memory_search"}]
    (definition,) = executor.mcp_tool_definitions()
    assert definition.name == "memory_search"
    assert [(p.name, p.required) for p in definition.parameters] == [("query", True), ("bot_id", False)]


def test_discovered_tool_defaults_omitted_bot_id_to_the_caller(client):
    executor = _executor(client)
    executor.execute(ToolCall(name="tool_search", arguments={"query": "select:memory_search,messages_find"},
                              raw_text=""))
    executor.execute(ToolCall(name="memory_search", arguments={"query": "dogs"}, raw_text=""))
    assert client.calls[-1][3] == ("memory_search", {"query": "dogs", "bot_id": "fixture-bot"})
    executor.execute(ToolCall(name="memory_search", arguments={"query": "dogs", "bot_id": "other"},
                              raw_text=""))
    assert client.calls[-1][3] == ("memory_search", {"query": "dogs", "bot_id": "other"})
    executor.execute(ToolCall(name="messages_find", arguments={}, raw_text=""))
    assert client.calls[-1][3] == ("messages_find", {})


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


def test_replaced_local_wrappers_and_aliases_are_refused_in_mcp_mode(client):
    memory = Mock()
    executor = ToolExecutor(user_id="fixture-owner", bot_id="fixture-bot", memory_client=memory,
                            bawthub_mcp=client, turn_capability="turn-token")
    for name, args in (("memory", {"action": "search", "query": "x"}),
                       ("search_memories", {"query": "x"}),
                       ("history", {"action": "recent"}),
                       ("get_user_profile", {})):
        result = executor.execute(ToolCall(name=name, arguments=args, raw_text=""))
        assert tool_result_is_error(result) and "not enabled" in result, name
    assert memory.method_calls == [] and client.calls == []


def test_exact_mcp_name_dispatches_before_legacy_aliases(client, monkeypatch):
    from llm_bawt.tools import executor as executor_module

    monkeypatch.setitem(executor_module.normalize_legacy_tool_call.__globals__["LEGACY_TOOL_MAP"],
                        "messages_find", ("history", {"action": "search"}))
    _executor(client).execute(ToolCall(name="messages_find", arguments={"query": "x"}, raw_text=""))
    assert client.calls[-1][3] == ("messages_find", {"query": "x"})


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
    names = {t.name for t in BaseLLMBawt._get_tool_definitions(bot, bawthub_mcp=True)}
    assert {"memory", "history", "profile"} <= names and "tool_search" not in names
    bot.bawthub_mcp = object()
    # ReAct and non-streamed turns can't run MCP discovery: they keep the wrappers.
    names = {t.name for t in BaseLLMBawt._get_tool_definitions(bot)}
    assert {"memory", "history", "profile"} <= names and "tool_search" not in names
    names = {t.name for t in BaseLLMBawt._get_tool_definitions(bot, bawthub_mcp=True)}
    assert "tool_search" in names and not names & {"memory", "history", "profile"}
    assert "time" in names  # Local tools without an MCP equivalent stay.


def test_prompt_advertises_discovery_only_on_the_native_stream_path():
    from llm_bawt.core.base import BaseLLMBawt

    client = SimpleNamespace(stream_with_tools=object(), supports_native_tools=lambda: True)
    bot = SimpleNamespace(bawthub_mcp=object(), client=client, tool_format="native")
    assert BaseLLMBawt._uses_bawthub_mcp_native_stream(bot)
    bot.tool_format = "react"
    assert not BaseLLMBawt._uses_bawthub_mcp_native_stream(bot)
    bot.tool_format, bot.bawthub_mcp = "native", None
    assert not BaseLLMBawt._uses_bawthub_mcp_native_stream(bot)


def test_identity_only_turn_owns_no_conversation(monkeypatch):
    from uuid import uuid4

    from cryptography.fernet import Fernet

    from llm_bawt import task_turn_context as codec
    from llm_bawt.mcp_server import bot_access_identity, task_association

    fernet = Fernet(Fernet.generate_key())
    monkeypatch.setattr(codec, "_get_fernet", lambda: fernet)
    fields = dict(session_id=str(uuid4()), turn_id="turn-" + "c" * 32,
                  trigger_message_id=str(uuid4()), bot_id="fixture-bot", user_id="fixture-owner")
    ordinary = codec.mint_task_turn_context(**fields)
    identity = codec.mint_task_turn_context(**fields, identity_only=True)
    # Ordinary tokens keep their exact payload shape.
    assert "identity_only" not in json.loads(fernet.decrypt(ordinary.encode()))
    assert codec.open_task_turn_context(identity).identity_only is True
    with pytest.raises(codec.TaskTurnContextError):
        codec.require_conversation_authority(codec.open_task_turn_context(identity))

    caller = bot_access_identity.TrustedCaller("fixture-bot", "fixture-owner", "turn")
    for token, owner in ((ordinary, ("fixture-bot", "fixture-owner")), (identity, None)):
        binding = task_association.set_current_task_turn_capability(token)
        try:
            assert bot_access_identity.trusted_session_owner(fields["session_id"], caller) == owner
            if owner is None:
                with pytest.raises(codec.TaskTurnContextError):
                    asyncio.run(task_association.associate_current_task("TASK-1"))
        finally:
            task_association.reset_current_task_turn_capability(binding)



def test_stream_and_nonstream_prompts_match_their_executors():
    """Same opted-in native bot: only the streamed turn's prompt offers discovery."""
    from llm_bawt.core.base import BaseLLMBawt
    from llm_bawt.core.prompt_sections import PromptSectionsMixin, bawthub_mcp_prompt_turn

    client = SimpleNamespace(stream_with_tools=object(), supports_native_tools=lambda: True)
    bot = SimpleNamespace(search_client=None, news_client=None, web_fetch_client=None,
                          home_client=None, model_lifecycle=None, ha_native_client=None,
                          memory=object(), bawthub_mcp=object(), client=client,
                          tool_format="native", bot=SimpleNamespace(uses_tools=True))
    bot._uses_bawthub_mcp_native_stream = lambda: BaseLLMBawt._uses_bawthub_mcp_native_stream(bot)
    bot._get_tool_definitions = lambda **kw: BaseLLMBawt._get_tool_definitions(bot, **kw)

    def tools_prompt():
        builder = Mock()
        PromptSectionsMixin._sec_tools(bot, builder, "hello")
        return builder.add_section.call_args.args[1]

    nonstream = tools_prompt()
    with bawthub_mcp_prompt_turn():
        stream = tools_prompt()
    assert "tool_search" in stream and "tool_search" not in nonstream
    assert "tool_search" not in tools_prompt()  # the mark is scoped to the turn
