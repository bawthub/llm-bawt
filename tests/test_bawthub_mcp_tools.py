"""Regular bots: allowlisted BawtHub MCP tools with deferred discovery (TASK-2599)."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock

import mcp.types as mcp_types
from agent_bridge.mcp_call_context import MCP_CALL_CONTEXT_KEY
import pytest

from llm_bawt.integrations.bawthub_mcp import client as mcp_client
from llm_bawt.integrations.bawthub_mcp.client import (
    BawtHubMcpClient, parse_allowlist,
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
    calls, stamps = [], []
    # The real stamp is covered against the real interceptor below.
    monkeypatch.setattr(BawtHubMcpClient, "_call_context",
                        staticmethod(lambda capability, name, arguments, tool_use_id: {"stamp": name}))

    def bounded(coro, _timeout):
        return asyncio.run(coro)

    def fake(kind, value):
        async def run(self, headers, *args):
            if kind == "call":
                name, arguments = args
                stamps.append(arguments.get(MCP_CALL_CONTEXT_KEY))
                args = (name, {k: v for k, v in arguments.items() if k != MCP_CALL_CONTEXT_KEY})
            calls.append((kind, value, headers, args))
            return value
        return run

    monkeypatch.setattr(mcp_client, "run_bounded", bounded)
    monkeypatch.setattr(BawtHubMcpClient, "_list_async", fake("list", CATALOG))
    result = SimpleNamespace(content=[SimpleNamespace(type="text", text='{"ok": true}')], isError=False)
    monkeypatch.setattr(BawtHubMcpClient, "_call_async", fake("call", result))
    out = BawtHubMcpClient("http://mcp.example:8001", parse_allowlist("memory_search, messages_find"))
    out.calls, out.result, out.stamps = calls, result, stamps
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


def test_multi_word_search_ranks_partial_matches_instead_of_returning_nothing(client):
    # No tool has every word; a model's keyword list must still find tools.
    query = "memory store search facts recall history profile"
    assert [t.name for t in client.search_tools("turn-token", query)] == ["memory_search"]
    assert [t.name for t in client.search_tools("turn-token", "find past memories", limit=5)] == [
        "messages_find", "memory_search"]
    assert client.search_tools("turn-token", "service operation") == []


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
    # Memory, history and profiles come only from BawtHub MCP (TASK-2614).
    assert {"memory_search", "messages_find", "profile"} <= mcp_client.DEFAULT_ALLOWLIST

    def init(allowlist):
        settings = {"bawthub_mcp_tool_allowlist": allowlist}
        bot = SimpleNamespace(bot=SimpleNamespace(uses_tools=True), model_definition={},
                              bot_id="fixture-bot",
                              _resolve_setting=lambda key, fallback=None: settings.get(key, fallback))
        OrchestratorInitializationMixin._init_bawthub_mcp(
            bot, SimpleNamespace(MCP_SERVER_URL="http://mcp.example:8001"))
        return bot.bawthub_mcp.allowlist

    # No opt-in switch: every regular tool-using bot gets BawtHub MCP.
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



def _sdk_result(text=None, structured=None, is_error=False):
    content = [mcp_types.TextContent(type="text", text=text)] if text is not None else []
    return mcp_types.CallToolResult(content=content, structured_content=structured, is_error=is_error)


@pytest.mark.parametrize("result, expected", [
    (_sdk_result(text="found 2", structured={"results": [1, 2]}), "found 2"),
    (_sdk_result(structured={"results": []}), '{"results": []}'),
    (_sdk_result(structured={}), "{}"),
    (_sdk_result(text="   ", structured=[]), "[]"),
    (_sdk_result(), "memory_search returned no content"),
])
def test_result_prefers_text_then_structured_json_never_sdk_repr(client, monkeypatch, result, expected):
    async def call(self, headers, name, arguments):
        return result
    monkeypatch.setattr(BawtHubMcpClient, "_call_async", call)
    out = client.call_tool("turn-token", "memory_search", {"query": "x"}, tool_use_id="call-1")
    assert out == expected and "meta=" not in out


def test_structured_only_result_end_to_end_and_policy_refusal(client, monkeypatch):
    async def call(self, headers, name, arguments):
        return state["result"]
    state = {"result": _sdk_result(structured={"results": []})}
    monkeypatch.setattr(BawtHubMcpClient, "_call_async", call)
    executor = _executor(client)
    ok = executor.execute(ToolCall(name="memory_search", arguments={"query": "x"}, raw_text=""))
    assert strip_tool_result_tags(ok) == '{"results": []}' and not tool_result_is_error(ok)
    state["result"] = _sdk_result(structured={"is_error": True, "error": "access_denied"})
    refused = executor.execute(ToolCall(name="memory_search", arguments={"query": "x"}, raw_text=""))
    assert tool_result_is_error(refused) and "access_denied" in refused

def test_deleted_local_wrappers_are_unknown_tools(client):
    executor = _executor(client)
    for name, args in (("memory", {"action": "search", "query": "x"}),
                       ("search_memories", {"query": "x"}),
                       ("history", {"action": "recent"}),
                       ("self", {"action": "get"}),
                       ("get_user_profile", {})):
        result = executor.execute(ToolCall(name=name, arguments=args, raw_text=""))
        assert tool_result_is_error(result) and "Unknown tool" in result, name
    assert client.calls == []


def test_profile_defaults_to_this_bot_or_this_turns_user(client):
    executor = _executor(client)
    client._allowlist = client._allowlist | {"profile"}
    executor.execute(ToolCall(name="profile", arguments={"action": "get", "entity_type": "bot"},
                              raw_text=""))
    assert client.calls[-1][3] == ("profile", {"action": "get", "entity_type": "bot",
                                               "entity_id": "fixture-bot"})
    executor.execute(ToolCall(name="profile", arguments={"action": "get"}, raw_text=""))
    assert client.calls[-1][3] == ("profile", {"action": "get", "entity_id": "fixture-owner"})


def test_react_tool_search_returns_loaded_usage_inline(client):
    executor = ToolExecutor(user_id="fixture-owner", bot_id="fixture-bot", bawthub_mcp=client,
                            turn_capability="turn-token", inline_loaded_schemas=True)
    result = strip_tool_result_tags(executor.execute(
        ToolCall(name="tool_search", arguments={"query": "select:memory_search"}, raw_text="")))
    assert "memory_search" in result and "What to find" in result


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


def test_mcp_tool_search_replaces_local_memory_history_and_profile_tools():
    from llm_bawt.core.base import BaseLLMBawt

    bot = SimpleNamespace(search_client=None, news_client=None, web_fetch_client=None,
                          home_client=None, model_lifecycle=None, ha_native_client=None,
                          memory=object(), bawthub_mcp=object())
    names = {t.name for t in BaseLLMBawt._get_tool_definitions(bot)}
    assert "tool_search" in names and not names & {"memory", "history", "profile", "self"}
    assert "time" in names  # Local tools without an MCP equivalent stay.


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



def test_every_tool_path_prompt_offers_discovery():
    """ReAct, streamed and non-streamed turns share one prompt and executor setup."""
    from llm_bawt.core.base import BaseLLMBawt
    from llm_bawt.core.prompt_sections import PromptSectionsMixin

    bot = SimpleNamespace(search_client=None, news_client=None, web_fetch_client=None,
                          home_client=None, model_lifecycle=None, ha_native_client=None,
                          memory=object(), bawthub_mcp=object(),
                          bot=SimpleNamespace(uses_tools=True))
    bot._get_tool_definitions = lambda: BaseLLMBawt._get_tool_definitions(bot)
    for tool_format in ("native", "react"):
        bot.tool_format = tool_format
        builder = Mock()
        PromptSectionsMixin._sec_tools(bot, builder, "hello")
        assert "tool_search" in builder.add_section.call_args.args[1], tool_format


# --- Real approval interceptor: per-call stamp from the server-owned turn ---

def _interceptor_rig(monkeypatch):
    """The client calling the real ApprovalAwareFastMCP in-process."""
    from cryptography.fernet import Fernet

    from llm_bawt import task_turn_context as turn_codec
    from llm_bawt.mcp_server import task_association
    from llm_bawt.mcp_server.approval_interceptor import ApprovalAwareFastMCP

    fernet = Fernet(Fernet.generate_key())
    monkeypatch.setattr(turn_codec, "_get_fernet", lambda: fernet)
    capability = turn_codec.mint_task_turn_context(
        session_id="00000000-0000-4000-8000-000000000001", turn_id="turn-" + "a" * 32,
        trigger_message_id="00000000-0000-4000-8000-000000000002", bot_id="fixture-bot", user_id="fixture-owner",
        identity_only=True)
    server = ApprovalAwareFastMCP("test", approval_store_provider=lambda: None)
    executed = []

    @server.tool(name="memory_search")
    async def memory_search(query: str, bot_id: str | None = None):
        owner = task_association.current_task_turn_context()
        executed.append((query, bot_id, owner.bot_id, owner.user_id))
        return {"hits": [query]}

    sent = []

    async def call(self, headers, name, arguments):
        sent.append(dict(arguments))
        binding = task_association.set_current_task_turn_capability(headers[TASK_TURN_CONTEXT_HEADER])
        try:
            return await server.call_tool(name, arguments)
        finally:
            task_association.reset_current_task_turn_capability(binding)

    monkeypatch.setattr(mcp_client, "run_bounded", lambda coro, _timeout: asyncio.run(coro))
    monkeypatch.setattr(BawtHubMcpClient, "_call_async", call)
    client = BawtHubMcpClient("http://mcp.example:8001", parse_allowlist("memory_search ops_run"))
    return client, capability, executed, sent


def test_stamped_read_runs_with_the_signed_owner_and_clean_args(monkeypatch):
    client, capability, executed, sent = _interceptor_rig(monkeypatch)
    executor = ToolExecutor(user_id="fixture-owner", bot_id="fixture-bot",
                            bawthub_mcp=client, turn_capability=capability)
    call = ToolCall(name="memory_search", arguments={"query": "dogs"}, raw_text="", tool_use_id="call_1")
    result = executor.execute(call)
    assert not tool_result_is_error(result), result
    assert executed == [("dogs", None, "fixture-bot", "fixture-owner")]
    # The stamp rides only on the wire, bound to the finalized args.
    stamp = sent[0][MCP_CALL_CONTEXT_KEY]
    assert stamp["tool_use_id"] == "call_1" and stamp["backend"] == mcp_client.CALL_CONTEXT_BACKEND
    assert call.arguments == {"query": "dogs"} and MCP_CALL_CONTEXT_KEY not in result
    assert capability not in json.dumps(sent) and capability not in result


@pytest.mark.parametrize("stamp, status", [
    (None, "approval_context_missing"),
    ({"version": 1, "tool_use_id": "x"}, "approval_context_invalid"),
    ("mismatch", "approval_context_invalid"),
])
def test_missing_malformed_or_mismatched_stamp_is_refused(monkeypatch, stamp, status):
    from agent_bridge.mcp_call_context import mint_mcp_call_context

    client, capability, executed, _ = _interceptor_rig(monkeypatch)
    if stamp == "mismatch":
        stamp = mint_mcp_call_context(
            capability=capability, tool_name="memory_search", tool_input={"query": "cats"},
            tool_use_id="call_1", agent_request_id="turn-" + "a" * 32,
            session_key="s", backend=mcp_client.CALL_CONTEXT_BACKEND)
    monkeypatch.setattr(BawtHubMcpClient, "_call_context",
                        staticmethod(lambda *_args: stamp))
    args = {"query": "dogs"}
    if stamp is None:
        monkeypatch.setattr(mcp_client, "MCP_CALL_CONTEXT_KEY", "_unused_key")
    with pytest.raises(mcp_client.BawtHubMcpToolError) as error:
        client.call_tool(capability, "memory_search", args, tool_use_id="call_1")
    assert status in str(error.value) or "not executed" in str(error.value)
    assert executed == []


def test_without_provider_call_id_or_for_gated_tools_nothing_is_sent(monkeypatch):
    client, capability, executed, sent = _interceptor_rig(monkeypatch)
    for name, call_id in (("memory_search", ""), ("ops_run", "call_2")):
        with pytest.raises(mcp_client.BawtHubMcpToolError):
            client.call_tool(capability, name, {"query": "x"}, tool_use_id=call_id)
    assert sent == [] and executed == []


def test_gated_tool_stays_out_of_discovery_even_when_allowlisted(monkeypatch):
    client, capability, _, _ = _interceptor_rig(monkeypatch)
    monkeypatch.setattr(BawtHubMcpClient, "_list_async",
                        lambda self, headers: _return(CATALOG))
    assert [t.name for t in client.search_tools(capability, "", limit=10)] == ["memory_search"]


async def _return(value):
    return value


# --- Every loop signs real MCP calls (TASK-2614 review) ---

def _react_replies(*replies):
    queue = iter(replies)
    return lambda *_a, **_k: next(queue)


def test_react_tool_loop_signs_mcp_calls_with_a_turn_bound_id(monkeypatch):
    from llm_bawt.models.message import Message
    from llm_bawt.tools.loop import ToolLoop

    client_mcp, capability, executed, sent = _interceptor_rig(monkeypatch)
    loop = ToolLoop(user_id="fixture-owner", bot_id="fixture-bot", tool_format="react",
                    bawthub_mcp=client_mcp, turn_capability=capability)
    llm = SimpleNamespace(supports_native_tools=lambda: False, query=_react_replies(
        'Action: memory_search\nAction Input: {"query": "dogs"}',
        "Final Answer: done"))
    assert "done" in loop.run([Message(role="user", content="hi")], llm)
    assert executed == [("dogs", None, "fixture-bot", "fixture-owner")]
    assert sent[0][MCP_CALL_CONTEXT_KEY]["tool_use_id"] == "react_1_0"


def test_native_tool_loop_forwards_the_provider_call_id(monkeypatch):
    from llm_bawt.models.message import Message
    from llm_bawt.tools.loop import ToolLoop

    client_mcp, capability, executed, sent = _interceptor_rig(monkeypatch)
    loop = ToolLoop(user_id="fixture-owner", bot_id="fixture-bot", tool_format="native",
                    bawthub_mcp=client_mcp, turn_capability=capability)
    rounds = iter([
        ("", [{"id": "call_provider_7", "function": {
            "name": "memory_search", "arguments": '{"query": "cats"}'}}]),
        ("done", None),
    ])
    llm = SimpleNamespace(supports_native_tools=lambda: True,
                          query_with_tools=lambda *_a, **_k: next(rounds))
    assert loop.run([Message(role="user", content="hi")], llm) == "done"
    assert executed == [("cats", None, "fixture-bot", "fixture-owner")]
    assert sent[0][MCP_CALL_CONTEXT_KEY]["tool_use_id"] == "call_provider_7"


def test_streamed_react_signs_mcp_calls(monkeypatch):
    from llm_bawt.models.message import Message
    from llm_bawt.tools.streaming import stream_with_tools

    client_mcp, capability, executed, sent = _interceptor_rig(monkeypatch)
    replies = iter(['Action: memory_search\nAction Input: {"query": "fish"}', "All done."])
    out = list(stream_with_tools(
        [Message(role="user", content="hi")], lambda _m, _s=None: iter([next(replies)]),
        user_id="fixture-owner", bot_id="fixture-bot", tool_format="react",
        bawthub_mcp=client_mcp, turn_capability=capability))
    assert executed == [("fish", None, "fixture-bot", "fixture-owner")]
    assert sent[0][MCP_CALL_CONTEXT_KEY]["tool_use_id"] == "react_1"
    assert "All done." in "".join(chunk for chunk in out if isinstance(chunk, str))
