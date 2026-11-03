"""TASK-1966: per-tool strict Responses schemas, null round-trip, stall attribution."""
import asyncio
import json
import logging
from types import SimpleNamespace as NS

import pytest

from claude_code_bridge.proxy.adapters.base import _held_tool_fields
from claude_code_bridge.proxy.request_context import ProxyRequestContext, custom_header_env
from claude_code_bridge.proxy.stream import responses_to_anthropic_sse
from claude_code_bridge.proxy.strict_tools import (
    STRICT_TOOLS_HEADER, parse_strict_tools, strip_strict_nulls, to_strict_parameters,
)
from claude_code_bridge.proxy.translate import anthropic_to_responses
from tests import virtual_clock
from tests.test_proxy_progress_recovery import THINKING, TOOL, Harness, success_frames

READ = {"type": "object", "properties": {
    "file_path": {"type": "string"},
    "offset": {"type": "integer", "minimum": 0},
    "limit": {"type": "integer", "default": 2000},
}, "required": ["file_path"], "additionalProperties": False,
    "$schema": "http://json-schema.org/draft-07/schema#"}
EDIT = {"type": "object", "properties": {
    "file_path": {"type": "string"}, "old_string": {"type": "string"},
    "new_string": {"type": "string"}, "replace_all": {"type": "boolean"},
}, "required": ["file_path", "old_string", "new_string"]}
TASKS = {"type": "object", "properties": {
    "subject": {"type": "string"},
    "metadata": {"type": "object", "additionalProperties": {}},
}, "required": ["subject"]}
NESTED = {"type": "object", "properties": {
    "todos": {"type": "array", "items": {"type": "object", "properties": {
        "content": {"type": "string"}, "note": {"type": "string"},
    }, "required": ["content"]}},
}, "required": ["todos"]}


def test_convertible_schema_is_closed_and_optionals_nullable():
    out = to_strict_parameters(READ)
    assert out["additionalProperties"] is False
    assert out["required"] == ["file_path", "offset", "limit"]
    assert out["properties"]["file_path"] == {"type": "string"}
    assert out["properties"]["offset"] == {"anyOf": [
        {"type": "integer", "minimum": 0}, {"type": "null"}]}
    assert "default" not in json.dumps(out) and "$schema" not in out
    assert READ["required"] == ["file_path"]  # input not mutated


@pytest.mark.parametrize("schema", [
    TASKS,
    {"type": "object", "properties": {"a": {"type": "string"}}, "allOf": [{}]},
    {"type": "object", "properties": {"a": {"$ref": "#/$defs/x"}}},
    {"type": "object"},
    {"type": "object", "properties": {"m": {"type": "object", "propertyNames": {}}}},
    {"type": "string"},
    None,
])
def test_unconvertible_schema_falls_back(schema):
    assert to_strict_parameters(schema) is None


def test_no_argument_tool_converts():
    assert to_strict_parameters({"type": "object", "properties": {}}) == {
        "type": "object", "properties": {}, "required": [], "additionalProperties": False}


def test_nested_items_convert_and_one_of_becomes_any_of():
    out = to_strict_parameters(NESTED)
    item = out["properties"]["todos"]["items"]
    assert item["required"] == ["content", "note"] and item["additionalProperties"] is False
    assert to_strict_parameters({"type": "object", "properties": {"v": {"oneOf": [
        {"type": "string"}, {"type": "integer"}]}}, "required": ["v"]}
    )["properties"]["v"] == {"anyOf": [{"type": "string"}, {"type": "integer"}]}


def _body(*tools):
    return {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "x"}],
            "tools": [{"name": n, "description": n, "input_schema": s} for n, s in tools]}


def _tools(body, **kw):
    payload = anthropic_to_responses(body, "gpt-test", **kw)
    return {t["name"]: t for t in payload["tools"]}


def test_translate_marks_strict_per_tool_and_flag_off_is_unchanged():
    body = _body(("Read", READ), ("TaskCreate", TASKS))
    on = _tools(body, strict_tools=True)
    assert on["Read"]["strict"] is True
    assert on["Read"]["parameters"] == to_strict_parameters(READ)
    assert "strict" not in on["TaskCreate"]
    assert on["TaskCreate"]["parameters"] == _tools(body)["TaskCreate"]["parameters"]
    off = _tools(body)
    assert all("strict" not in t for t in off.values())
    assert off["Read"]["parameters"]["required"] == ["file_path"]


def test_flag_parses_and_rides_the_custom_header():
    assert parse_strict_tools("1") and parse_strict_tools(True) and parse_strict_tools("on")
    assert not parse_strict_tools(None) and not parse_strict_tools("0") and not parse_strict_tools(1)
    ctx = ProxyRequestContext(request_id="r", provider="openai_chatgpt", strict_tools=True)
    headers = dict(line.split(": ", 1) for line in custom_header_env(ctx).splitlines())
    assert parse_strict_tools(headers[STRICT_TOOLS_HEADER]) is True
    off = ProxyRequestContext(request_id="r", provider="openai_chatgpt")
    assert STRICT_TOOLS_HEADER not in custom_header_env(off)


def test_strip_nulls_only_for_optional_keys_recursively():
    value = {"todos": [{"content": None, "note": None}], "extra": None}
    assert strip_strict_nulls(value, NESTED) == {"todos": [{"content": None}], "extra": None}


def _cli_input(frames):
    """Reassemble what the CLI receives as the tool_use input."""
    parts = []
    for chunk in frames:
        for line in chunk.decode().splitlines():
            if line.startswith("data: "):
                event = json.loads(line[6:])
                delta = event.get("delta") or {}
                if delta.get("type") == "input_json_delta":
                    parts.append(delta["partial_json"])
    return json.loads("".join(parts))


@pytest.mark.parametrize("strict", [True, False])
def test_strict_nulls_round_trip_to_cli_as_absent(strict):
    """Regression: the CLI sees exactly what a non-strict call would have sent."""
    args = {"file_path": "/tmp/a", "old_string": "x", "new_string": "", "replace_all": None}

    async def run():
        item = NS(type="function_call", id="item", call_id="call-0", name="Edit")
        events = [NS(type="response.output_item.added", item=item),
                  NS(type="response.function_call_arguments.delta", item_id="item",
                     delta=json.dumps(args)),
                  NS(type="response.function_call_arguments.done", item_id="item",
                     arguments=json.dumps(args)),
                  NS(type="response.output_item.done", item=item),
                  NS(type="response.completed", response=NS(status="completed", usage=None))]

        async def upstream():
            for event in events:
                yield event

        ctx = ProxyRequestContext(request_id="r", provider="openai_chatgpt", strict_tools=strict)
        return [c async for c in responses_to_anthropic_sse(
            upstream(), anthropic_model="fixture",
            tool_schemas=[{"name": "Edit", "input_schema": EDIT}], request_context=ctx)]

    got = _cli_input(asyncio.run(run()))
    # new_string="" is a required deletion and must survive either way.
    expected = {"file_path": "/tmp/a", "old_string": "x", "new_string": ""}
    if not strict:
        expected["replace_all"] = None  # non-strict path never touches nulls
    assert got == expected


def test_required_null_is_kept_for_cli_validation():
    assert strip_strict_nulls({"file_path": None, "offset": None}, READ) == {"file_path": None}


@pytest.mark.parametrize("block,expected", [
    (None, ("-", "-", "-")),
    ({"kind": "text"}, ("-", "-", "-")),
    ({"kind": "tool", "name": "Bash"}, ("Bash", "0", "no")),
    ({"kind": "tool", "name": "Bash", "held_args": '{"command": "ls"'}, ("Bash", "16", "no")),
    ({"kind": "tool", "name": "Read", "held_args": '{"file_path":"/a"}  \n'}, ("Read", "21", "yes")),
])
def test_held_tool_fields(block, expected):
    assert _held_tool_fields(block) == expected


def test_whitespace_stall_log_names_tool_without_argument_content(monkeypatch, caplog):
    async def run():
        h = Harness(monkeypatch, "sse", [
            THINKING, {"type": "response.output_item.added", "item": TOOL},
            {"type": "response.function_call_arguments.delta", "item_id": "fc_new",
             "delta": '{"value":"secret"}'},
            {"type": "response.function_call_arguments.delta", "item_id": "fc_new",
             "delta": "   "},
        ], success_frames())
        try:
            await h.collect()
        finally:
            await h.adapter.close()
    with caplog.at_level(logging.WARNING):
        virtual_clock.run_timing(run())
    lines = [r.getMessage() for r in caplog.records
             if r.getMessage().startswith("proxy_retry") and "boundary=stream" in r.getMessage()]
    assert lines, "stream-boundary proxy_retry log missing"
    assert "tool=check held_arg_chars=21 held_args_json=yes" in lines[-1]
    assert "secret" not in lines[-1]


# --- flag plumbing: catalog -> init_bot -> command -> header -> context ----

def test_catalog_serving_config_validates_strict_flag():
    from pydantic import ValidationError

    from llm_bawt.service.routes.model_catalog import EndpointWrite

    assert EndpointWrite(serving_config={"responses_strict_tools": True}).serving_config == {
        "responses_strict_tools": True}
    for bad in ("true", 1, 0):
        with pytest.raises(ValidationError, match="responses_strict_tools"):
            EndpointWrite(serving_config={"responses_strict_tools": bad})


def test_init_bot_threads_strict_flag_only_when_true(monkeypatch):
    from tests.test_model_consolidation import _run_init_bot

    bot = NS(agent_backend="claude-code", agent_backend_config={}, default_model="sol")
    model = {"type": "claude-code", "model_id": "openai_chatgpt/gpt-6.1-sol"}
    on = _run_init_bot(monkeypatch, bot, {"sol": {**model, "responses_strict_tools": True}})
    assert on.client._bot_config["responses_strict_tools"] is True
    for value in (False, "true", None):
        off = _run_init_bot(monkeypatch, bot, {"sol": {**model, "responses_strict_tools": value}})
        assert "responses_strict_tools" not in off.client._bot_config


def test_strict_flag_command_to_proxy_context_round_trip(monkeypatch):
    from claude_code_bridge.proxy.routes import messages
    from claude_code_bridge.send_request import SendRequest
    from claude_code_bridge.send_stream import ClaudeStreamMixin
    from tests.test_openrouter_provider_routing import _FIELDS, _published_command

    command = _published_command(responses_strict_tools=True)
    assert command["responses_strict_tools"] == "1"
    assert "responses_strict_tools" not in _published_command()

    req = SendRequest.from_fields({**_FIELDS, "responses_strict_tools": "1"})
    assert req.strict_tools is True
    assert SendRequest.from_fields(_FIELDS).strict_tools is False

    class Mixin(ClaudeStreamMixin):
        _proxy_base_url = "http://127.0.0.1:12345"

    env = Mixin()._build_sdk_env(
        use_proxy=True, model=req.model, subagent_model=None, force_refresh=False,
        bot_id="snark", session_key="snark:nick", thread_session_id="thread-1",
        request_id="request-1", strict_tools=req.strict_tools,
    )
    headers = dict(line.split(": ", 1) for line in env["ANTHROPIC_CUSTOM_HEADERS"].splitlines())
    assert headers[STRICT_TOOLS_HEADER] == "1"

    captured = []

    class Adapter:
        def account_hash(self):
            return "mock-account"

        async def call(self, body, upstream_model, context):
            captured.append(context)
            yield b'event: message_stop\ndata: {"type":"message_stop"}\n\n'

    def request_with(header_map):
        class Request:
            headers = header_map

            async def json(self):
                return {"model": req.model, "messages": [{"role": "user", "content": "hi"}],
                        "stream": True}
        return Request()

    monkeypatch.setattr("claude_code_bridge.proxy.routes.lookup", lambda _p: Adapter())

    async def run_route(request):
        response = await messages(request)
        return b"".join([chunk async for chunk in response.body_iterator])

    asyncio.run(run_route(request_with(headers)))
    asyncio.run(run_route(request_with({"X-LLM-Bawt-Request-ID": "request-2"})))
    assert [c.strict_tools for c in captured] == [True, False]
