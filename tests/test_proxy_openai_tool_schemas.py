"""TASK-2586: OpenAI-bound MCP schemas must not contain oneOf keywords."""

import asyncio
import copy
import json
from unittest.mock import AsyncMock

import httpx
from openai import AsyncOpenAI
import pytest

from claude_code_bridge.proxy.adapters.chatgpt_app import ChatGPTAppAdapter
from claude_code_bridge.proxy.adapters.openai import OpenAIAdapter
from claude_code_bridge.proxy.adapters.openai_chatgpt import OpenAIChatGPTAdapter
from claude_code_bridge.proxy.adapters.xai import XaiAdapter
from claude_code_bridge.proxy.openai_tool_schemas import normalize_openai_tool_schemas
from claude_code_bridge.proxy.request_context import ProxyRequestContext


TOOL = "mcp__fixture__capture_scenario"
SCHEMA = {
    "type": "object",
    "properties": {"steps": {"type": "array", "items": {"$ref": "#/$defs/Step"}}},
    "required": ["steps"],
    "$defs": {
        "Step": {"description": "A step", "oneOf": [
            {"type": "object", "properties": {"label": {"type": "string"}}},
            {"$ref": "#/$defs/Wait"},
        ]},
        "Wait": {"type": "object", "properties": {"duration": {"oneOf": [
            {"type": "integer"}, {"type": "string"},
        ]}}},
    },
    "additionalProperties": False,
}


def _body():
    return {"tools": [{"name": TOOL, "input_schema": copy.deepcopy(SCHEMA)}],
            "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100, "stream": True}


@pytest.mark.parametrize("adapter_type,strict", [
    (OpenAIChatGPTAdapter, False),
    (OpenAIChatGPTAdapter, True),  # $ref makes this particular tool fall back.
    (OpenAIAdapter, False),
    (ChatGPTAppAdapter, False),
])
def test_openai_wire_accepts_referenced_unions(adapter_type, strict, monkeypatch):
    monkeypatch.setattr("claude_code_bridge.proxy.usage_capture.schedule_capture", lambda _: None)

    async def run():
        calls = []

        def respond(request):
            assert request.url.path == "/v1/responses"
            payload = json.loads(request.content)
            calls.append(payload)
            (tool,) = payload["tools"]
            if adapter_type is ChatGPTAppAdapter:
                assert tool["type"] == "namespace"
                (tool,) = tool["tools"]
            assert tool["type"] == "function" and tool["name"] == TOOL
            assert tool.get("strict") is not True
            expected = copy.deepcopy(SCHEMA)
            for node in (expected["$defs"]["Step"], expected["$defs"]["Wait"]["properties"]["duration"]):
                node["anyOf"] = node.pop("oneOf")
            if tool["parameters"] != expected:
                return httpx.Response(400, json={"error": {"code": "invalid_json_schema",
                    "message": "Unsupported tool schema"}})
            done = {"type": "response.completed", "response": {"status": "completed"}}
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  text=f"data: {json.dumps(done)}\n\ndata: [DONE]\n\n")

        adapter = adapter_type()
        adapter.authorize = AsyncMock(return_value=("fixture", "http://openai.test/v1"))
        adapter.start = AsyncMock()
        adapter._responses_client = AsyncOpenAI(
            api_key="fixture", http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        )
        body = _body()
        ctx = ProxyRequestContext(request_id="fixture", provider=adapter.name, strict_tools=strict)
        try:
            frames = b"".join([frame async for frame in adapter.call(body, "fixture", ctx)])
        finally:
            await adapter.close()
        assert b'"type":"message_stop"' in frames
        assert b'"type":"error"' not in frames
        assert len(calls) == 1
        assert body == _body()

    asyncio.run(run())


def test_other_provider_keeps_original_schema():
    adapter = XaiAdapter()
    adapter.authorize = AsyncMock(return_value=("fixture", "http://provider.test/v1"))
    adapter.start = AsyncMock()
    call = asyncio.run(adapter.prepare_call(_body(), "fixture", None))
    assert call.body["tools"][0]["parameters"] == SCHEMA


def test_rewrites_only_schema_keywords_and_preserves_sibling_constraints():
    literal = {"oneOf": ["literal key"]}
    a = [{"type": "string"}, {"type": "integer"}]
    b = [{"const": "chosen"}, {"type": "integer"}]
    schema = {"type": "object", "properties": {
        "oneOf": {"const": literal, "default": literal, "examples": [literal]},
        "choice": {"anyOf": a, "oneOf": b, "allOf": [{"description": "Existing"}]},
    }}
    custom = {"type": "custom", "name": "apply_patch"}
    body = {"tools": [custom, {"type": "function", "name": TOOL, "parameters": schema}]}
    normalize_openai_tool_schemas(body)
    assert body["tools"][0] is custom
    properties = body["tools"][1]["parameters"]["properties"]
    assert properties["oneOf"] == schema["properties"]["oneOf"]
    assert properties["choice"] == {"anyOf": a, "allOf": [
        {"description": "Existing"}, {"anyOf": b},
    ]}
    assert schema["properties"]["choice"]["oneOf"] == b
