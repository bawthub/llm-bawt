"""Hermetic coverage for the Kimi For Coding Chat Completions adapter."""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from claude_code_bridge.proxy.adapters import lookup
from claude_code_bridge.proxy.adapters.kimi_coding import KimiCodingAdapter
from claude_code_bridge.proxy.stream_cc import chat_completions_to_anthropic_sse
from claude_code_bridge.proxy.translate_cc import anthropic_to_chat_completions
from llm_bawt.service.usage.adapters import kimi_coding as kimi_usage
from llm_bawt.service.usage.adapters.kimi_coding import KimiCodingUsageAdapter
from tests.stored_provider_keys import stub_stored_keys
from claude_code_bridge.proxy.adapters import kimi_coding as kimi_proxy
from claude_code_bridge.provider_key_broker import BrokerKeyNotInstalled


def _no_broker_key(monkeypatch):
    """The app broker reports no stored Kimi key, so the legacy env applies."""

    def fake_fetch(provider_id):
        assert provider_id == "kimi-coding"
        raise BrokerKeyNotInstalled("503")

    monkeypatch.setattr(kimi_proxy, "fetch_provider_api_key", fake_fetch)


def _payloads(frames: list[bytes]) -> list[dict]:
    out: list[dict] = []
    for frame in frames:
        data = next(
            line[6:]
            for line in frame.decode().splitlines()
            if line.startswith("data: ")
        )
        out.append(json.loads(data))
    return out


def _collect(lines: list[str], *, tools: list[dict] | None = None) -> list[bytes]:
    async def source():
        for line in lines:
            yield line

    async def run():
        return [
            frame
            async for frame in chat_completions_to_anthropic_sse(
                source(), anthropic_model="kimi_coding/k3", tool_schemas=tools
            )
        ]

    return asyncio.run(run())


def test_kimi_coding_registered_and_separate_from_moonshot():
    assert isinstance(lookup("kimi_coding"), KimiCodingAdapter)
    assert lookup("kimi_coding") is not lookup("moonshot")
    assert KimiCodingAdapter._base_url() == "https://api.kimi.com/coding/v1"


def test_translate_anthropic_history_tools_and_effort():
    body = {
        "model": "kimi_coding/k3",
        "system": "Use the tools.",
        "messages": [
            {"role": "user", "content": "calculate"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "old", "signature": "local"},
                    {
                        "type": "tool_use",
                        "id": "tool_1",
                        "name": "calc",
                        "input": {"expr": "17*23"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_1",
                        "content": "391",
                    }
                ],
            },
        ],
        "tools": [
            {
                "name": "calc",
                "description": "calculate",
                "input_schema": {
                    "type": "object",
                    "properties": {"expr": {"type": "string"}},
                    "required": ["expr"],
                },
            }
        ],
        "tool_choice": {"type": "tool", "name": "calc"},
        "thinking": {"type": "enabled", "budget_tokens": 12000},
        "max_tokens": 2048,
        "temperature": 0.2,
        "stream": True,
    }

    out = anthropic_to_chat_completions(body, "k3")

    assert out["model"] == "k3"
    assert out["messages"][0] == {"role": "system", "content": "Use the tools."}
    assert out["messages"][2]["tool_calls"][0]["function"]["arguments"] == (
        '{"expr":"17*23"}'
    )
    assert out["messages"][3] == {
        "role": "tool",
        "tool_call_id": "tool_1",
        "content": "391",
    }
    assert out["tools"][0]["function"]["name"] == "calc"
    assert out["tool_choice"] == "auto"
    assert out["reasoning_effort"] == "high"
    assert out["max_tokens"] == 2048
    assert "temperature" not in out  # Kimi rejects every explicit value except 1.
    assert out["stream_options"] == {"include_usage": True}


def test_stream_translates_reasoning_text_usage_and_stop():
    lines = [
        'data: {"choices":[{"delta":{"role":"assistant","content":null},"finish_reason":null}]}',
        'data: {"choices":[{"delta":{"reasoning_content":"We need"},"finish_reason":null}]}',
        'data: {"choices":[{"delta":{"reasoning_content":" to answer."},"finish_reason":null}]}',
        'data: {"choices":[{"delta":{"content":"Done"},"finish_reason":null}]}',
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
        'data: {"choices":[],"usage":{"prompt_tokens":88,"completion_tokens":29,"cached_tokens":88,"prompt_tokens_details":{"cached_tokens":88}}}',
        "data: [DONE]",
    ]

    payloads = _payloads(_collect(lines))
    starts = [p for p in payloads if p["type"] == "content_block_start"]
    assert [p["content_block"]["type"] for p in starts] == ["thinking", "text"]
    assert "".join(
        p["delta"]["thinking"]
        for p in payloads
        if p["type"] == "content_block_delta"
        and p["delta"].get("type") == "thinking_delta"
    ) == "We need to answer."
    assert any(
        p.get("delta", {}).get("type") == "signature_delta" for p in payloads
    )
    assert "".join(
        p["delta"]["text"]
        for p in payloads
        if p["type"] == "content_block_delta"
        and p["delta"].get("type") == "text_delta"
    ) == "Done"
    final = next(p for p in payloads if p["type"] == "message_delta")
    assert final["delta"]["stop_reason"] == "end_turn"
    assert final["usage"] == {
        "input_tokens": 0,
        "output_tokens": 29,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 88,
    }


def test_stream_reassembles_fragmented_tool_call_and_sanitizes_optional_empties():
    tools = [
        {
            "name": "Read",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "pages": {"type": "string"},
                },
                "required": ["file_path"],
            },
        }
    ]
    lines = [
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"tool_abc","type":"function","function":{"name":"Read","arguments":"{\\"file_path\\":\\"/tmp/x\\""}}]},"finish_reason":null}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":",\\"pages\\":\\"\\"}"}}]},"finish_reason":null}]}',
        'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
        'data: {"choices":[],"usage":{"prompt_tokens":20,"completion_tokens":5}}',
        "data: [DONE]",
    ]

    payloads = _payloads(_collect(lines, tools=tools))
    start = next(
        p
        for p in payloads
        if p["type"] == "content_block_start"
        and p["content_block"]["type"] == "tool_use"
    )
    assert start["content_block"] == {
        "type": "tool_use",
        "id": "tool_abc",
        "name": "Read",
        "input": {},
    }
    args = next(
        p["delta"]["partial_json"]
        for p in payloads
        if p["type"] == "content_block_delta"
        and p["delta"].get("type") == "input_json_delta"
    )
    assert json.loads(args) == {"file_path": "/tmp/x"}
    final = next(p for p in payloads if p["type"] == "message_delta")
    assert final["delta"]["stop_reason"] == "tool_use"


@pytest.fixture
def fake_kimi_upstream():
    captured: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("content-length", 0))
            captured["path"] = self.path
            captured["headers"] = {k.lower(): v for k, v in self.headers.items()}
            captured["body"] = json.loads(self.rfile.read(length) or b"{}")
            frames = (
                'data: {"choices":[{"delta":{"content":"OK"},"finish_reason":null}]}\n\n'
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
                'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":1}}\n\n'
                "data: [DONE]\n\n"
            ).encode()
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            self.wfile.write(frames)
            self.wfile.flush()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", captured
    finally:
        server.shutdown()
        server.server_close()


def test_adapter_posts_chat_completions_with_bearer(fake_kimi_upstream, monkeypatch):
    base_url, captured = fake_kimi_upstream
    monkeypatch.setenv("KIMI_CODING_BASE_URL", base_url)
    monkeypatch.setenv("KIMI_CODING_API_KEY", "coding-key")
    _no_broker_key(monkeypatch)
    body = {
        "model": "kimi_coding/k3",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 32,
        "stream": True,
    }

    async def run():
        return b"".join(
            [chunk async for chunk in KimiCodingAdapter().call(body, "k3")]
        )

    out = asyncio.run(run())
    assert captured["path"] == "/chat/completions"
    assert captured["headers"]["authorization"] == "Bearer coding-key"
    assert captured["body"]["model"] == "k3"
    assert captured["body"]["stream"] is True
    assert captured["body"]["stream_options"] == {"include_usage": True}
    assert b'"model":"kimi_coding/k3"' in out
    assert b'"text":"OK"' in out


def test_kimi_coding_usage_parses_live_subscription_shape(monkeypatch):
    payload = {
        "limits": [
            {
                "detail": {
                    "limit": "100",
                    "remaining": "99",
                    "resetTime": "2026-07-27T21:06:02.913444Z",
                    "used": "1",
                },
                "window": {
                    "duration": 300,
                    "timeUnit": "TIME_UNIT_MINUTE",
                },
            }
        ],
        "parallel": {"limit": "10"},
        "usage": {
            "limit": "100",
            "remaining": "100",
            "resetTime": "2026-08-03T16:06:02.913444Z",
        },
        "user": {"membership": {"level": "LEVEL_BASIC"}},
    }

    class _Response:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return payload

    class _Client:
        def __init__(self, *, timeout):
            assert timeout == 20.0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, url, *, headers):
            assert url == "https://api.kimi.com/coding/v1/usages"
            assert headers == {
                "Authorization": "Bearer coding-key",
                "Accept": "application/json",
            }
            return _Response()

    monkeypatch.setenv("KIMI_CODING_API_KEY", "coding-key")
    stub_stored_keys(monkeypatch, {})
    monkeypatch.setattr(kimi_usage.httpx, "AsyncClient", _Client)

    snap = asyncio.run(KimiCodingUsageAdapter().fetch())

    assert snap.provider == "kimi_coding"
    assert snap.status == "ok"
    assert snap.available is True
    assert snap.raw == payload
    assert [limit.id for limit in snap.limits] == ["session_5h", "weekly_all"]
    session, weekly = snap.limits
    assert session.label == "5-hour request limit"
    assert session.used == 1
    assert session.limit == 100
    assert session.used_pct == 1.0
    assert session.unit == "requests"
    assert session.window == "5h"
    assert session.resets_at == 1785186362
    assert weekly.label == "Weekly request limit"
    assert weekly.used == 0
    assert weekly.limit == 100
    assert weekly.used_pct == 0.0
    assert weekly.window == "7d"
    assert weekly.resets_at == 1785773162


def test_kimi_coding_usage_supports_remaining_only_and_drifted_reset_name():
    limits = kimi_usage._parse_limits(
        {
            "limits": [
                {
                    "detail": {
                        "limit": 200,
                        "remaining": 150,
                        "resetAt": "2026-07-28T00:00:00Z",
                    },
                    "window": {"duration": 120, "timeUnit": "MINUTE"},
                }
            ]
        }
    )

    assert len(limits) == 1
    assert limits[0].id == "window_2h"
    assert limits[0].label == "2-hour request limit"
    assert limits[0].used == 50
    assert limits[0].used_pct == 25.0
    assert limits[0].resets_at == 1785196800


def test_kimi_subscription_cost_is_zero_not_unknown_model_fallback():
    from claude_code_bridge._bridge_helpers import _estimate_proxy_cost_usd

    usage = {"input_tokens": 100_000, "output_tokens": 10_000}
    assert _estimate_proxy_cost_usd("kimi_coding/k3", usage) == 0.0
