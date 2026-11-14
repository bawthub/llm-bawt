"""Deterministic native-SSE telemetry; no provider calls, clocks or secrets."""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

import pytest

from claude_code_bridge.proxy.adapters import anthropic_passthrough
from claude_code_bridge.proxy.adapters.zai import ZaiAdapter
from claude_code_bridge.proxy.passthrough_telemetry import PassthroughTelemetry
from claude_code_bridge.proxy.request_context import ProxyRequestContext


def frame(kind, **fields):
    return ("event: " + kind + "\ndata: " + json.dumps(
        {"type": kind, **fields}, ensure_ascii=False,
    ) + "\n\n").encode()


def context():
    return ProxyRequestContext(request_id="fixture", provider="zai", started_at=10.0)


def test_late_input_cache_and_cumulative_output_reach_context():
    ctx = context()
    tap = PassthroughTelemetry(ctx)
    tap.feed(frame("message_start", message={"usage": {
        "input_tokens": 0, "output_tokens": 1,
    }}), 10)
    tap.feed(frame("message_delta", usage={
        "input_tokens": 40, "cache_read_input_tokens": 150,
        "cache_creation_input_tokens": 10, "output_tokens": 20,
    }), 100)
    tap.feed(frame("message_delta", usage={"output_tokens": 22}), 110)
    tap.feed(frame("message_delta", delta={"stop_reason": "end_turn"}), 120)
    assert (ctx.input_tokens, ctx.cached_tokens, ctx.cache_creation_tokens,
            ctx.output_tokens) == (200, 150, 10, 22)
    assert tap.cache_hit_pct == ctx.cache_hit_pct == 75.0
    assert ctx.first_output_ms is None  # token counts alone aren't output timing


@pytest.mark.parametrize("value", [None, True, False, -1, 1.5, "bad", {}, [], "42"])
def test_omitted_or_invalid_usage_does_not_erase_previous_values(value):
    tap = PassthroughTelemetry()
    tap.feed(frame("message_start", message={"usage": {"input_tokens": 50}}), 1)
    tap.feed(frame("message_delta", usage={"input_tokens": value}), 2)
    assert tap.usage == {"input_tokens": 50}
    tap.feed(frame("message_delta", usage={"input_tokens": 0}), 3)
    assert tap.usage == {"input_tokens": 0}  # explicit zero is a valid correction


@pytest.mark.parametrize("separator", [b"\n", b"\r\n"])
@pytest.mark.parametrize("chunk_size", [1, 2, 7, 65536])
def test_chunked_utf8_delimiters_and_multiline_data(separator, chunk_size):
    tap = PassthroughTelemetry(context())
    raw = (b":comment\n\ndata: {\"type\":\"message_delta\",\n"
           b'data: "usage":{"input_tokens":50,"output_tokens":12}}\n\n'
           + frame("content_block_delta", delta={"text": "café 🐾"}))
    raw = raw.replace(b"\n", separator)
    for offset in range(0, len(raw), chunk_size):
        tap.feed(raw[offset:offset + chunk_size], 25.0)
    assert tap.total_input == 50 and tap.usage["output_tokens"] == 12
    assert tap.first_byte_ms == tap.first_output_ms == 25.0


@pytest.mark.parametrize("separator", [b"\n", b"\r\n"])
@pytest.mark.parametrize("chunk_size", [1, 1024, 200000])
def test_oversized_event_is_skipped_and_later_usage_survives(separator, chunk_size):
    tap = PassthroughTelemetry()
    raw = frame("content_block_delta", delta={"text": "x" * 150000})
    raw += frame("message_delta", usage={"output_tokens": 7})
    raw = raw.replace(b"\n", separator)
    for offset in range(0, len(raw), chunk_size):
        tap.feed(raw[offset:offset + chunk_size], 12)
        assert len(tap._line) <= 65536
        assert sum(map(len, tap._data)) <= 65536
    assert tap.usage == {"output_tokens": 7}
    assert tap.first_output_ms is None  # skipped event cannot supply a timing


@pytest.mark.parametrize("payload", [
    b"not json", b"null", b"[]", b"42", b'"secret"',
    b'{"type":[],"usage":{}}', b'{"type":"message_start","message":[]}',
    b'{"type":"message_delta","usage":[]}', b'\xff',
])
def test_malformed_metadata_recovers_without_content_logging(payload, caplog):
    tap = PassthroughTelemetry()
    tap.feed(b"data: " + payload + b"\n\n", 5)
    tap.feed(frame("message_delta", usage={"output_tokens": 9}), 6)
    assert tap.usage == {"output_tokens": 9}
    assert not caplog.records


@pytest.mark.parametrize("event", [
    {"type": "content_block_delta", "delta": {"text": "hello"}},
    {"type": "content_block_delta", "delta": {"thinking": "thinking"}},
    {"type": "content_block_delta", "delta": {"partial_json": "{"}},
    {"type": "content_block_start", "content_block": {"type": "tool_use"}},
    {"type": "content_block_start", "content_block": {"type": "server_tool_use"}},
    {"type": "content_block_start", "content_block": {"type": "text", "text": "hi"}},
])
def test_first_output_excludes_pings_envelopes_empty_blocks_and_signatures(event):
    ctx = context()
    tap = PassthroughTelemetry(ctx)
    tap.feed(b":keepalive\n\n", 10)
    tap.feed(frame("message_start", message={}), 20)
    tap.feed(frame("content_block_start", content_block={"type": "text", "text": ""}), 30)
    tap.feed(frame("content_block_delta", delta={"text": "", "signature": "opaque"}), 40)
    assert ctx.upstream_ttfb_ms == 10 and ctx.first_output_ms is None
    kind = event["type"]
    tap.feed(frame(kind, **{k: v for k, v in event.items() if k != "type"}), 50)
    tap.feed(frame("content_block_delta", delta={"text": "more"}), 60)
    assert tap.first_output_ms == ctx.first_output_ms == 50


@pytest.mark.parametrize("where", ["message", "event"])
def test_only_explicit_serving_provider_is_recorded(where):
    ctx = context()
    tap = PassthroughTelemetry(ctx)
    fields = {"message": {"model": "provider/model"}}
    if where == "message":
        fields["message"]["provider"] = "Example Provider"
    else:
        fields["provider"] = "Example Provider"
    tap.feed(frame("message_start", **fields), 1)
    assert tap.serving_provider == ctx.serving_provider == "Example Provider"


@pytest.mark.parametrize("provider", [None, {}, [], "", "unsafe\nlog", "x" * 129])
def test_provider_absent_invalid_or_unsafe_remains_unknown(provider):
    ctx = context()
    tap = PassthroughTelemetry(ctx)
    tap.feed(frame("message_start", message={"model": "provider/model"}, provider=provider), 1)
    assert ctx.serving_provider is None and tap.serving_provider is None


def test_missing_telemetry_is_logged_as_unknown_not_proof_of_zero(caplog):
    tap = PassthroughTelemetry()
    with caplog.at_level("INFO"):
        ZaiAdapter._log_usage(tap, "fixture-model")
    assert "input=na" in caplog.text and "output=na" in caplog.text
    assert "cached=na" in caplog.text and "cache_hit=na" in caplog.text
    assert "upstream_ttfb_ms=na" in caplog.text
    assert "first_output_ms=na" in caplog.text and "serving_provider=na" in caplog.text


def install_upstream(monkeypatch, chunks, *, fail=False):
    """In-memory transport preserves chunk boundaries and never opens sockets."""
    class Response:
        status_code = 200

        async def aiter_bytes(self):
            for chunk in chunks:
                yield chunk
            if fail:
                raise RuntimeError("fixture upstream disconnect")

    class Client:
        @asynccontextmanager
        async def stream(self, *args, **kwargs):
            yield Response()

    async def http_client(self):
        return Client()

    monkeypatch.setattr(ZaiAdapter, "_api_key", lambda self: "fixture-unused")
    monkeypatch.setattr(ZaiAdapter, "http_client", http_client)
    return ZaiAdapter()


def test_adapter_observes_late_usage_and_logs_once_preserving_wire_bytes(monkeypatch, caplog):
    raw = (frame("message_start", message={"model": "fixture", "usage": {"input_tokens": 0}})
           + frame("content_block_delta", delta={"text": "private café 🐾"})
           + frame("message_delta", provider="Example", usage={
               "input_tokens": 30, "cache_read_input_tokens": 70,
               "cache_creation_input_tokens": 5, "output_tokens": 15,
           }) + frame("message_stop"))
    chunks = [raw[i:i+3] for i in range(0, len(raw), 3)]
    adapter = install_upstream(monkeypatch, chunks)
    ticks = iter([10.1] + [10.2 + i * 0.01 for i in range(len(chunks))])
    monkeypatch.setattr(anthropic_passthrough.time, "perf_counter", lambda: next(ticks))
    ctx = context()

    async def run():
        return [chunk async for chunk in adapter.call({"model": "fixture"}, "fixture", ctx)]

    with caplog.at_level("INFO"):
        out = asyncio.run(run())
    # TASK-2240 frames complete SSE events for structural rewriting. Transport
    # boundaries may change, but unrelated payload bytes and events must not.
    assert b"".join(out) == raw
    assert ctx.local_setup_ms == pytest.approx(100)
    assert ctx.upstream_ttfb_ms == pytest.approx(100)
    assert ctx.first_output_ms is not None and ctx.first_output_ms > ctx.upstream_ttfb_ms
    assert (ctx.input_tokens, ctx.cached_tokens, ctx.output_tokens) == (105, 70, 15)
    assert ctx.serving_provider == "Example"
    assert len([r for r in caplog.records if " usage model=" in r.message]) == 1
    assert "private" not in caplog.text and "café" not in caplog.text
    assert "output=15" in caplog.text and "cached=70" in caplog.text


@pytest.mark.parametrize("fail", [False, True])
def test_usage_retained_on_close_or_upstream_error(monkeypatch, caplog, fail):
    chunks = [frame("message_delta", usage={"input_tokens": 50, "output_tokens": 5})]
    adapter = install_upstream(monkeypatch, chunks, fail=fail)
    ctx = context()

    async def run():
        stream = adapter.call({"model": "fixture"}, "fixture", ctx)
        assert await anext(stream) == chunks[0]
        if fail:
            with pytest.raises(RuntimeError, match="fixture upstream disconnect"):
                await anext(stream)
        else:
            await stream.aclose()

    with caplog.at_level("INFO"):
        asyncio.run(run())
    assert ctx.input_tokens == 50 and ctx.output_tokens == 5
    assert len([r for r in caplog.records if " usage model=" in r.message]) == 1


def test_telemetry_failure_cannot_change_relay(monkeypatch):
    chunks = [frame("message_start", message={}), b"data: malformed\n\n"]
    adapter = install_upstream(monkeypatch, chunks)

    def fail(*args):
        raise ValueError("private diagnostic must not be logged")

    monkeypatch.setattr(PassthroughTelemetry, "feed", fail)

    async def run():
        return [chunk async for chunk in adapter.call({"model": "fixture"}, "fixture")]

    assert asyncio.run(run()) == chunks


@pytest.mark.parametrize("stream", [False, True])
def test_route_completion_uses_same_final_snapshot_without_content_logging(monkeypatch, caplog, stream):
    import httpx
    from fastapi import FastAPI
    from claude_code_bridge.proxy import routes

    raw = (frame("message_start", message={
        "id": "msg_fixture", "type": "message", "role": "assistant",
        "model": "fixture", "content": [], "usage": {"input_tokens": 0},
    }) + frame("content_block_start", index=0, content_block={"type": "text", "text": ""})
        + frame("content_block_delta", index=0, delta={"type": "text_delta", "text": "private output"})
        + frame("content_block_stop", index=0)
        + frame("message_delta", delta={"stop_reason": "end_turn"}, provider="Example", usage={
            "input_tokens": 40, "cache_read_input_tokens": 150,
            "cache_creation_input_tokens": 10, "output_tokens": 22,
        }) + frame("message_stop"))
    adapter = install_upstream(monkeypatch, [raw])
    monkeypatch.setattr(routes, "lookup", lambda _: adapter)
    app = FastAPI()
    app.include_router(routes.router)

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
            return await client.post("/v1/messages", json={
                "model": "zai/fixture", "messages": [{"role": "user", "content": "private prompt"}],
                "stream": stream,
            })

    with caplog.at_level("INFO"):
        response = asyncio.run(run())
    assert response.status_code == 200
    completion = [r.message for r in caplog.records if "proxy_stream_complete" in r.message]
    assert len(completion) == 1
    assert "input=200 cached=150 output=22 cache_hit=75.0%" in completion[0]
    assert "cache_create=10" in completion[0] and 'serving_provider="Example"' in completion[0]
    assert "upstream_ttfb_ms=na" not in completion[0] and "first_output_ms=na" not in completion[0]
    assert "private output" not in caplog.text and "private prompt" not in caplog.text
