"""TASK-1915: route clients never receive executable output-limit tool calls."""
import asyncio
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from claude_code_bridge.proxy import routes
from claude_code_bridge.proxy.adapters.anthropic_passthrough import AnthropicPassthroughAdapter
from claude_code_bridge.proxy.retry import RetryBudgetRegistry, RetryPolicy
from claude_code_bridge.proxy.stream import responses_to_anthropic_sse
from claude_code_bridge.proxy.stream_cc import chat_completions_to_anthropic_sse
from claude_code_bridge.proxy.tool_guard import ToolCallGuard


def frame(kind, **fields):
    return (f"event: {kind}\ndata: " + json.dumps(dict(type=kind, **fields), ensure_ascii=False) + "\n\n").encode()


def tool_frames(index=0, value="café"):
    return [frame("content_block_start", index=index, content_block=dict(type="tool_use", id=f"call-{index}", name="work", input={})),
            frame("content_block_delta", index=index, delta=dict(type="input_json_delta", partial_json=json.dumps({"value": value}, ensure_ascii=False))),
            frame("content_block_stop", index=index)]


def terminal(reason):
    return [frame("message_delta", delta=dict(stop_reason=reason, stop_sequence=None), usage={"output_tokens": 7}),
            frame("message_stop")]


def native_frames(reason, *, value="café"):
    return [frame("message_start", message=dict(id="msg_fixture", type="message", role="assistant", model="fixture", content=[], usage={})),
            *tool_frames(value=value), *terminal(reason)]


class Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


class Passthrough(AnthropicPassthroughAdapter):
    LABEL = "Fixture"
    DEFAULT_BASE_URL = "https://fixture.invalid"
    BASE_URL_ENV = "UNUSED_FIXTURE_URL"

    def _api_key(self):
        return "test-only-key"


async def adapter_source(mode, reason):
    if mode == "native":
        # Exercise the real native adapter, not a pre-parsed replacement. Split
        # JSON/UTF-8/SSE boundaries and use CRLF just like arbitrary HTTP chunks.
        wire = b"".join(native_frames(reason)).replace(b"\n", b"\r\n")
        chunks = [wire[i:i + 7] for i in range(0, len(wire), 7)]
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda _req: httpx.Response(200, stream=Chunks(chunks))
        )) as client:
            adapter = Passthrough()
            adapter.http_client = AsyncMock(return_value=client)
            async for chunk in adapter.call({"model": "fixture", "messages": []}, "fixture"):
                yield chunk
    elif mode == "cc":
        payloads = [dict(choices=[dict(delta=dict(tool_calls=[dict(index=0, id="call-0", function=dict(name="work", arguments='{"value":"café"}'))]), finish_reason=None)]),
                    dict(choices=[dict(delta={}, finish_reason="tool_calls" if reason == "tool_use" else reason)])]
        response = httpx.Response(200, content="".join("data: " + json.dumps(p) + "\n\n" for p in payloads) + "data: [DONE]\n\n")
        async for chunk in chat_completions_to_anthropic_sse(response.aiter_lines(), anthropic_model="fixture"):
            yield chunk
    else:
        item = NS(type="function_call", id="item", call_id="call-0", name="work")
        events = [NS(type="response.output_item.added", item=item),
                  NS(type="response.function_call_arguments.done", item_id="item", arguments='{"value":"café"}'),
                  NS(type="response.output_item.done", item=item)]
        if reason == "tool_use":
            events.append(NS(type="response.completed", response=NS(status="completed", usage=None)))
        else:
            events.append(NS(type="response.incomplete", response=NS(incomplete_details=NS(reason=reason), usage=None)))

        async def upstream():
            for event in events:
                yield event

        async for chunk in responses_to_anthropic_sse(upstream(), anthropic_model="fixture"):
            yield chunk


@pytest.mark.parametrize("mode", ["native", "cc", "responses"])
@pytest.mark.parametrize("reason", ["tool_use", "length", "max_tokens", "max_output_tokens"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("budgeted", [False, True])
def test_routes_block_truncation_and_preserve_valid_calls(monkeypatch, mode, reason, stream, budgeted):
    async def run():
        class Adapter:
            calls = 0

            def account_hash(self):
                return "fixture"

            def retry_policy(self, *_):
                return RetryPolicy(max_attempts=2, jitter_s=0)

            async def call(self, *_):
                self.calls += 1
                async for chunk in adapter_source(mode, reason):
                    yield chunk

        adapter = Adapter()
        monkeypatch.setattr(routes, "lookup", lambda _provider: adapter)
        app = FastAPI()
        app.include_router(routes.router)
        app.state.proxy_retry_budgets = RetryBudgetRegistry() if budgeted else None
        headers = {"X-LLM-Bawt-Request-ID": "fixture-turn"} if budgeted else {}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
            body = dict(model="fixture/model", stream=stream, messages=[dict(role="user", content="work")])
            result = await client.post("/v1/messages", json=body, headers=headers)
            if stream:
                events = [json.loads(line[5:]) for line in result.text.splitlines() if line.startswith("data:")]
                tools = [e for e in events if e.get("content_block", {}).get("type") == "tool_use"]
                if reason == "tool_use":
                    assert len(tools) == 1
                    assert not any(e["type"] == "error" for e in events)
                    assert events[-1]["type"] == "message_stop"
                else:
                    assert tools == []
                    assert events[-1]["type"] == "error"
                    assert events[-1]["error"]["type"] == "api_error"
                    assert "truncated" in events[-1]["error"]["message"]
            elif reason == "tool_use":
                assert result.status_code == 200
                assert result.json()["content"] == [dict(type="tool_use", id="call-0", name="work", input={"value": "café"})]
                assert result.json()["stop_reason"] == "tool_use"
            else:
                assert result.status_code == 500
                assert result.json()["type"] == "error"
                assert "truncated" in result.json()["error"]["message"]
                assert "content" not in result.json()
            assert adapter.calls == 1  # never automatically replay this response
            if budgeted:
                # Keep the original conservative tool tombstone, including
                # across SDK stream→JSON fallback and valid completed calls.
                again = await client.post("/v1/messages", json={**body, "stream": False}, headers=headers)
                assert again.status_code == 500
                assert "tool" in again.json()["error"]["message"]
                assert adapter.calls == 1

    asyncio.run(run())


@pytest.mark.parametrize("ending", ["eof", "error", "missing_reason", "truncated"])
def test_entire_parallel_batch_withheld_on_unsafe_terminal(ending):
    guard = ToolCallGuard()
    frames = [*tool_frames(0), *tool_frames(1)]
    if ending == "error":
        frames.append(frame("error", error=dict(type="api_error", message="upstream failed")))
    elif ending == "missing_reason":
        frames.append(frame("message_stop"))
    elif ending == "truncated":
        frames.extend(terminal("max_tokens"))
    output = []
    try:
        if ending == "error":
            for value in frames:
                output.extend(guard.feed(value, tool_start=routes._tool_start(value)))
            guard.finish()
            assert output == [frames[-1]]
        else:
            with pytest.raises(RuntimeError, match="no tool calls forwarded"):
                for value in frames:
                    output.extend(guard.feed(value, tool_start=routes._tool_start(value)))
                guard.finish()
            assert output == []
    finally:
        guard.close()


def test_valid_large_arguments_spill_without_size_cap_and_retain_exact_bytes():
    guard = ToolCallGuard()
    frames = native_frames("tool_use", value="z" * (1024 * 1024 + 1))
    output = []
    try:
        for value in frames[:-1]:
            output.extend(guard.feed(value, tool_start=routes._tool_start(value)))
        assert b"".join(output) == frames[0]  # no executable tool bytes yet
        held = guard._held
        assert held._rolled  # bounded memory, not a blanket rejection of Write/code
        output.extend(guard.feed(frames[-1], tool_start=False))
        guard.finish()
        assert b"".join(output) == b"".join(frames)
        assert held.closed
    finally:
        guard.close()


def test_text_only_max_tokens_and_pretool_text_are_not_suppressed():
    text = [frame("content_block_start", index=0, content_block=dict(type="text", text="")),
            frame("content_block_delta", index=0, delta=dict(type="text_delta", text="Still visible")),
            frame("content_block_stop", index=0)]
    guard = ToolCallGuard()
    output = [part for value in [*text, *terminal("max_tokens")]
              for part in guard.feed(value, tool_start=False)]
    guard.finish()
    assert b"".join(output) == b"".join([*text, *terminal("max_tokens")])
    # Text before the first tool remains visible even when the tool is refused.
    with pytest.raises(RuntimeError, match="truncated"):
        for value in [*tool_frames(1), *terminal("max_tokens")]:
            output.extend(guard.feed(value, tool_start=routes._tool_start(value)))
    guard.close()
    assert b"tool_use" not in b"".join(output)


@pytest.mark.parametrize("ping_interval", [0, 0.001])
def test_pending_tools_keep_heartbeats_and_close_upstream_on_cancel(monkeypatch, ping_interval):
    async def run():
        entered = asyncio.Event()
        closed = asyncio.Event()

        class Adapter:
            def account_hash(self):
                return "fixture"

            async def call(self, *_):
                try:
                    yield native_frames("tool_use")[0]
                    for value in tool_frames():
                        yield value
                    entered.set()
                    yield frame("ping")
                    await asyncio.Event().wait()
                finally:
                    closed.set()

        monkeypatch.setattr(routes, "_PING_INTERVAL", ping_interval)
        source = routes._proxy_iter(Adapter(), {}, "fixture", provider="fixture", context=routes.ProxyRequestContext(request_id="fixture", provider="fixture"))
        assert b"message_start" in await anext(source)
        assert b'"ping"' in await asyncio.wait_for(anext(source), 1)
        assert entered.is_set()
        await source.aclose()
        assert closed.is_set()

    asyncio.run(run())


def test_responses_incomplete_without_response_stays_truncated():
    async def run():
        item = NS(type="function_call", id="item", call_id="call-0", name="work")

        async def upstream():
            yield NS(type="response.output_item.added", item=item)
            yield NS(type="response.incomplete", response=None)

        frames = [chunk async for chunk in responses_to_anthropic_sse(upstream(), anthropic_model="fixture")]
        delta = next(json.loads(line[5:]) for chunk in frames for line in chunk.splitlines()
                     if line.startswith(b"data:") and b'"stop_reason"' in line and b'"message_delta"' in line)
        assert delta["delta"]["stop_reason"] == "max_tokens"

    asyncio.run(run())
