"""Standard HTTP models must not bypass progress bounds (Nova Sol incident)."""
import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from claude_code_bridge.proxy.responses_supervisor import ResponsesSSEStream
from claude_code_bridge.proxy.chatgpt_transport import ChatGPTEventTimeout
from claude_code_bridge.proxy.request_context import ProxyRequestContext


class HTTPStream:
    response = NS(headers={})

    def __init__(self, events):
        self.events = events
        self.close = AsyncMock()

    def __aiter__(self):
        return self.events


async def silent():
    await asyncio.sleep(10)
    yield None


async def done():
    yield NS(type="response.completed", response=NS(usage=None, status="completed"))


@pytest.mark.parametrize("phase", ["headers", "first", "productive"])
def test_http_progress_bounds_close_stream(phase):
    async def run():
        async def chatter():
            while True:
                await asyncio.sleep(.001)
                yield NS(type="response.in_progress")
        http = HTTPStream(chatter() if phase == "productive" else silent())
        async def open_http():
            if phase == "headers":
                await asyncio.sleep(10)
            return http
        stream = ResponsesSSEStream(open_http, first_event_timeout=.02,
                                    productive_idle_timeout=.03)
        try:
            with pytest.raises(ChatGPTEventTimeout) as error:
                async for _ in stream:
                    pass
            assert error.value.phase == ("first" if phase == "headers" else phase)
            assert error.value.transport == "sse"
            assert error.value.fallback_transport == "sse"
        finally:
            await stream.close()
        assert http.close.await_count == (0 if phase == "headers" else 1)
    asyncio.run(run())


@pytest.mark.parametrize("model", ["gpt-5.6-sol", "gpt-5.4", "future-model"])
@pytest.mark.parametrize("committed", [None, "text", "reasoning", "tool", "exhausted"])
def test_standard_adapter_supervised_recovery(monkeypatch, model, committed):
    from claude_code_bridge.proxy.adapters.openai_chatgpt import OpenAIChatGPTAdapter
    from claude_code_bridge.proxy import responses_supervisor, retry
    monkeypatch.setattr(retry, "compute_backoff", lambda *a, **k: 0)
    monkeypatch.setattr(responses_supervisor, "ResponsesSSEStream", lambda opener, **kw: ResponsesSSEStream(
        opener, first_event_timeout=.01, productive_idle_timeout=.015, **kw))

    async def run():
        async def first_events():
            if committed == "text":
                yield NS(type="response.output_text.delta", delta="visible")
            elif committed == "reasoning":
                yield NS(type="response.reasoning_summary_text.delta", delta="thinking")
            elif committed == "tool":
                yield NS(type="response.output_item.added", item=NS(type="function_call", id="fc", call_id="call", name="check"))
            await asyncio.sleep(10)
        streams = [HTTPStream(first_events()), HTTPStream(silent() if committed == "exhausted" else done())]
        create = AsyncMock(side_effect=streams)
        options = []
        request_client = NS(responses=NS(create=create))
        request_client.with_options = lambda **kw: (options.append(kw) or request_client)
        adapter = OpenAIChatGPTAdapter()
        adapter.authorize = AsyncMock(return_value=("test-token", "https://example.test"))
        adapter._http_client = NS()
        adapter._responses_client = NS(with_options=lambda **kw: (options.append(kw) or request_client), close=AsyncMock())
        statuses = []
        context = ProxyRequestContext(request_id="test", provider="openai_chatgpt", conversation_id="test",
                                      status_callback=lambda _, s: statuses.append(s))
        try:
            output = b"".join([chunk async for chunk in adapter.call(
                {"model": model, "max_tokens": 100, "messages": [{"role": "user", "content": "ping"}]}, model, context)])
            should_retry = committed in (None, "exhausted")
            assert create.await_count == (2 if should_retry else 1)
            assert options and all(o["max_retries"] == 0 for o in options)
            # Retain ordinary Responses payload; no Lite developer item migration.
            assert create.call_args.kwargs["model"] == model
            assert "instructions" in create.call_args.kwargs
            assert adapter._chatgpt_transport is None
            assert streams[0].close.await_count == 1
            if should_retry:
                assert statuses[0]["state"] == "reconnecting"
                assert statuses[0]["transport"] == "sse"
                assert streams[1].close.await_count == 1
            if committed is None:
                assert b"message_stop" in output
                assert statuses[-1]["state"] == "recovered"
            else:
                assert b'"type":"api_error"' in output
                assert b"message_stop" not in output
        finally:
            await adapter.close()
    asyncio.run(run())


def test_proxy_cancellation_is_not_logged_as_success(caplog):
    from claude_code_bridge.proxy.routes import _proxy_iter

    async def call(*args):
        raise asyncio.CancelledError()
        yield b""

    async def run():
        context = ProxyRequestContext(request_id="cancelled-test", provider="test")
        with pytest.raises(asyncio.CancelledError):
            async for _ in _proxy_iter(NS(call=call, account_hash=lambda: "test"), {}, "sol", provider="test", context=context):
                pass
    with caplog.at_level("INFO", logger="claude_code_bridge.proxy.routes"):
        asyncio.run(run())
    record = next(r.message for r in caplog.records if "proxy_stream_complete request_id=cancelled-test" in r.message)
    assert "status=499" in record


def test_cancellation_closes_http_without_replay():
    async def run():
        http = HTTPStream(silent())
        stream = ResponsesSSEStream(AsyncMock(return_value=http))
        task = asyncio.create_task(anext(stream))
        await asyncio.sleep(.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await stream.close()
        await stream.close()
        assert http.close.await_count == 1
    asyncio.run(run())
