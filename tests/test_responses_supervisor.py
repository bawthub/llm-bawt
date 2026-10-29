"""Standard HTTP models must not bypass progress bounds (Nova Sol incident)."""
import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from claude_code_bridge.proxy.responses_supervisor import ResponsesSSEStream
from claude_code_bridge.proxy.chatgpt_transport import (
    ChatGPTEventTimeout, DEFAULT_ARGUMENT_STREAM_TIMEOUT, DEFAULT_PRODUCTIVE_IDLE_TIMEOUT,
)
from claude_code_bridge.proxy.request_context import ProxyRequestContext
from tests import virtual_clock


class HTTPStream:
    response = NS(headers={})

    def __init__(self, events):
        self.events = events
        self.close = AsyncMock()

    def __aiter__(self):
        return self.events


async def silent(started=None):
    if started is not None:
        started.set()
    await asyncio.Event().wait()  # the supervisor must interrupt this pending receive
    yield None


async def done():
    yield NS(type="response.completed", response=NS(usage=None, status="completed"))


@pytest.mark.parametrize("phase", ["headers", "first", "productive"])
def test_http_progress_bounds_close_stream(phase):
    async def run():
        async def chatter():
            # An output item rules out the earlier first-output deadline;
            # only productive idle can end this pending receive.
            yield NS(type="response.output_item.added", item=NS(type="reasoning"))
            yield NS(type="response.in_progress")
            await asyncio.Event().wait()
        http = HTTPStream(chatter() if phase == "productive" else silent())
        async def open_http():
            if phase == "headers":
                await asyncio.Event().wait()
            return http
        stream = ResponsesSSEStream(open_http)
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
    virtual_clock.run_timing(run())


@pytest.mark.parametrize("model", ["gpt-5.6-sol", "gpt-5.4", "future-model"])
@pytest.mark.parametrize("committed", [None, "text", "thinking", "reasoning", "tool", "exhausted"])
def test_standard_adapter_supervised_recovery(monkeypatch, model, committed):
    from claude_code_bridge.proxy.adapters.openai_chatgpt import OpenAIChatGPTAdapter
    from claude_code_bridge.proxy import retry
    monkeypatch.setattr(retry, "compute_backoff", lambda *a, **k: 0)

    async def run():
        async def first_events():
            if committed == "text":
                yield NS(type="response.output_text.delta", delta="visible")
            elif committed in ("thinking", "reasoning"):
                yield NS(type="response.reasoning_summary_text.delta", delta="thinking")
                if committed == "reasoning":
                    yield NS(type="response.output_item.done", item=NS(
                        type="reasoning", id="rs_done", summary=[], encrypted_content="opaque",
                    ))
            elif committed == "tool":
                yield NS(type="response.output_item.added", item=NS(type="function_call", id="fc", call_id="call", name="check"))
            await asyncio.Event().wait()
        streams = [HTTPStream(first_events()), HTTPStream(silent() if committed == "exhausted" else done())]
        create = AsyncMock(side_effect=streams)
        options = []
        request_client = NS(responses=NS(create=create))
        request_client.with_options = lambda **kw: (options.append(kw) or request_client)
        adapter = OpenAIChatGPTAdapter()
        adapter._cached_account_id = "fixture-account"
        adapter.authorize = AsyncMock(return_value=("test-token", "https://example.test"))
        adapter._http_client = NS()
        adapter._responses_client = NS(with_options=lambda **kw: (options.append(kw) or request_client), close=AsyncMock())
        statuses = []
        context = ProxyRequestContext(request_id="test", provider="openai_chatgpt", conversation_id="test",
                                      status_callback=lambda _, s: statuses.append(s))
        try:
            output = b"".join([chunk async for chunk in adapter.call(
                {"model": model, "max_tokens": 100, "messages": [{"role": "user", "content": "ping"}]}, model, context)])
            # Text may retry, but this fixture returns no matching prefix:
            # it must still fail without forwarding a completion or tool.
            # TASK-1095: a reasoning-only checkpoint (no visible output, no
            # tool) is continuable and needs no prefix, so it recovers.
            should_retry = committed in (None, "thinking", "reasoning", "text", "exhausted")
            assert create.await_count == (2 if should_retry else 1)
            assert options and all(o["max_retries"] == 0 for o in options)
            # Retain ordinary Responses payload; no Lite developer item migration.
            assert create.call_args.kwargs["model"] == model
            assert "instructions" in create.call_args.kwargs
            assert adapter._chatgpt_transport is None
            assert streams[0].close.await_count == 1
            if should_retry:
                states = [status["state"] for status in statuses]
                # The production 30s notice precedes the 60/90s retry deadline.
                assert states[:3] == ["waiting", "stalled", "retrying"]
                retry_status = statuses[2]
                assert retry_status["transport"] == "sse"
                # TASK-1079: an HTTPS stall is not "reconnecting over HTTPS".
                assert "HTTPS" not in retry_status["message"]
                assert streams[1].close.await_count == 1
            if committed in (None, "thinking", "reasoning"):
                assert b"message_stop" in output
                assert statuses[-1]["state"] == "recovered"
            else:
                assert b'"type":"api_error"' in output
                assert b"message_stop" not in output
        finally:
            await adapter.close()
    virtual_clock.run_timing(run())


def test_stall_log_reports_idle_event_types_without_payloads(caplog):
    async def run():
        async def chatter():
            yield NS(type="response.output_text.delta", delta="secret-user-text")
            yield NS(type="response.in_progress", payload="secret-opaque-content")
            await asyncio.Event().wait()
        stream = ResponsesSSEStream(AsyncMock(return_value=HTTPStream(chatter())))
        try:
            with pytest.raises(ChatGPTEventTimeout):
                async for _ in stream:
                    pass
        finally:
            await stream.close()
    with caplog.at_level("WARNING"):
        virtual_clock.run_timing(run())
    assert "last_event='response.in_progress'" in caplog.text
    assert "idle_events={'response.in_progress':" in caplog.text
    assert "secret-user-text" not in caplog.text
    assert "secret-opaque-content" not in caplog.text


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
        started = asyncio.Event()
        http = HTTPStream(silent(started))
        stream = ResponsesSSEStream(AsyncMock(return_value=http))
        task = asyncio.create_task(anext(stream))
        await started.wait()  # cancel while the HTTP receive is actually pending
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await stream.close()
        await stream.close()
        assert http.close.await_count == 1
    virtual_clock.run_timing(run())


def _adapter_call(streams, monkeypatch, *, model="gpt-6-sol"):
    """Drive the real ChatGPT adapter over scripted upstream attempts."""
    from claude_code_bridge.proxy.adapters.openai_chatgpt import OpenAIChatGPTAdapter
    from claude_code_bridge.proxy import retry
    monkeypatch.setattr(retry, "compute_backoff", lambda *a, **k: 0)

    async def run():
        create = AsyncMock(side_effect=streams)
        request_client = NS(responses=NS(create=create))
        request_client.with_options = lambda **kw: request_client
        adapter = OpenAIChatGPTAdapter()
        adapter._cached_account_id = "fixture-account"
        adapter.authorize = AsyncMock(return_value=("test-token", "https://example.test"))
        adapter._http_client = NS()
        adapter._responses_client = NS(with_options=lambda **kw: request_client, close=AsyncMock())
        statuses = []
        context = ProxyRequestContext(request_id="test", provider="openai_chatgpt", conversation_id="test",
                                      status_callback=lambda _, s: statuses.append(s))
        loop = asyncio.get_running_loop()
        started = loop.time()
        try:
            output = b"".join([chunk async for chunk in adapter.call(
                {"model": model, "max_tokens": 100, "messages": [{"role": "user", "content": "ping"}]},
                model, context)])
        finally:
            await adapter.close()
        return output, create.await_count, loop.time() - started, statuses
    return virtual_clock.run_timing(run())


@pytest.mark.parametrize("after_output", ["silent", "lifecycle_only"])
def test_first_output_then_no_progress_is_a_stall_retried_once(monkeypatch, after_output):
    """TASK-1511: output starts, then the upstream goes quiet or only chatters."""
    async def stalled():
        yield NS(type="response.reasoning_summary_text.delta", delta="thinking")
        while True:
            await asyncio.sleep(5 if after_output == "lifecycle_only" else 3600)
            yield NS(type="response.in_progress")

    output, attempts, elapsed, statuses = _adapter_call(
        [HTTPStream(stalled()), HTTPStream(done())], monkeypatch)
    assert attempts == 2
    assert b"message_stop" in output
    assert [s["state"] for s in statuses][:3] == ["waiting", "stalled", "retrying"]
    # Detected on the 90s productive deadline, far inside the app's 600s watchdog.
    assert elapsed == pytest.approx(DEFAULT_PRODUCTIVE_IDLE_TIMEOUT, abs=1)


def test_tool_argument_runaway_fails_visibly_inside_app_watchdog(monkeypatch):
    """TASK-1511 (Nova 2026-10-29): productive-looking tool-argument deltas are
    invisible downstream, so the 90s productive timer never fires and the app's
    600s no-event watchdog killed the turn silently. A committed tool call is
    never replayed, so the bound must surface an error, not retry."""
    async def runaway():
        yield NS(type="response.output_item.added",
                 item=NS(type="function_call", id="fc", call_id="call", name="Write"))
        while True:
            await asyncio.sleep(1)
            yield NS(type="response.function_call_arguments.delta", delta="ab")

    output, attempts, elapsed, _ = _adapter_call([HTTPStream(runaway())], monkeypatch)
    assert attempts == 1
    assert b'"type":"api_error"' in output
    assert b"message_stop" not in output
    # The argument clock starts at the first delta, one second in.
    assert elapsed == pytest.approx(1 + DEFAULT_ARGUMENT_STREAM_TIMEOUT, abs=1)


def test_argument_deadline_is_per_tool_call():
    """Sequential tool calls each get the full argument budget."""
    async def run():
        async def calls():
            for _ in range(3):
                yield NS(type="response.output_item.added", item=NS(type="function_call"))
                # Each call alone stays under the budget; together they exceed it.
                for _ in range(int(DEFAULT_ARGUMENT_STREAM_TIMEOUT) - 40):
                    await asyncio.sleep(1)
                    yield NS(type="response.function_call_arguments.delta", delta="ab")
                yield NS(type="response.output_item.done", item=NS(type="function_call"))
            yield NS(type="response.completed", response=NS(usage=None, status="completed"))

        stream = ResponsesSSEStream(AsyncMock(return_value=HTTPStream(calls())))
        await stream.prepare()
        try:
            events = [event async for event in stream]
        finally:
            await stream.close()
        assert events[-1].type == "response.completed"
    virtual_clock.run_timing(run())
