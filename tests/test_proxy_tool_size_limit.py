"""TASK-2330: a productive oversize tool call fails once, clearly; SDK resends wait.

The per-call argument cap (``argument_stream_timeout``) is a deliberate size
limit, not a liveness stall: a call still streaming arguments at the cap would
hit it again on every re-sample. It must not burn stall retries, and the model
must be told its tool call was too large. An identical SDK resend that arrives
while the proxy's own attempt is still running waits for it instead of getting
an immediate 500.
"""
import asyncio
import copy
import hashlib
import json

import pytest

from claude_code_bridge.proxy import routes
from claude_code_bridge.proxy.chatgpt_transport import (
    DEFAULT_ARGUMENT_STREAM_TIMEOUT,
    ChatGPTEventTimeout,
)
from claude_code_bridge.proxy.request_context import ProxyRequestContext
from claude_code_bridge.proxy.retry import (
    FailureBucket,
    RetryBudgetRegistry,
    RetryPolicy,
    classify_stream_exception,
)
from claude_code_bridge.proxy.tool_recovery import CARRIER_NAME
from tests import virtual_clock
from tests.test_proxy_progress_recovery import BODY, THINKING, TOOL, Feed, Harness, parse_message, success_frames
from tests.test_proxy_tool_recovery import collect, events

DELTAS = int(DEFAULT_ARGUMENT_STREAM_TIMEOUT) + 60


class PacedFeed(Feed):
    """Argument deltas arrive every second: productive, never idle."""

    async def next_frame(self):
        frame = await super().next_frame()
        if frame.get("type") == "response.function_call_arguments.delta":
            await asyncio.sleep(1.0)
        return frame


def oversize_frames():
    frames = [THINKING, {"type": "response.output_item.added", "item": TOOL}]
    # Distinct content per delta: a long, healthy argument, not a repetition loop.
    frames.extend(
        {"type": "response.function_call_arguments.delta", "item_id": TOOL["id"],
         "delta": '{"value":"' if i == 0 else hashlib.sha256(str(i).encode()).hexdigest() * 4}
        for i in range(DELTAS)
    )
    return frames


def oversize_harness(monkeypatch, mode):
    h = Harness(monkeypatch, mode, [], success_frames())
    paced = PacedFeed(oversize_frames())
    h.feeds[0] = paced
    if mode == "sse":
        h.create.side_effect = [paced, h.feeds[1]]
    else:
        h.connector.return_value = paced
    return h


def _timeout(phase):
    return ChatGPTEventTimeout(phase, transport="sse", attempt=1, elapsed_seconds=243.0,
                               productive_idle_seconds=0.0, fallback_transport="sse")


def test_argument_cap_is_not_bucketed_as_a_progress_stall():
    assert classify_stream_exception(_timeout("arguments")) is not FailureBucket.G_PROGRESS_STALL
    # Genuine liveness failures keep their stall bucket and bounded retry.
    for phase in ("productive", "next", "first_output", "whitespace"):
        assert classify_stream_exception(_timeout(phase)) is FailureBucket.G_PROGRESS_STALL


@pytest.mark.parametrize("mode", ["sse", "lite_ws"])
def test_productive_oversize_call_tells_model_once_without_resampling(monkeypatch, mode):
    async def run():
        h = oversize_harness(monkeypatch, mode)
        loop = asyncio.get_running_loop()
        started = loop.time()
        try:
            frames = await collect(h, copy.deepcopy(BODY), RetryBudgetRegistry())
            elapsed = loop.time() - started
            message = await parse_message(frames)
            tools = [b for b in message.content if b.type == "tool_use"]
            assert [t.name for t in tools] == [CARRIER_NAME]
            text = "".join(b.text for b in message.content if b.type == "text")
            assert "too large" in text
            assert "smaller" in text
            assert h.attempts == 1, "an oversize call must not be re-sampled"
            assert not any(s.get("state") in {"retrying", "reconnecting"} for s in h.statuses)
            assert elapsed < DEFAULT_ARGUMENT_STREAM_TIMEOUT + 30
        finally:
            await h.adapter.close()
    virtual_clock.run_timing(run())


def test_oversize_call_without_carrier_fails_once_with_size_cause(monkeypatch):
    async def run():
        h = oversize_harness(monkeypatch, "sse")
        budgets = RetryBudgetRegistry()
        assert budgets.claim_tool_recovery(h.context.request_id)  # carrier already used this turn
        body = copy.deepcopy(BODY)
        try:
            result = events(await collect(h, body, budgets))
            assert result[-1]["type"] == "error"
            assert result[-1]["error"]["type"] == "invalid_request_error"
            cause = result[-1]["error"]["message"]
            assert "too large" in cause
            again = events(await collect(h, body, budgets))
            assert again[-1]["error"]["message"] == cause
            assert h.attempts == 1
        finally:
            await h.adapter.close()
    virtual_clock.run_timing(run())


def _frame(event):
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


OK_FRAMES = [
    _frame({"type": "message_start", "message": {
        "id": "msg_fixture", "type": "message", "role": "assistant", "model": "fixture",
        "content": [], "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 0}}}),
    _frame({"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 1}}),
    _frame({"type": "message_stop"}),
]


class HeldAdapter:
    """The first sample stays in flight until released; later samples finish."""

    def __init__(self):
        self.calls = 0
        self.release = asyncio.Event()

    def account_hash(self):
        return "fixture-account"

    def retry_policy(self, model, context):
        return RetryPolicy()

    async def call(self, body, model, context):
        self.calls += 1
        if self.calls == 1:
            await self.release.wait()
        for frame in OK_FRAMES:
            yield frame


async def _drain(adapter, budgets, body, sink):
    context = ProxyRequestContext(request_id="turn", provider="openai_chatgpt")
    async for chunk in routes._proxy_iter(
        adapter, body, "fixture", provider="openai_chatgpt", context=context,
        budgets=budgets, fingerprint=budgets.fingerprint(body, "openai_chatgpt", "fixture"),
    ):
        sink.append(chunk)


def test_identical_resend_waits_for_running_attempt_instead_of_failing():
    async def run():
        adapter, budgets, body = HeldAdapter(), RetryBudgetRegistry(), {"messages": []}
        first, second = [], []
        original = asyncio.create_task(_drain(adapter, budgets, body, first))
        while adapter.calls == 0:
            await asyncio.sleep(0.1)
        resend = asyncio.create_task(_drain(adapter, budgets, body, second))
        await asyncio.sleep(60)
        assert not resend.done(), "the resend must wait, not fail in milliseconds"
        assert second and all(b'"ping"' in chunk for chunk in second), "keep the waiter alive"
        adapter.release.set()
        await asyncio.gather(original, resend)
        assert not any(b"event: error" in chunk for chunk in second)
        assert b"message_stop" in b"".join(second)
    virtual_clock.run_timing(run())


def test_resend_wait_is_bounded_and_says_what_is_happening():
    async def run():
        adapter, budgets, body = HeldAdapter(), RetryBudgetRegistry(), {"messages": []}
        first, second = [], []
        original = asyncio.create_task(_drain(adapter, budgets, body, first))
        while adapter.calls == 0:
            await asyncio.sleep(0.1)
        await _drain(adapter, budgets, body, second)
        error = events(second)[-1]
        assert error["type"] == "error"
        assert "already running" in error["error"]["message"]
        assert "server-side" not in error["error"]["message"]
        assert adapter.calls == 1
        original.cancel()
        with pytest.raises(asyncio.CancelledError):
            await original
    virtual_clock.run_timing(run())
