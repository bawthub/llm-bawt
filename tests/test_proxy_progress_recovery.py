"""TASK-950: recover an unfinished thinking stream, never replay committed work."""
import asyncio
import copy
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import httpx
import pytest
from anthropic import AsyncAnthropic
# Warm the SDK's lazy fallback type import before starting millisecond test clocks.
from openai.types.responses import ResponseStreamEvent  # noqa: F401

from claude_code_bridge.proxy.adapters.openai_chatgpt import OpenAIChatGPTAdapter
from claude_code_bridge.proxy.chatgpt_transport import ChatGPTResponsesTransport, _event_object
from claude_code_bridge.proxy.reasoning import ReasoningCodec
from claude_code_bridge.proxy.request_context import ProxyRequestContext
from claude_code_bridge.proxy.responses_supervisor import ResponsesSSEStream
from claude_code_bridge.proxy.retry import FailureBucket, RetryPhase, RetryPolicy, decide
from claude_code_bridge.proxy.translate import anthropic_to_responses

MODEL = "test-progress-model"
BASE = "https://provider.test/codex"
NATIVE = {"type": "reasoning", "id": "rs_recovered", "summary": [],
          "encrypted_content": "opaque-test-state", "status": "completed"}
THINKING = {"type": "response.reasoning_summary_text.delta", "delta": "Working on it"}
TOOL = {"type": "function_call", "id": "fc_new", "call_id": "call_new", "name": "check"}
DONE = {"type": "response.completed", "response": {"status": "completed", "usage": {
    "input_tokens": 20, "output_tokens": 10, "input_tokens_details": {"cached_tokens": 5},
}}}
BODY = {"model": "openai_chatgpt/" + MODEL, "max_tokens": 100, "messages": [
    {"role": "user", "content": "Continue after the completed action"},
    {"role": "assistant", "content": [
        {"type": "tool_use", "id": "call_old", "name": "check", "input": {"value": "old"}},
    ]},
    {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "call_old", "content": "already done"},
    ]},
], "tools": [{"name": "check", "description": "Fixture tool", "input_schema": {
    "type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"],
}}]}


class Feed:
    """The same scripted upstream frames over HTTP or a fake WebSocket."""

    def __init__(self, frames):
        self.frames = iter(frames)
        self.response = NS(headers={})
        self.close = AsyncMock()
        self.send = AsyncMock()

    async def next_frame(self):
        frame = next(self.frames, None)
        if frame is None:
            await asyncio.sleep(10)  # supervisor/cancellation must interrupt this
            raise AssertionError("Unsupervised fixture stream")
        return frame

    def __aiter__(self):
        return self

    async def __anext__(self):
        return _event_object(await self.next_frame())

    async def recv(self):
        return json.dumps(await self.next_frame())


def success_frames():
    return [THINKING, {"type": "response.output_item.done", "item": NATIVE},
            {"type": "response.output_item.added", "item": TOOL},
            {"type": "response.function_call_arguments.delta", "item_id": "fc_new",
             "delta": '{"value":"new"}'},
            {"type": "response.function_call_arguments.done", "item_id": "fc_new"},
            {"type": "response.output_item.done", "item": TOOL}, DONE]


class Harness:
    def __init__(self, monkeypatch, mode, first, second, *, idle=.025):
        from claude_code_bridge.proxy import responses_supervisor, retry, usage_capture
        monkeypatch.setattr(retry, "compute_backoff", lambda *a, **kw: 0)
        monkeypatch.setattr(usage_capture, "schedule_capture", lambda _: None)
        monkeypatch.setattr(responses_supervisor, "ResponsesSSEStream", lambda opener, **kw:
                            ResponsesSSEStream(opener, first_event_timeout=idle,
                                               productive_idle_timeout=idle, **kw))
        self.feeds = [Feed(first), Feed(second)]
        self.create = AsyncMock(side_effect=self.feeds if mode == "sse" else self.feeds[1:])
        self.options = []
        client = NS(responses=NS(create=self.create), post=self.create)
        self.adapter = OpenAIChatGPTAdapter()
        self.adapter._cached_account_id = "account-test"
        self.adapter.authorize = AsyncMock(return_value=("fixture-token", BASE))
        self.adapter._http_client = NS()
        self.adapter._responses_client = NS(
            with_options=lambda **kw: (self.options.append(kw) or client), close=AsyncMock(),
        )
        self.connector = AsyncMock(return_value=self.feeds[0])
        if mode == "lite_ws":
            self.adapter._chatgpt_transport = ChatGPTResponsesTransport(
                connector=self.connector, first_event_timeout=idle,
                productive_idle_timeout=idle,
            )
        self.statuses = []
        self.context = ProxyRequestContext(
            request_id="fixture", provider="openai_chatgpt", conversation_id="fixture-thread",
            responses_transport=mode, status_callback=lambda _, status: self.statuses.append(status),
        )
        self.frames = []
        self.mode = mode

    async def collect(self):
        async for frame in self.adapter.call(copy.deepcopy(BODY), MODEL, self.context):
            self.frames.append(frame)

    @property
    def attempts(self):
        return self.create.await_count + self.connector.await_count

    def check_closed(self):
        assert self.feeds[0].close.await_count == 1
        assert self.feeds[1].close.await_count == (1 if self.attempts == 2 else 0)
        transport = self.adapter._chatgpt_transport
        if transport is not None:
            assert not transport._active
            # A successful fallback/cancellation may retain a routing token,
            # but never the unfinished WebSocket generation.
            assert all(s.socket is None for s in transport._idle.values())


async def parse_message(frames):
    transport = httpx.MockTransport(lambda _: httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content=b"".join(frames),
    ))
    async with AsyncAnthropic(api_key="fixture", http_client=httpx.AsyncClient(transport=transport)) as client:
        async with client.messages.stream(model="fixture", max_tokens=100, messages=[
            {"role": "user", "content": "fixture"},
        ]) as stream:
            return await stream.get_final_message()


@pytest.mark.parametrize("mode", ["sse", "lite_ws"])
def test_thinking_stall_recovers_with_valid_wire_and_no_replayed_tools(monkeypatch, mode):
    async def run():
        h = Harness(monkeypatch, mode, [THINKING], success_frames())
        try:
            await h.collect()
            assert h.attempts == 2
            assert [s["state"] for s in h.statuses] == ["reconnecting", "recovered"]
            assert h.statuses[0]["stall_phase"] == "productive"
            assert h.statuses[0]["max_attempts"] == 2
            assert all(o["max_retries"] == 0 for o in h.options)
            h.check_closed()
            message = await parse_message(h.frames)
            assert message.stop_reason == "tool_use"
            blocks = json.loads(message.model_dump_json())["content"]
            assert [b["id"] for b in blocks if b["type"] == "tool_use"] == ["call_new"]
            assert blocks[-1]["input"] == {"value": "new"}
            # The abandoned summary is display-only; only completed native state
            # from the successful attempt is replayable on the following hop.
            codec = ReasoningCodec.for_account("openai_chatgpt", MODEL, BASE, "account-test")
            replay = anthropic_to_responses({"messages": [{"role": "assistant", "content": blocks}]},
                                             MODEL, reasoning_codec=codec)
            assert [i for i in replay["input"] if i.get("type") == "reasoning"] == [NATIVE]
            assert b"".join(h.frames).count(b"event: message_start\n") == 1
            assert b"".join(h.frames).count(b"event: message_stop\n") == 1
            wire = [json.loads(f.split(b"data: ", 1)[1]) for f in h.frames if b"data: " in f]
            starts = [f["index"] for f in wire if f["type"] == "content_block_start"]
            stops = [f["index"] for f in wire if f["type"] == "content_block_stop"]
            assert starts == stops == list(range(len(starts)))
            if mode == "sse":
                calls = [c.kwargs for c in h.create.await_args_list]
            else:
                first = json.loads(h.feeds[0].send.call_args.args[0])
                first.pop("type")
                # WS carries the Lite flag in metadata; HTTP carries it in headers.
                assert first.pop("client_metadata") == {
                    "ws_request_header_x_openai_internal_codex_responses_lite": "true",
                }
                assert h.create.call_args.kwargs["options"]["headers"][
                    "x-openai-internal-codex-responses-lite"
                ] == "true"
                calls = [first, h.create.call_args.kwargs["body"]]
            assert calls[0] == calls[1]  # same history, routing/cache identity
            assert any(i.get("type") == "function_call_output" and i["call_id"] == "call_old"
                       for i in calls[1]["input"])
        finally:
            await h.adapter.close()
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["sse", "lite_ws"])
@pytest.mark.parametrize("barrier", ["text", "tool", "native_reasoning", "empty_native_reasoning"])
def test_committed_output_still_prevents_stall_replay(monkeypatch, mode, barrier):
    async def run():
        frames = [] if barrier == "empty_native_reasoning" else [THINKING]
        if barrier == "text":
            frames.append({"type": "response.output_text.delta", "delta": "Visible answer"})
        elif barrier == "tool":
            frames.append({"type": "response.output_item.added", "item": TOOL})
        else:
            frames.append({"type": "response.output_item.done", "item": NATIVE})
        h = Harness(monkeypatch, mode, frames, success_frames())
        try:
            await h.collect()
            assert h.attempts == 1
            assert not h.statuses
            assert b'"type":"api_error"' in b"".join(h.frames)
            assert b"event: message_stop" not in b"".join(h.frames)
            h.check_closed()
        finally:
            await h.adapter.close()
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["sse", "lite_ws"])
def test_thinking_stall_exhaustion_is_bounded_and_terminal(monkeypatch, mode):
    async def run():
        h = Harness(monkeypatch, mode, [THINKING], [THINKING])
        try:
            await h.collect()
            assert h.attempts == 2
            assert h.statuses[0]["state"] == "reconnecting"
            output = b"".join(h.frames)
            assert output.count(b"event: error\n") == 1
            assert b'"type":"api_error"' in output
            assert b"overloaded_error" not in output  # no outer CLI retry loop
            assert b"event: message_stop" not in output
            h.check_closed()
        finally:
            await h.adapter.close()
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["sse", "lite_ws"])
def test_cancelled_thinking_is_not_retried(monkeypatch, mode):
    async def run():
        h = Harness(monkeypatch, mode, [THINKING], success_frames(), idle=10)
        task = asyncio.create_task(h.collect())
        try:
            for _ in range(100):
                if b"thinking_delta" in b"".join(h.frames):
                    break
                await asyncio.sleep(.001)
            assert b"thinking_delta" in b"".join(h.frames)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert h.attempts == 1
            assert not h.statuses
            h.check_closed()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await h.adapter.close()
    asyncio.run(run())


@pytest.mark.parametrize("attempt", [1, 2, 3])
@pytest.mark.parametrize("budget", [1, 2, 3])
@pytest.mark.parametrize("phase", [RetryPhase.PRE_OUTPUT, RetryPhase.THINKING])
def test_supervised_retry_budget_is_owned_by_policy(attempt, budget, phase):
    policy = RetryPolicy(max_attempts=budget)
    for _ in range(attempt):
        policy.start_attempt()
    decision = decide(bucket=FailureBucket.G_PROGRESS_STALL, phase=phase, policy=policy)
    assert decision.retry is (attempt == 1 and budget > 1)
    if not decision.retry:
        assert decision.final_error_type == "api_error"
