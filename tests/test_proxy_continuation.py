"""TASK-1064: real retry-loop splice, native replay, and no-tool-replay safety."""
import asyncio
import copy
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from claude_code_bridge.proxy.adapters.base import ProviderAdapter
from claude_code_bridge.proxy.adapters.openai_chatgpt import OpenAIChatGPTAdapter
from claude_code_bridge.proxy.chatgpt_transport import ChatGPTEventTimeout, _event_object
from claude_code_bridge.proxy.continuation import (
    CommittedOutput, ContinuationMismatchError, MAX_CAPTURE_CHARS, validated_replay,
)
from claude_code_bridge.proxy import retry
from claude_code_bridge.proxy.request_context import ProxyRequestContext
from claude_code_bridge.proxy.translate import anthropic_to_responses
from tests.test_proxy_reasoning import CODEC, NATIVE, DONE, native_events, parse_sdk_message

BODY = {"model": "openai_chatgpt/test", "max_tokens": 100,
        "messages": [{"role": "user", "content": "Answer"}]}


def text(value):
    return {"type": "response.output_text.delta", "delta": value}


def tool():
    return {"type": "response.output_item.added", "item": {
        "type": "function_call", "id": "fc", "call_id": "call_fixture", "name": "check",
    }}


def stall():
    return ChatGPTEventTimeout("productive", transport="sse", attempt=1,
                               elapsed_seconds=90, productive_idle_seconds=90,
                               fallback_transport="sse")


class FixtureStream:
    def __init__(self, events):
        self.events = events
        self.close = AsyncMock()

    async def __aiter__(self):
        for event in self.events:
            if isinstance(event, BaseException):
                raise event
            yield _event_object(event)


class FixtureAdapter(ProviderAdapter):
    name = "fixture"

    def __init__(self, attempts):
        super().__init__()
        self.attempts = attempts
        self.bodies = []
        self.streams = []
        self._responses_client = NS(with_options=lambda **kw: None)

    async def start(self):
        pass

    async def authorize(self):
        return "fake", "https://fixture.test"

    def reasoning_codec(self, *args):
        return CODEC

    def supports_text_continuation(self, context):
        return True

    def retry_policy(self, *args):
        return retry.RetryPolicy(base_backoff_s=0, jitter_s=0)

    async def open_stream(self, **kwargs):
        self.bodies.append(copy.deepcopy(kwargs["body"]))
        attempt = self.attempts[len(self.bodies) - 1]
        if isinstance(attempt, BaseException):
            raise attempt
        stream = FixtureStream(attempt)
        self.streams.append(stream)
        return stream


@pytest.fixture(autouse=True)
def no_external_side_effects(monkeypatch):
    monkeypatch.setattr("claude_code_bridge.proxy.usage_capture.schedule_capture", lambda _: None)
    monkeypatch.setattr("claude_code_bridge.proxy.cache_diag.record", lambda _: None)


async def collect(adapter):
    return [frame async for frame in adapter.call(BODY, "test")]


def visible(frames):
    output = []
    for frame in frames:
        for line in frame.decode().splitlines():
            if line.startswith("data: "):
                payload = json.loads(line[6:])
                delta = payload.get("delta", {})
                if delta.get("type") == "text_delta":
                    output.append(delta["text"])
    return "".join(output)


@pytest.mark.parametrize("failure", [ConnectionError("cut"), stall()])
def test_native_reasoning_text_splice_and_next_request(failure):
    async def run():
        second = {**NATIVE, "id": "rs_second", "encrypted_content": "opaque-second"}
        adapter = FixtureAdapter([
            native_events(NATIVE) + [text("hello"), failure],
            native_events(second) + [text("he"), text("llo world"), DONE],
        ])
        frames = await collect(adapter)
        assert visible(frames) == "hello world"
        assert b'event: error' not in b"".join(frames)
        assert b"".join(frames).count(b"event: message_start") == 1
        blocks = await parse_sdk_message(frames)
        replay = anthropic_to_responses({"messages": BODY["messages"] + [
            {"role": "assistant", "content": blocks}, {"role": "user", "content": "Next"},
        ]}, "test", reasoning_codec=CODEC)["input"]
        assert replay[1:5] == [NATIVE, {"role": "assistant", "content": "hello"},
                               second, {"role": "assistant", "content": " world"}]
        assert adapter.bodies[1]["input"][1:3] == [NATIVE, {"role": "assistant", "content": "hello"}]
        assert len(adapter.bodies[0]["input"]) == 1  # original never mutated
        for stream in adapter.streams:
            stream.close.assert_awaited_once()
    asyncio.run(run())


@pytest.mark.parametrize("suffix", [
    [text("wrong"), DONE], [text("hel"), DONE], [tool(), DONE],
    [{"type": "response.refusal.delta", "delta": "No"}, DONE],
])
def test_mismatch_never_forwards_new_reasoning_text_or_tools(suffix):
    adapter = FixtureAdapter([[text("hello"), stall()], native_events(NATIVE) + suffix])
    frames = asyncio.run(collect(adapter))
    wire = b"".join(frames)
    assert visible(frames) == "hello"
    assert b'"type":"api_error"' in wire
    assert b"tool_use" not in wire and b"signature_delta" not in wire
    assert b"message_stop" not in wire
    assert len(adapter.bodies) == 2


@pytest.mark.parametrize("events", [
    [text("hello"), tool()],
    [{"type": "response.refusal.delta", "delta": "No"}],
    [text("x" * (MAX_CAPTURE_CHARS + 1))],
])
def test_nonrecoverable_committed_output_keeps_original_barrier(events):
    adapter = FixtureAdapter([events + [stall()]])
    frames = asyncio.run(collect(adapter))
    assert len(adapter.bodies) == 1
    assert b'"type":"api_error"' in b"".join(frames)


@pytest.mark.parametrize("failure", [ConnectionError("cut"), stall()])
def test_reasoning_only_checkpoint_continues_instead_of_killing_the_turn(failure):
    """TASK-1095 incident: Snark's reasoning item completed (signature forwarded,
    no text) and then the upstream went silent for 90s. That is phase TEXT with
    an empty text checkpoint, which used to fail upward as text_partial_no_retry
    and terminated the whole SDK turn."""
    async def run():
        second = {**NATIVE, "id": "rs_second", "encrypted_content": "opaque-second"}
        adapter = FixtureAdapter([
            native_events(NATIVE) + [failure],
            native_events(second) + [text("done"), tool(), DONE],
        ])
        frames = await collect(adapter)
        wire = b"".join(frames)
        assert b"event: error" not in wire
        assert wire.count(b"event: message_start") == 1
        assert wire.count(b'"type":"tool_use"') == 1
        assert visible(frames) == "done"
        recovery = adapter.bodies[1]["input"]
        assert recovery[:2] == adapter.bodies[0]["input"] + [NATIVE]
        assert recovery[2]["role"] == "developer"
        assert "before any visible output or tool call" in recovery[2]["content"]
        assert "EXACTLY" not in recovery[2]["content"]  # no prefix to reproduce
        assert len(recovery) == 3
        # The SDK message the CLI persists replays both native items in order.
        blocks = await parse_sdk_message(frames)
        replay = anthropic_to_responses({"messages": BODY["messages"] + [
            {"role": "assistant", "content": blocks}, {"role": "user", "content": "Next"},
        ]}, "test", reasoning_codec=CODEC)["input"]
        assert replay[1:4] == [NATIVE, second, {"role": "assistant", "content": "done"}]
    asyncio.run(run())


def test_reasoning_only_checkpoint_is_still_barred_after_overflow_or_refusal():
    checkpoint = CommittedOutput()
    checkpoint.add_reasoning(NATIVE)
    assert checkpoint.can_continue and checkpoint.text == ""
    checkpoint.blocked = True
    assert not checkpoint.can_continue
    with pytest.raises(ContinuationMismatchError):
        checkpoint.request({"input": []})
    assert not CommittedOutput().can_continue


def test_new_tool_after_validated_prefix_is_forwarded_once_and_never_retried():
    adapter = FixtureAdapter([[text("hello"), stall()], [text("hello"), tool(), stall()]])
    frames = asyncio.run(collect(adapter))
    assert visible(frames) == "hello"
    assert b"".join(frames).count(b'"type":"tool_use"') == 1
    assert len(adapter.bodies) == 2


def test_multiple_network_retries_rebuild_from_original_without_duplicate_checkpoints():
    adapter = FixtureAdapter([
        [text("one"), ConnectionError("cut")],
        [text("one two"), ConnectionError("cut")],
        [text("one two three"), DONE],
    ])
    frames = asyncio.run(collect(adapter))
    assert visible(frames) == "one two three"
    assert adapter.bodies[2]["input"][1] == {"role": "assistant", "content": "one two"}
    assert len(adapter.bodies[2]["input"]) == 3


def test_open_failure_during_continuation_reuses_checkpoint():
    adapter = FixtureAdapter([[text("one"), ConnectionError("cut")],
                              ConnectionError("open"), [text("one two"), DONE]])
    frames = asyncio.run(collect(adapter))
    assert visible(frames) == "one two"
    assert adapter.bodies[1] == adapter.bodies[2]


def test_stall_budget_stays_two_attempts():
    adapter = FixtureAdapter([[text("one"), stall()], [text("one two"), stall()]])
    frames = asyncio.run(collect(adapter))
    assert visible(frames) == "one two"
    assert b'"type":"api_error"' in b"".join(frames)
    assert len(adapter.bodies) == 2


def test_cancellation_is_never_retried():
    adapter = FixtureAdapter([[text("one"), asyncio.CancelledError()]])
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(collect(adapter))
    assert len(adapter.bodies) == 1
    adapter.streams[0].close.assert_awaited_once()


@pytest.mark.parametrize("bucket", list(retry.FailureBucket))
@pytest.mark.parametrize("phase", [retry.RetryPhase.TEXT, retry.RetryPhase.TOOL_COMMITTED])
def test_continuation_policy_keeps_permanent_rate_auth_bug_and_tool_barriers(bucket, phase):
    policy = retry.RetryPolicy()
    policy.start_attempt()
    decision = retry.decide(bucket=bucket, phase=phase, policy=policy, allow_text_continuation=True)
    assert decision.retry == (phase is retry.RetryPhase.TEXT and bucket in {
        retry.FailureBucket.A_TRANSIENT_NETWORK, retry.FailureBucket.B_UPSTREAM_5XX,
        retry.FailureBucket.G_PROGRESS_STALL,
    })


def test_adapter_opt_in_excludes_other_providers_and_lite():
    adapter = OpenAIChatGPTAdapter()
    assert adapter.supports_text_continuation(None)
    assert not adapter.supports_text_continuation(ProxyRequestContext(
        request_id="test", provider="openai_chatgpt", responses_transport="lite_ws"))
    assert not ProviderAdapter.supports_text_continuation(adapter, None)


def test_checkpoint_bound_and_copy():
    checkpoint = CommittedOutput()
    checkpoint.add_text("first")
    checkpoint.add_reasoning(NATIVE)
    checkpoint.add_text("second")
    request = checkpoint.request({"input": []})
    request["input"][1]["id"] = "mutated"
    assert checkpoint.items[1]["id"] == NATIVE["id"]
    checkpoint.add_text("x" * MAX_CAPTURE_CHARS)
    assert not checkpoint.can_continue and not checkpoint.items


def test_validation_buffer_is_bounded(monkeypatch):
    monkeypatch.setattr("claude_code_bridge.proxy.continuation.MAX_PENDING_EVENTS", 2)
    async def run():
        stream = FixtureStream([{"type": "response.in_progress"}] * 3)
        with pytest.raises(ContinuationMismatchError, match="buffer"):
            async for _ in validated_replay(stream, "hello"):
                pass
    asyncio.run(run())
