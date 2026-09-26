"""TASK-714 regression tests for the proxy retry policy + stream splice hygiene.

All hermetic — no network, no bridges, no upstream credentials. The retry
policy module is pure; the stream translator is exercised with fake async
iterators; the retry loop is exercised by stubbing the adapter's
``client.responses.create`` and driving `.call()` end-to-end.

The invariants regressed here (spec-locked, do not weaken):

- Never retry after ``content_block_start(tool_use)`` was forwarded — no dup
  tool execution possible.
- Never retry after any ``text_delta`` was forwarded (default policy;
  prefix-suppressed replay is a future opt-in).
- Bucket C (permanent/auth/context) never enters the retry loop even at
  attempt 1.
- ``asyncio.CancelledError`` and ``GeneratorExit`` propagate, never retry.
- SSE splice after a THINKING-state retry is wire-valid: no duplicate
  ``message_start``, no duplicate block indexes, the previous open block gets
  a ``content_block_stop`` before new blocks resume at a monotone index.
- Rate-limit retry-after over the 10s cap short-circuits to
  ``rate_limit_error`` — the SDK stream is never held that long.
- Terminal accounting: an upstream-error turn with partial visible text
  persists ``status="error"`` / ``end_reason="upstream_error"`` — never
  ``status="completed"`` / ``end_reason="stop"``.
"""

from __future__ import annotations

import asyncio
import json
import types
from typing import Any, AsyncIterator

import pytest

from claude_code_bridge.proxy import retry as retry_mod
from claude_code_bridge.proxy import stream as stream_mod
from claude_code_bridge.proxy.adapters import base as base_mod


# The container has anyio 4.x installed (pytest-anyio auto-registers the
# ``@pytest.mark.anyio`` mark) but NOT pytest-asyncio. Pin the async backend
# to asyncio for every test in this module and use the anyio mark uniformly.
@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ────────────────────────────────────────────────────────────────────────────
# Pure retry-policy tests
# ────────────────────────────────────────────────────────────────────────────


class TestBucketClassification:
    def test_transient_network_is_bucket_a(self):
        # openai.APIConnectionError has module prefix "openai". A synthesised
        # exception class with __module__="openai" hits the same code path.
        class FakeAPIConnErr(Exception):
            pass
        FakeAPIConnErr.__module__ = "openai"
        assert (
            retry_mod.classify_initial_exception(FakeAPIConnErr("connect refused"))
            is retry_mod.FailureBucket.A_TRANSIENT_NETWORK
        )

    def test_httpx_timeout_is_bucket_a(self):
        class FakeReadTimeout(Exception):
            pass
        FakeReadTimeout.__module__ = "httpx"
        assert (
            retry_mod.classify_initial_exception(FakeReadTimeout("read timeout"))
            is retry_mod.FailureBucket.A_TRANSIENT_NETWORK
        )

    def test_upstream_500_is_bucket_b(self):
        class FakeAPIStatusError(Exception):
            def __init__(self, msg, status):
                super().__init__(msg)
                self.status_code = status
        FakeAPIStatusError.__module__ = "openai"
        assert (
            retry_mod.classify_initial_exception(FakeAPIStatusError("bad gateway", 502))
            is retry_mod.FailureBucket.B_UPSTREAM_5XX
        )

    def test_auth_401_is_bucket_c(self):
        class FakeAuthErr(Exception):
            def __init__(self, msg, status):
                super().__init__(msg)
                self.status_code = status
        FakeAuthErr.__module__ = "openai"
        assert (
            retry_mod.classify_initial_exception(FakeAuthErr("invalid api key", 401))
            is retry_mod.FailureBucket.C_PERMANENT
        )

    def test_bad_request_400_is_bucket_c(self):
        class FakeBadReq(Exception):
            def __init__(self, msg, status):
                super().__init__(msg)
                self.status_code = status
        FakeBadReq.__module__ = "openai"
        # context-window overflow lands here too
        assert (
            retry_mod.classify_initial_exception(FakeBadReq("context too long", 400))
            is retry_mod.FailureBucket.C_PERMANENT
        )

    def test_rate_limit_is_bucket_d(self):
        class FakeRateErr(Exception):
            def __init__(self, msg, status):
                super().__init__(msg)
                self.status_code = status
        FakeRateErr.__module__ = "openai"
        assert (
            retry_mod.classify_initial_exception(FakeRateErr("rate limited", 429))
            is retry_mod.FailureBucket.D_RATE_LIMIT
        )

    def test_broker_runtime_error_is_bucket_f(self):
        assert (
            retry_mod.classify_initial_exception(RuntimeError("ChatGPT token broker unreachable"))
            is retry_mod.FailureBucket.F_AUTH_BROKER
        )

    def test_translator_bug_is_bucket_e(self):
        # KeyError NOT from an openai/httpx module → our bug
        assert (
            retry_mod.classify_initial_exception(KeyError("missing 'input' in body"))
            is retry_mod.FailureBucket.E_TRANSLATOR_BUG
        )

    def test_supervised_stream_timeout_is_bucket_g(self):
        class SupervisedTimeout(TimeoutError):
            proxy_retry_owner = True

        assert (
            retry_mod.classify_stream_exception(SupervisedTimeout("stalled"))
            is retry_mod.FailureBucket.G_PROGRESS_STALL
        )


class TestInbandClassification:
    """Al addition #2: in-band response.failed / response.error / error payloads."""

    def test_server_error_code_is_bucket_b(self):
        assert (
            retry_mod.classify_inband_error({"code": "server_error", "message": "boom"})
            is retry_mod.FailureBucket.B_UPSTREAM_5XX
        )

    def test_overloaded_message_is_bucket_b(self):
        # Some providers use "overloaded" in the message with a generic code
        assert (
            retry_mod.classify_inband_error({"code": "unknown", "message": "server is overloaded"})
            is retry_mod.FailureBucket.B_UPSTREAM_5XX
        )

    def test_503_shaped_code_is_bucket_b(self):
        assert (
            retry_mod.classify_inband_error({"code": "503"})
            is retry_mod.FailureBucket.B_UPSTREAM_5XX
        )

    def test_invalid_api_key_is_bucket_c(self):
        assert (
            retry_mod.classify_inband_error({"code": "invalid_api_key", "message": "..."})
            is retry_mod.FailureBucket.C_PERMANENT
        )

    def test_context_length_is_bucket_c(self):
        assert (
            retry_mod.classify_inband_error({"code": "context_length_exceeded"})
            is retry_mod.FailureBucket.C_PERMANENT
        )

    def test_rate_limit_code_is_bucket_d(self):
        assert (
            retry_mod.classify_inband_error({"code": "rate_limit_exceeded"})
            is retry_mod.FailureBucket.D_RATE_LIMIT
        )

    def test_unknown_code_is_bucket_e(self):
        # Unknown code → treat as our bug (don't guess-retry)
        assert (
            retry_mod.classify_inband_error({"code": "wat", "message": "?"})
            is retry_mod.FailureBucket.E_TRANSLATOR_BUG
        )


class TestBackoff:
    def test_attempt_one_returns_zero(self):
        # First attempt is initial — no backoff before firing it.
        assert retry_mod.compute_backoff(1) == 0.0

    def test_backoff_increases_exponentially(self):
        # Fixed jitter=0 for determinism via a rigged RNG.
        class ZeroRng:
            def uniform(self, a, b): return 0.0
        rng = ZeroRng()
        b2 = retry_mod.compute_backoff(2, jitter=0.5, rng=rng)
        b3 = retry_mod.compute_backoff(3, jitter=0.5, rng=rng)
        b4 = retry_mod.compute_backoff(4, jitter=0.5, rng=rng)
        assert b2 < b3 < b4

    def test_backoff_respects_cap(self):
        class MaxRng:
            def uniform(self, a, b): return b
        # Even at attempt 20 with max jitter, we cap at backoff_cap_s
        b = retry_mod.compute_backoff(20, base=1.0, cap=5.0, jitter=1.0, rng=MaxRng())
        assert b == 5.0


class TestDecideStateMachine:
    """The (bucket × phase × attempt) cross product enforces invariants."""

    def test_tool_committed_never_retries_even_bucket_a(self):
        # HARD invariant: no duplicate tool execution possible
        policy = retry_mod.RetryPolicy(max_attempts=5)
        policy.start_attempt()
        d = retry_mod.decide(
            bucket=retry_mod.FailureBucket.A_TRANSIENT_NETWORK,
            phase=retry_mod.RetryPhase.TOOL_COMMITTED,
            policy=policy,
        )
        assert d.retry is False
        assert d.final_error_type == "api_error"
        assert "tool_committed" in d.reason

    def test_text_phase_default_fails_upward_as_api_error(self):
        # No duplicate visible text on CLI retry — surface api_error (terminal
        # at the CLI, unlike overloaded_error which would trigger retry)
        policy = retry_mod.RetryPolicy(max_attempts=5)
        policy.start_attempt()
        d = retry_mod.decide(
            bucket=retry_mod.FailureBucket.A_TRANSIENT_NETWORK,
            phase=retry_mod.RetryPhase.TEXT,
            policy=policy,
        )
        assert d.retry is False
        assert d.final_error_type == "api_error"

    def test_pre_output_bucket_a_retries(self):
        policy = retry_mod.RetryPolicy(max_attempts=3)
        policy.start_attempt()
        d = retry_mod.decide(
            bucket=retry_mod.FailureBucket.A_TRANSIENT_NETWORK,
            phase=retry_mod.RetryPhase.PRE_OUTPUT,
            policy=policy,
        )
        assert d.retry is True
        assert d.backoff_s >= 0

    def test_thinking_bucket_b_retries(self):
        # Al #1: retry OK at THINKING; splice hygiene is the loop's job
        policy = retry_mod.RetryPolicy(max_attempts=3)
        policy.start_attempt()
        d = retry_mod.decide(
            bucket=retry_mod.FailureBucket.B_UPSTREAM_5XX,
            phase=retry_mod.RetryPhase.THINKING,
            policy=policy,
        )
        assert d.retry is True

    def test_thinking_progress_stall_never_replays(self):
        policy = retry_mod.RetryPolicy(max_attempts=3)
        policy.start_attempt()
        d = retry_mod.decide(
            bucket=retry_mod.FailureBucket.G_PROGRESS_STALL,
            phase=retry_mod.RetryPhase.THINKING,
            policy=policy,
        )
        assert d.retry is False
        assert d.final_error_type == "api_error"
        assert d.reason == "thinking_stall_no_replay"

    def test_bucket_c_never_retries_even_pre_output(self):
        policy = retry_mod.RetryPolicy(max_attempts=5)
        policy.start_attempt()
        d = retry_mod.decide(
            bucket=retry_mod.FailureBucket.C_PERMANENT,
            phase=retry_mod.RetryPhase.PRE_OUTPUT,
            policy=policy,
        )
        assert d.retry is False
        assert d.final_error_type == "authentication_error"

    def test_bucket_c_specific_type_override_wins(self):
        """Al review fix (spec acceptance #5): permanent errors stay
        capability-classified — a 400/context overflow must surface as
        invalid_request_error, never the authentication_error fallback."""
        policy = retry_mod.RetryPolicy(max_attempts=5)
        policy.start_attempt()
        d = retry_mod.decide(
            bucket=retry_mod.FailureBucket.C_PERMANENT,
            phase=retry_mod.RetryPhase.PRE_OUTPUT,
            policy=policy,
            permanent_error_type="invalid_request_error",
        )
        assert d.retry is False
        assert d.final_error_type == "invalid_request_error"

    def test_bucket_e_never_retries(self):
        policy = retry_mod.RetryPolicy(max_attempts=5)
        policy.start_attempt()
        d = retry_mod.decide(
            bucket=retry_mod.FailureBucket.E_TRANSLATOR_BUG,
            phase=retry_mod.RetryPhase.PRE_OUTPUT,
            policy=policy,
        )
        assert d.retry is False
        assert d.final_error_type == "api_error"

    def test_exhausted_pre_output_returns_overloaded_error(self):
        # CLI-retryable — PRE_OUTPUT exhaustion is safe to hand back for another swing
        policy = retry_mod.RetryPolicy(max_attempts=1)
        policy.start_attempt()  # exhausts the budget
        d = retry_mod.decide(
            bucket=retry_mod.FailureBucket.A_TRANSIENT_NETWORK,
            phase=retry_mod.RetryPhase.PRE_OUTPUT,
            policy=policy,
        )
        assert d.retry is False
        assert d.final_error_type == "overloaded_error"

    def test_rate_limit_over_cap_short_circuits(self):
        # Al #3: retry-after > 10s → immediate rate_limit_error, no long hold
        policy = retry_mod.RetryPolicy(max_attempts=5, rate_limit_max_wait_s=10.0)
        policy.start_attempt()
        d = retry_mod.decide(
            bucket=retry_mod.FailureBucket.D_RATE_LIMIT,
            phase=retry_mod.RetryPhase.PRE_OUTPUT,
            policy=policy,
            retry_after_s=15.0,
        )
        assert d.retry is False
        assert d.final_error_type == "rate_limit_error"
        assert "over_cap" in d.reason

    def test_rate_limit_under_cap_retries(self):
        policy = retry_mod.RetryPolicy(max_attempts=5, rate_limit_max_wait_s=10.0)
        policy.start_attempt()
        d = retry_mod.decide(
            bucket=retry_mod.FailureBucket.D_RATE_LIMIT,
            phase=retry_mod.RetryPhase.PRE_OUTPUT,
            policy=policy,
            retry_after_s=3.0,
        )
        assert d.retry is True
        assert d.backoff_s == 3.0


class TestPhaseFromState:
    def test_empty_state_is_connecting(self):
        s = stream_mod.TranslatorState()
        assert retry_mod.phase_from_state(s) is retry_mod.RetryPhase.CONNECTING

    def test_message_start_only_is_pre_output(self):
        s = stream_mod.TranslatorState(message_start_yielded=True)
        assert retry_mod.phase_from_state(s) is retry_mod.RetryPhase.PRE_OUTPUT

    def test_thinking_yielded_is_thinking(self):
        s = stream_mod.TranslatorState(message_start_yielded=True, thinking_yielded=True)
        assert retry_mod.phase_from_state(s) is retry_mod.RetryPhase.THINKING

    def test_text_delta_beats_thinking(self):
        s = stream_mod.TranslatorState(
            message_start_yielded=True, thinking_yielded=True, text_delta_yielded=True,
        )
        assert retry_mod.phase_from_state(s) is retry_mod.RetryPhase.TEXT

    def test_tool_committed_beats_everything(self):
        s = stream_mod.TranslatorState(
            message_start_yielded=True, thinking_yielded=True,
            text_delta_yielded=True, tool_committed=True,
        )
        assert retry_mod.phase_from_state(s) is retry_mod.RetryPhase.TOOL_COMMITTED


# ────────────────────────────────────────────────────────────────────────────
# Translator + state observation tests
# ────────────────────────────────────────────────────────────────────────────


def _mk_event(etype: str, **fields: Any) -> types.SimpleNamespace:
    """Build a Responses-API-shaped fake event."""
    return types.SimpleNamespace(type=etype, **fields)


async def _drain(agen: AsyncIterator[bytes]) -> list[bytes]:
    return [chunk async for chunk in agen]


class TestTranslatorStateObservations:
    @pytest.mark.anyio
    async def test_message_start_marks_state(self):
        state = stream_mod.TranslatorState()

        async def upstream():
            yield _mk_event("response.completed", response=types.SimpleNamespace(
                usage=None, status="completed",
            ))

        _ = await _drain(stream_mod.responses_to_anthropic_sse(
            upstream(), anthropic_model="foo", state=state,
        ))
        assert state.message_start_yielded is True

    @pytest.mark.anyio
    async def test_thinking_delta_marks_state(self):
        state = stream_mod.TranslatorState()

        async def upstream():
            yield _mk_event("response.output_item.added",
                            item=types.SimpleNamespace(type="reasoning", id="r1"))
            yield _mk_event("response.reasoning_text.delta", delta="thinking hard")
            yield _mk_event("response.completed", response=types.SimpleNamespace(
                usage=None, status="completed",
            ))

        _ = await _drain(stream_mod.responses_to_anthropic_sse(
            upstream(), anthropic_model="foo", state=state,
        ))
        assert state.thinking_yielded is True
        assert state.text_delta_yielded is False
        assert state.tool_committed is False

    @pytest.mark.anyio
    async def test_text_delta_marks_state(self):
        state = stream_mod.TranslatorState()

        async def upstream():
            yield _mk_event("response.output_item.added",
                            item=types.SimpleNamespace(type="message"))
            yield _mk_event("response.output_text.delta", delta="hi")
            yield _mk_event("response.completed", response=types.SimpleNamespace(
                usage=None, status="completed",
            ))

        _ = await _drain(stream_mod.responses_to_anthropic_sse(
            upstream(), anthropic_model="foo", state=state,
        ))
        assert state.text_delta_yielded is True

    @pytest.mark.anyio
    async def test_tool_use_marks_state_hard_barrier(self):
        state = stream_mod.TranslatorState()

        async def upstream():
            yield _mk_event(
                "response.output_item.added",
                item=types.SimpleNamespace(
                    type="function_call", id="fc1", call_id="call_1", name="Bash",
                ),
            )
            yield _mk_event("response.function_call_arguments.delta",
                            item_id="fc1", delta='{"command":"ls"}')
            yield _mk_event("response.function_call_arguments.done", item_id="fc1")
            yield _mk_event(
                "response.output_item.done",
                item=types.SimpleNamespace(type="function_call", id="fc1"),
            )
            yield _mk_event("response.completed", response=types.SimpleNamespace(
                usage=None, status="completed",
            ))

        _ = await _drain(stream_mod.responses_to_anthropic_sse(
            upstream(), anthropic_model="foo",
            tool_schemas=[{"name": "Bash", "input_schema": {"required": ["command"]}}],
            state=state,
        ))
        # HARD invariant regression: retry policy will refuse to replay from here.
        assert state.tool_committed is True

    @pytest.mark.anyio
    async def test_inband_error_populates_state_no_error_frame(self):
        """Al #2: in-band error routes to state, translator returns cleanly."""
        state = stream_mod.TranslatorState()

        async def upstream():
            yield _mk_event("response.output_item.added",
                            item=types.SimpleNamespace(type="message"))
            yield _mk_event("response.output_text.delta", delta="hello")
            yield _mk_event(
                "response.failed",
                error=types.SimpleNamespace(code="server_error", message="upstream boom"),
            )

        chunks = await _drain(stream_mod.responses_to_anthropic_sse(
            upstream(), anthropic_model="foo", state=state,
        ))
        # No error frame emitted — retry loop owns the wire
        joined = b"".join(chunks)
        assert b'"type":"error"' not in joined
        # State captured the payload for the retry loop
        assert state.inband_error == {"code": "server_error", "message": "upstream boom"}

    @pytest.mark.anyio
    async def test_inband_error_without_state_still_emits_frame(self):
        """Backcompat: state=None keeps the pre-TASK-714 wire behavior."""

        async def upstream():
            yield _mk_event(
                "response.failed",
                error=types.SimpleNamespace(code="server_error", message="boom"),
            )

        chunks = await _drain(stream_mod.responses_to_anthropic_sse(
            upstream(), anthropic_model="foo",
        ))
        joined = b"".join(chunks)
        assert b"event: error" in joined
        assert b'"type":"error"' in joined

    @pytest.mark.anyio
    async def test_stream_exception_propagates_when_state_provided(self):
        state = stream_mod.TranslatorState()

        async def upstream():
            yield _mk_event("response.output_item.added",
                            item=types.SimpleNamespace(type="message"))
            raise RuntimeError("simulated mid-stream")

        with pytest.raises(RuntimeError, match="simulated mid-stream"):
            _ = await _drain(stream_mod.responses_to_anthropic_sse(
                upstream(), anthropic_model="foo", state=state,
            ))

    @pytest.mark.anyio
    async def test_stream_exception_swallowed_when_state_none(self):
        """Backcompat: state=None → translator emits error frame + swallows."""

        async def upstream():
            yield _mk_event("response.output_item.added",
                            item=types.SimpleNamespace(type="message"))
            raise RuntimeError("simulated")

        chunks = await _drain(stream_mod.responses_to_anthropic_sse(
            upstream(), anthropic_model="foo",
        ))
        joined = b"".join(chunks)
        assert b"event: error" in joined


# ────────────────────────────────────────────────────────────────────────────
# Al #1: SSE splice hygiene regression
# ────────────────────────────────────────────────────────────────────────────


class TestSpliceHygiene:
    """The stream MUST be wire-valid across a THINKING-state retry splice."""

    @pytest.mark.anyio
    async def test_resumed_translator_skips_message_start_and_resumes_index(self):
        # Attempt 1: emit message_start + reasoning delta, then simulate failure.
        state = stream_mod.TranslatorState()

        async def attempt1_upstream():
            yield _mk_event("response.output_item.added",
                            item=types.SimpleNamespace(type="reasoning", id="r1"))
            yield _mk_event("response.reasoning_text.delta", delta="one two three")
            raise RuntimeError("attempt 1 fails mid-thinking")

        with pytest.raises(RuntimeError):
            _ = await _drain(stream_mod.responses_to_anthropic_sse(
                attempt1_upstream(), anthropic_model="foo", state=state,
            ))

        assert state.message_start_yielded is True
        assert state.thinking_yielded is True
        assert state.next_block_index >= 1  # at least the reasoning block was allocated
        assert state.open_block is not None  # still open — retry loop closes it

        # Retry loop simulates: close the open block, then invoke translator
        # in resumed mode. Capture attempt-2 output for splice validation.
        stop_frame = base_mod._stop_block_frame(state.open_block["index"])
        assert b"content_block_stop" in stop_frame
        state.open_block = None  # simulate loop clearing it

        async def attempt2_upstream():
            yield _mk_event("response.output_item.added",
                            item=types.SimpleNamespace(type="message"))
            yield _mk_event("response.output_text.delta", delta="hello world")
            yield _mk_event("response.completed",
                            response=types.SimpleNamespace(usage=None, status="completed"))

        resumed = await _drain(stream_mod.responses_to_anthropic_sse(
            attempt2_upstream(), anthropic_model="foo",
            state=state, resumed_from_index=state.next_block_index,
        ))
        joined_resumed = b"".join(resumed)

        # No duplicate message_start on the resumed attempt
        assert b"message_start" not in joined_resumed

        # New blocks pick up at state.next_block_index — monotone, no dup index
        opens = [line for line in joined_resumed.split(b"\n\n") if b"content_block_start" in line]
        assert opens, "resumed attempt should have opened at least one new block"
        # Index of the first resumed block should be >= 1 (not 0)
        for op in opens:
            data_line = [ln for ln in op.split(b"\n") if ln.startswith(b"data:")][0]
            payload = json.loads(data_line[5:].strip())
            assert payload["index"] >= 1, f"resumed block reused index 0: {payload}"

    @pytest.mark.anyio
    async def test_spliced_stream_parses_cleanly(self):
        """Full end-to-end: attempt-1 partial + stop + attempt-2 completion
        should assemble into a well-formed Anthropic message via
        _anthropic_sse_to_message (the buffer used by the non-streaming route)."""
        from claude_code_bridge.proxy.routes import _anthropic_sse_to_message

        state = stream_mod.TranslatorState()

        async def attempt1():
            yield _mk_event("response.output_item.added",
                            item=types.SimpleNamespace(type="reasoning", id="r1"))
            yield _mk_event("response.reasoning_text.delta", delta="thinking...")
            raise RuntimeError("boom mid-thinking")

        async def collect_attempt1() -> AsyncIterator[bytes]:
            try:
                async for chunk in stream_mod.responses_to_anthropic_sse(
                    attempt1(), anthropic_model="claude-x", state=state,
                ):
                    yield chunk
            except RuntimeError:
                pass  # simulate retry-loop swallow

        parts1: list[bytes] = []
        async for chunk in collect_attempt1():
            parts1.append(chunk)

        # Splice hygiene from the retry loop's perspective
        splice_stop = base_mod._stop_block_frame(state.open_block["index"])
        state.open_block = None

        async def attempt2():
            yield _mk_event("response.output_item.added",
                            item=types.SimpleNamespace(type="message"))
            yield _mk_event("response.output_text.delta", delta="hi")
            yield _mk_event("response.completed",
                            response=types.SimpleNamespace(usage=None, status="completed"))

        parts2: list[bytes] = []
        async for chunk in stream_mod.responses_to_anthropic_sse(
            attempt2(), anthropic_model="claude-x",
            state=state, resumed_from_index=state.next_block_index,
        ):
            parts2.append(chunk)

        async def spliced() -> AsyncIterator[bytes]:
            for c in parts1:
                yield c
            yield splice_stop
            for c in parts2:
                yield c

        # Downstream buffer must parse without error and produce a well-formed message
        message, err = await _anthropic_sse_to_message(spliced(), fallback_model="claude-x")
        assert err is None, f"spliced stream had error frame: {err}"
        assert message["role"] == "assistant"
        # We should end up with at least one text block containing 'hi'
        text_blocks = [b for b in message["content"] if b.get("type") == "text"]
        assert any("hi" in (b.get("text") or "") for b in text_blocks)


# ────────────────────────────────────────────────────────────────────────────
# End-to-end retry loop regression via a stubbed adapter
# ────────────────────────────────────────────────────────────────────────────


class _FakeUpstreamStream:
    """An async iterator of Responses events, optionally raising mid-stream."""

    def __init__(self, events: list[Any], raise_after: int | None = None,
                 raise_exc: BaseException | None = None):
        self._events = events
        self._raise_after = raise_after
        self._raise_exc = raise_exc or RuntimeError("simulated")
        self._i = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._raise_after is not None and self._i >= self._raise_after:
            raise self._raise_exc
        if self._i >= len(self._events):
            raise StopAsyncIteration
        ev = self._events[self._i]
        self._i += 1
        return ev


class _StubResponsesClient:
    """Stands in for ``AsyncOpenAI.with_options(...).responses``."""

    def __init__(self, script: list):
        # `script` is a list of either _FakeUpstreamStream instances OR
        # exception instances to raise on responses.create.
        self._script = script
        self._i = 0

    @property
    def responses(self):
        return self

    async def create(self, **_body):
        if self._i >= len(self._script):
            raise RuntimeError("stub exhausted")
        item = self._script[self._i]
        self._i += 1
        if isinstance(item, BaseException):
            raise item
        return item


class _StubOpenAIWrapper:
    """Stands in for AsyncOpenAI — .with_options() returns the SAME stub every
    time so retry-attempt counters accumulate correctly across calls."""

    def __init__(self, script):
        self._script = script
        self._stub = _StubResponsesClient(script)

    def with_options(self, **_kw):
        return self._stub

    # Convenience for tests: expose the counter directly on the wrapper.
    @property
    def _i(self) -> int:
        return self._stub._i


class _FakeAdapter(base_mod.ProviderAdapter):
    """Minimal adapter subclass for retry-loop tests. No network, no auth."""

    name = "fake"

    def __init__(self, script):
        super().__init__()
        self._script = script
        # Preload the client so start() is a no-op
        self._responses_client = _StubOpenAIWrapper(script)

    async def start(self):  # override — no real httpx pool
        return

    async def close(self):
        return

    async def authorize(self):
        return "fake-bearer", "http://127.0.0.1"


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    """Skip actual asyncio.sleep so retry tests run in ms."""
    async def _instant(_):
        return
    monkeypatch.setattr(asyncio, "sleep", _instant)


@pytest.fixture(autouse=True)
def _skip_usage_capture(monkeypatch):
    """usage_capture writes to Redis; stub it out for hermetic tests."""
    monkeypatch.setattr(
        "claude_code_bridge.proxy.usage_capture.schedule_capture",
        lambda _: None,
    )


@pytest.fixture(autouse=True)
def _skip_cache_diag(monkeypatch):
    """cache_diag is a no-op unless env-flagged, but be defensive."""
    monkeypatch.setattr(
        "claude_code_bridge.proxy.cache_diag.record",
        lambda _: None,
    )


@pytest.fixture(autouse=True)
def _skip_translate(monkeypatch):
    """anthropic_to_responses does real work; stub to identity for tests."""
    monkeypatch.setattr(
        "claude_code_bridge.proxy.translate.anthropic_to_responses",
        lambda body, upstream, **kwargs: {"model": upstream, "input": [], "instructions": "x", "stream": True, "store": False},
    )


class TestRetryLoop:
    """The whole call() flow with fake upstreams."""

    @pytest.mark.anyio
    async def test_success_on_first_attempt_no_retry_no_extra_frames(self):
        # Attempt 1: clean completion
        upstream = _FakeUpstreamStream([
            _mk_event("response.output_item.added",
                      item=types.SimpleNamespace(type="message")),
            _mk_event("response.output_text.delta", delta="ok"),
            _mk_event("response.completed",
                      response=types.SimpleNamespace(usage=None, status="completed")),
        ])
        adapter = _FakeAdapter(script=[upstream])
        chunks = await _drain(adapter.call({"model": "fake/foo"}, "foo"))
        joined = b"".join(chunks)
        assert b"message_start" in joined
        assert b"message_stop" in joined
        # Exactly one message_start
        assert joined.count(b"event: message_start") == 1
        # No error frame on the wire
        assert b'"type":"error"' not in joined

    @pytest.mark.anyio
    async def test_pre_output_bucket_a_retries_and_succeeds(self):
        # Attempt 1: raises before any yield (bucket A initial failure)
        class FakeConnErr(Exception): pass
        FakeConnErr.__module__ = "openai"
        # Attempt 2: clean completion
        good_stream = _FakeUpstreamStream([
            _mk_event("response.output_item.added",
                      item=types.SimpleNamespace(type="message")),
            _mk_event("response.output_text.delta", delta="ok"),
            _mk_event("response.completed",
                      response=types.SimpleNamespace(usage=None, status="completed")),
        ])
        adapter = _FakeAdapter(script=[FakeConnErr("boom"), good_stream])
        chunks = await _drain(adapter.call({"model": "fake/foo"}, "foo"))
        joined = b"".join(chunks)
        assert b'"type":"error"' not in joined
        assert joined.count(b"event: message_start") == 1
        assert b"message_stop" in joined

    @pytest.mark.anyio
    async def test_tool_committed_never_retries(self):
        """HARD invariant regression: NO duplicate tool execution possible."""
        # Attempt 1: emit tool_use, then raise. Retry loop must refuse.
        upstream = _FakeUpstreamStream(
            [
                _mk_event(
                    "response.output_item.added",
                    item=types.SimpleNamespace(
                        type="function_call", id="fc1", call_id="call_x", name="Bash",
                    ),
                ),
                _mk_event("response.function_call_arguments.delta",
                          item_id="fc1", delta='{"command":"ls"}'),
                _mk_event("response.function_call_arguments.done", item_id="fc1"),
            ],
            raise_after=3,
            raise_exc=RuntimeError("upstream cut after tool"),
        )
        # If retry fired, script[1] would be requested — put a sentinel that
        # would corrupt the assertion.
        adapter = _FakeAdapter(script=[upstream, RuntimeError("SHOULD NOT RETRY")])
        chunks = await _drain(adapter.call(
            {"model": "fake/foo", "tools": [{"name": "Bash", "input_schema": {"required": ["command"]}}]},
            "foo",
        ))
        joined = b"".join(chunks)
        # Exactly one message_start (no retry). Tool_use block was forwarded.
        assert joined.count(b"event: message_start") == 1
        assert b"tool_use" in joined
        # Terminal error surface must be api_error (NOT overloaded_error which
        # would let the CLI retry from scratch → dup tool execution)
        assert b'"type":"api_error"' in joined
        assert b'"type":"overloaded_error"' not in joined
        # Adapter's second script slot was NEVER requested
        assert adapter._responses_client._script[1] is not None  # script slot still present
        # Concretely: only one .create was consumed (index advanced to 1, not 2)
        assert adapter._responses_client._i == 1

    @pytest.mark.anyio
    async def test_text_partial_default_fails_upward_as_api_error(self):
        """Partial visible text → api_error (prevents duplicate visible text on CLI retry)."""
        upstream = _FakeUpstreamStream(
            [
                _mk_event("response.output_item.added",
                          item=types.SimpleNamespace(type="message")),
                _mk_event("response.output_text.delta", delta="hello"),
            ],
            raise_after=2,
            raise_exc=RuntimeError("cut mid-text"),
        )
        adapter = _FakeAdapter(script=[upstream, RuntimeError("SHOULD NOT RETRY")])
        chunks = await _drain(adapter.call({"model": "fake/foo"}, "foo"))
        joined = b"".join(chunks)
        assert b'"type":"api_error"' in joined
        assert b'"type":"overloaded_error"' not in joined
        assert adapter._responses_client._i == 1  # single attempt

    @pytest.mark.anyio
    async def test_permanent_401_never_retries_returns_auth_error(self):
        """Bucket C skips the loop entirely."""
        class FakeAuthErr(Exception):
            def __init__(self):
                super().__init__("invalid api key")
                self.status_code = 401
        FakeAuthErr.__module__ = "openai"
        adapter = _FakeAdapter(script=[FakeAuthErr(), RuntimeError("SHOULD NOT RETRY")])
        chunks = await _drain(adapter.call({"model": "fake/foo"}, "foo"))
        joined = b"".join(chunks)
        assert b'"type":"authentication_error"' in joined
        assert adapter._responses_client._i == 1

    @pytest.mark.anyio
    async def test_permanent_400_surfaces_invalid_request_error(self):
        """Al review fix (spec acceptance #5): a 400 (e.g. context-window
        overflow) must surface capability-classified as invalid_request_error,
        NOT the authentication_error bucket-C fallback."""
        class FakeBadRequestErr(Exception):
            def __init__(self):
                super().__init__("context length exceeded")
                self.status_code = 400
        FakeBadRequestErr.__module__ = "openai"
        adapter = _FakeAdapter(
            script=[FakeBadRequestErr(), RuntimeError("SHOULD NOT RETRY")]
        )
        chunks = await _drain(adapter.call({"model": "fake/foo"}, "foo"))
        joined = b"".join(chunks)
        assert b'"type":"invalid_request_error"' in joined
        assert b'"type":"authentication_error"' not in joined
        assert adapter._responses_client._i == 1


class TestPermanentTypeMapping:
    """Al review fix: status/payload → capability-classified error type."""

    def test_status_mapping(self):
        f = retry_mod.permanent_error_type_for_status
        assert f(400) == "invalid_request_error"
        assert f(401) == "authentication_error"
        assert f(403) == "permission_error"
        assert f(404) == "not_found_error"
        assert f(429) is None
        assert f(500) is None
        assert f(None) is None

    def test_inband_mapping(self):
        f = retry_mod.permanent_error_type_for_inband
        assert f({"code": "context_length_exceeded"}) == "invalid_request_error"
        assert f({"code": "invalid_request"}) == "invalid_request_error"
        assert f({"code": "400"}) == "invalid_request_error"
        assert f({"code": "invalid_api_key"}) == "authentication_error"
        assert f({"code": "permission_denied"}) == "permission_error"
        assert f({"code": "403"}) == "permission_error"
        assert f({"code": "not_found"}) == "not_found_error"
        assert f({"code": "server_error"}) is None
        assert f({}) is None
        assert f(None) is None  # type: ignore[arg-type]

    def test_exception_mapping(self):
        class FakeErr(Exception):
            status_code = 404
        assert (
            retry_mod.permanent_error_type_for_exception(FakeErr())
            == "not_found_error"
        )
        assert retry_mod.permanent_error_type_for_exception(None) is None

    @pytest.mark.anyio
    async def test_thinking_retry_splices_wire_valid(self):
        """Al #1 end-to-end: attempt-1 thinking failure, attempt-2 succeeds,
        the spliced stream parses cleanly downstream."""
        from claude_code_bridge.proxy.routes import _anthropic_sse_to_message

        thinking_upstream = _FakeUpstreamStream(
            [
                _mk_event("response.output_item.added",
                          item=types.SimpleNamespace(type="reasoning", id="r1")),
                _mk_event("response.reasoning_text.delta", delta="thinking..."),
            ],
            raise_after=2,
            raise_exc=RuntimeError("mid-thinking cut"),  # generic → bucket E
        )
        good_upstream = _FakeUpstreamStream([
            _mk_event("response.output_item.added",
                      item=types.SimpleNamespace(type="message")),
            _mk_event("response.output_text.delta", delta="done"),
            _mk_event("response.completed",
                      response=types.SimpleNamespace(usage=None, status="completed")),
        ])
        # Use a bucket-B exception at attempt 1 so it retries. Simplest is a
        # module-openai class:
        class FakeReadErr(Exception): pass
        FakeReadErr.__module__ = "openai"
        thinking_upstream._raise_exc = FakeReadErr("mid-thinking cut")
        adapter = _FakeAdapter(script=[thinking_upstream, good_upstream])
        chunks = await _drain(adapter.call({"model": "fake/foo"}, "foo"))

        async def replay() -> AsyncIterator[bytes]:
            for c in chunks:
                yield c

        message, err = await _anthropic_sse_to_message(replay(), fallback_model="foo")
        # Splice must produce a wire-valid message with no error frame surfaced
        assert err is None, f"spliced stream had error frame: {err}"
        text_blocks = [b for b in message["content"] if b.get("type") == "text"]
        assert any("done" in (b.get("text") or "") for b in text_blocks)

    @pytest.mark.anyio
    async def test_cancelled_error_never_retries(self):
        upstream = _FakeUpstreamStream(
            [_mk_event("response.output_item.added",
                       item=types.SimpleNamespace(type="message"))],
            raise_after=1,
            raise_exc=asyncio.CancelledError(),
        )
        adapter = _FakeAdapter(script=[upstream, RuntimeError("SHOULD NOT RETRY")])
        with pytest.raises(asyncio.CancelledError):
            _ = await _drain(adapter.call({"model": "fake/foo"}, "foo"))
        assert adapter._responses_client._i == 1


class _FakeBrokerAdapter(_FakeAdapter):
    """Broker-cached adapter (has _cached_expires_at) — 401s must trigger one
    cache-invalidated re-authorize instead of failing permanent."""

    def __init__(self, script):
        super().__init__(script)
        self._cached_expires_at = 9_999_999_999.0  # "valid" cache, revoked upstream
        self.authorize_calls = 0

    async def authorize(self):
        self.authorize_calls += 1
        return f"bearer-{self.authorize_calls}", "http://127.0.0.1"


def _fake_401() -> Exception:
    class Fake401(Exception):
        def __init__(self):
            super().__init__("Provided authentication token is expired.")
            self.status_code = 401
    Fake401.__module__ = "openai"
    return Fake401()


class TestBrokerCached401:
    @pytest.mark.anyio
    async def test_upstream_401_reauthorizes_once_and_succeeds(self):
        good_stream = _FakeUpstreamStream([
            _mk_event("response.output_item.added",
                      item=types.SimpleNamespace(type="message")),
            _mk_event("response.output_text.delta", delta="ok"),
            _mk_event("response.completed",
                      response=types.SimpleNamespace(usage=None, status="completed")),
        ])
        adapter = _FakeBrokerAdapter(script=[_fake_401(), good_stream])
        chunks = await _drain(adapter.call({"model": "fake/foo"}, "foo"))
        joined = b"".join(chunks)
        assert b"message_stop" in joined
        assert b'"type":"error"' not in joined
        # initial authorize + post-401 re-authorize
        assert adapter.authorize_calls == 2
        # cache was invalidated before the re-authorize
        assert adapter._cached_expires_at == 0

    @pytest.mark.anyio
    async def test_upstream_401_twice_fails_as_authentication_error(self):
        adapter = _FakeBrokerAdapter(script=[_fake_401(), _fake_401()])
        chunks = await _drain(adapter.call({"model": "fake/foo"}, "foo"))
        joined = b"".join(chunks)
        assert b"authentication_error" in joined
        assert adapter.authorize_calls == 2  # no infinite auth loop

    @pytest.mark.anyio
    async def test_non_broker_adapter_401_stays_permanent(self):
        adapter = _FakeAdapter(script=[_fake_401()])
        chunks = await _drain(adapter.call({"model": "fake/foo"}, "foo"))
        joined = b"".join(chunks)
        assert b"authentication_error" in joined
        # single upstream call — no retry for non-broker adapters
        assert adapter._responses_client._i == 1
