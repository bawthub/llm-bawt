"""ProviderAdapter ABC.

Adapters are stateful (cached credentials, refresh timers) so the registry
stores live instances, not classes. The default ``call`` implementation
targets the OpenAI Responses API and is shared by every Responses-API
provider — adapters only override when their upstream shape differs.

TASK-714: ``call()`` now wraps the initial-request + stream iteration in a
bounded retry loop consulting :mod:`proxy.retry`. The state machine + backoff
+ failure-bucket classification live there; this module only ORCHESTRATES the
retry (close open block on splice, invoke translator in resumed mode with a
monotone starting index, honor rate-limit cap, gate final-attempt usage
accounting).

Request identity across retries is load-bearing for prompt-cache routing —
the ``responses_body`` (including ``prompt_cache_key``), the ``session_id``
header, and the ``chatgpt-account-id`` header are built ONCE outside the loop.
Pre-output retries reuse that body verbatim. Opted-in text recovery builds from
that immutable original plus the ordered committed output; routing stays stable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, ClassVar

import httpx

from ..errors import final_error_frame as _final_error_frame
from ..request_context import ProxyRequestContext
from .. import retry as retry_mod
from ..stream import TranslatorState
from ..repetition_breaker import RepetitionLoopError
from ..tool_guard import UnsafeToolCallError

logger = logging.getLogger(__name__)

def _held_tool_fields(block: dict | None) -> tuple[str, str, str]:
    """TASK-1966 stall attribution: open tool name, held-argument size, and
    whether the held arguments already parse as JSON. Never the content."""
    if not isinstance(block, dict) or block.get("kind") != "tool":
        return "-", "-", "-"
    held = block.get("held_args") or ""
    try:
        json.loads(held)
        parses = "yes"
    except (TypeError, ValueError):
        parses = "no"
    name = block.get("name") or "unknown"
    return str(name)[:64], str(len(held)), parses


def _stop_block_frame(index: int) -> bytes:
    """Emit a well-formed content_block_stop frame for the splice (review #1).

    Kept in adapters/base.py rather than exported from stream.py to avoid
    coupling the retry loop to the translator's internals — the shape is
    trivial and stable per the Anthropic Messages API spec.
    """
    payload = {"type": "content_block_stop", "index": index}
    return f"event: content_block_stop\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def _is_unflushed_tool(block: dict) -> bool:
    """A tool block the SDK has seen start but never received arguments for."""
    return block.get("kind") == "tool" and not block.get("args_flushed")


def _retry_status(
    context: ProxyRequestContext,
    exc: BaseException | None,
    *,
    bucket: retry_mod.FailureBucket,
    phase: retry_mod.RetryPhase,
    attempt: int,
    policy: retry_mod.RetryPolicy,
) -> dict:
    """The single live status for "this attempt failed; trying again".

    TASK-1079: wording must match what is actually happening — only a
    WebSocket attempt falls back to HTTPS; an HTTPS stall simply retries.
    """
    transport = getattr(exc, "transport", None) or context.responses_transport or "sse"
    fallback = getattr(exc, "fallback_transport", None)
    max_attempts = policy.max_attempts
    if bucket is retry_mod.FailureBucket.G_PROGRESS_STALL:
        max_attempts = min(max_attempts, retry_mod.MAX_PROGRESS_STALL_ATTEMPTS)
    counter = f"attempt {attempt + 1}/{max_attempts}"
    state = "retrying"
    if phase is retry_mod.RetryPhase.TEXT:
        message = f"Upstream interrupted; resuming reply ({counter})"
    elif transport == "websocket" and fallback == "sse":
        state = "reconnecting"
        message = f"Upstream stalled; reconnecting over HTTPS ({counter})"
    elif bucket is retry_mod.FailureBucket.G_PROGRESS_STALL:
        message = f"Model stalled; retrying ({counter})"
    else:
        message = f"Model request failed; retrying ({counter})"
    return {
        "state": state,
        "message": message,
        "provider": context.provider,
        "attempt": attempt,
        "next_attempt": attempt + 1,
        "max_attempts": max_attempts,
        "transport": transport,
        "fallback_transport": fallback,
        "stall_phase": getattr(exc, "phase", None),
        "productive_idle_seconds": getattr(exc, "productive_idle_seconds", None),
        "elapsed_seconds": getattr(exc, "elapsed_seconds", None),
    }


@dataclass(frozen=True)
class PreparedCall:
    """Request-local credentials/routing, never reread from mutable adapter cache."""

    bearer: str = field(repr=False)
    base_url: str
    body: dict = field(repr=False)
    headers: tuple[tuple[str, str], ...] = field(repr=False)
    reasoning_codec: Any = field(repr=False)
    identity: tuple[str, ...] | None

    @property
    def credential_scope(self) -> str:
        return retry_mod.credential_scope(self.bearer, self.base_url)


class ProviderAdapter(ABC):
    """Abstract base for upstream-provider adapters.

    Subclasses MUST set ``name`` and implement ``authorize``.

    The default ``call`` translates Anthropic Messages → Responses API →
    Anthropic SSE using the shared translate/stream helpers. Override
    ``call`` only when an upstream doesn't speak Responses API (e.g. a
    legacy Chat Completions–only provider).
    """

    name: ClassVar[str]
    capture_codex_usage: ClassVar[bool] = True

    def __init__(self) -> None:
        self._lifecycle_lock = asyncio.Lock()
        self._http_client: httpx.AsyncClient | None = None
        self._responses_client = None

    async def start(self) -> None:
        """Create the adapter-owned connection pool once per proxy lifecycle."""
        if self._http_client is not None:
            return
        async with self._lifecycle_lock:
            if self._http_client is not None:
                return
            from openai import AsyncOpenAI

            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=15.0, read=600.0, write=60.0, pool=15.0),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            )
            # Request-local ``with_options`` wrappers below share this exact
            # underlying pool while carrying fresh credentials and headers.
            self._responses_client = AsyncOpenAI(
                api_key="lifecycle-managed",
                base_url="http://127.0.0.1",
                http_client=self._http_client,
            )

    async def close(self) -> None:
        """Close the lifecycle-owned pool exactly once."""
        async with self._lifecycle_lock:
            client = self._responses_client
            self._responses_client = None
            self._http_client = None
            if client is not None:
                await client.close()

    async def http_client(self) -> httpx.AsyncClient:
        """Return the reusable raw HTTP client for non-Responses adapters."""
        await self.start()
        assert self._http_client is not None
        return self._http_client

    def account_hash(self) -> str:
        """Non-secret account scope for concurrency telemetry."""
        return "default"

    @abstractmethod
    async def authorize(self) -> tuple[str, str]:
        """Return ``(bearer_token, base_url)``.

        Refresh credentials in-place when needed. ``base_url`` is the API
        root (e.g. ``https://api.openai.com/v1``), passed straight to the
        ``openai`` client.
        """

    def extra_headers(
        self,
        responses_body: dict,
        context: ProxyRequestContext | None = None,
    ) -> dict[str, str]:
        """Per-request headers merged into the upstream HTTP client.

        Default: none. Override for providers that need extra auth/routing
        headers (e.g. the ChatGPT backend's ``chatgpt-account-id``).
        """
        return {}

    def prepare_request(
        self,
        responses_body: dict,
        context: ProxyRequestContext | None = None,
    ) -> dict:
        """Last-chance hook to adapt the translated Responses body to an
        upstream's quirks (strip unsupported params, force defaults, etc.).

        Default: identity. Override only when an upstream rejects standard
        Responses fields. Mutates and returns the same dict for convenience.
        """
        return responses_body

    def reasoning_codec(self, upstream_model: str, base_url: str):
        """Opt-in native reasoning replay after credentials have been resolved."""
        return None

    async def open_stream(self, *, client, body, headers, bearer, base_url, context):
        """Ordinary HTTP, supervised independently of the selected model."""
        from ..responses_supervisor import ResponsesSSEStream

        stream = ResponsesSSEStream(
            lambda: client.responses.create(**body), context=context,
        )
        await stream.prepare()
        return stream

    def retry_policy(
        self,
        upstream_model: str,
        context: ProxyRequestContext | None = None,
    ) -> retry_mod.RetryPolicy:
        """Return the per-call retry budget; provider adapters may narrow it."""
        return retry_mod.RetryPolicy()

    def supports_text_continuation(self, context: ProxyRequestContext | None) -> bool:
        """Opt in only where native checkpoint replay has been verified."""
        return False

    def retry_auth_identity(self, base_url: str) -> tuple[str, ...] | None:
        """Opt in only for a known broker account, never an anonymous auth failure."""
        return None

    async def prepare_call(self, anthropic_body, upstream_model, context) -> PreparedCall:
        from .. import translate

        bearer, base_url = await self.authorize()
        # Capture all cache-derived values before another await can rotate them.
        codec = self.reasoning_codec(upstream_model, base_url)
        identity = self.retry_auth_identity(base_url)
        body = translate.anthropic_to_responses(
            anthropic_body, upstream_model, reasoning_codec=codec,
            allowed_skill_names=context.skill_names if context else (),
            strict_tools=bool(context and context.strict_tools),
            apply_patch=context.apply_patch if context else None,
        )
        if context is not None and context.apply_patch:
            from ..apply_patch import find_cwd
            context.patch_cwd = find_cwd(anthropic_body)
        body = self.prepare_request(body, context)
        headers = tuple((self.extra_headers(body, context) or {}).items())
        prepared = PreparedCall(bearer, base_url, body, headers, codec, identity)
        await self.start()
        return prepared

    async def call(
        self,
        anthropic_body: dict,
        upstream_model: str,
        context: ProxyRequestContext | None = None,
    ) -> AsyncIterator[bytes]:
        """Default Responses API call. Yields Anthropic-shaped SSE bytes.

        TASK-714: wraps the initial request + stream iteration in a bounded
        retry loop. State machine + budget in :mod:`proxy.retry`; loop-level
        responsibilities documented in the module docstring.
        """
        # Local imports keep the module import-graph trivial (the proxy
        # subpackage imports adapters, which imports openai, which is heavy).
        from .. import stream as stream_mod
        from ..continuation import validated_replay

        call_started = time.perf_counter()
        # authorize() runs once outside the loop — subsequent attempts reuse the
        # bearer unless bucket F (auth-broker) fires, in which case we invalidate
        # the adapter's cached token and re-authorize. Response body is also
        # built ONCE so prompt_cache_key / session_id / account-id stay stable
        # across retries (load-bearing for cache routing).
        lease = context.retry_lease if context is not None else None
        prepared = (lease.prepared_call if lease is not None else None)
        if prepared is None:
            prepared = await self.prepare_call(anthropic_body, upstream_model, context)
        assert self._responses_client is not None
        bearer, base_url = prepared.bearer, prepared.base_url
        reasoning_codec = prepared.reasoning_codec
        responses_body = prepared.body
        original_body = responses_body
        continuation_prefix = None
        headers = dict(prepared.headers) or None
        if lease is not None:
            lease.record_authorization(prepared.identity, prepared.credential_scope)
        client = self._responses_client.with_options(
            api_key=bearer,
            base_url=base_url,
            set_default_headers=headers,
            max_retries=0,
        )
        if context is not None:
            context.local_setup_ms = (
                time.perf_counter() - call_started
            ) * 1000
        from .. import cache_diag
        cache_diag.record(responses_body)
        logger.debug(
            "Proxy → Responses API model=%s tools=%d input_items=%d",
            responses_body.get("model"),
            len(responses_body.get("tools", []) or []),
            len(responses_body.get("input", []) or []),
        )

        state = TranslatorState()
        policy = lease.policy if lease is not None else self.retry_policy(upstream_model, context)
        route_attempt = context.attempt if lease is not None else 0
        # Per review #4: the on_usage callback wired into context.record_usage
        # fires INSIDE the translator whenever the upstream emits usage. On
        # retry we want the FINAL SUCCESSFUL attempt's usage to win. The
        # translator populates ``state.usage_snapshot`` on every attempt; we
        # gate the callback to fire only for the terminal successful attempt
        # by using a local wrapper that we swap out on retry.
        record_usage_fn = context.record_usage if context is not None else None
        anthropic_model_for_translator = anthropic_body.get("model", upstream_model)
        tool_schemas = anthropic_body.get("tools")

        while True:
            # Routes reserve the first attempt before entering any adapter;
            # direct adapter callers reserve it here. Subsequent attempts all
            # consume the SAME policy before another upstream sample.
            attempt = route_attempt or policy.claim_attempt()
            route_attempt = 0
            if attempt is None:
                if context is not None:
                    context.report_status({"state": "failed", "provider": context.provider})
                yield _final_error_frame("api_error", "Proxy retry budget exhausted")
                return
            attempt_started = time.perf_counter()
            if context is not None:
                context.attempt = attempt
                if attempt == 1 and lease is None:
                    context.report_status({
                        "state": "waiting", "provider": context.provider,
                    })
            # Only treat this as a resumed attempt if the PRIOR attempt actually
            # emitted the ``message_start`` envelope downstream. A retry after a
            # CONNECTING failure (initial request never returned; translator
            # never ran) is functionally a fresh start — the translator must
            # emit ``message_start`` normally on attempt 2, not skip it.
            resumed_from_index = (
                state.next_block_index
                if attempt > 1 and state.message_start_yielded
                else None
            )
            upstream_stream = None
            initial_exc: BaseException | None = None
            try:
                # ── initial request boundary ────────────────────────────────
                upstream_started = time.perf_counter()
                upstream_stream = await self.open_stream(
                    client=client, body=responses_body, headers=headers or {},
                    bearer=bearer, base_url=base_url, context=context,
                )
                if context is not None and attempt == 1:
                    # Only the FIRST attempt's TTFB is meaningful; later attempts
                    # measure retry latency, not upstream health.
                    context.upstream_ttfb_ms = (
                        time.perf_counter() - upstream_started
                    ) * 1000
                # Usage capture on final attempt only — retries would double-
                # count headers. Fire only when we know this attempt is the last
                # one; instead of predicting, we schedule on every attempt and
                # rely on state.usage_snapshot for the authoritative final value.
                # (The header-capture path writes to Redis and is idempotent.)
                if self.capture_codex_usage:
                    from .. import usage_capture as _usage_capture
                    _usage_capture.schedule_capture(upstream_stream)
            except (asyncio.CancelledError, GeneratorExit):
                # Client is gone (aclose from FastAPI's StreamingResponse, or a
                # task cancellation). Do not retry, do not emit an error frame
                # — the receiver isn't there to see it. Both propagate as
                # BaseException subclasses; re-raise BEFORE the generic catch.
                raise
            except BaseException as exc:  # noqa: BLE001
                initial_exc = exc

            if initial_exc is not None:
                bucket = retry_mod.classify_initial_exception(initial_exc)
                # An upstream 401 from a broker-cached adapter can mean OUR
                # cached bearer went stale (the app rotates the token pair out
                # from under long-lived proxy processes; the revoked token
                # reports "expired" upstream while its exp claim looks fine).
                # Route it through bucket F — one cache-invalidated re-authorize
                # against the broker — instead of failing permanent. Bucket F's
                # own attempt cap keeps a genuinely dead credential bounded.
                if (
                    bucket is retry_mod.FailureBucket.C_PERMANENT
                    and retry_mod._extract_status_code(initial_exc) == 401
                    and hasattr(self, "_cached_expires_at")
                ):
                    bucket = retry_mod.FailureBucket.F_AUTH_BROKER
                retry_after = retry_mod.extract_retry_after_seconds(initial_exc)
                phase = retry_mod.phase_from_state(state)
                decision = retry_mod.decide(
                    bucket=bucket, phase=phase, policy=policy,
                    allow_text_continuation=(self.supports_text_continuation(context)
                                             and state.committed_output.can_continue),
                    retry_after_s=retry_after,
                    permanent_error_type=retry_mod.permanent_error_type_for_exception(
                        initial_exc
                    ),
                )
                if lease is not None:
                    if lease.recovery_sample:
                        decision = retry_mod.RetryDecision(
                            False, reason="auth_recovery_sample_failed", final_error_type="api_error",
                        )
                    if (bucket is retry_mod.FailureBucket.F_AUTH_BROKER and not decision.retry
                            and (retry_mod._extract_status_code(initial_exc) == 401
                                 or isinstance(initial_exc, retry_mod.BrokerAuthorizationError))):
                        lease.mark_auth_failure()
                    elif bucket is not retry_mod.FailureBucket.F_AUTH_BROKER or not decision.retry:
                        lease.clear_auth_failure()
                policy.record_decision(bucket, decision)
                logger.warning(
                    "proxy_retry request_id=%s bot_id=%s boundary=open "
                    "transport=%s bucket=%s phase=%s stall_phase=%s "
                    "attempt=%d/%d decision=%s reason=%r "
                    "attempt_elapsed_ms=%.0f total_elapsed_ms=%.0f "
                    "backoff_ms=%.0f exc=%s",
                    context.request_id if context else "unknown",
                    context.bot_id if context else "unknown",
                    context.responses_transport if context else "unknown",
                    bucket.value, phase.value,
                    getattr(initial_exc, "phase", "none"),
                    attempt, policy.max_attempts,
                    "retry" if decision.retry else "final", decision.reason,
                    (time.perf_counter() - attempt_started) * 1000,
                    (time.perf_counter() - call_started) * 1000,
                    decision.backoff_s * 1000, type(initial_exc).__name__,
                )
                if decision.retry:
                    # Bucket F (auth-broker) → invalidate cached token so the
                    # next authorize() re-fetches. Adapter-agnostic: guard with
                    # getattr for adapters that don't have the attribute.
                    if bucket is retry_mod.FailureBucket.F_AUTH_BROKER:
                        if hasattr(self, "_cached_expires_at"):
                            self._cached_expires_at = 0  # type: ignore[attr-defined]
                        refreshed = await self.prepare_call(anthropic_body, upstream_model, context)
                        bearer, base_url = refreshed.bearer, refreshed.base_url
                        if (refreshed.reasoning_codec != reasoning_codec
                                or refreshed.identity != prepared.identity):
                            if lease is not None:
                                lease.clear_auth_failure()
                            if context is not None:
                                context.report_status({"state": "failed", "provider": context.provider})
                            yield _final_error_frame(
                                "authentication_error",
                                "Reasoning account scope changed during authorization; start a new request",
                            )
                            return
                        headers = dict(refreshed.headers) or None
                        if lease is not None:
                            lease.record_authorization(refreshed.identity, refreshed.credential_scope)
                        client = self._responses_client.with_options(
                            api_key=bearer, base_url=base_url,
                            set_default_headers=headers,
                            max_retries=0,
                        )
                    await asyncio.sleep(decision.backoff_s)
                    continue
                # Terminal — surface the final error type. On CONNECTING
                # failure state.message_start_yielded is still False, so the
                # client sees ONLY an error frame (no orphan message_start).
                if context is not None:
                    context.report_status({"state": "failed", "provider": context.provider})
                yield _final_error_frame(
                    decision.final_error_type or "api_error",
                    f"Proxy exhausted retries: {initial_exc}",
                )
                return

            # ── stream iteration boundary ───────────────────────────────────
            stream_exc: BaseException | None = None
            try:
                state.terminal_event_seen = False
                translated_stream = (
                    validated_replay(upstream_stream, continuation_prefix)
                    if continuation_prefix is not None else upstream_stream
                )
                async for chunk in stream_mod.responses_to_anthropic_sse(
                    translated_stream,
                    anthropic_model=anthropic_model_for_translator,
                    tool_schemas=tool_schemas,
                    on_usage=record_usage_fn,
                    state=state,
                    resumed_from_index=resumed_from_index,
                    reasoning_codec=reasoning_codec,
                    allowed_skill_names=context.skill_names if context else (),
                    request_context=context,
                ):
                    progress = getattr(upstream_stream, "progress", None)
                    if progress is not None:
                        progress.feed("exported", chunk)
                    yield chunk
            except (asyncio.CancelledError, GeneratorExit):
                raise
            except BaseException as exc:  # noqa: BLE001
                stream_exc = exc
            finally:
                close_stream = getattr(upstream_stream, "close", None)
                if close_stream is not None:
                    await close_stream()

            # Clean return path — either translator finished successfully OR
            # populated state.inband_error (review #2). Same decision path.
            if stream_exc is None and state.inband_error is None:
                return  # success

            if stream_exc is None:
                # In-band terminal — classify by payload code.
                bucket = retry_mod.classify_inband_error(state.inband_error or {})
                exc_repr = f"in-band {state.inband_error.get('code')}: {state.inband_error.get('message', '')[:120]}"
                exc_hint = None
                permanent_type = retry_mod.permanent_error_type_for_inband(
                    state.inband_error or {}
                )
            else:
                bucket = retry_mod.classify_stream_exception(stream_exc)
                exc_repr = f"{type(stream_exc).__name__}: {str(stream_exc)[:120]}"
                exc_hint = stream_exc
                permanent_type = retry_mod.permanent_error_type_for_exception(
                    stream_exc
                )

            retry_after = retry_mod.extract_retry_after_seconds(exc_hint)
            phase = retry_mod.phase_from_state(state)
            decision = retry_mod.decide(
                bucket=bucket, phase=phase, policy=policy,
                allow_text_continuation=(self.supports_text_continuation(context)
                                         and state.committed_output.can_continue),
                retry_after_s=retry_after,
                permanent_error_type=permanent_type,
            )
            if lease is not None:
                # A recovery sample never enters continuation/internal retries.
                if lease.recovery_sample:
                    decision = retry_mod.RetryDecision(
                        False, reason="auth_recovery_sample_failed", final_error_type="api_error",
                    )
                lease.clear_auth_failure()  # stream failures are not initial auth proof
            policy.record_decision(bucket, decision)
            logger.warning(
                "proxy_retry request_id=%s bot_id=%s boundary=stream "
                "transport=%s bucket=%s phase=%s stall_phase=%s "
                "attempt=%d/%d decision=%s reason=%r "
                "attempt_elapsed_ms=%.0f total_elapsed_ms=%.0f "
                "productive_idle_ms=%s backoff_ms=%.0f detail=%r "
                "tool=%s held_arg_chars=%s held_args_json=%s",
                context.request_id if context else "unknown",
                context.bot_id if context else "unknown",
                getattr(stream_exc, "transport", None)
                or (context.responses_transport if context else "unknown"),
                bucket.value, phase.value,
                getattr(stream_exc, "phase", "none"),
                attempt, policy.max_attempts,
                "retry" if decision.retry else "final", decision.reason,
                (time.perf_counter() - attempt_started) * 1000,
                (time.perf_counter() - call_started) * 1000,
                round(stream_exc.productive_idle_seconds * 1000)
                if stream_exc is not None
                and hasattr(stream_exc, "productive_idle_seconds") else "unknown",
                decision.backoff_s * 1000, exc_repr,
                *_held_tool_fields(state.open_block),
            )

            if decision.retry:
                if phase is retry_mod.RetryPhase.TEXT:
                    responses_body = state.committed_output.request(original_body)
                    continuation_prefix = state.committed_output.text
                if context is not None:
                    context.report_status(_retry_status(
                        context, stream_exc, bucket=bucket, phase=phase,
                        attempt=attempt, policy=policy,
                    ))
                # Review #1 block-index hygiene: close any block that was open at
                # failure time. The translator has already set state.open_block
                # to None if it emitted the close naturally; if we exited via
                # exception mid-block, state.open_block is still the last-opened
                # block and needs one content_block_stop before the resumed
                # translator emits new blocks starting at state.next_block_index.
                if state.open_block is not None:
                    yield _stop_block_frame(state.open_block["index"])
                    state.open_block = None
                # Clear the in-band error so the next attempt starts clean.
                state.inband_error = None
                await asyncio.sleep(decision.backoff_s)
                continue

            # The route's dispatch guard can replace one withheld, incomplete
            # ordinary tool with a non-executable error carrier. No resampling.
            if (context is not None and getattr(context, "recover_incomplete_tools", False)
                    and state.open_block is not None and state.open_block.get("kind") == "tool"
                    and (getattr(stream_exc, "phase", None) == "whitespace"
                         or isinstance(stream_exc, RepetitionLoopError))):
                from ..tool_recovery import ToolArgumentGenerationError
                name, chars, parses = _held_tool_fields(state.open_block)
                discard_stream = getattr(upstream_stream, "discard", None)
                if discard_stream is not None:
                    await discard_stream()
                cause = ("upstream whitespace stall" if getattr(stream_exc, "phase", None) == "whitespace"
                         else "upstream argument repetition")
                raise ToolArgumentGenerationError(
                    name, int(chars), f"{type(stream_exc).__name__}: {cause}",
                    incomplete=_is_unflushed_tool(state.open_block) and parses == "no",
                )

            # Terminal — emit the state-appropriate final error type. On
            # TEXT/TOOL_COMMITTED phases, the decision returns api_error (not
            # overloaded_error) so the CLI does NOT retry from scratch — this
            # is the invariant against duplicate visible text / duplicate tool
            # execution. On PRE_OUTPUT/THINKING, overloaded_error lets the CLI
            # take one more swing.
            #
            # TASK-1079: a tool block whose arguments were never flushed stays
            # OPEN. Closing it hands the CLI a complete ``{}`` call, which it
            # executes (verified against the installed CLI). Left open, the
            # CLI discards the unfinished block and makes its single
            # non-streaming retry of the same request — nothing executes twice.
            if state.open_block is not None and not _is_unflushed_tool(state.open_block):
                yield _stop_block_frame(state.open_block["index"])
            state.open_block = None
            discard_stream = getattr(upstream_stream, "discard", None)
            if discard_stream is not None:
                await discard_stream()
            final_error_type = decision.final_error_type or "api_error"
            if isinstance(stream_exc, UnsafeToolCallError):
                # TASK-1915: this path absorbs guard/breaker errors before the
                # route's handler, so emit the same counter line here.
                logger.warning(
                    "%s request_id=%s provider=%s model=%s reason=%s",
                    "proxy_repetition_break" if isinstance(stream_exc, RepetitionLoopError)
                    else "proxy_tool_rejected",
                    context.request_id if context else "unknown",
                    context.provider if context else "unknown",
                    original_body.get("model"), stream_exc,
                )
            if context is not None:
                context.report_status({"state": "failed", "provider": context.provider})
            yield _final_error_frame(
                final_error_type,
                f"Proxy stream failed: {exc_repr}",
            )
            return
