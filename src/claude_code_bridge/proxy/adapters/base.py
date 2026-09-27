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
header, and the ``chatgpt-account-id`` header are built ONCE outside the loop
and reused verbatim on every attempt.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import AsyncIterator, ClassVar

import httpx

from ..request_context import ProxyRequestContext
from .. import retry as retry_mod
from ..stream import TranslatorState

logger = logging.getLogger(__name__)

def _stop_block_frame(index: int) -> bytes:
    """Emit a well-formed content_block_stop frame for the splice (Al #1).

    Kept in adapters/base.py rather than exported from stream.py to avoid
    coupling the retry loop to the translator's internals — the shape is
    trivial and stable per the Anthropic Messages API spec.
    """
    payload = {"type": "content_block_stop", "index": index}
    return f"event: content_block_stop\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def _final_error_frame(error_type: str, message: str) -> bytes:
    """Emit a well-formed Anthropic-shaped SSE error frame."""
    payload = {"type": "error", "error": {"type": error_type, "message": message}}
    return f"event: error\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


class ProviderAdapter(ABC):
    """Abstract base for upstream-provider adapters.

    Subclasses MUST set ``name`` and implement ``authorize``.

    The default ``call`` translates Anthropic Messages → Responses API →
    Anthropic SSE using the shared translate/stream helpers. Override
    ``call`` only when an upstream doesn't speak Responses API (e.g. a
    legacy Chat Completions–only provider).
    """

    name: ClassVar[str]

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
        from .. import translate

        call_started = time.perf_counter()
        # authorize() runs once outside the loop — subsequent attempts reuse the
        # bearer unless bucket F (auth-broker) fires, in which case we invalidate
        # the adapter's cached token and re-authorize. Response body is also
        # built ONCE so prompt_cache_key / session_id / account-id stay stable
        # across retries (load-bearing for cache routing).
        bearer, base_url = await self.authorize()
        await self.start()
        assert self._responses_client is not None
        reasoning_codec = self.reasoning_codec(upstream_model, base_url)
        responses_body = translate.anthropic_to_responses(
            anthropic_body, upstream_model, reasoning_codec=reasoning_codec,
        )
        responses_body = self.prepare_request(responses_body, context)
        headers = self.extra_headers(responses_body, context) or None
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
        policy = self.retry_policy(upstream_model, context)
        # Per Al #4: the on_usage callback wired into context.record_usage
        # fires INSIDE the translator whenever the upstream emits usage. On
        # retry we want the FINAL SUCCESSFUL attempt's usage to win. The
        # translator populates ``state.usage_snapshot`` on every attempt; we
        # gate the callback to fire only for the terminal successful attempt
        # by using a local wrapper that we swap out on retry.
        record_usage_fn = context.record_usage if context is not None else None
        anthropic_model_for_translator = anthropic_body.get("model", upstream_model)
        tool_schemas = anthropic_body.get("tools")

        while True:
            attempt = policy.start_attempt()
            if context is not None:
                context.attempt = attempt
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
                    retry_after_s=retry_after,
                    permanent_error_type=retry_mod.permanent_error_type_for_exception(
                        initial_exc
                    ),
                )
                logger.warning(
                    "proxy_retry attempt=%d/%d bucket=%s phase=%s decision=%s "
                    "reason=%r backoff_ms=%.0f exc=%s",
                    attempt, policy.max_attempts, bucket.value, phase.value,
                    "retry" if decision.retry else "final",
                    decision.reason, decision.backoff_s * 1000,
                    type(initial_exc).__name__,
                )
                if decision.retry:
                    # Bucket F (auth-broker) → invalidate cached token so the
                    # next authorize() re-fetches. Adapter-agnostic: guard with
                    # getattr for adapters that don't have the attribute.
                    if bucket is retry_mod.FailureBucket.F_AUTH_BROKER:
                        if hasattr(self, "_cached_expires_at"):
                            self._cached_expires_at = 0  # type: ignore[attr-defined]
                        bearer, base_url = await self.authorize()
                        if self.reasoning_codec(upstream_model, base_url) != reasoning_codec:
                            yield _final_error_frame(
                                "authentication_error",
                                "Reasoning account scope changed during authorization; start a new request",
                            )
                            return
                        headers = self.extra_headers(responses_body, context) or None
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
                yield _final_error_frame(
                    decision.final_error_type or "api_error",
                    f"Proxy exhausted retries: {initial_exc}",
                )
                return

            # ── stream iteration boundary ───────────────────────────────────
            stream_exc: BaseException | None = None
            try:
                state.terminal_event_seen = False
                async for chunk in stream_mod.responses_to_anthropic_sse(
                    upstream_stream,
                    anthropic_model=anthropic_model_for_translator,
                    tool_schemas=tool_schemas,
                    on_usage=record_usage_fn,
                    state=state,
                    resumed_from_index=resumed_from_index,
                    reasoning_codec=reasoning_codec,
                ):
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
            # populated state.inband_error (Al #2). Same decision path.
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
                retry_after_s=retry_after,
                permanent_error_type=permanent_type,
            )
            logger.warning(
                "proxy_retry attempt=%d/%d bucket=%s phase=%s decision=%s "
                "reason=%r backoff_ms=%.0f detail=%r",
                attempt, policy.max_attempts, bucket.value, phase.value,
                "retry" if decision.retry else "final",
                decision.reason, decision.backoff_s * 1000, exc_repr,
            )

            if decision.retry:
                # Al #1 block-index hygiene: close any block that was open at
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
                fallback_transport = getattr(
                    stream_exc, "fallback_transport", None
                )
                if context is not None and fallback_transport:
                    context.report_status({
                        "state": "reconnecting",
                        "message": "Upstream stalled; reconnecting over HTTPS…",
                        "provider": context.provider,
                        "attempt": attempt,
                        "next_attempt": attempt + 1,
                        "max_attempts": min(
                            policy.max_attempts, retry_mod.MAX_PROGRESS_STALL_ATTEMPTS,
                        ),
                        "transport": getattr(stream_exc, "transport", "unknown"),
                        "fallback_transport": fallback_transport,
                        "stall_phase": getattr(stream_exc, "phase", "unknown"),
                        "productive_idle_seconds": getattr(
                            stream_exc, "productive_idle_seconds", None
                        ),
                        "elapsed_seconds": getattr(
                            stream_exc, "elapsed_seconds", None
                        ),
                    })
                await asyncio.sleep(decision.backoff_s)
                continue

            # Terminal — emit the state-appropriate final error type. On
            # TEXT/TOOL_COMMITTED phases, the decision returns api_error (not
            # overloaded_error) so the CLI does NOT retry from scratch — this
            # is the invariant against duplicate visible text / duplicate tool
            # execution. On PRE_OUTPUT/THINKING, overloaded_error lets the CLI
            # take one more swing.
            if state.open_block is not None:
                yield _stop_block_frame(state.open_block["index"])
                state.open_block = None
            discard_stream = getattr(upstream_stream, "discard", None)
            if discard_stream is not None:
                await discard_stream()
            final_error_type = decision.final_error_type or "api_error"
            yield _final_error_frame(
                final_error_type,
                f"Proxy stream failed: {exc_repr}",
            )
            return
