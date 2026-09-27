"""Bounded, side-effect-aware retry policy for the Anthropic-compat proxy.

TASK-714. Pure module — no I/O, no network. The retry loop in ``adapters/base.py``
consumes:

- :class:`RetryPolicy` — configuration + budget bookkeeping (max attempts,
  backoff schedule, cap).
- :class:`RetryDecision` — the yes/no/how of retrying a given failure at a given
  translator state, plus the Anthropic error type to surface at exhaustion.
- :func:`classify_initial_exception` / :func:`classify_stream_exception` — map
  exception classes to failure buckets.
- :func:`classify_inband_error` — map upstream in-band ``response.failed`` /
  ``response.error`` / ``error`` payloads to buckets (Al's addition #2 —
  those events currently short-circuit without raising).
- :func:`decide` — the state-machine cross product: (bucket × translator state
  × attempt count) → RetryDecision.
- :func:`compute_backoff` — bounded exponential backoff with jitter, hard cap.

State classification uses the observation-based :class:`TranslatorState` in
``proxy.stream``; this module intentionally imports it lazily (only in TYPE_CHECKING)
so the retry logic stays testable without pulling the whole translator.

Invariants (spec-locked, enforced here + regressed in tests):

- **Never retry after a tool call has been forwarded to the SDK** — TOOL_COMMITTED
  is a terminal-for-retry state regardless of bucket.
- **Never retry on ``asyncio.CancelledError``** — client is gone; nothing to serve.
- **Permanent 4xx / auth / context errors stay capability-classified** — bucket C
  never enters the retry loop even at attempt 1.
- **Rate-limit cap** (Al #3): if a ``retry-after`` hint exceeds
  ``RATE_LIMIT_MAX_WAIT_S`` we emit ``rate_limit_error`` immediately and let the
  CLI's own backoff take over — never hold the SDK's stream open on a long wait.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .stream import TranslatorState

logger = logging.getLogger(__name__)


# ── Buckets — failure classification the state machine consumes ─────────────

class FailureBucket(Enum):
    """Coarse failure class. See module docstring for wire behavior per bucket."""

    A_TRANSIENT_NETWORK = "A_transient_network"      # retryable, connect/read errors
    B_UPSTREAM_5XX = "B_upstream_5xx"                # retryable, upstream degraded
    C_PERMANENT = "C_permanent"                      # never retry, capability-classified
    D_RATE_LIMIT = "D_rate_limit"                    # retry-after honored with cap
    E_TRANSLATOR_BUG = "E_translator_bug"            # our bug — never retry
    F_AUTH_BROKER = "F_auth_broker"                  # one force-refresh retry allowed
    G_PROGRESS_STALL = "G_progress_stall"            # supervised Responses liveness failure


# ── Translator state — the retry decision needs to know what was yielded ────

class RetryPhase(Enum):
    """What we've emitted downstream so far. Drives retry legality.

    Kept parallel to :class:`stream.TranslatorState` observations so the retry
    module doesn't import the translator eagerly.
    """

    CONNECTING = "CONNECTING"        # upstream request not yet returned
    PRE_OUTPUT = "PRE_OUTPUT"        # message_start yielded, no content blocks yet
    THINKING = "THINKING"            # thinking content forwarded (safe to replay per Al #1 caveat)
    TEXT = "TEXT"                    # text_delta forwarded (partial visible output — fail up)
    TOOL_COMMITTED = "TOOL_COMMITTED"  # content_block_start(tool_use) — HARD no-retry


def phase_from_state(state: "TranslatorState | None") -> RetryPhase:
    """Derive the retry phase from a translator state snapshot.

    Precedence: TOOL_COMMITTED > TEXT > THINKING > PRE_OUTPUT > CONNECTING.
    """
    if state is None:
        return RetryPhase.CONNECTING
    if state.tool_committed:
        return RetryPhase.TOOL_COMMITTED
    # Native reasoning already persisted downstream cannot be retracted by a
    # splice. Use the same fail-upward barrier as visible text.
    if state.text_delta_yielded or getattr(state, "reasoning_committed", False):
        return RetryPhase.TEXT
    if state.thinking_yielded:
        return RetryPhase.THINKING
    if state.message_start_yielded:
        return RetryPhase.PRE_OUTPUT
    return RetryPhase.CONNECTING


# ── Decision — what the retry loop does with (bucket × phase × attempt) ─────

@dataclass(frozen=True)
class RetryDecision:
    """Outcome of :func:`decide`.

    ``retry=True`` — the loop should back off ``backoff_s`` seconds and re-attempt.
    ``retry=False`` — surface an error frame with ``final_error_type`` and stop.
    """

    retry: bool
    backoff_s: float = 0.0
    reason: str = ""                     # human-readable classification for logs
    final_error_type: str | None = None  # Anthropic error type when retry=False


# ── Budget + backoff schedule ──────────────────────────────────────────────

# Chosen conservatively — the outer Claude CLI already has its own retry loop,
# so proxy retries pay for themselves by preserving prompt-cache identity, not
# by unlimited stubbornness. See TASK-714 spec.
DEFAULT_MAX_ATTEMPTS = 3
# One supervised reconnect, then a terminal error (no outer CLI retry loop).
MAX_PROGRESS_STALL_ATTEMPTS = 2
DEFAULT_BASE_BACKOFF_S = 0.2
DEFAULT_BACKOFF_CAP_S = 5.0
DEFAULT_JITTER_S = 0.2

# Al #3: never hold the SDK's SSE stream open longer than this for a rate-limit
# retry-after. Beyond the cap we emit rate_limit_error immediately and hand off
# to the CLI's own backoff.
RATE_LIMIT_MAX_WAIT_S = 10.0


@dataclass
class RetryPolicy:
    """Retry configuration + per-call() budget bookkeeping.

    One instance per :meth:`ProviderAdapter.call` invocation. Not thread-safe;
    proxy handles one attempt at a time per call.
    """

    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    base_backoff_s: float = DEFAULT_BASE_BACKOFF_S
    backoff_cap_s: float = DEFAULT_BACKOFF_CAP_S
    jitter_s: float = DEFAULT_JITTER_S
    rate_limit_max_wait_s: float = RATE_LIMIT_MAX_WAIT_S
    _attempts_used: int = field(default=0, init=False)

    @property
    def attempt(self) -> int:
        """1-based number of the CURRENT attempt (0 before any start)."""
        return self._attempts_used

    def start_attempt(self) -> int:
        """Consume one attempt slot. Returns the new attempt number (1-based)."""
        self._attempts_used += 1
        return self._attempts_used

    @property
    def attempts_remaining(self) -> int:
        return max(0, self.max_attempts - self._attempts_used)


def compute_backoff(
    attempt: int,
    *,
    base: float = DEFAULT_BASE_BACKOFF_S,
    cap: float = DEFAULT_BACKOFF_CAP_S,
    jitter: float = DEFAULT_JITTER_S,
    rng: random.Random | None = None,
) -> float:
    """Exponential backoff with additive jitter, hard cap.

    Attempt is 1-based (first RETRY is attempt 2). Attempt 1 (initial) returns 0.
    Cap includes jitter — never sleeps longer than ``cap``.
    """
    if attempt <= 1:
        return 0.0
    r = rng if rng is not None else random
    exp = base * (2 ** (attempt - 2))
    jittered = exp + r.uniform(0.0, max(0.0, jitter))
    return min(jittered, cap)


# ── Bucket classification — exceptions from the initial request ────────────

def classify_initial_exception(exc: BaseException) -> FailureBucket:
    """Bucket a failure raised BEFORE any Anthropic bytes were yielded downstream.

    Called on exceptions escaping ``await client.responses.create(**body)``. The
    initial request boundary is the only place we distinguish "connect failed"
    from "upstream responded with 5xx"; once we're inside the SSE body it's all
    :func:`classify_stream_exception`.

    Never called on ``asyncio.CancelledError`` — the retry loop must check for
    that BEFORE consulting this classifier (cancellation isn't a bucket, it's a
    control-flow signal).
    """
    module = type(exc).__module__.split(".", 1)[0]
    status = _extract_status_code(exc)

    # E: our own translation/prep code raising a data error before we even
    # asked the upstream. Never retry — we're not going to translate better
    # by trying again.
    if module in ("json",) or isinstance(exc, (KeyError, ValueError)):
        # BUT: openai raises openai.APIError subclasses that inherit from
        # nothing special — check module first.
        if module not in ("openai", "httpx", "httpcore"):
            return FailureBucket.E_TRANSLATOR_BUG

    # F: auth broker RuntimeError from adapter.authorize(). Distinct from
    # upstream 401 because a fresh broker fetch may succeed.
    if isinstance(exc, RuntimeError) and "broker" in str(exc).lower():
        return FailureBucket.F_AUTH_BROKER

    # C: permanent 4xx from upstream. Auth/permission/not-found/bad-request
    # (including context-window overflow) cannot be fixed by retrying.
    if status in (400, 401, 403, 404):
        return FailureBucket.C_PERMANENT

    # D: rate limit. Distinct bucket so the caller can consult retry-after.
    if status == 429:
        return FailureBucket.D_RATE_LIMIT

    # B: retryable upstream 5xx.
    if status is not None and 500 <= status < 600:
        return FailureBucket.B_UPSTREAM_5XX

    # A: transient network. Everything openai/httpx/httpcore-shaped that
    # isn't a status error, plus OS-level timeouts.
    if module in ("openai", "httpx", "httpcore") or isinstance(
        exc, (OSError, ConnectionError, TimeoutError, asyncio.TimeoutError)
    ):
        return FailureBucket.A_TRANSIENT_NETWORK

    # Fallback: treat unknown exceptions as translator bugs (safer than
    # retry-looping on something we don't understand).
    return FailureBucket.E_TRANSLATOR_BUG


def classify_stream_exception(exc: BaseException) -> FailureBucket:
    """Bucket a failure raised DURING SSE iteration (mid-stream).

    Same class hierarchy as :func:`classify_initial_exception` but we're past
    the initial-request boundary — the caller has already checked the phase and
    decided whether phase permits any retry at all.
    """
    # Supervised liveness failures get one reconnect before commitment, including
    # unfinished thinking summaries. Their exhausted budget is terminal to the
    # CLI; ordinary transient failures retain their existing retry policy.
    if getattr(exc, "proxy_retry_owner", False):
        return FailureBucket.G_PROGRESS_STALL

    # Same table works — mid-stream we just don't get status codes as often,
    # but the openai/httpx/httpcore module test is what carries the weight.
    return classify_initial_exception(exc)


def classify_inband_error(payload: dict[str, Any]) -> FailureBucket:
    """Al addition #2: bucket an in-band ``response.failed``/``error`` payload.

    These currently short-circuit the translator without raising. The retry loop
    consults this via ``TranslatorState.inband_error`` after a "clean" return.

    Recognized shapes:

    - ``{"code": "server_error"|"internal_error"}`` → B_UPSTREAM_5XX
    - ``{"code": "insufficient_quota"|"rate_limit"|"429"}`` → D_RATE_LIMIT
    - ``{"code": "invalid_api_key"|"401"|"403"|"authentication"|"permission"}`` → C_PERMANENT
    - ``{"code": "context_length"|"invalid_request"|"400"|"404"}`` → C_PERMANENT
    - anything with a 5xx-shaped code (``5\\d\\d``) → B_UPSTREAM_5XX
    - anything with a 4xx-shaped code → C_PERMANENT
    - fallback → E_TRANSLATOR_BUG (unknown code, treat as our problem to look at)
    """
    if not isinstance(payload, dict):
        return FailureBucket.E_TRANSLATOR_BUG

    code = str(payload.get("code") or payload.get("type") or "").strip().lower()
    message = str(payload.get("message") or "").lower()

    if code in ("insufficient_quota", "rate_limit", "rate_limit_exceeded", "429"):
        return FailureBucket.D_RATE_LIMIT

    if code in (
        "invalid_api_key",
        "authentication",
        "authentication_error",
        "permission",
        "permission_denied",
        "401",
        "403",
    ):
        return FailureBucket.C_PERMANENT

    if code in (
        "context_length_exceeded",
        "context_length",
        "invalid_request",
        "invalid_request_error",
        "not_found",
        "404",
        "400",
    ):
        return FailureBucket.C_PERMANENT

    # Numeric-shaped status: allow "5xx" / "4xx" style detection.
    if code and code.isdigit() and len(code) == 3:
        n = int(code)
        if 500 <= n < 600:
            return FailureBucket.B_UPSTREAM_5XX
        if 400 <= n < 500:
            return FailureBucket.C_PERMANENT

    if any(kw in code for kw in ("server_error", "internal_error", "overloaded", "unavailable")):
        return FailureBucket.B_UPSTREAM_5XX
    if any(kw in message for kw in ("overloaded", "server error", "internal error", "unavailable")):
        return FailureBucket.B_UPSTREAM_5XX

    return FailureBucket.E_TRANSLATOR_BUG


# Spec acceptance #5: permanent errors stay CAPABILITY-CLASSIFIED. A context
# window overflow (400) must surface as invalid_request_error — NOT the
# authentication_error fallback — or the CLI and llm-bawt's overflow-recovery
# path misread the failure. These helpers preserve status specificity that the
# bucket alone throws away; callers pass the result to decide()'s
# ``permanent_error_type`` override.
_PERMANENT_TYPE_BY_STATUS: dict[int, str] = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
}


def permanent_error_type_for_status(status: int | None) -> str | None:
    """Anthropic error type for a permanent upstream HTTP status, or None."""
    if status is None:
        return None
    return _PERMANENT_TYPE_BY_STATUS.get(status)


def permanent_error_type_for_exception(exc: BaseException | None) -> str | None:
    """Status-specific permanent error type pulled from an exception, or None."""
    if exc is None:
        return None
    return permanent_error_type_for_status(_extract_status_code(exc))


def permanent_error_type_for_inband(payload: dict[str, Any]) -> str | None:
    """Status-specific permanent error type for an in-band error payload."""
    if not isinstance(payload, dict):
        return None
    code = str(payload.get("code") or payload.get("type") or "").strip().lower()
    if code in (
        "invalid_api_key",
        "authentication",
        "authentication_error",
        "401",
    ):
        return "authentication_error"
    if code in ("permission", "permission_denied", "403"):
        return "permission_error"
    if code in ("not_found", "404"):
        return "not_found_error"
    if code in (
        "context_length_exceeded",
        "context_length",
        "invalid_request",
        "invalid_request_error",
        "400",
    ):
        return "invalid_request_error"
    if code.isdigit() and len(code) == 3:
        return permanent_error_type_for_status(int(code))
    return None


def _extract_status_code(exc: BaseException) -> int | None:
    """Best-effort pull of an HTTP status code from openai/httpx exceptions."""
    for attr in ("status_code", "http_status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and 100 <= value < 600:
            return value
    response = getattr(exc, "response", None)
    if response is not None:
        value = getattr(response, "status_code", None)
        if isinstance(value, int) and 100 <= value < 600:
            return value
    return None


# ── The core decision function ──────────────────────────────────────────────

def decide(
    *,
    bucket: FailureBucket,
    phase: RetryPhase,
    policy: RetryPolicy,
    retry_after_s: float | None = None,
    permanent_error_type: str | None = None,
) -> RetryDecision:
    """Cross product of (failure bucket × translator phase × attempt count).

    Returns a :class:`RetryDecision`. When ``retry=True``, the caller must
    (a) close any open content_block before backoff, (b) sleep ``backoff_s``,
    (c) preserve request identity (prompt_cache_key / session_id / account-id
    across the retry), and (d) resume the translator with a monotone block
    index (Al #1).

    When ``retry=False``, ``final_error_type`` is the Anthropic error type to
    surface downstream:

    - ``overloaded_error`` at PRE_OUTPUT exhaustion → CLI may retry
    - ``api_error`` at TEXT/TOOL exhaustion (or post-partial failures) → CLI
      terminates the turn honestly (prevents duplicate visible text on CLI
      retry, prevents duplicate tool execution)
    - Bucket-C types (``authentication_error``, ``rate_limit_error``,
      ``invalid_request_error``) surface verbatim regardless of phase
    """
    # Hard invariant: TOOL_COMMITTED is a no-retry state, always. Even a bucket A
    # network hiccup here means the SDK has already been handed a tool call —
    # replaying could dup-execute.
    if phase is RetryPhase.TOOL_COMMITTED:
        return RetryDecision(
            retry=False,
            reason=f"tool_committed_no_retry bucket={bucket.value}",
            final_error_type="api_error",
        )

    # Partial visible text — spec permits prefix-suppressed replay behind an
    # env flag; default is honest fail-upward. Emit api_error so the CLI does
    # NOT retry from scratch (which would produce duplicate visible text).
    if phase is RetryPhase.TEXT:
        return RetryDecision(
            retry=False,
            reason=f"text_partial_no_retry bucket={bucket.value}",
            final_error_type="api_error",
        )

    # TASK-950: partial thinking summaries are not committed native reasoning.
    # The existing splice closes their open block before starting a fresh sample.
    # Completed replayable reasoning is already protected by the TEXT barrier
    # above (phase_from_state), as are visible text and dispatched tools.
    # Own the supervised budget here, not in a second adapter-level override.
    if bucket is FailureBucket.G_PROGRESS_STALL and (
        policy.attempt >= MAX_PROGRESS_STALL_ATTEMPTS or policy.attempts_remaining <= 0
    ):
        return RetryDecision(
            retry=False,
            reason="progress-stall retry budget exhausted",
            final_error_type="api_error",
        )

    # Bucket C — permanent. Never retry, classify honestly. The caller passes
    # ``permanent_error_type`` (from permanent_error_type_for_status /
    # _for_exception / _for_inband) to preserve status specificity — spec
    # acceptance #5 requires context overflow (400) to surface as
    # invalid_request_error, not the phase-based fallback.
    if bucket is FailureBucket.C_PERMANENT:
        return RetryDecision(
            retry=False,
            reason="permanent",
            final_error_type=permanent_error_type
            or _permanent_error_type_for_phase(phase),
        )

    # Bucket E — our bug. Never retry.
    if bucket is FailureBucket.E_TRANSLATOR_BUG:
        return RetryDecision(
            retry=False,
            reason="translator_bug",
            final_error_type="api_error",
        )

    # Bucket D — rate limit. Al #3: cap retry-after so we don't hold the SDK
    # stream open on a long wait.
    if bucket is FailureBucket.D_RATE_LIMIT:
        wait = retry_after_s if retry_after_s is not None else policy.base_backoff_s
        if wait > policy.rate_limit_max_wait_s:
            return RetryDecision(
                retry=False,
                reason=f"rate_limit_wait_over_cap retry_after={wait:.1f}s cap={policy.rate_limit_max_wait_s:.1f}s",
                final_error_type="rate_limit_error",
            )
        if policy.attempts_remaining <= 0:
            return RetryDecision(
                retry=False,
                reason="rate_limit_exhausted",
                final_error_type="rate_limit_error",
            )
        return RetryDecision(
            retry=True,
            backoff_s=wait,
            reason="rate_limit_backoff",
        )

    # Bucket F — auth broker. ONE force-refresh retry allowed.
    # The adapter must invoke authorize() with force=True next attempt; that
    # coordination lives in the retry loop, not here.
    if bucket is FailureBucket.F_AUTH_BROKER:
        if policy.attempt >= 2:
            return RetryDecision(
                retry=False,
                reason="auth_broker_force_refresh_exhausted",
                final_error_type="authentication_error",
            )
        return RetryDecision(
            retry=True,
            backoff_s=compute_backoff(policy.attempt + 1, base=policy.base_backoff_s,
                                       cap=policy.backoff_cap_s, jitter=policy.jitter_s),
            reason="auth_broker_force_refresh",
        )

    # Buckets A / B / G — transient / 5xx / supervised stall before commitment.
    # THINKING phase: replay this model request; the retry loop is responsible for the
    # block-index hygiene (Al #1) — closing the open thinking block before the
    # retry resumes, then re-indexing new blocks from state.next_block_index.
    if policy.attempts_remaining <= 0:
        return RetryDecision(
            retry=False,
            reason=f"exhausted bucket={bucket.value} phase={phase.value}",
            final_error_type=_exhausted_error_type_for_phase(phase),
        )
    return RetryDecision(
        retry=True,
        backoff_s=compute_backoff(policy.attempt + 1, base=policy.base_backoff_s,
                                   cap=policy.backoff_cap_s, jitter=policy.jitter_s),
        reason=f"retry bucket={bucket.value} phase={phase.value}",
    )


def _permanent_error_type_for_phase(phase: RetryPhase) -> str:
    """Bucket-C surface type — mostly ``authentication_error`` / ``invalid_request_error``.

    Callers may override with the specific classification when they have more
    info (e.g. 404 vs 401). Default is authentication_error since that's the
    most common permanent-mid-request failure we see in the openai_chatgpt path.
    """
    return "authentication_error"


def _exhausted_error_type_for_phase(phase: RetryPhase) -> str:
    """Post-exhaustion surface type.

    PRE_OUTPUT / CONNECTING → overloaded_error (CLI may still retry the whole
    /v1/messages call). THINKING → overloaded_error too — no visible content
    was committed downstream. TEXT / TOOL_COMMITTED handled earlier.
    """
    if phase in (RetryPhase.CONNECTING, RetryPhase.PRE_OUTPUT, RetryPhase.THINKING):
        return "overloaded_error"
    return "api_error"


def extract_retry_after_seconds(exc: BaseException | None) -> float | None:
    """Best-effort pull of a ``retry-after`` header from an openai/httpx exception.

    Returns seconds as a float, or None if no hint is present. Values are
    treated as absolute seconds (delta from now) — HTTP-date form is not parsed
    (rare in the ChatGPT-backend responses we see).
    """
    if exc is None:
        return None
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if headers is None:
        return None
    try:
        raw = headers.get("retry-after") or headers.get("Retry-After")
    except Exception:  # noqa: BLE001
        return None
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None
