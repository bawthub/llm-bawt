"""Responses API stream events → Anthropic Messages API SSE events.

This translator is **feature-complete** against the OpenAI Responses streaming
event set (`openai.types.responses.ResponseStreamEvent`, 53 event types as of
openai-python 2.x) and emits a fully spec-compliant Anthropic Messages SSE
stream so the Claude Agent SDK consumes a non-Anthropic provider exactly as if
it were talking to api.anthropic.com.

Anthropic event sequence we produce:

    message_start
      content_block_start (index, thinking)      ← reasoning summary / CoT
        content_block_delta (thinking_delta)*
        content_block_delta (signature_delta)     ← reasoning encrypted_content
      content_block_stop  (index)
      content_block_start (index, text)
        content_block_delta (text_delta)*
      content_block_stop  (index)
      content_block_start (index, tool_use)
        content_block_delta (input_json_delta)*
      content_block_stop  (index)
    message_delta (stop_reason, usage)
    message_stop

plus periodic ``ping`` frames (Anthropic emits these too) so a slow upstream
never reads as a dead connection.

Why reasoning matters: high-effort reasoning models (gpt-5.x via the codex
backend, GLM, …) stream a long burst of ``response.reasoning*`` events BEFORE
any visible text. The previous translator dropped them, so the whole reasoning
window was silent SSE — the Claude SDK and the UI saw a frozen turn until the
first text token. Surfacing reasoning as Anthropic ``thinking`` blocks (a) keeps
the stream alive and (b) shows the model's thinking, matching native Anthropic.

Responses events consumed (grouped):

    lifecycle:   response.created / .in_progress / .queued
                 response.completed / .incomplete / .failed / .error / error
    reasoning:   response.output_item.added(type=reasoning)
                 response.reasoning_text.delta / .done
                 response.reasoning_summary_text.delta / .done
                 response.reasoning_summary_part.added / .done
                 response.output_item.done(type=reasoning)   → encrypted_content
    text:        response.content_part.added / .done
                 response.output_text.delta / .done
                 response.output_text.annotation.added
                 response.refusal.delta / .done
    tools:       response.output_item.added(type=function_call|custom_tool_call)
                 response.function_call_arguments.delta / .done
                 response.custom_tool_call_input.delta / .done
                 response.output_item.done(type=function_call|custom_tool_call)
    server-side: web_search_call / code_interpreter_call / mcp_call /
                 image_generation_call / file_search_call (kept alive, see note)
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

from .tool_sanitizers import recover_trailing_json, sanitize_tool_arguments
from .reasoning import ReasoningCodec

logger = logging.getLogger(__name__)


# ── TASK-714: TranslatorState — live view of what's been yielded ────────────
#
# The retry loop in adapters/base.py consults this on failure to pick a policy:
#   - TOOL_COMMITTED (state.tool_committed=True) → hard no-retry
#   - TEXT (state.text_delta_yielded=True) → default fail-upward with api_error
#   - THINKING (state.thinking_yielded=True) → retry OK; block-index hygiene
#     across the splice is the loop's responsibility (Al #1)
#   - PRE_OUTPUT (state.message_start_yielded=True, no content) → retry OK
#   - CONNECTING (state.message_start_yielded=False) → retry OK
#
# When ``state`` is passed to :func:`responses_to_anthropic_sse` the translator
# ALSO changes error behavior: exceptions from the upstream iterator PROPAGATE
# to the caller (retry loop catches), and in-band ``response.failed`` /
# ``response.error`` / ``error`` payloads populate ``state.inband_error`` and
# short-circuit WITHOUT emitting an SSE error frame — the retry loop consults
# the payload and either retries (bucket B) or emits its own final error frame
# with a bucket-appropriate Anthropic error type.
#
# Backward compat: ``state=None`` preserves the pre-TASK-714 behavior
# (translator emits its own error frame and swallows the exception). Nothing
# outside this proxy calls this generator, so backcompat is only for tests /
# the non-streaming route buffer in routes.py that treats the generator as
# opaque.

class IncompleteResponseStreamError(ConnectionError):
    """Upstream stream ended without a Responses terminal event."""


@dataclass
class TranslatorState:
    """Observation of what the translator has yielded so far, for retry decisions."""

    # ── Yield markers (drive RetryPhase classification in retry.py) ─────────
    message_start_yielded: bool = False
    thinking_yielded: bool = False       # any thinking_delta emitted
    reasoning_committed: bool = False    # replayable state forwarded; no splice retry
    text_delta_yielded: bool = False     # ANY text_delta emitted (partial visible output)
    tool_committed: bool = False         # content_block_start(tool_use) emitted — hard no-retry
    terminal_event_seen: bool = False    # response.completed / response.incomplete

    # ── Block index bookkeeping (Al #1: monotone indexes across a retry splice) ─
    # ``next_block_index`` is the index that WILL be assigned to the next block
    # opened by the translator. When the retry loop closes the current open
    # block and re-invokes the translator with ``resumed_from_index=state.next_block_index``,
    # new blocks pick up from there — no restart at 0, no duplicate indexes.
    next_block_index: int = 0
    open_block: dict | None = None       # {"index", "kind", "item_id", "call_id"?, "name"?}

    # ── In-band terminal events (Al #2) ─────────────────────────────────────
    # Populated by response.failed / response.error / error SSE events from the
    # upstream. When present after a "clean" return from the translator, the
    # retry loop treats the run as failed and consults classify_inband_error.
    inband_error: dict | None = field(default=None)

    # ── Usage snapshot (Al #4) ──────────────────────────────────────────────
    # Only the final SUCCESSFUL attempt's usage should count toward turn
    # accounting. The retry loop resets this per attempt; on_usage on the
    # translator populates it. See adapters/base.py::call for the reset dance.
    usage_snapshot: tuple[int, int, int, int] | None = None    # (input, output, cache_read, cache_create)


def _sse(event: str, data: dict) -> bytes:
    """Format one SSE frame. Anthropic streams use both ``event:`` and
    ``data:`` lines so curious clients can switch on either."""
    payload = json.dumps(data, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n".encode()


def _ping() -> bytes:
    return _sse("ping", {"type": "ping"})


def _stream_error_type(exc: BaseException) -> str:
    """Map a mid-stream failure onto a canonical Anthropic error type.

    Only types in Anthropic's canonical set are recognized by the Claude CLI's
    retry classifier; anything else ends the turn outright. Transient upstream
    faults (the provider accepted the request, then injected an error into the
    already-committed SSE body — which ``AsyncOpenAI(max_retries=...)`` cannot
    cover, since retries only guard the initial request) are reported as
    ``overloaded_error`` so the CLI retries. A genuine bug in our own
    translation logic is reported as ``api_error``.
    """
    module = type(exc).__module__.split(".", 1)[0]
    if module in ("openai", "httpx", "httpcore") or isinstance(exc, (OSError, TimeoutError)):
        return "overloaded_error"
    return "api_error"


# Responses ``status`` / ``incomplete reason`` → Anthropic stop_reason.
_STOP_REASON_MAP = {
    "stop": "end_turn",
    "completed": "end_turn",
    "length": "max_tokens",
    "max_output_tokens": "max_tokens",
    "max_tokens": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
    "refusal": "refusal",
}


def _anthropic_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def _usage_get(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from an object or dict (Responses usage is either)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_usage(resp: Any) -> tuple[int, int, int, int]:
    """Pull (input, output, cache_read, cache_create) from a Response object.

    Responses API reports cached prompt tokens under
    ``usage.input_tokens_details.cached_tokens``; ``input_tokens`` is the FULL
    prompt (cached + uncached), unlike Anthropic where ``input_tokens`` is the
    uncached remainder. The caller normalises to Anthropic's split.

    Also accepts dict-shaped usage (some SDKs / xAI edge paths) and the
    chat-completions alias ``prompt_tokens_details.cached_tokens``.
    """
    usage = _usage_get(resp, "usage", None)
    if usage is None and isinstance(resp, dict):
        usage = resp.get("usage")
    if usage is None:
        return 0, 0, 0, 0
    input_tokens = int(
        _usage_get(usage, "input_tokens", None)
        or _usage_get(usage, "prompt_tokens", 0)
        or 0
    )
    output_tokens = int(
        _usage_get(usage, "output_tokens", None)
        or _usage_get(usage, "completion_tokens", 0)
        or 0
    )
    cache_read = 0
    details = (
        _usage_get(usage, "input_tokens_details", None)
        or _usage_get(usage, "prompt_tokens_details", None)
    )
    if details is not None:
        cache_read = int(_usage_get(details, "cached_tokens", 0) or 0)
    return input_tokens, output_tokens, cache_read, 0


async def responses_to_anthropic_sse(
    upstream_stream: AsyncIterator[Any],
    anthropic_model: str,
    tool_schemas: list[dict] | None = None,
    on_usage: Callable[[int, int, int, int], None] | None = None,
    *,
    state: TranslatorState | None = None,
    resumed_from_index: int | None = None,
    reasoning_codec: ReasoningCodec | None = None,
) -> AsyncIterator[bytes]:
    """Translate a Responses API event stream into Anthropic SSE bytes.

    TASK-714 additions (kwargs-only, backward compat when both omitted):

    - ``state`` — a :class:`TranslatorState` the translator updates as it yields.
      When provided, ALSO changes error behavior: exceptions from the upstream
      iterator PROPAGATE (retry loop catches), and in-band ``response.failed``/
      ``response.error``/``error`` payloads populate ``state.inband_error`` and
      short-circuit WITHOUT emitting an SSE error frame.
    - ``resumed_from_index`` — if not None, skip ``message_start`` (previous
      attempt already emitted it), start block indexes at this integer, and
      assume the caller has already emitted a ``content_block_stop`` for any
      block open at retry time. Al #1: keeps indexes monotone across the
      splice; the SDK's Anthropic-SSE parser is strict about this.
    """
    message_id = _anthropic_message_id()
    required_args_by_tool: dict[str, frozenset[str]] = {}
    for tool in tool_schemas or []:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        schema = tool.get("input_schema")
        if not isinstance(name, str) or not isinstance(schema, dict):
            continue
        required = schema.get("required")
        required_args_by_tool[name] = frozenset(
            value for value in required or [] if isinstance(value, str)
        )

    # TASK-714: skip message_start on resumed attempts — the previous attempt
    # already emitted it and the SDK's parser rejects a second one. When state
    # is provided we also mark message_start_yielded=True for the initial
    # emission so the retry loop's phase classifier sees PRE_OUTPUT immediately.
    if resumed_from_index is None:
        # Emit message_start up front so the SDK has the envelope before any
        # content arrives. Our Responses-backed providers only know real usage at
        # response.completed, so we start with zeroes and refine at message_delta.
        yield _sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": anthropic_model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                    },
                },
            },
        )
        if state is not None:
            state.message_start_yielded = True

    # Anthropic permits at most one *open* content block at a time and requires
    # strict open→delta→stop ordering. Responses emits output items
    # sequentially (reasoning, then message text, then function calls), so we
    # model a single ``open_block`` and a per-item index map for routing arg
    # deltas to the right tool block.
    # TASK-714 (Al #1): on a resumed attempt, block indexes start from where
    # the previous attempt left off — NOT at 0 — so the spliced stream stays
    # monotone. The caller is responsible for emitting content_block_stop on
    # any block that was open at the failure point BEFORE calling us again.
    next_index = resumed_from_index if resumed_from_index is not None else 0
    open_block: dict | None = None          # {"index", "kind", "item_id", "call_id"?}
    blocks_by_item: dict[str, dict] = {}    # item_id → block dict (tools)
    # Buffer tool argument deltas so we can sanitize before the SDK sees them.
    # GPT models sometimes fill optional params with empty strings (e.g.
    # pages: "") which the SDK tools reject. We buffer, strip empties, then
    # emit on the .done event.
    tool_arg_buffers: dict[str, str] = {}   # item_id → accumulated JSON string

    terminal_event_seen = False
    saw_tool_use = False
    saw_refusal = False
    explicit_stop: str | None = None
    input_tokens = output_tokens = cache_read = cache_create = 0

    def _open_start_frame(kind: str, idx: int, **extra: Any) -> bytes:
        if kind == "thinking":
            block = {"type": "thinking", "thinking": "", "signature": ""}
        elif kind == "tool":
            block = {
                "type": "tool_use",
                "id": extra.get("call_id", ""),
                "name": extra.get("name", ""),
                "input": {},
            }
        else:  # text
            block = {"type": "text", "text": ""}
        return _sse(
            "content_block_start",
            {"type": "content_block_start", "index": idx, "content_block": block},
        )

    def _stop_frame(idx: int) -> bytes:
        return _sse("content_block_stop", {"type": "content_block_stop", "index": idx})

    def _alloc_index() -> int:
        """Allocate the next monotone block index, syncing to state (TASK-714)."""
        nonlocal next_index
        idx = next_index
        next_index += 1
        if state is not None:
            state.next_block_index = next_index
        return idx

    def _set_open(block: dict | None) -> None:
        """Set the open block, mirroring to state (TASK-714) so the retry loop
        knows what content_block_stop to emit before a splice."""
        nonlocal open_block
        open_block = block
        if state is not None:
            state.open_block = block

    try:
        async for event in upstream_stream:
            etype = getattr(event, "type", "") or ""

            # ── lifecycle: pre-content keepalive ────────────────────────────
            if etype in ("response.created", "response.in_progress", "response.queued"):
                yield _ping()
                continue

            # ── output item opened ──────────────────────────────────────────
            if etype == "response.output_item.added":
                item = getattr(event, "item", None)
                itype = getattr(item, "type", "") if item else ""
                if itype == "reasoning":
                    if open_block is not None:
                        yield _stop_frame(open_block["index"])
                        _set_open(None)
                    idx = _alloc_index()
                    _set_open({"index": idx, "kind": "thinking",
                               "item_id": getattr(item, "id", "") or ""})
                    yield _open_start_frame("thinking", idx)
                elif itype in ("function_call", "custom_tool_call"):
                    if open_block is not None:
                        yield _stop_frame(open_block["index"])
                        _set_open(None)
                    item_id = getattr(item, "id", "") or ""
                    call_id = (
                        getattr(item, "call_id", "")
                        or item_id
                        or f"call_{uuid.uuid4().hex[:16]}"
                    )
                    name = getattr(item, "name", "") or ""
                    idx = _alloc_index()
                    block = {
                        "index": idx,
                        "kind": "tool",
                        "item_id": item_id,
                        "call_id": call_id,
                        "name": name,
                    }
                    _set_open(block)
                    blocks_by_item[item_id] = block
                    saw_tool_use = True
                    # TASK-714 HARD BARRIER: a tool call has been forwarded to
                    # the SDK. Retry policy will refuse to replay from this
                    # point regardless of failure bucket — replaying could
                    # duplicate tool execution.
                    if state is not None:
                        state.tool_committed = True
                    yield _open_start_frame("tool", idx, call_id=call_id, name=name)
                elif itype == "message":
                    # Text container — the actual text block opens lazily on the
                    # first content_part.added / output_text.delta.
                    pass
                else:
                    # Server-side tool calls (web_search, code_interpreter, mcp,
                    # image_generation, file_search). The Claude Agent SDK drives
                    # this proxy with function tools only, so these are not
                    # reachable in practice — but keep the stream alive and log
                    # if a provider ever emits one so we notice.
                    logger.debug("Unhandled output item type=%r — keepalive", itype)
                    yield _ping()
                continue

            # ── reasoning deltas ────────────────────────────────────────────
            if etype in ("response.reasoning_text.delta",
                         "response.reasoning_summary_text.delta"):
                delta = getattr(event, "delta", "") or ""
                if not delta:
                    continue
                if open_block is None or open_block["kind"] != "thinking":
                    if open_block is not None:
                        yield _stop_frame(open_block["index"])
                    idx = _alloc_index()
                    _set_open({"index": idx, "kind": "thinking", "item_id": ""})
                    yield _open_start_frame("thinking", idx)
                # TASK-714: mark THINKING phase — retry loop may still replay
                # this whole call (thinking is display-only per Al's decision).
                if state is not None:
                    state.thinking_yielded = True
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": open_block["index"],
                        "delta": {"type": "thinking_delta", "thinking": delta},
                    },
                )
                continue

            if etype == "response.reasoning_summary_part.added":
                # A new summary section. Separate sections with a blank line so
                # multi-part reasoning summaries stay readable.
                summary_index = getattr(event, "summary_index", 0) or 0
                if (summary_index > 0 and open_block is not None
                        and open_block["kind"] == "thinking"):
                    yield _sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": open_block["index"],
                            "delta": {"type": "thinking_delta", "thinking": "\n\n"},
                        },
                    )
                continue

            # ── text / refusal deltas ───────────────────────────────────────
            if etype in ("response.output_text.delta", "response.refusal.delta"):
                delta = getattr(event, "delta", "") or ""
                if not delta:
                    continue
                if open_block is None or open_block["kind"] != "text":
                    if open_block is not None:
                        yield _stop_frame(open_block["index"])
                    idx = _alloc_index()
                    _set_open({"index": idx, "kind": "text", "item_id": ""})
                    yield _open_start_frame("text", idx)
                if etype == "response.refusal.delta":
                    saw_refusal = True
                # TASK-714 HARD BARRIER for retry: any visible text_delta means
                # partial visible content has been forwarded to the SDK. Retry
                # policy defaults to honest fail-upward from this state to
                # prevent duplicated visible text on a CLI-side retry.
                if state is not None:
                    state.text_delta_yielded = True
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": open_block["index"],
                        "delta": {"type": "text_delta", "text": delta},
                    },
                )
                continue

            if etype == "response.content_part.added":
                # Open the text block eagerly when a text/refusal part starts so
                # an empty part still produces a well-formed block pair.
                part = getattr(event, "part", None)
                ptype = getattr(part, "type", "") if part else ""
                if ptype in ("output_text", "text", "refusal"):
                    if open_block is None or open_block["kind"] != "text":
                        if open_block is not None:
                            yield _stop_frame(open_block["index"])
                        idx = _alloc_index()
                        _set_open({"index": idx, "kind": "text", "item_id": ""})
                        yield _open_start_frame("text", idx)
                continue

            # ── function / custom tool argument deltas ──────────────────────
            if etype in ("response.function_call_arguments.delta",
                         "response.custom_tool_call_input.delta"):
                item_id = getattr(event, "item_id", "") or ""
                partial = getattr(event, "delta", "") or ""
                block = blocks_by_item.get(item_id)
                if block is None or not partial:
                    continue
                # Buffer instead of emitting — we sanitize on .done.
                tool_arg_buffers[item_id] = tool_arg_buffers.get(item_id, "") + partial
                continue

            # ── output item closed ──────────────────────────────────────────
            if etype == "response.output_item.done":
                item = getattr(event, "item", None)
                itype = getattr(item, "type", "") if item else ""
                if itype == "reasoning":
                    # Persist a native, origin-tagged replay item in the SDK
                    # signature, independently of whether a summary was shown.
                    signature = reasoning_codec.encode(item) if reasoning_codec else None
                    if signature and (open_block is None or open_block["kind"] != "thinking"):
                        if open_block is not None:
                            yield _stop_frame(open_block["index"])
                        idx = _alloc_index()
                        _set_open({"index": idx, "kind": "thinking",
                                   "item_id": getattr(item, "id", "")})
                        yield _open_start_frame("thinking", idx)
                    if open_block is not None and open_block["kind"] == "thinking":
                        sig = signature or (
                            getattr(item, "encrypted_content", None)
                            if reasoning_codec is None else None
                        ) or f"reasoning:{getattr(item, 'id', '') or message_id}"
                        if signature and state is not None:
                            # A retry splice cannot retract committed native
                            # state; fail upward rather than mixing attempts.
                            state.reasoning_committed = True
                        yield _sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": open_block["index"],
                                "delta": {"type": "signature_delta", "signature": sig},
                            },
                        )
                        yield _stop_frame(open_block["index"])
                        _set_open(None)
                elif itype in ("function_call", "custom_tool_call"):
                    item_id = getattr(item, "id", "") or ""
                    block = blocks_by_item.get(item_id) or open_block
                    if block is not None:
                        yield _stop_frame(block["index"])
                        if open_block is block:
                            _set_open(None)
                    saw_tool_use = True
                elif itype == "message":
                    if open_block is not None and open_block["kind"] == "text":
                        yield _stop_frame(open_block["index"])
                        _set_open(None)
                continue

            # ── terminal: success ───────────────────────────────────────────
            if etype == "response.completed":
                terminal_event_seen = True
                if state is not None:
                    state.terminal_event_seen = True
                resp = getattr(event, "response", None)
                if resp is not None:
                    input_tokens, output_tokens, cache_read, cache_create = _extract_usage(resp)
                    # TASK-714 (Al #4): stash usage on state so the retry loop
                    # can pick the FINAL successful attempt's numbers, not the
                    # sum across aborted attempts. Callback still fires for
                    # side effects; the loop is responsible for gating which
                    # attempt's callback actually credits the turn.
                    if state is not None:
                        state.usage_snapshot = (input_tokens, output_tokens, cache_read, cache_create)
                    if on_usage is not None:
                        on_usage(input_tokens, output_tokens, cache_read, cache_create)
                    logger.info(
                        "Responses usage: input=%d cached=%d uncached=%d output=%d cache_hit=%.1f%%",
                        input_tokens, cache_read,
                        max(input_tokens - cache_read, 0), output_tokens,
                        (100.0 * cache_read / input_tokens) if input_tokens else 0.0,
                    )
                    raw_stop = (
                        getattr(resp, "status", "")
                        or getattr(resp, "stop_reason", "")
                    )
                    if raw_stop in _STOP_REASON_MAP:
                        explicit_stop = _STOP_REASON_MAP[raw_stop]
                # Match Codex CLI: the Responses terminal event ends the stream.
                # Do not wait for transport EOF after a valid completion.
                break

            # ── terminal: incomplete (e.g. hit max_output_tokens) ───────────
            if etype == "response.incomplete":
                terminal_event_seen = True
                if state is not None:
                    state.terminal_event_seen = True
                resp = getattr(event, "response", None)
                if resp is not None:
                    input_tokens, output_tokens, cache_read, cache_create = _extract_usage(resp)
                    if state is not None:
                        state.usage_snapshot = (input_tokens, output_tokens, cache_read, cache_create)
                    if on_usage is not None:
                        on_usage(input_tokens, output_tokens, cache_read, cache_create)
                    details = getattr(resp, "incomplete_details", None)
                    reason = getattr(details, "reason", "") if details else ""
                    explicit_stop = _STOP_REASON_MAP.get(reason, "max_tokens")
                break

            # ── terminal: failure / error ───────────────────────────────────
            # TASK-714 Al #2: in-band terminal errors historically short-circuited
            # with a wire error frame and cleanly returned — the exception-only
            # retry loop never saw them. When a state is provided, we instead
            # populate state.inband_error and return cleanly WITHOUT emitting an
            # error frame; the retry loop consults classify_inband_error() and
            # either retries (bucket B, 5xx-shaped) or emits its own final error
            # frame (bucket C, auth/4xx-shaped). Backcompat: state=None keeps
            # the pre-TASK-714 behavior.
            if etype in ("response.failed", "response.error", "error"):
                err = getattr(event, "error", None)
                if err is None:
                    resp = getattr(event, "response", None)
                    err = getattr(resp, "error", None) if resp is not None else None
                msg = getattr(err, "message", "") or "upstream error"
                code = getattr(err, "code", "") or "upstream_error"
                if state is not None:
                    # Retry loop owns the wire response — pass through cleanly
                    # WITHOUT yielding an error frame here. Close-open-block
                    # hygiene (Al #1) is also the loop's responsibility.
                    state.inband_error = {"code": code, "message": msg}
                    return
                # Close any open block before the error so the SDK's parser
                # doesn't choke on a dangling block.
                if open_block is not None:
                    yield _stop_frame(open_block["index"])
                    open_block = None
                yield _sse(
                    "error",
                    {"type": "error", "error": {"type": code, "message": msg}},
                )
                return

            # ── tool argument .done: emit sanitized buffered JSON ────────────
            if etype in ("response.function_call_arguments.done",
                         "response.custom_tool_call_input.done"):
                item_id = getattr(event, "item_id", "") or ""
                block = blocks_by_item.get(item_id) or open_block
                raw = tool_arg_buffers.pop(item_id, "{}")
                tool_name = block.get("name", "") if block else ""
                # Sanitize tool arguments before the SDK sees them. GPT fills
                # optional params with empty strings (pages: ""), but empty
                # strings can also be intentional required values (notably
                # Edit.new_string="" for deletion). Only remove empties from
                # fields that the matching tool schema declares optional.
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict):
                        if tool_name in required_args_by_tool:
                            required_args = required_args_by_tool[tool_name]
                            parsed = {
                                key: value
                                for key, value in parsed.items()
                                if value != "" or key in required_args
                            }
                        # Never request worktree isolation — the bridge CWD
                        # is not a git repo, so it always fails.
                        if parsed.get("isolation") == "worktree":
                            del parsed["isolation"]
                        # Tool-specific sanitizers (JS trailing-token strip, etc.)
                        parsed = sanitize_tool_arguments(parsed, tool_name)
                    cleaned = json.dumps(parsed, separators=(",", ":"))
                except (json.JSONDecodeError, TypeError):
                    # Try to recover valid JSON with trailing leaked reasoning.
                    recovered = recover_trailing_json(raw)
                    cleaned = recovered if recovered is not None else raw
                if block is not None and cleaned:
                    yield _sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": block["index"],
                            "delta": {"type": "input_json_delta", "partial_json": cleaned},
                        },
                    )
                continue

            # ── events with no Anthropic analog: close-out / annotations /
            #    *.done markers / server-tool progress / audio. Safe to skip,
            #    but emit a keepalive so a quiet provider never stalls. ───────
            if etype in (
                "response.output_text.done",
                "response.refusal.done",
                "response.content_part.done",
                "response.output_text.annotation.added",
                "response.reasoning_text.done",
                "response.reasoning_summary_text.done",
                "response.reasoning_summary_part.done",
            ):
                continue

            # Anything else (server-tool progress, audio, image partials, …):
            # keep the connection warm and move on.
            logger.debug("Passthrough keepalive for event type=%r", etype)
            yield _ping()

        # ── finalize ────────────────────────────────────────────────────────
        # Codex CLI treats EOF before response.completed/response.incomplete as
        # a stream error. Without this guard, a zero-event HTTP 200 becomes a
        # successful empty Anthropic message and the agent silently stops.
        if not terminal_event_seen:
            raise IncompleteResponseStreamError(
                "stream closed before response.completed"
            )

        # Flush any remaining buffered tool args (e.g. stream cut off before
        # .done event). Emit as-is since we can't guarantee well-formed JSON.
        for item_id, leftover in tool_arg_buffers.items():
            block = blocks_by_item.get(item_id) or open_block
            if block is not None and leftover:
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": block["index"],
                        "delta": {"type": "input_json_delta", "partial_json": leftover},
                    },
                )
        tool_arg_buffers.clear()

        if open_block is not None:
            yield _stop_frame(open_block["index"])
            _set_open(None)

        # Precedence: a tool call ends the turn for the agent loop; an explicit
        # non-default terminal (max_tokens, content-filter refusal) outranks a
        # generic completion; a refusal content block outranks a plain
        # end_turn (the upstream still reports status=completed on refusals).
        if saw_tool_use:
            stop_reason = "tool_use"
        elif explicit_stop is not None and explicit_stop != "end_turn":
            stop_reason = explicit_stop
        elif saw_refusal:
            stop_reason = "refusal"
        elif explicit_stop is not None:
            stop_reason = explicit_stop
        else:
            stop_reason = "end_turn"

        uncached_input_tokens = max((input_tokens or 0) - (cache_read or 0), 0)
        yield _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {
                    "input_tokens": uncached_input_tokens,
                    "output_tokens": output_tokens or 0,
                    "cache_creation_input_tokens": cache_create or 0,
                    "cache_read_input_tokens": cache_read or 0,
                },
            },
        )
        yield _sse("message_stop", {"type": "message_stop"})

    except Exception as exc:  # noqa: BLE001
        # TASK-714: when a state is provided the retry loop owns the wire
        # response — we PROPAGATE the exception so the loop can decide bucket +
        # phase policy. Block-close hygiene (Al #1) becomes the loop's job.
        # Backcompat: state=None keeps the pre-TASK-714 behavior (emit an SSE
        # error frame with a canonical Anthropic error type and swallow).
        if state is not None:
            raise
        logger.exception("Stream translation failed")
        if open_block is not None:
            try:
                yield _stop_frame(open_block["index"])
            except Exception:  # noqa: BLE001
                pass
        # The error `type` MUST come from Anthropic's canonical set — the
        # Claude CLI switches on it to decide retryability (it matches the
        # literal `"type":"overloaded_error"` substring to classify a
        # retryable server overload). A non-canonical type matches no branch,
        # so the CLI treats the turn as terminally failed and stops mid-work.
        # See errors.py for the same contract on the non-streaming path.
        yield _sse(
            "error",
            {
                "type": "error",
                "error": {
                    "type": _stream_error_type(exc),
                    "message": f"Proxy stream translation failed: {exc}",
                },
            },
        )
