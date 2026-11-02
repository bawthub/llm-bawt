"""Stop degenerate repetition in streamed model output on every proxy route.

Some upstreams loop on a short fragment (e.g. a closing tag, or whitespace)
inside a tool argument until max_tokens, which costs a long silent wait and
tens of kilobytes of junk. This watches the unified Anthropic SSE that every
adapter emits and raises as soon as the tail of one content block is a short
period repeated across a large span. Raising before ToolCallGuard releases a
held tool suffix means no executable call is forwarded.

Detection only; it does not retry. The existing retry lease keeps its
conservative no-replay tombstone for streams that started a tool block.
Thresholds come from the environment (``0`` disables a check).
"""
from __future__ import annotations

import json
import os

from .tool_guard import UnsafeToolCallError

ENV_MIN_SPAN = "CLAUDE_CODE_BRIDGE_PROXY_REPETITION_MIN_SPAN"
ENV_MAX_PERIOD = "CLAUDE_CODE_BRIDGE_PROXY_REPETITION_MAX_PERIOD"
ENV_MAX_TOOL_ARG_CHARS = "CLAUDE_CODE_BRIDGE_PROXY_MAX_TOOL_ARG_CHARS"

# A run this long of one fragment repeated is degenerate; the 2026-11-01
# incident looped a ~30-char fragment for ~59KB per call.
DEFAULT_MIN_SPAN = 8192
DEFAULT_MAX_PERIOD = 512
# Legitimate Write/Edit arguments can be very large; size cap is opt-in.
DEFAULT_MAX_TOOL_ARG_CHARS = 0
_CHECK_STRIDE = 256

_DELTA_FIELDS = {
    "input_json_delta": "partial_json",
    "text_delta": "text",
    "thinking_delta": "thinking",
}


class RepetitionLoopError(UnsafeToolCallError):
    """Terminal rejection of degenerate output; never a retryable overload."""


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default
    return max(value, 0)


def periodic_tail(text: str, min_span: int, max_period: int) -> int | None:
    """Smallest period <= max_period that the last ``min_span`` chars repeat."""
    if min_span <= 0 or len(text) < min_span:
        return None
    span = text[-min_span:]
    probe = min(64, min_span // 2)
    last = span[-probe:]
    for period in range(1, min(max_period, min_span - probe, min_span // 2) + 1):
        # Cheap probe first; non-periodic text rejects nearly every candidate here.
        if span[-probe - period:-period] != last:
            continue
        if span[period:] == span[:-period]:
            return period
    return None


class _Block:
    __slots__ = ("kind", "tail", "chars", "pending")

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.tail = ""
        self.chars = 0
        self.pending = 0


class RepetitionBreaker:
    def __init__(self, *, min_span: int | None = None, max_period: int | None = None,
                 max_tool_arg_chars: int | None = None) -> None:
        self.min_span = _env_int(ENV_MIN_SPAN, DEFAULT_MIN_SPAN) if min_span is None else min_span
        self.max_period = (_env_int(ENV_MAX_PERIOD, DEFAULT_MAX_PERIOD)
                           if max_period is None else max_period)
        self.max_tool_arg_chars = (
            _env_int(ENV_MAX_TOOL_ARG_CHARS, DEFAULT_MAX_TOOL_ARG_CHARS)
            if max_tool_arg_chars is None else max_tool_arg_chars
        )
        self._blocks: dict[object, _Block] = {}

    @property
    def enabled(self) -> bool:
        return (self.min_span > 0 and self.max_period > 0) or self.max_tool_arg_chars > 0

    def feed(self, frame: bytes) -> None:
        if not self.enabled or b"content_block_" not in frame:
            return
        data = b"\n".join(line[5:].lstrip() for line in frame.splitlines()
                          if line.startswith(b"data:"))
        try:
            payload = json.loads(data) if data else None
        except (ValueError, UnicodeDecodeError):
            return  # framing errors belong to ToolCallGuard
        if not isinstance(payload, dict):
            return
        index = payload.get("index")
        if not isinstance(index, int):
            return
        if payload.get("type") == "content_block_start":
            block = payload.get("content_block") or {}
            self._blocks[index] = _Block(str(block.get("type") or "unknown"))
            return
        if payload.get("type") == "content_block_stop":
            block = self._blocks.pop(index, None)
            if block is not None and block.pending:
                self._check(block)
            return
        if payload.get("type") != "content_block_delta":
            return
        delta = payload.get("delta") or {}
        field = _DELTA_FIELDS.get(delta.get("type"))
        piece = delta.get(field) if field else None
        if not isinstance(piece, str) or not piece:
            return
        block = self._blocks.setdefault(index, _Block("unknown"))
        self._observe(block, piece)

    def observe(self, key: object, piece: str, *, kind: str) -> None:
        """Watch text a translator buffers before it reaches the unified SSE."""
        if self.enabled and piece:
            self._observe(self._blocks.setdefault(key, _Block(kind)), piece)

    def discard(self, key: object) -> None:
        self._blocks.pop(key, None)

    def _observe(self, block: _Block, piece: str) -> None:
        block.chars += len(piece)
        if (self.max_tool_arg_chars and block.kind in {"tool_use", "server_tool_use"}
                and block.chars > self.max_tool_arg_chars):
            raise RepetitionLoopError(
                f"Tool arguments exceeded {self.max_tool_arg_chars} chars; stream stopped, "
                "no tool calls forwarded"
            )
        if self.min_span <= 0 or self.max_period <= 0:
            return
        keep = self.min_span + _CHECK_STRIDE
        # Check at stride boundaries inside large pieces too: a translator may
        # deliver a whole buffered argument at once, and its closing bytes
        # would otherwise hide the periodic run behind them.
        for start in range(0, len(piece), _CHECK_STRIDE):
            part = piece[start:start + _CHECK_STRIDE]
            block.tail = (block.tail + part)[-keep:]
            block.pending += len(part)
            if block.pending >= _CHECK_STRIDE:
                self._check(block)

    def _check(self, block: _Block) -> None:
        block.pending = 0
        period = periodic_tail(block.tail, self.min_span, self.max_period)
        if period is not None:
            raise RepetitionLoopError(
                f"Model output looped: {block.kind} block repeated a {period}-char "
                f"fragment for {self.min_span}+ chars (block_chars={block.chars}); "
                "stream stopped, no tool calls forwarded"
            )
