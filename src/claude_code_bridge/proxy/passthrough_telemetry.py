"""Bounded, observational telemetry for Anthropic-native SSE streams.

Usage fields are cumulative snapshots, not increments. Providers may supply
input/cache counts only in message_delta; omitted/invalid fields must not erase
previous counts. This tap never yields, rewrites, retries or logs stream content.
"""
from __future__ import annotations

import json
import re

from .request_context import ProxyRequestContext

_MAX_EVENT_BYTES = 64 * 1024
_PROVIDER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._/-]{0,127}\Z")
_USAGE_FIELDS = (
    "input_tokens", "output_tokens", "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


class SSEObserver:
    """Shared bounded framing only; subclasses consume data or fixed skip codes."""

    def __init__(self) -> None:
        self._line = bytearray()
        self._data: list[bytes] = []
        self._size = 0
        self._discarding = False

    def feed(self, chunk: bytes, elapsed_ms: float) -> None:
        """Observe original bytes; elapsed time is from the upstream POST start.

        Parse only complete, bounded SSE events. LF and CRLF (including split
        delimiters/UTF-8) are supported. Oversize/malformed events are skipped,
        resynchronizing at the next blank line without retaining their bodies.
        """
        offset = 0
        while offset < len(chunk):
            end = chunk.find(b"\n", offset)
            part = chunk[offset:] if end < 0 else chunk[offset:end]
            self._size = min(_MAX_EVENT_BYTES + 1, self._size + len(part))
            if self._size > _MAX_EVENT_BYTES:
                if not self._discarding:
                    self._skip("oversize", elapsed_ms)
                self._discarding = True
                self._data.clear()
                # Keep only the bounded current line, so it cannot be mistaken
                # for a blank delimiter when the rest arrives in a later chunk.
            if not self._discarding:
                self._line.extend(part)
            elif part:
                self._line[:] = b"\r" if part == b"\r" and not self._line else b"x"
            if end < 0:
                break
            line = bytes(self._line).removesuffix(b"\r")
            self._line.clear()
            if not line:
                if not self._discarding and self._data:
                    self._observe(b"\n".join(self._data), elapsed_ms)
                self._data.clear()
                self._size = 0
                self._discarding = False
            elif not self._discarding and line.startswith(b"data:"):
                self._data.append(line[5:].removeprefix(b" "))
            elif not self._discarding and line.startswith(b":"):
                self._skip("comment", elapsed_ms)
            offset = end + 1

    def _skip(self, reason: str, elapsed_ms: float) -> None:
        pass

    def _observe(self, data: bytes, elapsed_ms: float) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        self._line.clear()
        self._data.clear()
        self._size = 0
        self._discarding = False


class PassthroughTelemetry(SSEObserver):
    def __init__(self, context: ProxyRequestContext | None = None) -> None:
        super().__init__()
        self.context = context
        self.usage: dict[str, int] = {}
        self.serving_provider: str | None = None
        self.first_byte_ms: float | None = None
        self.first_output_ms: float | None = None

    def feed(self, chunk: bytes, elapsed_ms: float) -> None:
        if chunk and self.first_byte_ms is None:
            self.first_byte_ms = elapsed_ms
            if self.context is not None and self.context.upstream_ttfb_ms is None:
                self.context.upstream_ttfb_ms = elapsed_ms
        super().feed(chunk, elapsed_ms)
        self._sync_context()

    def _observe(self, data: bytes, elapsed_ms: float) -> None:
        try:
            event = json.loads(data)
        except (ValueError, UnicodeDecodeError):
            return
        if not isinstance(event, dict):
            return
        kind = event.get("type")
        message = event.get("message")
        if not isinstance(message, dict):
            message = {}
        if kind in ("message_start", "message_delta"):
            usage = message.get("usage") if kind == "message_start" else event.get("usage")
            if isinstance(usage, dict):
                for field in _USAGE_FIELDS:
                    value = usage.get(field)
                    if type(value) is int and value >= 0:
                        self.usage[field] = value
            # Only an explicit upstream metadata field identifies the serving
            # provider. Never substitute the adapter/routing provider or model.
            provider = event.get("provider") or message.get("provider")
            if isinstance(provider, str) and _PROVIDER_NAME.fullmatch(provider):
                self.serving_provider = provider
        if self.first_output_ms is None and self._has_output(event):
            self.first_output_ms = elapsed_ms
            if self.context is not None and self.context.first_output_ms is None:
                self.context.first_output_ms = elapsed_ms

    @staticmethod
    def _has_output(event: dict) -> bool:
        kind = event.get("type")
        if kind == "content_block_start":
            block = event.get("content_block")
            if not isinstance(block, dict):
                return False
            # A tool header is meaningful output even before its arguments.
            return block.get("type") in ("tool_use", "server_tool_use") or any(
                isinstance(block.get(key), str) and bool(block[key])
                for key in ("text", "thinking")
            )
        if kind == "content_block_delta":
            delta = event.get("delta")
            return isinstance(delta, dict) and any(
                isinstance(delta.get(key), str) and bool(delta[key])
                for key in ("text", "thinking", "partial_json")
            )
        return False

    @property
    def total_input(self) -> int:
        return sum(self.usage.get(field, 0) for field in (
            "input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens",
        ))

    @property
    def cache_hit_pct(self) -> float:
        return (100.0 * self.usage.get("cache_read_input_tokens", 0) / self.total_input
                if self.total_input else 0.0)

    def _sync_context(self) -> None:
        if self.context is None:
            return
        if self.usage:
            self.context.record_usage(
                self.total_input, self.usage.get("output_tokens", 0),
                self.usage.get("cache_read_input_tokens", 0),
                self.usage.get("cache_creation_input_tokens", 0),
            )
        if self.serving_provider is not None:
            self.context.serving_provider = self.serving_provider
