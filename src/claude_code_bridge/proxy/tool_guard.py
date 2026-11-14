"""Withhold executable tool blocks until the response's stop reason is known.

A closed tool block is executable even when a later message_delta says
max_tokens. Keep the entire suffix from the first tool start in order, and
release it only at a non-truncated message_stop. Text before tools and pings
still stream. This is a dispatch guard, not a retry policy or argument cap.
"""
from __future__ import annotations

import json
from tempfile import SpooledTemporaryFile
from typing import Iterator


class UnsafeToolCallError(RuntimeError):
    """Terminal dispatch rejection; never advertise a retryable overload."""


class ToolCallGuard:
    def __init__(self) -> None:
        self._held = None
        self._stop_reason: str | None = None
        self._tool_starts: list[tuple[int, str]] = []

    def replacement_slot(self) -> tuple[int, str] | None:
        """Only one still-withheld tool is eligible; never drop completed siblings."""
        if self._held is not None and len(self._tool_starts) == 1:
            return self._tool_starts[0]
        return None

    def feed(self, frame: bytes, *, tool_start: bool) -> Iterator[bytes]:
        data = b"\n".join(line[5:].lstrip() for line in frame.splitlines()
                          if line.startswith(b"data:"))
        try:
            payload = json.loads(data) if data else {}
            if not isinstance(payload, dict):
                raise ValueError("Invalid SSE payload")
        except (ValueError, UnicodeDecodeError) as exc:
            if self._held is not None:
                raise UnsafeToolCallError("Invalid SSE frame while validating tool calls") from exc
            yield frame
            return
        event = payload.get("type")
        if event == "message_delta":
            reason = (payload.get("delta") or {}).get("stop_reason")
            if reason:
                self._stop_reason = reason
        if tool_start:
            self._tool_starts.append((payload["index"], payload["content_block"]["id"]))
        if tool_start and self._held is None:
            # Spill large, legitimate code/file arguments to an anonymous file;
            # the threshold bounds RAM, NOT accepted argument/response size.
            self._held = SpooledTemporaryFile(max_size=1024 * 1024, mode="w+b")
        if self._held is None or event == "ping" or not data:
            yield frame
            return
        if event == "error":
            self.close()
            yield frame
            return
        if self._stop_reason in {
            "max_tokens", "length", "max_output_tokens", "model_context_window_exceeded",
        }:
            raise UnsafeToolCallError("Upstream truncated tool calls at its output/context limit; no tool calls forwarded")
        self._held.write(frame)
        if event == "message_stop":
            if self._stop_reason is None:
                raise UnsafeToolCallError("Tool response ended without a stop reason; no tool calls forwarded")
            self._held.seek(0)
            while chunk := self._held.read(64 * 1024):
                yield chunk
            self.close()

    def finish(self) -> None:
        if self._held is not None:
            raise UnsafeToolCallError("Tool response ended before message_stop; no tool calls forwarded")

    def close(self) -> None:
        if self._held is not None:
            self._held.close()
            self._held = None
