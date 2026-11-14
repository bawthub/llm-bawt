"""Per-attempt metadata only, before SDK decode and after translation.

Never retains payloads beyond the existing bounded SSE framing buffers. Labels
are static, counters saturate, and observation failures cannot change the stream.
An export count means yielded by the translator, not consumed by the SDK/UI.
HTTP raw counts cover decompressed body bytes and bounded JSON type families,
not network packets or hidden provider work. WS reuses the existing JSON parse
before event-object conversion. Oversize/unframed content has no inferred family.
"""
from __future__ import annotations

import json
import logging
import re

from .passthrough_telemetry import SSEObserver

logger = logging.getLogger(__name__)
_CAP = 2**31 - 1
_REQUEST_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_DELTA_FAMILIES = {
    "response.reasoning_text.delta": "reasoning_delta",
    "response.reasoning_summary_text.delta": "reasoning_summary_delta",
    "response.output_text.delta": "text_delta",
    "response.refusal.delta": "text_delta",
    "response.function_call_arguments.delta": "tool_delta",
    "response.custom_tool_call_input.delta": "tool_delta",
}
_RESPONSE_FAMILIES = frozenset({
    "reasoning_delta", "reasoning_summary_delta", "text_delta", "tool_delta", "empty_delta", "reasoning_open",
    "reasoning_native", "reasoning_empty", "tool_item", "message_item", "metadata",
    "terminal", "unknown", "malformed", "oversize", "comment",
})
_EXPORT_DELTAS = {
    "thinking_delta": "thinking", "text_delta": "text",
    "signature_delta": "signature", "input_json_delta": "tool_arguments",
}
_EXPORT_KINDS = {
    "message_start": "message_start", "message_delta": "message_meta",
    "message_stop": "message_stop", "content_block_stop": "block_stop",
    "ping": "ping", "error": "error",
}
_EXPORT_FAMILIES = frozenset({
    *_EXPORT_DELTAS.values(), *_EXPORT_KINDS.values(), "tool_header", "block_start",
    "unknown", "malformed", "oversize", "comment",
})


def field(event, name):
    return event.get(name) if isinstance(event, dict) else getattr(event, name, None)


def response_family(event):
    kind = field(event, "type")
    if not isinstance(kind, str):
        return "malformed"
    if kind in _DELTA_FAMILIES:
        delta = field(event, "delta")
        if not isinstance(delta, str):
            return "malformed"
        return _DELTA_FAMILIES[kind] if delta and not delta.isspace() else "empty_delta"
    if kind in {"response.output_item.added", "response.output_item.done"}:
        item = field(event, "item")
        item_type = field(item, "type")
        if item_type == "reasoning":
            if kind.endswith("added"):
                return "reasoning_open"
            native = field(item, "encrypted_content")
            return ("reasoning_native" if isinstance(native, str) and native and not native.isspace()
                    else "reasoning_empty")
        if item_type in ("function_call", "custom_tool_call"):
            return "tool_item"
        return "message_item" if item_type == "message" else "unknown"
    if kind in {"response.created", "response.in_progress", "response.queued",
                "response.metadata", "keepalive", "ping"}:
        return "metadata"
    if kind in {"response.completed", "response.incomplete", "response.failed",
                "response.error", "error"}:
        return "terminal"
    return "unknown"


def export_family(event):
    kind = field(event, "type")
    if not isinstance(kind, str):
        return "malformed"
    if kind == "content_block_delta":
        subtype = field(field(event, "delta"), "type")
        return _EXPORT_DELTAS.get(subtype, "unknown") if isinstance(subtype, str) else "malformed"
    if kind == "content_block_start":
        return ("tool_header" if field(field(event, "content_block"), "type") == "tool_use"
                else "block_start")
    return _EXPORT_KINDS.get(kind, "unknown")


class _Boundary(SSEObserver):
    def __init__(self, owner, name):
        super().__init__()
        self.owner, self.name = owner, name

    def _skip(self, reason, elapsed_ms):
        self.owner.record(self.name, reason, elapsed_ms)

    def _observe(self, data, elapsed_ms):
        if data == b"[DONE]":
            self.owner.record(self.name, "terminal", elapsed_ms)
            return
        try:
            event = json.loads(data)
            family = (response_family if self.name == "raw" else export_family)(event)
        except (ValueError, RecursionError):
            family = "malformed"
        self.owner.record(self.name, family, elapsed_ms)


class ResponsesProgress:
    def __init__(self, started_at, context, attempt, clock):
        self.started_at, self.clock, self.attempt = started_at, clock, attempt
        request_id = getattr(context, "request_id", None)
        self.request_id = (request_id if isinstance(request_id, str)
                           and _REQUEST_ID.fullmatch(request_id) else "unknown")
        self.counts = {name: {} for name in ("raw", "decoded", "exported")}
        self.last_ms = {name: {} for name in self.counts}
        self.first_body_ms = self.last_body_ms = None
        self.body_bytes = self.body_chunks = self.ws_chars = self.observer_errors = 0
        self.raw_attached = False
        self.productive = self.nonproductive = 0
        self.last_productive_ms = None
        self._raw, self._exported = _Boundary(self, "raw"), _Boundary(self, "exported")
        self._reported = set()

    def elapsed(self):
        return round(min(_CAP, max(0.0, (self.clock() - self.started_at) * 1000)), 1)

    def record(self, boundary, family, elapsed_ms):
        allowed = _EXPORT_FAMILIES if boundary == "exported" else _RESPONSE_FAMILIES
        label = family if family in allowed else "unknown"
        counts = self.counts[boundary]
        counts[label] = min(_CAP, counts.get(label, 0) + 1)
        self.last_ms[boundary][label] = elapsed_ms

    def received(self, raw):
        elapsed = self.elapsed()
        if self.first_body_ms is None:
            self.first_body_ms = elapsed
        self.last_body_ms = elapsed
        self.body_chunks = min(_CAP, self.body_chunks + 1)
        if isinstance(raw, str):
            self.ws_chars = min(_CAP, self.ws_chars + len(raw))
        else:
            self.body_bytes = min(_CAP, self.body_bytes + len(raw))

    def feed(self, boundary, chunk):
        observer = self._raw if boundary == "raw" else self._exported
        try:
            if boundary == "raw" and chunk:
                self.received(chunk)
            observer.feed(chunk, self.elapsed())
        except Exception:
            self.observer_errors = min(_CAP, self.observer_errors + 1)
            observer.clear()

    def attach_http(self, upstream):
        """Tap decompressed HTTP body chunks before the SDK's SSE/JSON decoder.

        Patch only this response instance's public byte iterator, not the SDK or
        the socket stream (which could still be compressed). Preserve each chunk.
        """
        try:
            response = getattr(upstream, "response", None)
            original = getattr(response, "aiter_bytes", None)
            if self.raw_attached or not callable(original):
                return

            async def observed(*args, **kwargs):
                async for chunk in original(*args, **kwargs):
                    self.feed("raw", chunk)
                    yield chunk

            response.aiter_bytes = observed
            self.raw_attached = True
        except Exception:
            self.observer_errors = min(_CAP, self.observer_errors + 1)

    def received_ws(self, raw, payload):
        """Reuse already parsed WS JSON; do not parse/copy a second large body."""
        try:
            self.raw_attached = True
            self.received(raw)
            self.record("raw", response_family(payload), self.elapsed())
        except Exception:
            self.observer_errors = min(_CAP, self.observer_errors + 1)

    def decoded(self, event, productive):
        try:
            elapsed = self.elapsed()
            family = response_family(event)
            self.record("decoded", family, elapsed)
            if productive:
                self.productive = min(_CAP, self.productive + 1)
                self.last_productive_ms = elapsed
            else:
                self.nonproductive = min(_CAP, self.nonproductive + 1)
            return family
        except Exception:
            self.observer_errors = min(_CAP, self.observer_errors + 1)
            return "unknown"

    def report(self, trigger, transport, complete=False):
        """At most one stall and one close summary per attempt; never payloads."""
        try:
            if trigger not in {"stall", "close"} or trigger in self._reported:
                return
            self._reported.add(trigger)
            summary = {
                "counts": self.counts, "last_ms": self.last_ms,
                "body_bytes": self.body_bytes, "body_chunks": self.body_chunks,
                "first_body_ms": self.first_body_ms, "last_body_ms": self.last_body_ms,
                "ws_chars": self.ws_chars, "raw_attached": self.raw_attached,
                "productive": self.productive, "nonproductive": self.nonproductive,
                "last_productive_ms": self.last_productive_ms,
                "observer_errors": self.observer_errors, "elapsed_ms": self.elapsed(),
                "complete": bool(complete),
                "partial": {
                    "raw": bool(self._raw._line or self._raw._data or self._raw._discarding),
                    "exported": bool(self._exported._line or self._exported._data
                                     or self._exported._discarding),
                },
            }
            # Healthy completed streams are DEBUG; anything incomplete or with
            # observer errors stays INFO for forensics (TASK-2225).
            healthy = bool(complete) and not self.observer_errors
            (logger.debug if healthy else logger.info)("responses_progress request_id=%s attempt=%d transport=%s trigger=%s summary=%s",
                        self.request_id, self.attempt,
                        transport if transport in {"sse", "websocket"} else "unknown", trigger,
                        json.dumps(summary, sort_keys=True, separators=(",", ":")))
        except Exception:
            self.observer_errors = min(_CAP, self.observer_errors + 1)
        finally:
            if trigger == "close":
                self._raw.clear()
                self._exported.clear()
