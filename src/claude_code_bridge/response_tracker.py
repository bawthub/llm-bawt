"""Per-LLM-response identity and attributable usage for one bridge run (TASK-2110).

Each upstream model response has a provider id: ``message_start.message.id`` on
the stream and ``AssistantMessage.message_id`` on the content-block snapshots
(several snapshots share one id). Tool calls are grouped by that id, never by
run/turn id, adjacency or timing.

Usage attribution (verified against live SDK 0.2.177 snapshots):

* ``AssistantMessage.stop_reason`` is null on every live snapshot, so it is
  never a finality signal here.
* Snapshot ``output_tokens`` is preliminary (the message_start value); the real
  count arrives on the id-less ``message_delta``.
* The main thread is serialized: at most one main response is open, so its
  id-less ``message_delta``/``message_stop`` are unambiguous. Those are the only
  signals that make usage final.
* Subagent stream events are not forwarded. Subagent usage comes only from
  snapshots keyed by ``message_id``; it stays provisional, and consumers must
  not present its output count as final. Sibling Workflow tasks share one
  parent, so nothing parent-keyed is used for attribution.

Counters are max-merged per field, never summed: repeated or partial snapshots
of the same response must not double-count it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Anthropic-shaped counters as reported; normalized input (fresh + cache read +
# cache write) is derived by readers, so a missing field is never added as 0.
USAGE_FIELDS = (
    "input_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "output_tokens",
)

PROVISIONAL = "provisional"
FINAL = "final"


def _count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


@dataclass
class ResponseRecord:
    response_id: str
    host_parent: str | None
    first_seen_seq: int
    model: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    usage_state: str = PROVISIONAL

    def payload(self) -> dict:
        return {
            "response_id": self.response_id,
            "host_parent": self.host_parent,
            "first_seen_seq": self.first_seen_seq,
            "model": self.model,
            "usage": dict(self.usage),
            "usage_state": self.usage_state,
        }


class ResponseTracker:
    """Request-local response ledger. Returns the records that changed."""

    def __init__(self) -> None:
        self._records: dict[str, ResponseRecord] = {}
        self._open_main: str | None = None
        self._published: dict[str, tuple] = {}

    def record(self, response_id: str | None) -> ResponseRecord | None:
        return self._records.get(response_id or "")

    def _ensure(self, response_id: str, host_parent: str | None, seq: int) -> ResponseRecord:
        record = self._records.get(response_id)
        if record is None:
            # First sighting fixes the scope and order; later sightings never move it.
            record = ResponseRecord(response_id, host_parent, seq)
            self._records[response_id] = record
        return record

    @staticmethod
    def _merge_usage(record: ResponseRecord, usage: Any) -> None:
        if not isinstance(usage, dict):
            return
        for key in USAGE_FIELDS:
            value = usage.get(key)
            if _count(value) and value > record.usage.get(key, -1):
                record.usage[key] = value

    def on_stream_event(self, event: dict, *, source: str | None, seq: int) -> ResponseRecord | None:
        """Main-thread stream frames only (``source`` is the parent tool id)."""
        if source is not None or not isinstance(event, dict):
            return None
        kind = event.get("type")
        if kind == "message_start":
            message = event.get("message") if isinstance(event.get("message"), dict) else {}
            response_id = message.get("id")
            if not isinstance(response_id, str) or not response_id:
                # No provider id: nothing is attributable until the next start.
                self._open_main = None
                return None
            record = self._ensure(response_id, None, seq)
            if isinstance(message.get("model"), str) and not record.model:
                record.model = message["model"]
            self._merge_usage(record, message.get("usage"))
            self._open_main = response_id
            return self._changed(record)
        record = self._records.get(self._open_main or "")
        if record is None:
            return None
        if kind == "message_delta":
            self._merge_usage(record, event.get("usage"))
            return self._changed(record)
        if kind == "message_stop":
            record.usage_state = FINAL
            self._open_main = None
            return self._changed(record)
        return None

    def on_assistant_message(
        self, msg: Any, *, host_parent: str | None, seq: int,
    ) -> ResponseRecord | None:
        response_id = getattr(msg, "message_id", None)
        if not isinstance(response_id, str) or not response_id:
            return None
        model = getattr(msg, "model", None)
        if isinstance(model, str) and model.startswith("<"):
            # CLI-synthesized messages (``<synthetic>``) are not model responses.
            return None
        record = self._ensure(response_id, host_parent, seq)
        if isinstance(model, str) and model and not record.model:
            record.model = model
        self._merge_usage(record, getattr(msg, "usage", None))
        return self._changed(record)

    def _changed(self, record: ResponseRecord) -> ResponseRecord | None:
        signature = (record.model, tuple(sorted(record.usage.items())), record.usage_state)
        if self._published.get(record.response_id) == signature:
            return None
        self._published[record.response_id] = signature
        return record
