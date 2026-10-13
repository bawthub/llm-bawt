"""Upstream-status relay contract + live turn phase (TASK-1074).

Single owner of the ``upstream_status`` field list: the bridge-event relay
(``agent_backends/agent_bridge.py``) and the unified-SSE publisher
(``service/turn_stream_worker.py``) both go through here instead of each
enumerating fields by hand.

Turn phase is transient display state. It is never persisted or replayed;
:class:`LiveTurnPhases` keeps only the latest phase per in-flight turn in
memory so ``/v1/turn-logs?active_only=true`` can hand a mid-turn joiner the
current phase. Turns die with the app process, so memory is the right lifetime.
"""

from __future__ import annotations

import threading
import time
from typing import Any

# Fields copied from the bridge's ``upstream_status`` raw payload, in order.
# ``phase``/``since``/``model``/``context_tokens`` are the TASK-1074 phase
# contract; ``state`` mirrors ``phase`` for pre-phase clients; the rest is
# proxy/retry attempt detail.
UPSTREAM_STATUS_FIELDS: tuple[str, ...] = (
    "phase",
    "state",
    "since",
    "model",
    "context_tokens",
    "attempt",
    "next_attempt",
    "max_attempts",
    "error_status",
    "transport",
    "fallback_transport",
    "stall_phase",
    "productive_idle_seconds",
    "elapsed_seconds",
    "provider",
)

# What a mid-turn joiner gets as ``live_phase``.
LIVE_PHASE_FIELDS: tuple[str, ...] = ("phase", "since", "text", "model", "context_tokens")

# Safety net for a turn whose ``turn_complete`` never reached the drain.
LIVE_PHASE_TTL_SECONDS = 6 * 60 * 60


def upstream_status_queue_item(event: Any) -> dict[str, Any]:
    """Bridge ``AgentEvent`` → the worker's result-queue item."""
    raw = event.raw if isinstance(event.raw, dict) else {}
    meta = raw.get("upstream_status")
    meta = meta if isinstance(meta, dict) else {}
    item: dict[str, Any] = {"event": "upstream_status", "text": event.text or ""}
    item.update({field: meta.get(field) for field in UPSTREAM_STATUS_FIELDS})
    item["provider"] = item["provider"] or event.provider
    item["trigger_message_id"] = event.trigger_message_id
    return item


def upstream_status_sse_payload(
    item: dict[str, Any],
    *,
    turn_id: str | None,
    trigger_message_id: str | None,
    bot_id: str,
    user_id: str,
) -> dict[str, Any]:
    """Result-queue item → unified SSE ``upstream_status`` event."""
    payload: dict[str, Any] = {
        "_type": "upstream_status",
        "turn_id": turn_id,
        "trigger_message_id": item.get("trigger_message_id") or trigger_message_id,
        "bot_id": bot_id,
        "user_id": user_id,
        "text": item.get("text") or "",
    }
    payload.update({field: item.get(field) for field in UPSTREAM_STATUS_FIELDS})
    payload["ts"] = time.time()
    return payload


class LiveTurnPhases:
    """Latest transient phase per in-flight turn, fed by the unified drain."""

    def __init__(self, *, ttl: float = LIVE_PHASE_TTL_SECONDS, clock=time.time) -> None:
        self._ttl = ttl
        self._clock = clock
        self._lock = threading.Lock()  # drain on the loop, reads in threadpool
        self._phases: dict[str, tuple[float, dict[str, Any]]] = {}

    def observe(self, event: dict[str, Any]) -> None:
        turn_id = event.get("turn_id")
        if not turn_id:
            return
        kind = event.get("_type")
        if kind == "upstream_status" and event.get("phase"):
            phase = {field: event.get(field) for field in LIVE_PHASE_FIELDS}
            with self._lock:
                self._phases[turn_id] = (self._clock() + self._ttl, phase)
                self._prune_locked()
        elif kind == "turn_complete":
            with self._lock:
                self._phases.pop(turn_id, None)

    def get(self, turn_id: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._phases.get(turn_id)
            if entry is None:
                return None
            expires_at, phase = entry
            if expires_at <= self._clock():
                del self._phases[turn_id]
                return None
            return dict(phase)

    def _prune_locked(self) -> None:
        now = self._clock()
        for turn_id in [t for t, (exp, _p) in self._phases.items() if exp <= now]:
            del self._phases[turn_id]
