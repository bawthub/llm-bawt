"""Process-local working-memory counters for the health report (docs/WORKING_MEMORY.md 14).

Counts since this process started, per bot. They reset on restart by design;
durable state (items, versions, sessions) is read from the database instead.
"""
from __future__ import annotations

import threading
from collections import Counter

RENDER_FAILURES = "render_failures"
NOTICES_FIRED = "notices_fired"
NO_PIN_DECISION = "no_pin_decision"
NOTICE_FAILURES = "notice_failures"
COUNTERS = (RENDER_FAILURES, NOTICES_FIRED, NO_PIN_DECISION, NOTICE_FAILURES)

_lock = threading.Lock()
_counts: dict[str, Counter[str]] = {name: Counter() for name in COUNTERS}


def _bump(name: str, bot_id: str) -> None:
    with _lock:
        _counts[name][str(bot_id or "").strip().lower()] += 1


def record_render_failure(bot_id: str) -> None:
    _bump(RENDER_FAILURES, bot_id)


def record_notice(bot_id: str) -> None:
    _bump(NOTICES_FIRED, bot_id)


def record_no_pin_decision(bot_id: str) -> None:
    _bump(NO_PIN_DECISION, bot_id)


def record_notice_failure(bot_id: str) -> None:
    _bump(NOTICE_FAILURES, bot_id)


def snapshot(bot_id: str) -> dict[str, int]:
    key = str(bot_id or "").strip().lower()
    with _lock:
        return {name: _counts[name][key] for name in COUNTERS}


def reset() -> None:
    """Tests only."""
    with _lock:
        for counter in _counts.values():
            counter.clear()


__all__ = [
    "COUNTERS", "NOTICES_FIRED", "NOTICE_FAILURES", "NO_PIN_DECISION", "RENDER_FAILURES",
    "record_no_pin_decision", "record_notice", "record_notice_failure", "record_render_failure",
    "reset", "snapshot",
]
