"""Aggregate retrieval stats for the app scrape; no per-query or per-user series."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from prometheus_client import CollectorRegistry, Counter

if TYPE_CHECKING:
    from .context import WeakOpen
    from .find import FindRequest

_FIND_OUTCOMES = Counter(
    "message_retrieval_find", "Completed initial message searches by outcome.",
    ("outcome",), registry=None,
)
_CARD_OPENS = Counter(
    "message_retrieval_weak_opens", "Successful context opens of a served card, by its original rung.",
    ("rung",), registry=None,
)
_RUNGS = frozenset({"browse", "exact_all", "exact_any", "parts", "parts_loose", "words", "words_loose"})
_OUTCOMES = frozenset({"ok", "no_match"})
for outcome in _OUTCOMES:
    _FIND_OUTCOMES.labels(outcome=outcome)


def register(registry: CollectorRegistry) -> None:
    registry.register(_FIND_OUTCOMES)
    registry.register(_CARD_OPENS)


def record_find(request: FindRequest, payload: str) -> None:
    if request.handle is not None or not (request.query or "").strip():
        return  # Exclude browsing and paging from the initial-search denominator.
    outcome = json.loads(payload)["status"]
    if outcome in _OUTCOMES:
        _FIND_OUTCOMES.labels(outcome=outcome).inc()


def record_open(label: WeakOpen) -> None:
    # Only consume the bounded served rung. No query, filters, ref, rank,
    # anchor, or handle is logged, persisted, or exported by these metrics.
    rung = label.card.rung
    _CARD_OPENS.labels(rung=rung if rung in _RUNGS else "other").inc()
