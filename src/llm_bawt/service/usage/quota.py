"""Quota state of one model on one access path, from cached usage only.

``quota_state`` answers "can this bot's subscription take more work right
now?" with ``available``, ``exhausted`` or ``unknown``. It reads the usage
cache and never calls upstream, so it is safe on a discovery hot path. Unknown
is the honest default: an unreported path, a non-ok or stale snapshot, or a
window with no known percentage never reads as available.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .canonical import STATUS_OK, ProviderUsage, UsageLimit

QUOTA_AVAILABLE = "available"
QUOTA_EXHAUSTED = "exhausted"
QUOTA_UNKNOWN = "unknown"

# A snapshot older than this says nothing about the account now. The UI
# refreshes usage every few minutes, so 15 minutes covers normal polling gaps.
QUOTA_FRESH_SECONDS = 900


@dataclass(frozen=True)
class Quota:
    state: str
    checked_at: int | None = None


def limit_applies(limit: UsageLimit, model_key: str | None) -> bool:
    """Whether ``limit`` counts against ``model_key``.

    An unscoped limit covers every model. A scope entry matches the exact model
    key or one of its ``-``-separated parts, so the family ``fable`` covers
    ``claude-fable-5-1``. An unknown model only matches unscoped limits.
    """
    if limit.model_scope is None:
        return True
    key = (model_key or "").strip().lower()
    if not key:
        return False
    parts = set(key.split("-"))
    return any(scope == key or scope in parts for scope in limit.model_scope)


def quota_state(snapshot: ProviderUsage | None, model_key: str | None,
                *, now: float | None = None) -> Quota:
    """Derive the tri-state quota for ``model_key`` from one cached snapshot."""
    if snapshot is None or snapshot.status != STATUS_OK:
        return Quota(QUOTA_UNKNOWN, getattr(snapshot, "fetched_at", None))
    checked = snapshot.fetched_at
    current = time.time() if now is None else now
    if checked is None or current - checked > QUOTA_FRESH_SECONDS:
        return Quota(QUOTA_UNKNOWN, checked)
    applicable = [lim for lim in snapshot.limits if limit_applies(lim, model_key)]
    if not applicable:
        return Quota(QUOTA_UNKNOWN, checked)
    under = True
    for lim in applicable:
        if lim.used_pct is None:
            under = False
            continue
        if lim.used_pct >= 100:
            if lim.resets_at is None or lim.resets_at > current:
                return Quota(QUOTA_EXHAUSTED, checked)
            # The window reset after this reading: no longer provably
            # exhausted, but nothing has measured it under 100 either.
            under = False
    return Quota(QUOTA_AVAILABLE if under else QUOTA_UNKNOWN, checked)


__all__ = [
    "QUOTA_AVAILABLE",
    "QUOTA_EXHAUSTED",
    "QUOTA_FRESH_SECONDS",
    "QUOTA_UNKNOWN",
    "Quota",
    "limit_applies",
    "quota_state",
]
