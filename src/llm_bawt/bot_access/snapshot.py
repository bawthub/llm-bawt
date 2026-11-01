"""Process-local immutable policy state; request paths perform no I/O.

Only the background refresh owner may install a validated snapshot. Runtime
snapshots are event-invalidated, with no periodic refresh or implicit expiry.
Explicit finite deadlines remain supported by this primitive, but the runtime
owner does not use them as a hidden requirement for polling.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

if TYPE_CHECKING:
    from .policy import BotAccess

logger = logging.getLogger(__name__)
_REASONS = frozenset({
    "snapshot_cold", "snapshot_validation", "snapshot_busy", "snapshot_timeout",
    "snapshot_database", "snapshot_disconnected", "snapshot_revoked", "snapshot_stopped",
    "snapshot_clock",
})


def elapsed_clock() -> float:
    """Suspend-aware elapsed time, independent of NTP/wall-clock changes.

    Explicit finite deadlines must include process/host suspension. Fail closed
    on platforms without this clock rather than silently weakening that bound.
    """
    if not hasattr(time, "CLOCK_BOOTTIME"):
        raise SnapshotUnavailable("snapshot_clock")
    return time.clock_gettime(time.CLOCK_BOOTTIME)


class SnapshotUnavailable(RuntimeError):
    """Internal, bounded diagnostic code; never contains policy identities."""


@dataclass(frozen=True)
class InstalledSnapshot:
    policy: BotAccess
    revision: int
    deadline: float
    generation: int


class PolicySnapshot:
    def __init__(self, *, clock=elapsed_clock):
        self._clock = clock
        self._lock = threading.Lock()
        self._generation = 0
        self._installed: InstalledSnapshot | None = None
        self._reason = "snapshot_cold"
        self._refreshes = 0
        self._failures = 0

    def generation(self) -> int:
        with self._lock:
            return self._generation

    def install(self, policy: BotAccess, revision: int, deadline: float, generation: int) -> bool:
        """Refuse invalid/caller-bound views, late refreshes and old revisions."""
        if policy.caller is not None or not policy.tags.available:
            raise ValueError("snapshot_validation")
        with self._lock:
            if (generation != self._generation or not deadline > self._clock()
                    or (self._installed is not None
                        and revision < self._installed.revision)):
                return False
            # Reconciliation validates fresh data but is not itself revocation.
            # In-flight calls recheck this identity after ownership work yields;
            # replacing an EQUAL policy would spuriously deny sessions_get (and
            # other guarded tools) every time a healthy refresh races that await.
            # Invalidation clears _installed, so equal data cannot revive an old
            # generation. Changed policy, expiry and refresh failures still deny.
            if (self._installed is not None
                    and self._installed.deadline > self._clock()
                    and self._installed.policy == policy):
                policy = self._installed.policy
            self._installed = InstalledSnapshot(policy, revision, deadline, generation)
            self._reason = "ready"
            self._refreshes += 1
            return True

    def invalidate(self, reason: str) -> None:
        reason = reason if reason in _REASONS else "snapshot_unavailable"
        with self._lock:
            changed = self._installed is not None or self._reason != reason
            self._generation += 1
            self._installed = None
            self._reason = reason
            self._failures += 1
        # Log transitions, not ten identical outage messages per second. Never
        # log exception text, row values, SQL parameters or identities.
        if changed:
            logger.warning("Bot policy snapshot unavailable reason=%s", reason)

    def get(self):
        with self._lock:
            installed = self._installed
            if installed is None:
                raise SnapshotUnavailable(self._reason)
            if self._clock() >= installed.deadline:
                raise SnapshotUnavailable("snapshot_expired")
            return installed.policy

    def require_current(self, policy) -> None:
        if self.get() is not policy:
            raise SnapshotUnavailable("snapshot_revoked")

    def diagnostics(self) -> dict:
        with self._lock:
            installed = self._installed
            remaining = max(0.0, installed.deadline - self._clock()) if installed else 0.0
            return {
                "state": "ready" if installed and remaining else (
                    "snapshot_expired" if installed else self._reason),
                "revision": installed.revision if installed else None,
                "lease_remaining_seconds": remaining,
                "refreshes": self._refreshes,
                "failures": self._failures,
            }


_snapshots: WeakKeyDictionary = WeakKeyDictionary()
_snapshots_lock = threading.Lock()


def policy_snapshot(engine) -> PolicySnapshot:
    """Get memory state only. No lazy initialization of the database/worker."""
    with _snapshots_lock:
        snapshot = _snapshots.get(engine)
        if snapshot is None:
            snapshot = PolicySnapshot()
            _snapshots[engine] = snapshot
        return snapshot
