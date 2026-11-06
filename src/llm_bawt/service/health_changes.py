"""App health change version for BawtHub's push-driven health (TASK-2105 S3).

BawtHub long-polls ``GET /v1/health/changes/{seen}`` and reruns its checks
when the version moves. Anything that can change a provider's, maintenance
job's or internal check's reported state calls :func:`publish`: credential
writes and deletes, connects and disconnects, a provider's expiry instant
(a timer armed from the last health read), an Ollama reachability flip and a
chat request that could not connect. Publishes are coalesced briefly so a
burst (several rows in one transaction) is one rerun and lands after commit.

Time-based internal checks (queue staleness, admission lag) have no write to
hook; BawtHub's 60s reconciliation run covers them.
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)

#: Coalescing window: a burst of writes is one rerun, after the writer commits.
PUBLISH_DELAY_SECONDS = 0.5


class HealthChanges:
    def __init__(self, *, delay: float = PUBLISH_DELAY_SECONDS) -> None:
        self._lock = threading.Lock()
        self._changed = threading.Condition(self._lock)
        # Wall-clock seed: a restart never reuses a version a client holds.
        self._version = time.time_ns() // 1_000_000
        self._delay = delay
        self._pending: threading.Timer | None = None
        self._expiry: tuple[int, threading.Timer] | None = None

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def publish(self, reason: str) -> None:
        """Bump the version after the coalescing window. Never raises."""
        logger.debug("health change: %s", reason)
        with self._lock:
            if self._pending is not None:
                return
            if self._delay <= 0:
                self._bump_locked()
                return
            timer = threading.Timer(self._delay, self._fire)
            timer.daemon = True
            self._pending = timer
        timer.start()

    def _fire(self) -> None:
        with self._lock:
            self._pending = None
            self._bump_locked()

    def _bump_locked(self) -> None:
        self._version += 1
        self._changed.notify_all()

    def wait_for_change(self, seen: int, timeout: float) -> int:
        with self._lock:
            self._changed.wait_for(lambda: self._version != seen, timeout)
            return self._version

    def arm_expiry(self, expires_at_ms: list[int | None], *, now_ms: int | None = None) -> None:
        """Publish at the earliest future expiry among ``expires_at_ms`` (epoch ms).

        Re-armed on every provider health read, so the timer always targets
        the next expiry the last read reported; none in the future cancels it.
        """
        now = time.time() * 1000 if now_ms is None else now_ms
        future = [int(at) for at in expires_at_ms if isinstance(at, (int, float)) and at > now]
        target = min(future) if future else None
        with self._lock:
            if self._expiry is not None and self._expiry[0] == target:
                return
            if self._expiry is not None:
                self._expiry[1].cancel()
                self._expiry = None
            if target is None:
                return
            # +1s so the provider reads as expired when BawtHub reruns.
            timer = threading.Timer((target - now) / 1000 + 1, self._expired, args=(target,))
            timer.daemon = True
            self._expiry = (target, timer)
        timer.start()

    def _expired(self, target: int) -> None:
        with self._lock:
            if self._expiry is not None and self._expiry[0] == target:
                self._expiry = None
        self.publish("provider credential expiry")


_changes = HealthChanges()


def health_changes() -> HealthChanges:
    return _changes


def publish(reason: str) -> None:
    try:
        _changes.publish(reason)
    except Exception:  # noqa: BLE001 — a health notice must never fail its writer
        logger.debug("health change publish failed", exc_info=True)
