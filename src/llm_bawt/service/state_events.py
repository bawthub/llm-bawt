"""Tell open UIs when shared state changed, instead of having them poll (TASK-2109).

One closed browser event, ``{"_type": "state_changed", "topics": [...]}``, goes
out on the global config stream (the one ``task_changed`` uses). It carries
topic names only, never content: the UI refetches what it shows.

Most of this state is written outside the app process (the ops-reconciler
container, the MCP subprocess, the local video server), so the app cannot
publish at each write. Instead those writers queue a commit notify on Al's
commit-wake channel (:mod:`.commit_wake`), and one app loop, the
:class:`StateWatcher`, re-reads a small fingerprint per topic on each wake and
publishes only the topics whose fingerprint moved. Writers the watcher cannot
fingerprint cheaply (maintenance job runs, 200k+ rows) ``mark`` their topic
directly. The watcher's backstop timer only repairs a missed notification.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Callable, Iterable

from sqlalchemy import text as sa_text

from . import commit_wake

log = logging.getLogger(__name__)

EVENT_TYPE = "state_changed"

OPS_JOBS = "ops_jobs"
OPS_RELEASES = "ops_releases"
GPU = "gpu"
QUEUE_HOLD = "queue_hold"
APPROVAL_POLICIES = "approval_policies"
MEDIA = "media"
SCHEDULES = "schedules"
MAINTENANCE_JOBS = "maintenance_jobs"

#: Every topic a browser may receive. Closed: the client ignores anything else.
TOPICS = frozenset({OPS_JOBS, OPS_RELEASES, GPU, QUEUE_HOLD, APPROVAL_POLICIES, MEDIA, SCHEDULES,
                    MAINTENANCE_JOBS})

#: Bursts of commits inside this window produce one fingerprint pass.
MIN_INTERVAL_SECONDS = 1.0

# One read per topic; each returns a row tuple that changes exactly when the
# UI's view of that topic does. All tables are small (hundreds of rows).
_FINGERPRINTS: dict[str, str] = {
    OPS_JOBS: (
        "SELECT count(*), max(finished_at),"
        " md5(coalesce(string_agg(id || ':' || state, ',' ORDER BY id)"
        "  FILTER (WHERE finished_at IS NULL), ''))"
        " FROM ops_jobs"
    ),
    OPS_RELEASES: "SELECT count(*), coalesce(sum(event_seq), 0), max(updated_at) FROM ops_release_runs",
    GPU: (
        "SELECT (SELECT md5(coalesce(string_agg(s::text, ','), '')) FROM gpu_handoff_state s),"
        " (SELECT count(*) FROM gpu_handoff_video_jobs),"
        " (SELECT md5(coalesce(string_agg(c::text, ','), '')) FROM gpu_video_calibrations c)"
    ),
    APPROVAL_POLICIES: "SELECT count(*), coalesce(sum(version), 0), max(updated_at) FROM tool_approval_policies",
    MEDIA: (
        "SELECT count(*), max(completed_at),"
        " md5(coalesce(string_agg(id || ':' || status || ':' || coalesce(progress::text, ''), ','"
        "  ORDER BY id) FILTER (WHERE completed_at IS NULL), ''))"
        " FROM media_generations"
    ),
    SCHEDULES: (
        "SELECT count(*), md5(coalesce(string_agg(run_id || ':' || state, ',' ORDER BY run_id)"
        "  FILTER (WHERE state IN ('pending', 'queued', 'running')), ''))"
        " FROM prompt_occurrences"
    ),
}


def state_changed_event(topics: Iterable[str]) -> dict | None:
    """The closed event for ``topics`` (unknown ones dropped); None when empty."""
    known = sorted(set(topics) & TOPICS)
    return {"_type": EVENT_TYPE, "topics": known} if known else None


async def publish_state_changed(service: Any, topics: Iterable[str]) -> bool:
    """Publish on the global config stream. Never raises; False when not sent."""
    event = state_changed_event(topics)
    subscriber = getattr(service, "_redis_subscriber", None)
    if event is None or subscriber is None:
        return False
    try:
        return await subscriber.publish_global_config_event(event) is not None
    except Exception:  # noqa: BLE001 - UIs reconcile on reconnect and on their slow timer
        log.warning("state_changed publish failed for %s", event["topics"], exc_info=True)
        return False


def read_fingerprints(engine: Any, extra: dict[str, Callable[[Any], Any]] | None = None) -> dict[str, Any]:
    """Each topic's current fingerprint; a topic whose read fails is omitted."""
    result: dict[str, Any] = {}
    with engine.connect() as conn:
        for topic, sql in _FINGERPRINTS.items():
            try:
                with conn.begin_nested():
                    result[topic] = tuple(conn.execute(sa_text(sql)).first() or ())
            except Exception:  # noqa: BLE001 - a missing table just isn't watched
                log.debug("state fingerprint %s unavailable", topic, exc_info=True)
        for topic, read in (extra or {}).items():
            try:
                with conn.begin_nested():
                    result[topic] = read(conn)
            except Exception:  # noqa: BLE001
                log.debug("state fingerprint %s unavailable", topic, exc_info=True)
    return result


def _queue_hold_fingerprint(conn) -> tuple:
    from .user_stop_hold import live_hold_counts

    return live_hold_counts(conn)


class StateWatcher:
    """Publish ``state_changed`` for topics whose fingerprint moved since the last pass."""

    def __init__(self, service: Any, engine: Any, *, min_interval: float = MIN_INTERVAL_SECONDS):
        self.service = service
        self.engine = engine
        self.min_interval = min_interval
        self._loop: asyncio.AbstractEventLoop | None = None
        self._event: asyncio.Event | None = None
        self._marked: set[str] = set()
        self._lock = threading.Lock()
        self._last: dict[str, Any] = {}
        self.passes = 0

    # Thread-safe: called from routes' threadpool, the scheduler and other loops.
    def nudge(self) -> None:
        """Re-read the fingerprints soon (state changed without a commit notify)."""
        loop, event = self._loop, self._event
        if loop is None or event is None:
            return
        try:
            loop.call_soon_threadsafe(event.set)
        except RuntimeError:  # loop closed (shutdown)
            pass

    def mark(self, *topics: str) -> None:
        """Publish these topics on the next pass even if no fingerprint moved."""
        with self._lock:
            self._marked.update(topics)
        self.nudge()

    def _take_marked(self) -> set[str]:
        with self._lock:
            marked, self._marked = self._marked, set()
        return marked

    def _read(self) -> dict[str, Any]:
        return read_fingerprints(self.engine, {QUEUE_HOLD: _queue_hold_fingerprint})

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._event = commit_wake.subscribe(self.service) or asyncio.Event()
        try:
            self._last = await asyncio.to_thread(self._read)
        except Exception:  # noqa: BLE001 - the first pass sets the baseline instead
            log.warning("state watcher baseline read failed", exc_info=True)
        last_pass = time.monotonic()
        while True:
            await commit_wake.wait_for_wake(self._event, commit_wake.backstop(self.service, commit_wake.BACKSTOP_SECONDS))
            # Coalesce a burst of commits into one pass.
            delay = self.min_interval - (time.monotonic() - last_pass)
            if delay > 0:
                await asyncio.sleep(delay)
            self._event.clear()
            last_pass = time.monotonic()
            await self.check()

    async def check(self) -> set[str]:
        """One pass: read, diff against the last pass, publish; returns the changed topics."""
        self.passes += 1
        try:
            current = await asyncio.to_thread(self._read)
        except Exception:  # noqa: BLE001 - the next wake retries
            log.warning("state watcher read failed", exc_info=True)
            return set()
        last = self._last
        changed = {topic for topic, value in current.items() if topic in last and last[topic] != value}
        # A topic that only now became readable has no baseline: not a change.
        changed |= self._take_marked()
        self._last = {**last, **current}
        if changed and not await publish_state_changed(self.service, changed):
            # Keep them for the next pass rather than losing the change.
            with self._lock:
                self._marked |= changed
        return changed


def watcher_of(service: Any) -> StateWatcher | None:
    return getattr(service, "_state_watcher", None)


def nudge(service: Any) -> None:
    watcher = watcher_of(service)
    if watcher is not None:
        watcher.nudge()


def mark(service: Any, *topics: str) -> None:
    watcher = watcher_of(service)
    if watcher is not None:
        watcher.mark(*topics)


async def run_state_watcher(service: Any, engine: Any) -> None:
    """The app loop. PostgreSQL only: fingerprints and commit notifies need it."""
    if engine is None or engine.dialect.name != "postgresql":
        return
    watcher = StateWatcher(service, engine)
    service._state_watcher = watcher
    await watcher.run()


__all__ = ["APPROVAL_POLICIES", "EVENT_TYPE", "GPU", "MAINTENANCE_JOBS", "MEDIA", "OPS_JOBS", "OPS_RELEASES",
           "QUEUE_HOLD", "SCHEDULES", "StateWatcher", "TOPICS", "mark", "nudge", "publish_state_changed",
           "read_fingerprints", "run_state_watcher", "state_changed_event", "watcher_of"]
