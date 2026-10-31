"""Lifecycle-owned, change-triggered policy refresh, never request-side I/O.

PG transactional notifications accelerate refresh across app/MCP processes.
Reconciliation catches lost hints and direct SQL. A disconnected/failed refresh
invalidates memory; a hung refresher cannot extend the five-second deadline.
"""
from __future__ import annotations

import asyncio
import logging
import select
import threading
from contextlib import asynccontextmanager, contextmanager

from sqlalchemy import text
from sqlalchemy.exc import TimeoutError as PoolTimeout

from llm_bawt.bot_policy_events import POLICY_CHANNEL
from llm_bawt.bot_tags import read_snapshot_rows
from .policy import BotAccessError, BotPolicyResolver
from .snapshot import SnapshotUnavailable, elapsed_clock, policy_snapshot

logger = logging.getLogger(__name__)
MAX_FRESHNESS_SECONDS = 5.0
RECONCILE_SECONDS = 2.0


def failure_reason(exc: Exception) -> str:
    if isinstance(exc, (BotAccessError, ValueError, TypeError)):
        return "snapshot_validation"
    if isinstance(exc, SnapshotUnavailable):
        return "snapshot_clock"
    if isinstance(exc, PoolTimeout):
        return "snapshot_busy"
    if isinstance(exc, TimeoutError):
        return "snapshot_timeout"
    # Inspect only SQLSTATE, never log driver messages/queries/row identities.
    code = getattr(getattr(exc, "orig", exc), "pgcode", None)
    if code in {"57014", "55P03"}:
        return "snapshot_timeout"
    return "snapshot_database"


class PolicyRefresher:
    def __init__(self, engine, *, clock=elapsed_clock):
        self.engine = engine
        self.snapshot = policy_snapshot(engine)
        self.clock = clock
        self._stop = threading.Event()
        self.ready = threading.Event()
        self._revision = 0
        self._thread = threading.Thread(target=self._run, name="bot-policy-refresh", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.snapshot.invalidate("snapshot_stopped")
        self._thread.join(timeout=3)

    @staticmethod
    def _drain(raw, schema: str) -> bool:
        raw.poll()
        changed = False
        while raw.notifies:
            event = raw.notifies.pop(0)
            changed = changed or event.channel == POLICY_CHANNEL and event.payload == schema
        return changed

    def refresh(self, raw=None, schema: str | None = None) -> bool:
        """Called only by the background owner (or isolated test harness)."""
        if self._stop.is_set():
            return False
        started = self.clock()
        generation = self.snapshot.generation()
        try:
            policy = BotPolicyResolver._compile(read_snapshot_rows(self.engine), None)
            # A notification racing the read may describe a commit newer than
            # that SELECT. Discard, invalidate and reread, never extend old data.
            if raw is not None and self._drain(raw, schema):
                self.snapshot.invalidate("snapshot_revoked")
                return False
            if self._stop.is_set():
                return False
            self._revision += 1
            installed = self.snapshot.install(
                policy, self._revision, started + MAX_FRESHNESS_SECONDS, generation)
            if installed:
                self.ready.set()
            return installed
        except Exception as exc:
            self.snapshot.invalidate(failure_reason(exc))
            return False

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                # LISTEN is established before the first snapshot on every
                # connection incarnation. SQLAlchemy resets AUTOCOMMIT on close.
                with self.engine.connect().execution_options(isolation_level="AUTOCOMMIT") as listener:
                    listener.exec_driver_sql(f"LISTEN {POLICY_CHANNEL}")
                    schema = listener.execute(text("SELECT current_schema()")).scalar_one()
                    raw = listener.connection.driver_connection
                    self.snapshot.invalidate("snapshot_disconnected")
                    due = self.clock()
                    while not self._stop.is_set():
                        now = self.clock()
                        if now >= due:
                            installed = self.refresh(raw, schema)
                            due = self.clock() + (RECONCILE_SECONDS if installed else 0.1)
                        timeout = min(0.25, max(0.0, due - self.clock()))
                        if select.select([raw], [], [], timeout)[0]:
                            if self._drain(raw, schema):
                                self.snapshot.invalidate("snapshot_revoked")
                                due = self.clock()
                    listener.exec_driver_sql(f"UNLISTEN {POLICY_CHANNEL}")
            except Exception as exc:
                self.snapshot.invalidate(failure_reason(exc))
                self._stop.wait(0.5)
        self.snapshot.invalidate("snapshot_stopped")


# App and threaded MCP can share a process/engine; standalone MCP has its own.
# Leases here are *only local lifecycle references*, not distributed authority.
_owners: dict = {}
_owners_lock = threading.Lock()


@contextmanager
def policy_refresh_lifecycle(engine):
    if engine is None:
        yield None
        return
    with _owners_lock:
        owner, refs = _owners.get(engine, (None, 0))
        if owner is None:
            owner = PolicyRefresher(engine)
            owner.start()
        _owners[engine] = (owner, refs + 1)
    try:
        yield owner
    finally:
        with _owners_lock:
            _, refs = _owners[engine]
            if refs == 1:
                del _owners[engine]
                owner.stop()
            else:
                _owners[engine] = (owner, refs - 1)


@asynccontextmanager
async def policy_refresh_async_lifecycle(engine):
    context = policy_refresh_lifecycle(engine)
    owner = context.__enter__()
    try:
        if owner is not None:
            await asyncio.to_thread(owner.ready.wait, 3)
        yield owner
    finally:
        await asyncio.to_thread(context.__exit__, None, None, None)
