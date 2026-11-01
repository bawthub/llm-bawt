"""Lifecycle-owned, change-triggered policy refresh, never request-side I/O.

Load at startup/reconnect and on committed policy-change notifications only.
There is no periodic policy read or expiry that requires one. Direct SQL and
missed hints require the explicit admin refresh. Known invalidation, listener
loss and load failure fail closed until recovery. Transient load failures get
only a finite retry budget attached to their triggering event, never healthy
polling or an expiry lease.
"""
from __future__ import annotations

import asyncio
import logging
import select
import socket
import threading
from contextlib import asynccontextmanager, contextmanager

from sqlalchemy import text
from sqlalchemy.exc import TimeoutError as PoolTimeout

from llm_bawt.bot_policy_events import POLICY_CHANNEL, policy_changed
from llm_bawt.bot_tags import read_snapshot_rows
from .policy import BotAccessError, BotPolicyResolver
from .snapshot import SnapshotUnavailable, elapsed_clock, policy_snapshot

logger = logging.getLogger(__name__)

# At most two retries per startup/change/reconnect/admin trigger, only after a
# transient failed load. Exhaustion waits for another explicit trigger.
LOAD_RETRY_DELAYS = (0.1, 0.5)
TRANSIENT_LOAD_FAILURES = frozenset({"snapshot_database", "snapshot_busy", "snapshot_timeout"})


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
        self._refresh_requested = False  # only a change seen during the load
        self._load_failure: str | None = None
        self._completion = threading.Condition()
        self._completed_generation = -1
        self._result: dict = {}
        self._wake: socket.socket | None = None
        self._thread = threading.Thread(target=self._run, name="bot-policy-refresh", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.snapshot.invalidate("snapshot_stopped")
        self._complete(self.snapshot.generation(), "failed", "snapshot_stopped")
        if self._wake is not None:
            try:
                self._wake.send(b"\0")
            except OSError:  # the owner may already have closed its wake socket
                pass
        self._thread.join(timeout=3)

    @staticmethod
    def _drain(raw, schema: str) -> bool:
        raw.poll()
        changed = False
        while raw.notifies:
            event = raw.notifies.pop(0)
            changed = changed or event.channel == POLICY_CHANNEL and event.payload == schema
        return changed

    def _complete(self, generation: int, status: str, reason: str | None = None) -> None:
        with self._completion:
            if generation >= self._completed_generation:
                self._completed_generation = generation
                self._result = {"status": status}
                if reason is not None:
                    self._result["reason"] = reason
            self._completion.notify_all()

    def wait_for_refresh(self, generation: int, timeout: float) -> dict:
        """Bounded local load receipt, not an acknowledgment by other processes."""
        with self._completion:
            if not self._completion.wait_for(
                    lambda: self._completed_generation >= generation, timeout):
                return {"status": "timeout"}
            result = dict(self._result)
            if result["status"] == "installed" and self.snapshot.generation() != self._completed_generation:
                return {"status": "failed", "reason": "snapshot_revoked"}
            return result

    def refresh(self, raw=None, schema: str | None = None) -> bool:
        """Called only by the background owner (or isolated test harness)."""
        self._load_failure = None
        if self._stop.is_set():
            return False
        generation = self.snapshot.generation()
        try:
            policy = BotPolicyResolver._compile(read_snapshot_rows(self.engine), None)
            # A notification racing the read may describe a commit newer than
            # that SELECT. Discard, invalidate and reread, never extend old data.
            if raw is not None and self._drain(raw, schema):
                self.snapshot.invalidate("snapshot_revoked")
                self._refresh_requested = True
                return False
            if self._stop.is_set():
                return False
            self._revision += 1
            installed = self.snapshot.install(
                policy, self._revision, float("inf"), generation)
            if installed:
                self.ready.set()
                self._complete(generation, "installed")
            return installed
        except Exception as exc:
            self._load_failure = failure_reason(exc)
            self.snapshot.invalidate(self._load_failure)
            self._complete(generation, "failed", self._load_failure)
            return False

    def _run(self) -> None:
        wake_reader, wake_writer = socket.socketpair()
        self._wake = wake_writer
        try:
            while not self._stop.is_set():
                try:
                    # LISTEN precedes the first load on every connection
                    # incarnation, so commits during that load cannot be lost.
                    with self.engine.connect().execution_options(isolation_level="AUTOCOMMIT") as listener:
                        listener.exec_driver_sql(f"LISTEN {POLICY_CHANNEL}")
                        schema = listener.execute(text("SELECT current_schema()")).scalar_one()
                        raw = listener.connection.driver_connection
                        self.snapshot.invalidate("snapshot_disconnected")
                        refresh_requested = True
                        retries = iter(LOAD_RETRY_DELAYS)
                        while not self._stop.is_set():
                            if refresh_requested:
                                self._refresh_requested = False
                                self.refresh(raw, schema)
                                refresh_requested = self._refresh_requested
                                if refresh_requested:
                                    retries = iter(LOAD_RETRY_DELAYS)
                                    continue
                                if self._load_failure in TRANSIENT_LOAD_FAILURES:
                                    delay = next(retries, None)
                                    if delay is not None:
                                        if self._stop.wait(delay):
                                            break
                                        refresh_requested = True
                                        continue
                            # No timer or policy polling. The local socket only
                            # wakes shutdown; PostgreSQL wakes policy changes or
                            # connection loss.
                            readable = select.select([raw, wake_reader], [], [])[0]
                            if wake_reader in readable:
                                break
                            if raw in readable and self._drain(raw, schema):
                                self.snapshot.invalidate("snapshot_revoked")
                                refresh_requested = True
                                retries = iter(LOAD_RETRY_DELAYS)
                        listener.exec_driver_sql(f"UNLISTEN {POLICY_CHANNEL}")
                except Exception as exc:
                    reason = failure_reason(exc)
                    self.snapshot.invalidate(reason)
                    self._complete(self.snapshot.generation(), "failed", reason)
                    # Connection recovery, not periodic healthy policy reads.
                    self._stop.wait(0.5)
        finally:
            self._wake = None
            wake_reader.close()
            wake_writer.close()
            self.snapshot.invalidate("snapshot_stopped")
            self._complete(self.snapshot.generation(), "failed", "snapshot_stopped")


def request_policy_refresh(engine, *, timeout: float = 3.0) -> dict:
    """Notify all listeners and report this process's bounded load outcome only.

    No policy SELECT/compile runs in this admin request. Remote consumers are
    requested, not acknowledged; a missing local lifecycle is reported as such.
    """
    if engine is None:
        raise SnapshotUnavailable("snapshot_database")
    snapshot = policy_snapshot(engine)
    snapshot.invalidate("snapshot_revoked")
    generation = snapshot.generation()
    with _owners_lock:
        owner, _refs = _owners.get(engine, (None, 0))
    try:
        with engine.begin() as conn:
            policy_changed(conn)
    except Exception as exc:
        reason = failure_reason(exc)
        snapshot.invalidate(reason)
        raise SnapshotUnavailable(reason) from None
    if owner is None:
        return {"status": "unavailable", "reason": "snapshot_cold"}
    return owner.wait_for_refresh(generation, timeout)


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
