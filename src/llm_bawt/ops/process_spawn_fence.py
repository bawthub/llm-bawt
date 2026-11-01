"""Spawn-side half of the receipt-volume self-config fence (TASK-1169).

The future isolated self-config action uses the same path exclusively. A
spawn takes a shared, nonblocking flock before a plain ledger read, and keeps
it until the child is registered. Never take the worker-slot advisory lock
here: claim-side ordering is operation row -> worker slots, whereas the
future drain owner retains its exclusive flock while claiming its hold.
"""
from __future__ import annotations

import fcntl
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .executor_types import ExecutorError, SpawnDeferred

PROCESS_SPAWN_LOCK = "process-spawn.lock"


class SpawnFenceDeadlineExceeded(ExecutorError):
    """Exclusive acquisition budget ended; not evidence that a holder is dead.

    This outcome never authorizes clearing a durable hold or terminalizing its
    owner. The acquisition deadline is not a lease on mutation authority.
    """


def _check_acquire_deadline(deadline: datetime, clock: Callable[[], datetime]) -> None:
    now = clock()
    if not isinstance(deadline, datetime) or deadline.utcoffset() is None:
        raise ValueError("exclusive spawn fence requires an aware acquisition deadline")
    if not isinstance(now, datetime) or now.utcoffset() is None:
        raise ValueError("exclusive spawn fence requires an aware clock")
    if now >= deadline:
        raise SpawnFenceDeadlineExceeded("exclusive process spawn fence acquisition deadline exceeded")


@contextmanager
def exclusive_process_spawn_guard(receipt_root: str | Path, *, acquire_deadline: datetime,
                                  clock: Callable[[], datetime] | None = None):
    """One nonblocking acquire; retry with the SAME absolute deadline.

    Busy/unavailable -> SpawnDeferred; expired -> SpawnFenceDeadlineExceeded.
    Callers schedule retries through their existing driver, not a sleeping loop
    under a database transaction. A future durable caller must persist the
    deadline before retrying and reuse it after restarts. No caller is enabled
    by this helper, which neither takes a ledger hold nor recovers an owner.

    Success retains EX through the caller's critical section. The deadline
    bounds acquisition ONLY: expiry does not revoke an acquired guard, prove
    that Docker stopped mutating, or permit clearing a durable hold.
    """
    clock = clock if clock is not None else lambda: datetime.now(timezone.utc)
    _check_acquire_deadline(acquire_deadline, clock)
    try:
        lock = (Path(receipt_root) / PROCESS_SPAWN_LOCK).open("a")
    except OSError as exc:
        _check_acquire_deadline(acquire_deadline, clock)
        raise SpawnDeferred("exclusive process spawn fence unavailable") from exc
    with lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            _check_acquire_deadline(acquire_deadline, clock)
            raise SpawnDeferred("exclusive process spawn fence busy or unavailable") from exc
        # Budget may have expired during file/lock I/O. Never enter the caller's
        # drain/claim section after it, and close the lock on every error.
        _check_acquire_deadline(acquire_deadline, clock)
        yield


@contextmanager
def process_spawn_guard(receipt_root: str | Path, hold_reader: Callable[[], bool] | None):
    """Defer, without a spawn attempt, if the lock/ledger cannot prove open."""
    try:
        lock = (Path(receipt_root) / PROCESS_SPAWN_LOCK).open("a")
    except OSError as exc:
        raise SpawnDeferred("process spawn fence unavailable") from exc
    with lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError as exc:
            raise SpawnDeferred("process spawn fence busy or unavailable") from exc
        if hold_reader is None:
            raise SpawnDeferred("process spawn hold reader unavailable")
        try:
            held = hold_reader()
        except Exception as exc:
            raise SpawnDeferred("process spawn hold read unavailable") from exc
        if held is not False:
            raise SpawnDeferred("process spawn hold active or unknown")
        yield
