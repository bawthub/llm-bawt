"""Observe a real PostgreSQL lock wait from a second session (TASK-1451).

PostgreSQL snapshots ``pg_stat_activity`` once per transaction: "information
about the current queries of all sessions is collected when any such
information is first requested within a transaction, and the same information
will be displayed throughout the transaction." A poll loop that reuses one
transaction therefore re-reads its first view forever. When the first poll
lands before the worker starts waiting (a slow, shared CI runner), the loop
never sees the wait, the worker's ``lock_timeout`` ends it, and the test fails
"did not wait". Each poll here runs in its own transaction instead.
"""

from __future__ import annotations

import time
from concurrent.futures import Future
from threading import Thread
from typing import Callable

import pytest
from sqlalchemy import text

POLL_INTERVAL_S = 0.02

_WAITING_ON_HOLDER = text("""
    SELECT EXISTS (
        SELECT 1 FROM pg_stat_activity
        WHERE datname = current_database() AND wait_event_type = 'Lock'
          AND :holder_pid = ANY(pg_blocking_pids(pid))
    )
""")


def _finished(worker: Thread | Future) -> bool:
    return worker.done() if isinstance(worker, Future) else not worker.is_alive()


def _outcome(worker: Thread | Future, detail: Callable[[], object] | None) -> str:
    if isinstance(worker, Future) and worker.done():
        error = worker.exception()
        return repr(error) if error is not None else f"returned {worker.result()!r}"
    return repr(detail()) if detail is not None else "no detail"


def _pause() -> None:
    """Between polls; tests replace it to order a waiter after the first poll."""
    time.sleep(POLL_INTERVAL_S)


def wait_blocked_by(
    engine,
    holder_pid: int,
    worker: Thread | Future,
    *,
    timeout: float = 5.0,
    detail: Callable[[], object] | None = None,
) -> None:
    """Return once some session waits on a lock held by ``holder_pid``.

    Fails at once if ``worker`` finishes first: it got past the holder without
    waiting, or gave up (``lock_timeout``). ``detail`` adds the worker's own
    record to that message; a Future reports its result or exception.
    """
    deadline = time.monotonic() + timeout
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as observer:
        while True:
            if observer.execute(_WAITING_ON_HOLDER, {"holder_pid": holder_pid}).scalar_one():
                assert not _finished(worker), "worker finished while PostgreSQL reported a lock wait"
                return
            if _finished(worker):
                pytest.fail(f"worker finished without waiting on the holder's lock: {_outcome(worker, detail)}")
            if time.monotonic() >= deadline:
                pytest.fail(f"no session waited on the holder's lock within {timeout}s")
            _pause()
