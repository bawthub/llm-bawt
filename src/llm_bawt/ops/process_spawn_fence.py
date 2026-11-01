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
from pathlib import Path
from typing import Callable

from .executor_types import SpawnDeferred

PROCESS_SPAWN_LOCK = "process-spawn.lock"


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
