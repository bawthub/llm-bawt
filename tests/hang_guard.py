"""Prove a call returns on its own instead of timing how long it took (TASK-1463).

A wall-clock upper bound (``elapsed < 3``) fails on a slow shared runner with no
regression. ``finishes`` runs the call on a daemon thread and only asks whether
it returned: a call that waits on something the test holds forever (a PG lock,
a stalled collector, a deadlocked loop) never returns, so the generous guard is
only reached by a real hang, never by a slow pass.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

HANG_GUARD_S = 30.0


def finishes(fn: Callable[[], Any], *, guard: float = HANG_GUARD_S, what: str = "call") -> Any:
    """Return ``fn()``'s result, re-raise its exception, or fail if it never returns."""
    result: list[Any] = []
    error: list[BaseException] = []

    def target() -> None:
        try:
            result.append(fn())
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
            error.append(exc)

    thread = threading.Thread(target=target, name=f"hang-guard:{what}", daemon=True)
    thread.start()
    thread.join(guard)
    if thread.is_alive():
        raise AssertionError(f"{what} never returned: it waited on something that is never released")
    if error:
        raise error[0]
    return result[0]
