"""Independent shutdown phases for the app lifespan (TASK-1311).

Shutdown is a sequence of unrelated teardowns (scheduler, dispatcher, task
cancels, Redis, service). Before this, one raising phase aborted the rest:
uvicorn logged "Application shutdown failed. Exiting." and the service's
own ``shutdown()``, the task cancels and the Redis close never ran. Each
phase now fails on its own, with a traceback, and the sequence always
reaches its terminal log line.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

logger = logging.getLogger(__name__)

ShutdownPhase = tuple[str, Callable[[], Awaitable[Any]]]


async def run_shutdown_phases(
    phases: Iterable[ShutdownPhase], *, log: Any = logger
) -> list[str]:
    """Run every phase in order; return the names of the ones that failed.

    A failing phase is logged with its traceback and never skips a later
    phase. ``CancelledError`` is also contained: a cancelled phase must not
    abort the teardown of everything after it.
    """
    failed: list[str] = []
    for name, phase in phases:
        try:
            await phase()
        except asyncio.CancelledError:
            failed.append(name)
            log.warning("lifespan shutdown: phase %r was cancelled; continuing", name)
        except Exception:
            failed.append(name)
            log.exception("lifespan shutdown: phase %r failed; continuing", name)
    return failed


async def cancel_tasks(owner: Any, attributes: Iterable[str]) -> None:
    """Cancel each task held in ``owner.<attribute>`` and wait for it to unwind."""
    for attribute in attributes:
        task = getattr(owner, attribute, None)
        if not task:
            continue
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
