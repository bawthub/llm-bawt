"""Run blocking request setup in a worker without losing its result (TASK-2541).

``asyncio.to_thread`` frees the event loop, but cancelling the awaiting task
(client disconnect, nonstream cancellation) abandons a worker that keeps
running: a lease or reservation it commits afterwards has no owner. These
helpers keep the old synchronous contract instead: the call always finishes
and its result reaches the caller, and cancellation is re-raised only after
that, so the request's normal cleanup sees exactly what was written.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any


async def finish_in_thread(fn: Callable[..., Any], /, *args, **kwargs) -> tuple[Any, bool]:
    """Run ``fn`` in a worker to completion; return ``(result, cancelled)``.

    Cancellations that arrive while the worker runs are absorbed until it
    finishes. A worker exception propagates; if the caller was also
    cancelled, the cancellation wins (the worker wrote nothing to own).
    """
    fut = asyncio.ensure_future(asyncio.to_thread(fn, *args, **kwargs))
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(fut)
            break
        except asyncio.CancelledError:
            cancelled = True
            if fut.done():
                if fut.cancelled() or fut.exception() is not None:
                    raise
                result = fut.result()
                break
        except BaseException:
            if cancelled:
                raise asyncio.CancelledError() from None
            raise
    if cancelled:
        task = asyncio.current_task()
        if task is not None:
            # Clear the absorbed requests; the caller re-raises explicitly.
            while task.uncancel():
                pass
    return result, cancelled


async def run_in_thread(fn: Callable[..., Any], /, *args, **kwargs) -> Any:
    """Like ``asyncio.to_thread``, but a cancellation waits for ``fn`` first."""
    result, cancelled = await finish_in_thread(fn, *args, **kwargs)
    if cancelled:
        raise asyncio.CancelledError()
    return result
