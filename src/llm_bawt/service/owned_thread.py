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
    try:
        return await asyncio.shield(fut), False
    except asyncio.CancelledError:
        pass
    # Cancelled while the worker runs. AnyIO (Starlette disconnects) cancels
    # level-triggered, re-cancelling every await, so drain inside a shielded
    # scope: one wait, no retry spin. A further native task.cancel() can
    # still land; each one costs one more await, not a loop.
    import anyio

    with anyio.CancelScope(shield=True):
        while True:
            try:
                await asyncio.shield(fut)
                break
            except asyncio.CancelledError:
                if fut.done():
                    break
    task = asyncio.current_task()
    if task is not None:
        # Clear the absorbed requests; the caller re-raises explicitly.
        while task.uncancel():
            pass
    if fut.cancelled() or fut.exception() is not None:
        # The worker failed: it wrote nothing to own, so the cancellation wins.
        raise asyncio.CancelledError()
    return fut.result(), True


async def run_in_thread(fn: Callable[..., Any], /, *args, **kwargs) -> Any:
    """Like ``asyncio.to_thread``, but a cancellation waits for ``fn`` first."""
    result, cancelled = await finish_in_thread(fn, *args, **kwargs)
    if cancelled:
        raise asyncio.CancelledError()
    return result
