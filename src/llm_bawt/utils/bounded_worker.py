"""Cancellation-safe off-loop execution with no unbounded work queue."""
from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor


class WorkerBusy(RuntimeError):
    """All execution slots are occupied."""


class BoundedWorker:
    """Cancelled waiters retain their slot until synchronous work actually ends."""

    busy_error = WorkerBusy

    def __init__(self, count: int, name: str):
        self._slots = threading.BoundedSemaphore(count)
        self._pool = ThreadPoolExecutor(max_workers=count, thread_name_prefix=name)

    async def run(self, function, *args):
        if not self._slots.acquire(blocking=False):
            raise self.busy_error("busy")
        try:
            future = self._pool.submit(function, *args)
        except BaseException:
            self._slots.release()
            raise
        future.add_done_callback(lambda _: self._slots.release())
        wrapped = asyncio.wrap_future(future)
        wrapped.add_done_callback(lambda f: None if f.cancelled() else f.exception())
        return await asyncio.shield(wrapped)
