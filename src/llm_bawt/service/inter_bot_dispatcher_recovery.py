"""The dispatcher's startup recovery check (TASK-1232).

The dispatcher holds the singleton leadership lock, so it is the one place
that can run recovery exactly once per cluster. Recovery runs once per
process, at startup, never on a timer:

- ``run_once`` starts the check at the first ``run()`` entry, after the
  initial ``recover_expired`` (recovery steps defer claimed deliveries to it,
  so the pass must see what it released). ``run()`` is re-entered after a
  crash under the same leadership; that re-entry never starts a second check.
- A check is never cancelled for being slow. Its database work may run in a
  worker thread that cancelling the await would not stop, and the sweep's
  lock would then be released while that write is still in flight. Dispatch
  waits a bounded time for it (less if shutdown starts), then continues while
  it runs on. Only ``stop()`` (shutdown) cancels it.
- When the check ends, ``on_done`` runs (the dispatcher wakes, so deliveries
  to a bot whose stuck turn was just settled drain without waiting a sweep).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class StartupCheck(Protocol):
    async def run_at_startup(self) -> Any: ...


class StartupRecoveryRunner:
    # How long dispatch waits for the startup check before it starts anyway.
    WAIT_SECONDS = 30.0

    def __init__(self, check: StartupCheck, *, on_done: Callable[[], None] | None = None):
        self.check = check
        self._on_done = on_done
        self._task: asyncio.Task | None = None

    @property
    def started(self) -> bool:
        return self._task is not None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def run_once(self, stop: asyncio.Event | None = None) -> None:
        """Start the check unless this process already did, and wait for it up to the bound.

        ``stop`` is the dispatcher's shutdown event. The wait ends as soon as it
        is set: shutdown cancels the check right after, so waiting on it longer
        would only delay ``stop()`` by up to the bound.
        """
        if self._task is not None:
            return  # startup only: a crash restart of run() never reruns it
        task = asyncio.create_task(self.check.run_at_startup(), name="inter-bot-startup-recovery")
        task.add_done_callback(self._check_done)
        self._task = task
        stop_waiter = asyncio.create_task(stop.wait()) if stop is not None else None
        try:
            done, _ = await asyncio.wait(
                {task} if stop_waiter is None else {task, stop_waiter},
                timeout=self.WAIT_SECONDS,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            if stop_waiter is not None and not stop_waiter.done():
                stop_waiter.cancel()
                await asyncio.gather(stop_waiter, return_exceptions=True)
        if task not in done and stop is not None and stop.is_set():
            return  # shutdown: stop() cancels the check
        if task not in done:
            # Expected when a bridge restarts with the app: its probes may wait
            # out their own timeout, which is longer than this bound.
            logger.warning(
                "startup recovery check still running after %.0fs; dispatch continues "
                "and the check is left to finish",
                self.WAIT_SECONDS,
            )

    async def stop(self) -> None:
        """Cancel a running check (shutdown only) and wait for it to unwind.

        A cancelled check may leave a worker-thread write in flight after its
        await unwinds. That is safe only because the check is idempotent by
        contract: every step re-checks under its own fence.
        """
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    def _check_done(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        if task.exception():
            logger.error("startup recovery check failed; dispatch continues", exc_info=task.exception())
        if self._on_done is not None:
            self._on_done()
