"""Leader-only recovery cadence for the inter-bot dispatcher (TASK-1232 N1/N2).

The dispatcher holds the singleton leadership lock, so it is the one place that
can run recovery exactly once per cluster. This module owns *when* recovery
runs; the hooks object owns *what* it does.

- ``on_leadership_acquired`` runs at every ``run()`` entry, after the initial
  ``recover_expired`` (recovery steps defer claimed deliveries to it, so the
  pass must see what it released). ``run()`` is re-entered after a crash under
  the same leadership, so the hook must be idempotent.
- Both passes share one single-flight slot: a pass never overlaps another pass
  started by this cadence.
- A pass is never cancelled for being slow. Its database work may run in a
  worker thread that cancelling the await would not stop, and the hook's lock
  would then be released while that write is still in flight. Dispatch waits a
  bounded time for the leadership pass (less if shutdown starts), then
  continues while it runs on.
  Only ``stop()`` (shutdown) cancels a pass.
- ``periodic`` is gated on monotonic elapsed time, not on loop iterations: the
  loop also wakes for every enqueue/announce, which must not multiply recovery.
  It runs in the background so a slow pass never stalls the delivery sweep.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class RecoveryHooks(Protocol):
    async def on_leadership_acquired(self) -> None: ...

    async def periodic(self) -> None: ...


class RecoveryCadence:
    # How long dispatch waits for the leadership pass before it starts anyway.
    LEADERSHIP_WAIT_SECONDS = 30.0

    def __init__(
        self,
        hooks: RecoveryHooks,
        *,
        interval_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.hooks = hooks
        self.interval_seconds = interval_seconds
        self._clock = clock
        self._last_periodic_at: float | None = None
        self._task: asyncio.Task | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def on_leadership(self, stop: asyncio.Event | None = None) -> None:
        """Start the leadership pass and wait for it up to the bound.

        Restarts the periodic interval from now. A pass still running from an
        earlier ``run()`` entry is the same leadership's recovery, so no second
        pass starts beside it.

        ``stop`` is the dispatcher's shutdown event. The wait ends as soon as it
        is set: shutdown cancels the pass right after, so waiting on it longer
        would only delay ``stop()`` by up to the bound.
        """
        self._last_periodic_at = self._clock()
        if self.running:
            logger.warning("recovery pass still running at run() entry; not starting another")
            return
        task = self._start(self.hooks.on_leadership_acquired(), "inter-bot-recovery-leadership")
        stop_waiter = asyncio.create_task(stop.wait()) if stop is not None else None
        try:
            done, _ = await asyncio.wait(
                {task} if stop_waiter is None else {task, stop_waiter},
                timeout=self.LEADERSHIP_WAIT_SECONDS,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            if stop_waiter is not None and not stop_waiter.done():
                stop_waiter.cancel()
                await asyncio.gather(stop_waiter, return_exceptions=True)
        if task not in done and stop is not None and stop.is_set():
            return  # shutdown: stop() cancels the pass
        if task not in done:
            logger.error(
                "recovery leadership pass still running after %.0fs; dispatch continues "
                "and the pass is left to finish",
                self.LEADERSHIP_WAIT_SECONDS,
            )

    def maybe_start_periodic(self) -> asyncio.Task | None:
        """Start one periodic pass if the interval elapsed and no pass is running."""
        if self.running:
            return None
        now = self._clock()
        if self._last_periodic_at is not None and now - self._last_periodic_at < self.interval_seconds:
            return None
        # Stamp before running: a failing pass retries next interval, not next wake.
        self._last_periodic_at = now
        return self._start(self.hooks.periodic(), "inter-bot-recovery-periodic")

    async def stop(self) -> None:
        """Cancel the running pass (shutdown only) and wait for it to unwind.

        A cancelled pass may leave a worker-thread write in flight after its
        await unwinds. That is safe only because the hooks are idempotent by
        contract (see the module docstring).
        """
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    def _start(self, pass_: Coroutine[Any, Any, None], name: str) -> asyncio.Task:
        task = asyncio.create_task(pass_, name=name)
        task.add_done_callback(self._pass_done)
        self._task = task
        return task

    @staticmethod
    def _pass_done(task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception():
            logger.error(
                "recovery pass %s failed; dispatch continues", task.get_name(),
                exc_info=task.exception(),
            )
