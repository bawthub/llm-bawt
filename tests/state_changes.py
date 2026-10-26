"""Wait for a fake's state without polling (TASK-1457).

A test that polls a condition with ``asyncio.sleep(0.005)`` takes as long as
the code under test does in real time, and fails on a slow runner once its
poll budget runs out. Here the fakes call :meth:`StateChanges.poke` whenever
they are called (from the loop or from a worker thread), and the test waits
with :meth:`StateChanges.until`, which re-checks its condition at each poke
and at nothing else. On ``tests/virtual_clock.run_timing`` the code under test
runs at its production intervals while the clock jumps between them.

The test owns the guarantee that every state it waits on is followed by a
poke: name the fake call that comes after it where the wait is written.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable


class StateChanges:
    """Pokes from any thread; one waiter at a time, on the loop it was made on."""

    def __init__(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._changed = asyncio.Event()

    def poke(self) -> None:
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            self._changed.set()
            return
        with contextlib.suppress(RuntimeError):  # the loop already closed
            self._loop.call_soon_threadsafe(self._changed.set)

    async def until(self, predicate: Callable[[], bool], *, within: float) -> None:
        """Return once ``predicate()`` holds; fail if it doesn't within ``within``
        loop seconds (virtual seconds on the virtual clock)."""
        async def changed() -> None:
            while True:
                self._changed.clear()  # before the check: a poke after it is kept
                if predicate():
                    return
                await self._changed.wait()

        try:
            await asyncio.wait_for(changed(), within)
        except TimeoutError:
            raise AssertionError(f"condition not reached within {within:g}s") from None
