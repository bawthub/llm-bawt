"""Run a coroutine on an event loop whose clock jumps instead of waiting (TASK-1457).

A test that asserts "this returned within its 0.8s budget" by reading the wall
clock passes on a fast machine and fails on a slow, shared CI runner, with no
regression. On this loop ``loop.time()`` is virtual: whenever every task is
blocked on a timer, the clock jumps straight to the next timer. Timeouts, sleeps
and budgets keep their real ordering, they take no real time, and the elapsed
time a test reads from ``loop.time()`` is exact, so it can assert the budget
itself (``elapsed == BUDGET``) instead of a margin around it.

Code under test that reads ``time.monotonic()`` / ``time.time()`` itself (to
compute its own deadlines) must read the same clock: pass its module as
``follow=[module]`` and, for the run, that module's ``time`` reads the loop's
virtual clock (wall time = the real wall time at start + virtual elapsed).

Only for code that waits on asyncio timers and other tasks. Real I/O or a real
thread could finish while the clock jumps past it, so the loop refuses
``run_in_executor`` (and so ``asyncio.to_thread``) and fails rather than block
when nothing is scheduled.
"""

from __future__ import annotations

import asyncio
import selectors
import time as real_time
from types import ModuleType
from typing import Any, Awaitable, Iterable, TypeVar

T = TypeVar("T")


class _JumpingSelector:
    """Polls without blocking; an idle wait advances the loop's clock instead."""

    def __init__(self, real: selectors.BaseSelector, loop: "VirtualClockLoop") -> None:
        self._real = real
        self._loop = loop

    def select(self, timeout: float | None = None):
        events = self._real.select(0)
        if events or timeout == 0:
            return events
        if timeout is None:
            raise RuntimeError("virtual clock: every task is waiting and no timer is scheduled (deadlock)")
        self._loop.now += timeout
        return events

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class VirtualClockLoop(asyncio.SelectorEventLoop):
    def __init__(self) -> None:
        super().__init__()
        self.now = 0.0
        self._selector = _JumpingSelector(self._selector, self)

    def time(self) -> float:
        return self.now

    def run_in_executor(self, executor, func, *args):
        raise RuntimeError("virtual clock: a real thread would race the clock jump")


class _FollowingTime:
    """A module's ``time`` that reads the loop's virtual clock."""

    def __init__(self, loop: VirtualClockLoop) -> None:
        self._loop = loop
        self._wall_start = real_time.time()

    def monotonic(self) -> float:
        return self._loop.now

    perf_counter = monotonic

    def time(self) -> float:
        return self._wall_start + self._loop.now

    def sleep(self, seconds: float) -> None:
        raise RuntimeError("virtual clock: a blocking time.sleep would stall the loop")

    def __getattr__(self, name: str) -> Any:
        return getattr(real_time, name)


def run(main: Awaitable[T], *, follow: Iterable[ModuleType] = ()) -> T:
    """``asyncio.run`` on a :class:`VirtualClockLoop`; each ``follow`` module's
    ``time`` reads the virtual clock until the run ends."""
    loop = VirtualClockLoop()
    modules = list(follow)
    originals = [module.time for module in modules]
    for module, original in zip(modules, originals):
        if original is not real_time:
            loop.close()
            raise RuntimeError(f"virtual clock: {module.__name__}.time is not the time module")
    try:
        for module in modules:
            module.time = _FollowingTime(loop)
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(main)
    finally:
        try:
            tasks = asyncio.all_tasks(loop)
            for task in tasks:
                task.cancel()
            if tasks:
                loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            for module, original in zip(modules, originals):
                module.time = original
            asyncio.set_event_loop(None)
            loop.close()
