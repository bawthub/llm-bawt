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

:class:`TimingLoop` (below) is the variant for code that does use threads,
subprocesses or real sockets: it follows real time while such work is
outstanding instead of refusing it.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import selectors
import threading
import time as real_time
from types import ModuleType
from typing import Any, Awaitable, Callable, Iterable, Iterator, TypeVar
import weakref

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


# ---------------------------------------------------------------------------
# TimingLoop: virtual time for the whole unit suite (TASK-1457 D1/D2).
#
# VirtualClockLoop above is the strict variant for a test that asserts exact
# virtual time: it refuses threads outright. TimingLoop is the one every async
# unit test can run on. Each idle wait is in exactly one of two modes:
#   JUMP:   nothing real is outstanding, so the clock jumps to the next timer
#           and no real time passes.
#   FOLLOW: something real is outstanding (an executor worker, a subprocess,
#           a watched fd other than the loop's own). select blocks in real time
#           for the timer delta and the clock advances by what was spent, plus
#           the real time callbacks ran between selects, so a deadline can't
#           fire ahead of real work it is legitimately waiting on, nor stand
#           still while a busy external fd keeps select from ever blocking.
# step_cost (D2) charges each scheduler turn a fixed slice of virtual time, so a
# test that only passes because the code "got there in time" fails every run.
# Two ideas come from looptime (github.com/nolar/looptime, evaluated and not
# adopted: it counts executor Futures, not workers, and jumps over external
# I/O): integer ticks, and a sub-tick timeout advancing one tick.
# ---------------------------------------------------------------------------

_REAL_MONOTONIC = real_time.monotonic
_REAL_TIME = real_time.time
_REAL_PERF_COUNTER = real_time.perf_counter
_REAL_WRAP_FUTURE = asyncio.wrap_future
# A power of two (~0.95us), so every loop.time() reading, ticks / 2**20, is
# exact in binary and so is the difference of two readings. With decimal
# microseconds the 400s wait from a CI start of 124.770643s read back as
# 399.99999999999994 and failed `elapsed >= 400`; now a wait reads back as at
# least what was asked, and a test bounds elapsed loop time with no tolerance.
TICKS_PER_SECOND = 1 << 20
IDLE_LIMIT_S = 30.0
Observer = Callable[[str, float], None]


def _ticks(seconds: float) -> int:
    return round(seconds * TICKS_PER_SECOND)


class TimingLoop(asyncio.SelectorEventLoop):
    """JUMP/FOLLOW virtual clock with an optional per-turn cost."""

    def __init__(self, *, step_cost: float = 0.0, idle_limit: float = IDLE_LIMIT_S,
                 observer: Observer | None = None) -> None:
        super().__init__()
        # Start at real monotonic time, not 0: code that caches a reading
        # (a TTL, a log-once window) across loops or outside one then sees a
        # clock that only moves forward, whichever loop ran before.
        self._now = _ticks(_REAL_MONOTONIC())
        self._step = _ticks(step_cost)
        self._idle_limit = idle_limit
        self.observer = observer
        self.wall_offset = _REAL_TIME() - self.time()
        self._workers = 0
        self._workers_lock = threading.Lock()
        self._subprocesses: weakref.WeakSet[Any] = weakref.WeakSet()
        # The self-wakeup socket and anything else asyncio registers while
        # constructing the loop are the loop's own, never "real work".
        self._own_fds = frozenset(key.fd for key in self._selector.get_map().values())
        self._real_select = self._selector.select
        self._selector.select = self._select  # type: ignore[method-assign]
        self._accounted_at = _REAL_MONOTONIC()

    def time(self) -> float:
        return self._now / TICKS_PER_SECOND

    def _advance(self, seconds: float) -> None:
        # A positive wait shorter than one tick still moves the clock, or the
        # loop would spin on a timer it can never reach (looptime issue #35).
        self._now += max(1, _ticks(seconds)) if seconds > 0 else 0

    def advance(self, seconds: float) -> None:
        """Move the clock forward now, at no real cost. A test uses it to take
        a deadline past real work it holds blocked (FOLLOW would otherwise
        wait that long in real time); timers due by then fire on the next
        scheduler turn."""
        if seconds < 0:
            raise ValueError("virtual clock: time only moves forward")
        self._advance(seconds)

    def real_work(self) -> str | None:
        """What real work is outstanding, if any: the FOLLOW cause."""
        if self._workers:
            return "executor"
        if any(transport.get_returncode() is None for transport in list(self._subprocesses)):
            return "subprocess"
        if any(key.fd not in self._own_fds for key in self._selector.get_map().values()):
            return "fd"
        return None

    def _follow_real_time(self) -> float:
        """Advance by the real time since the last accounting point, whole
        ticks only (the remainder carries over), and return it."""
        ticks = _ticks(_REAL_MONOTONIC() - self._accounted_at)
        self._now += ticks
        self._accounted_at += ticks / TICKS_PER_SECOND
        return ticks / TICKS_PER_SECOND

    def _select(self, timeout: float | None = None):
        cause = self.real_work()
        if cause is None:
            return self._jump(timeout)
        # FOLLOW counts every real second while real work is outstanding,
        # including the time callbacks ran between selects: an external fd
        # that is always ready (a writable socket) keeps select from ever
        # blocking, and the clock must still keep pace with the real actor.
        busy = self._follow_real_time()
        remaining = None if timeout is None else max(0.0, timeout - busy)
        events = self._real_select(0)
        if not events and remaining != 0:
            # Only a real actor (a thread calling call_soon_threadsafe, a
            # peer writing) can wake the loop before the timer is due.
            events = self._real_select(remaining if remaining is not None else self._idle_limit)
        spent = self._follow_real_time()
        if not events and remaining and spent < remaining:
            self._advance(remaining - spent)  # select returned a hair early
        if self.observer is not None and busy + spent > 0:
            self.observer(cause, busy + spent)
        return events

    def _jump(self, timeout: float | None):
        # Real time spent with no real work outstanding never counts.
        self._accounted_at = _REAL_MONOTONIC()
        events = self._real_select(0)
        if events or timeout == 0:
            return events
        if timeout is not None:
            self._advance(timeout)  # JUMP
            return events
        # Nothing scheduled at all: only a real actor can wake the loop.
        started = _REAL_MONOTONIC()
        events = self._real_select(self._idle_limit)
        if not events:
            raise RuntimeError(
                f"virtual clock: every task waited {self._idle_limit:g}s of real time "
                "with no timer scheduled and no real work outstanding (deadlock)")
        self._advance(_REAL_MONOTONIC() - started)
        return events

    def _run_once(self) -> None:
        super()._run_once()  # type: ignore[misc]
        self._now += self._step  # D2: every scheduler turn costs step_cost

    def run_in_executor(self, executor, func, *args):
        # Count the WORKER, not the asyncio Future: wait_for cancelling the
        # Future reports it done while the thread is still running.
        self._check_closed()
        if executor is None:
            executor = self._default_executor
            self._check_default_executor()
            if executor is None:
                executor = concurrent.futures.ThreadPoolExecutor(thread_name_prefix="asyncio")
                self._default_executor = executor

        inner = executor.submit(func, *args)
        wrapped = _REAL_WRAP_FUTURE(inner, loop=self)
        self.track(inner)
        return wrapped

    def track(self, future: concurrent.futures.Future) -> None:
        """Count a worker as real work until its outcome is on the loop.

        A concurrent future is done only when its function returned or it was
        cancelled before starting, so this counts the worker, not an asyncio
        wrapper (``wait_for`` cancelling the wrapper leaves the thread running).
        Call it after ``wrap_future``: done callbacks run in the order they
        were added, so the release is queued on the loop behind the result.
        Released any earlier, the loop could see no real work and nothing
        ready, and jump to the next timer before the result arrives."""
        with self._workers_lock:
            self._workers += 1
        future.add_done_callback(lambda _: self._release_on_loop())

    async def shutdown_default_executor(self, timeout=None):
        # asyncio joins the pool on a plain thread of its own; count it, or
        # the clock jumps past ``timeout`` and leaves the pool unjoined.
        if self._default_executor is not None:
            with self._workers_lock:
                self._workers += 1  # released by _do_shutdown
        await super().shutdown_default_executor(timeout)

    def _do_shutdown(self, future):
        try:
            super()._do_shutdown(future)  # type: ignore[misc]
        finally:
            self._release_on_loop()  # queued behind the result it just sent

    def _release_on_loop(self) -> None:
        """Release one worker on the loop, after whatever the worker's thread
        already queued there (its result)."""
        try:
            self.call_soon_threadsafe(self._release_worker)
        except RuntimeError:  # the loop is closed: nothing left to wake
            self._release_worker()

    def _release_worker(self) -> None:
        with self._workers_lock:
            self._workers -= 1
        if not self.is_closed():
            with contextlib.suppress(Exception):
                self._write_to_self()

    async def _make_subprocess_transport(self, *args, **kwargs):
        transport = await super()._make_subprocess_transport(*args, **kwargs)  # type: ignore[misc]
        self._subprocesses.add(transport)
        return transport


class TimingLoopPolicy(asyncio.DefaultEventLoopPolicy):
    """Every loop asyncio.run, pytest-asyncio or anyio creates is a TimingLoop."""

    def __init__(self, *, step_cost: float = 0.0, observer: Observer | None = None) -> None:
        super().__init__()
        self.step_cost = step_cost
        self.observer = observer

    def new_event_loop(self) -> TimingLoop:
        return TimingLoop(step_cost=self.step_cost, observer=self.observer)


def run_timing(main: Awaitable[T]) -> T:
    """``asyncio.run`` on a :class:`TimingLoop`, for a test that holds real
    work blocked and takes a deadline past it with ``loop.advance()``, or
    runs code at its real timeouts without waiting for them. Code on the loop
    thread that reads ``time.monotonic()`` for its own deadlines reads the
    same clock (``follow_time``). Under the timing report it is the report's
    loop (same step cost), so D2 still covers the test."""
    policy = asyncio.get_event_loop_policy()
    factory = policy.new_event_loop if isinstance(policy, TimingLoopPolicy) else TimingLoop
    with follow_time(), follow_wrapped_futures():
        return asyncio.run(main, loop_factory=factory)


def _tracking_wrap_future(future, *, loop=None):
    wrapped = _REAL_WRAP_FUTURE(future, loop=loop)
    if isinstance(future, concurrent.futures.Future) and not future.done():
        target = loop if loop is not None else asyncio._get_running_loop()
        if isinstance(target, TimingLoop):
            target.track(future)  # after the wrap: its result is queued first
    return wrapped


@contextlib.contextmanager
def follow_wrapped_futures() -> Iterator[None]:
    """While active, a concurrent future awaited through ``asyncio.wrap_future``
    on a TimingLoop counts as real work (FOLLOW) until it is done. Code with
    its own ThreadPoolExecutor (``BoundedWorker``) never calls run_in_executor,
    so without this the loop would jump past a worker it is waiting on."""
    saved = (asyncio.wrap_future, asyncio.futures.wrap_future)
    asyncio.wrap_future = asyncio.futures.wrap_future = _tracking_wrap_future
    try:
        yield
    finally:
        asyncio.wrap_future, asyncio.futures.wrap_future = saved


def _running_timing_loop() -> TimingLoop | None:
    loop = asyncio._get_running_loop()
    return loop if isinstance(loop, TimingLoop) else None


def _virtual_monotonic() -> float:
    loop = _running_timing_loop()
    return loop.time() if loop is not None else _REAL_MONOTONIC()


def _virtual_perf_counter() -> float:
    loop = _running_timing_loop()
    return loop.time() if loop is not None else _REAL_PERF_COUNTER()


def _virtual_time() -> float:
    loop = _running_timing_loop()
    return loop.wall_offset + loop.time() if loop is not None else _REAL_TIME()


@contextlib.contextmanager
def follow_time() -> Iterator[None]:
    """While active, ``time.monotonic/perf_counter/time`` read the virtual clock
    on a thread whose running loop is a TimingLoop; every other thread keeps
    real time. Code that bound the function itself (``from time import
    monotonic``) is not followed; D7 keeps that out of src/."""
    saved = (real_time.monotonic, real_time.perf_counter, real_time.time)
    real_time.monotonic, real_time.perf_counter, real_time.time = (
        _virtual_monotonic, _virtual_perf_counter, _virtual_time)
    try:
        yield
    finally:
        real_time.monotonic, real_time.perf_counter, real_time.time = saved
