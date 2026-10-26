"""tests/virtual_clock.py: exact elapsed time with no real waiting (TASK-1457)."""

from __future__ import annotations

import asyncio
import concurrent.futures
import socket
import sys
import threading
import time
import types

import pytest

from tests import virtual_clock


def test_timers_keep_their_order_and_take_no_real_time():
    async def main():
        loop = asyncio.get_running_loop()
        started = loop.time()
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.8):
                await asyncio.sleep(30)
        await asyncio.wait_for(asyncio.sleep(0.3), timeout=1)
        return loop.time() - started

    real_started = time.monotonic()
    assert virtual_clock.run(main()) == pytest.approx(1.1)
    # A real wait would take 1.1s; any bound below that proves none happened.
    assert time.monotonic() - real_started < 1.1  # timing-ok: the bound is the real wait it rules out


def test_a_followed_module_reads_the_virtual_clock_until_the_run_ends():
    module = types.ModuleType("followed")
    module.time = time

    async def main():
        monotonic, wall = module.time.monotonic(), module.time.time()
        await asyncio.sleep(2)
        return module.time.monotonic() - monotonic, module.time.time() - wall

    assert virtual_clock.run(main(), follow=[module]) == (pytest.approx(2), pytest.approx(2))
    assert module.time is time


def test_a_thread_or_a_wait_on_nothing_fails_instead_of_racing_or_hanging():
    async def thread():
        await asyncio.to_thread(time.sleep, 0)

    with pytest.raises(RuntimeError, match="real thread"):
        virtual_clock.run(thread())
    with pytest.raises(RuntimeError, match="deadlock"):
        virtual_clock.run(asyncio.Event().wait())


# ---------------------------------------------------------------------------
# TimingLoop (JUMP/FOLLOW). Each seed is one way a virtual clock goes wrong.
# JUMP is proven by the observer never firing (no real wait happened), FOLLOW
# by the real time it reports, as a lower bound. Each mutant below removes one
# rule, and the seed that rule exists for must then fail.
# ---------------------------------------------------------------------------

REAL = virtual_clock._REAL_MONOTONIC


class Run:
    """Runs one coroutine on a fresh loop and keeps what the observer saw."""

    def __init__(self, loop_class: type[virtual_clock.TimingLoop] = virtual_clock.TimingLoop, **options):
        self.follows: list[tuple[str, float]] = []
        self.loop = loop_class(observer=lambda cause, spent: self.follows.append((cause, spent)), **options)

    def __call__(self, main):
        try:
            return self.loop.run_until_complete(main())
        finally:
            self.loop.run_until_complete(self.loop.shutdown_default_executor())
            self.loop.close()

    def followed(self, cause: str) -> float:
        return sum(spent for seen, spent in self.follows if seen == cause)


async def seed1_timer_only():
    loop = asyncio.get_running_loop()
    started = loop.time()
    await asyncio.sleep(0.1)
    return loop.time() - started


def seed1_ok(run: Run, result) -> bool:
    return result == pytest.approx(0.1) and not run.follows


async def seed2_thread_released_by_a_timer():
    released = threading.Event()

    async def release():
        await asyncio.sleep(0.15)
        released.set()

    task = asyncio.create_task(release())
    worker_saw_it = await asyncio.to_thread(released.wait, 30)
    await task
    return worker_saw_it


def seed2_ok(run: Run, result) -> bool:
    return result is True and run.followed("executor") >= 0.15


async def seed3_real_work_inside_its_deadline():
    return await asyncio.wait_for(asyncio.to_thread(time.sleep, 0.1), timeout=10)


def seed3_ok(run: Run, result) -> bool:
    return result is None and run.followed("executor") >= 0.1


async def seed4_subprocess():
    process = await asyncio.create_subprocess_exec(sys.executable, "-c", "import time; time.sleep(0.2)")
    try:
        return await asyncio.wait_for(process.wait(), timeout=10)
    except TimeoutError:
        process.kill()
        await process.wait()
        raise


def seed4_ok(run: Run, result) -> bool:
    return result == 0 and run.followed("subprocess") > 0


async def seed5_external_fd():
    loop = asyncio.get_running_loop()
    ours, peer = socket.socketpair()
    ours.setblocking(False)
    arrived = loop.create_future()
    loop.add_reader(ours.fileno(), lambda: arrived.done() or arrived.set_result(ours.recv(8)))
    writer = threading.Timer(0.1, peer.send, (b"x",))
    writer.start()
    try:
        return await asyncio.wait_for(arrived, timeout=10)
    finally:
        loop.remove_reader(ours.fileno())
        writer.join(30)
        assert not writer.is_alive()
        ours.close()
        peer.close()


def seed5_ok(run: Run, result) -> bool:
    return result == b"x" and run.followed("fd") >= 0.1


async def seed6_cancelled_future_live_worker():
    """wait_for cancels the asyncio Future, but the worker thread keeps running:
    it is still real work, so the clock must keep following it (B2). The
    deadline is taken past on the virtual clock while the worker stays blocked
    on an Event released only after the check, so nothing races."""
    loop = asyncio.get_running_loop()
    release = threading.Event()
    waiter = asyncio.create_task(asyncio.wait_for(asyncio.to_thread(release.wait, 30), timeout=5))
    await asyncio.sleep(0)  # wait_for has armed its deadline
    loop.advance(5)
    with pytest.raises(TimeoutError):
        await waiter
    while_live = loop.real_work()
    release.set()
    while loop.real_work() is not None:
        await asyncio.sleep(0.01)
    loop.observed.clear()
    await asyncio.sleep(100)  # nothing real left: JUMP
    return while_live, list(loop.observed)


def seed6_ok(run: Run, result) -> bool:
    while_live, after_release = result
    return while_live == "executor" and not after_release


SEED7_REAL_S = 0.02


async def seed7_always_ready_external_fd():
    """A writable socket keeps select from ever blocking. While it is
    registered, the clock must still keep pace with the real time callbacks
    run between selects."""
    loop = asyncio.get_running_loop()
    ours, peer = socket.socketpair()
    ours.setblocking(False)
    loop.add_writer(ours.fileno(), lambda: None)
    try:
        await asyncio.sleep(0)  # a select accounts real time up to here
        virtual_started, real_started = loop.time(), REAL()
        # The 2 ticks cover rounding to whole ticks, not runner speed.
        while REAL() < real_started + SEED7_REAL_S + 2 / virtual_clock.TICKS_PER_SECOND:
            await asyncio.sleep(0)
        await asyncio.sleep(0)  # and accounts the spin
        return loop.time() - virtual_started
    finally:
        loop.remove_writer(ours.fileno())
        ours.close()
        peer.close()


def seed7_ok(run: Run, result) -> bool:
    # One-sided: the clock counted at least the real time the loop spun. A
    # slow runner only makes the spin longer, so it can't fail this bound.
    return result >= SEED7_REAL_S and run.followed("fd") > 0


SEEDS = [(seed1_timer_only, seed1_ok), (seed2_thread_released_by_a_timer, seed2_ok),
         (seed3_real_work_inside_its_deadline, seed3_ok), (seed4_subprocess, seed4_ok),
         (seed5_external_fd, seed5_ok), (seed6_cancelled_future_live_worker, seed6_ok),
         (seed7_always_ready_external_fd, seed7_ok)]


class Observed(virtual_clock.TimingLoop):
    """Also keeps the observer's causes on the loop, for seed 6 to read mid-run."""

    def __init__(self, *, observer, **options):
        self.observed: list[str] = []

        def both(cause: str, spent: float) -> None:
            self.observed.append(cause)
            observer(cause, spent)

        super().__init__(observer=both, **options)


@pytest.mark.real_process(reason="seed 4 proves FOLLOW tracks a real subprocess")
@pytest.mark.parametrize(("main", "ok"), SEEDS, ids=[main.__name__ for main, _ in SEEDS])
def test_timing_loop_seed(main, ok):
    run = Run(Observed)
    assert ok(run, run(main))


class OwnFdsCountAsReal(Observed):
    def __init__(self, **options):
        super().__init__(**options)
        self._own_fds = frozenset()


class CountsFuturesNotWorkers(Observed):
    """looptime's rule: the worker counts until its asyncio Future is done."""

    def run_in_executor(self, executor, func, *args):
        # asyncio's own run_in_executor, skipping TimingLoop's worker count
        future = asyncio.BaseEventLoop.run_in_executor(self, executor, func, *args)
        with self._workers_lock:
            self._workers += 1

        def done(_):
            with self._workers_lock:
                self._workers -= 1

        future.add_done_callback(done)
        return future


class FollowJumps(Observed):
    def real_work(self):
        return None


class FollowsOnlyBlockingSelects(Observed):
    """r1's rule: in FOLLOW only a blocking select advances the clock, so a
    select that finds events ready leaves it standing still."""

    def _select(self, timeout=None):
        events = self._real_select(0)
        if events or timeout == 0:
            return events
        cause = self.real_work()
        if cause is None:
            return self._jump(timeout)
        started = REAL()
        events = self._real_select(timeout if timeout is not None else self._idle_limit)
        spent = REAL() - started
        self.observer(cause, spent)
        self._advance(timeout if timeout is not None and not events else spent)
        return events


@pytest.mark.real_process(reason="seed 4 under the mutant")
@pytest.mark.parametrize(("mutant", "broken"), [
    (OwnFdsCountAsReal, seed1_timer_only),
    (CountsFuturesNotWorkers, seed6_cancelled_future_live_worker),
    (FollowJumps, seed2_thread_released_by_a_timer),
    (FollowJumps, seed3_real_work_inside_its_deadline),
    (FollowJumps, seed4_subprocess),
    (FollowJumps, seed5_external_fd),
    (FollowsOnlyBlockingSelects, seed7_always_ready_external_fd),
], ids=lambda value: value.__name__)
def test_each_rule_is_what_its_seed_needs(mutant, broken):
    ok = dict(SEEDS)[broken]
    run = Run(mutant)
    try:
        result = run(broken)
    except (TimeoutError, AssertionError, pytest.fail.Exception):
        return  # the mutant broke the seed outright
    assert not ok(run, result)


def test_a_worker_cancelled_before_it_starts_is_released():
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    gate = threading.Event()

    async def main():
        loop = asyncio.get_running_loop()
        busy = loop.run_in_executor(pool, gate.wait, 30)
        queued = loop.run_in_executor(pool, gate.wait, 30)
        assert loop._workers == 2
        queued.cancel()
        await asyncio.sleep(0)
        assert loop._workers == 1
        gate.set()
        assert await busy is True
        await asyncio.sleep(0)
        return loop._workers

    try:
        assert Run()(main) == 0
    finally:
        pool.shutdown(wait=True)


async def _answer_after_two_turns(timeout: float) -> str:
    answer = asyncio.get_running_loop().create_future()
    waiter = asyncio.create_task(asyncio.wait_for(answer, timeout))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    answer.set_result("ok")
    return await waiter


def test_step_cost_fails_code_that_only_got_there_in_time():
    assert Run()(lambda: _answer_after_two_turns(0.01)) == "ok"
    with pytest.raises(TimeoutError):
        Run(step_cost=0.02)(lambda: _answer_after_two_turns(0.01))
    assert Run(step_cost=0.02)(lambda: _answer_after_two_turns(5)) == "ok"


def test_advance_moves_the_clock_forward_only():
    async def main():
        loop = asyncio.get_running_loop()
        started = loop.time()
        loop.advance(2.5)
        with pytest.raises(ValueError, match="only moves forward"):
            loop.advance(-1)
        return loop.time() - started

    assert Run()(main) == pytest.approx(2.5)


def test_jump_takes_no_real_time():
    started = REAL()
    assert Run()(lambda: asyncio.sleep(100, "slept")) == "slept"
    assert REAL() - started < 50  # timing-ok: the bound is the 100s real wait it rules out


def test_follow_time_reads_the_loop_clock_on_the_loop_thread_only():
    before = time.monotonic

    async def main():
        with virtual_clock.follow_time():
            monotonic, wall, perf = time.monotonic(), time.time(), time.perf_counter()
            await asyncio.sleep(100)
            on_loop = (time.monotonic() - monotonic, time.time() - wall, time.perf_counter() - perf)
            other_thread = await asyncio.to_thread(time.monotonic)
            return on_loop, time.monotonic() - other_thread

    on_loop, ahead_of_the_worker = Run()(main)
    assert on_loop == (pytest.approx(100), pytest.approx(100), pytest.approx(100))
    # The jump put the loop 100s ahead of real time; the worker thread read real time.
    assert ahead_of_the_worker > 50
    assert time.monotonic is before


def test_a_loop_clock_starts_at_real_time_so_a_cached_reading_never_goes_back():
    async def started():
        with virtual_clock.follow_time():
            return asyncio.get_running_loop().time(), time.time()

    monotonic, wall = REAL(), virtual_clock._REAL_TIME()
    loop_monotonic, loop_wall = Run()(started)
    assert loop_monotonic >= monotonic and loop_wall >= wall


def test_timing_loop_fails_a_wait_on_nothing_instead_of_hanging():
    with pytest.raises(RuntimeError, match="deadlock"):
        Run(idle_limit=0.2)(lambda: asyncio.Event().wait())


def test_the_policy_makes_every_new_loop_a_timing_loop():
    policy = virtual_clock.TimingLoopPolicy(step_cost=0.5)
    loop = policy.new_event_loop()
    try:
        assert isinstance(loop, virtual_clock.TimingLoop) and loop._step == 500_000
    finally:
        loop.close()
