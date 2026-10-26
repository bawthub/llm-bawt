"""The dispatcher's startup recovery check (TASK-1232).

The check runs once per leadership term: at the first ``run()`` entry, after
the initial ``recover_expired``. A crash re-entry of ``run()`` never reruns it
and nothing reruns it on a timer; only a re-acquired leadership term does
(TASK-1311, ``tests/test_dispatcher_leadership.py``). Dispatch waits for it
only up to a bound; a slow check is never cancelled for being slow, only
``stop()`` cancels it, and its end wakes the dispatcher.

Anything timed runs on ``tests/virtual_clock.run_timing`` at the production
sweep interval and wait bound (TASK-1457): the clock jumps between timers and
waits are driven by the fakes (``tests/state_changes``), never by polling.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from types import SimpleNamespace

import pytest

from llm_bawt.service.inter_bot_dispatcher import InterBotDeliveryDispatcher
from llm_bawt.service.inter_bot_dispatcher_recovery import StartupRecoveryRunner
from tests import virtual_clock
from tests.state_changes import StateChanges

SWEEP_S = inspect.signature(InterBotDeliveryDispatcher.__init__).parameters[
    "recovery_interval_seconds"].default
RELEASE_S = InterBotDeliveryDispatcher.RELEASE_TIMEOUT_SECONDS
WAIT_S = StartupRecoveryRunner.WAIT_SECONDS
PROMPT_S = 1.0
"""Virtual seconds for a step that waits out no timer: a few scheduler turns
(20ms each under the timing report) and worker threads that answer at once.
Well under every interval it is told apart from (the sweep, the wait bound)."""


class _Check:
    def __init__(self, events: list[str] | None = None, changes: StateChanges | None = None):
        self.events = events if events is not None else []
        self.changes = changes
        self.calls = 0
        self.raises: Exception | None = None
        self.block: asyncio.Event | None = None
        self.cancelled = False

    async def run_at_startup(self) -> None:
        self.calls += 1
        self.events.append("startup_check")
        if self.changes is not None:
            self.changes.poke()
        if self.block is not None:
            try:
                await self.block.wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        if self.raises is not None:
            raise self.raises


class _Store:
    def __init__(self, events: list[str], changes: StateChanges | None = None):
        self.events = events
        self.changes = changes
        self.recover_expired_calls = 0
        self.released_with_check_running: bool | None = None
        self.check: _Check | None = None

    def recover_expired(self, *, live_owner=None):
        self.recover_expired_calls += 1
        self.events.append("recover_expired")
        if self.changes is not None:
            self.changes.poke()

    def eligible_targets(self):
        return []

    def probe_dispatcher_lock(self, _conn):
        return True  # the leadership session stays alive in these tests

    def release_dispatcher_lock(self, *_a, **_k):
        check = self.check
        self.released_with_check_running = bool(
            check is not None and check.calls and check.block is not None
            and not check.block.is_set() and not check.cancelled
        )


def _dispatcher(check: _Check, changes: StateChanges | None = None):
    """A leader on the fakes, at the production sweep interval and wait bound."""
    events = check.events
    dispatcher = InterBotDeliveryDispatcher.__new__(InterBotDeliveryDispatcher)
    dispatcher.service = SimpleNamespace(
        config=SimpleNamespace(DEFAULT_USER="nick"),
        _redis_subscriber=SimpleNamespace(connected=True),
    )
    dispatcher.store = _Store(events, changes)
    dispatcher.store.check = check
    dispatcher.recovery_interval_seconds = SWEEP_S
    dispatcher._wake_event = asyncio.Event()

    def on_done() -> None:
        dispatcher._wake_event.set()
        if changes is not None:
            changes.poke()

    dispatcher._recovery = StartupRecoveryRunner(check, on_done=on_done)
    dispatcher._stop_event = asyncio.Event()
    dispatcher._task = None
    dispatcher._leader_task = None
    dispatcher._loop = None
    dispatcher._target_tasks = {}
    dispatcher._dispatch_tasks = set()
    dispatcher.side_pumps = None
    dispatcher.claim_owner = "test-dispatcher"
    dispatcher._dispatcher_lock = object()
    dispatcher._lock_probed_at = None  # the session is probed on every sweep
    dispatcher._subscriber_gap_logged = False
    return dispatcher


async def _entered(dispatcher, changes: StateChanges, sweeps_before: int, *,
                   within: float = PROMPT_S) -> None:
    """A run() entry settled: its initial recover_expired, then the sweep its
    own wake starts once the check is done or its wait bound passed."""
    await changes.until(lambda: dispatcher.store.recover_expired_calls >= sweeps_before + 2,
                        within=within)


async def _iterate(dispatcher, changes: StateChanges, count: int) -> None:
    """Drive ``count`` loop iterations by wake, each well before the timed sweep."""
    for _ in range(count):
        before = dispatcher.store.recover_expired_calls
        dispatcher._wake_event.set()
        await changes.until(lambda: dispatcher.store.recover_expired_calls > before,
                            within=PROMPT_S)


async def _stop(dispatcher, task) -> None:
    dispatcher._stop_event.set()
    dispatcher._wake_event.set()
    await asyncio.wait_for(task, PROMPT_S)


async def _stop_dispatcher(dispatcher) -> None:
    """stop() is bounded by the release timeout, whatever is still running."""
    await asyncio.wait_for(dispatcher.stop(), RELEASE_S + PROMPT_S)


# --- runner unit contract -------------------------------------------------


@pytest.mark.asyncio
async def test_a_failed_check_is_contained_and_still_wakes_dispatch(caplog):
    check, woke = _Check(), []
    check.raises = RuntimeError("boom")
    runner = StartupRecoveryRunner(check, on_done=lambda: woke.append(True))
    caplog.set_level(logging.ERROR)
    await runner.run_once()
    await asyncio.sleep(0)
    assert any("startup recovery check failed" in r.getMessage() for r in caplog.records)
    assert not runner.running and woke == [True]


def test_a_slow_check_is_waited_on_boundedly_never_cancelled_and_never_restarted(caplog):
    """Cancelling the await would not stop a check's worker-thread write, and
    the sweep would release its lock with that write in flight."""
    caplog.set_level(logging.WARNING)

    async def main():
        loop = asyncio.get_running_loop()
        changes = StateChanges()
        check, woke = _Check(changes=changes), []
        check.block = asyncio.Event()

        def on_done() -> None:
            woke.append(True)
            changes.poke()

        runner = StartupRecoveryRunner(check, on_done=on_done)
        started = loop.time()
        await runner.run_once()
        elapsed = loop.time() - started
        # timing-ok: virtual clock (run_timing): loop.time() is exact and never waits
        assert WAIT_S <= elapsed < WAIT_S + PROMPT_S
        assert any("still running after" in r.getMessage() and r.levelno == logging.WARNING
                   for r in caplog.records)
        assert runner.running and not check.cancelled and woke == []

        started = loop.time()
        await runner.run_once()  # crash re-entry: returns at once
        # timing-ok: virtual clock (run_timing): loop.time() is exact and never waits
        assert loop.time() - started < PROMPT_S
        assert check.calls == 1

        check.block.set()
        await changes.until(lambda: not runner.running and woke, within=PROMPT_S)  # on_done pokes
        assert not check.cancelled and woke == [True]
        await runner.run_once()  # a finished check is not rerun either
        assert check.calls == 1

    virtual_clock.run_timing(main())


def test_stop_cancels_a_running_check_without_waking_dispatch():
    async def main():
        check, woke = _Check(), []
        check.block = asyncio.Event()
        runner = StartupRecoveryRunner(check, on_done=lambda: woke.append(True))
        await runner.run_once()  # the bound passes, the check runs on
        assert runner.running
        await asyncio.wait_for(runner.stop(), PROMPT_S)
        assert check.cancelled and not runner.running and woke == []

    virtual_clock.run_timing(main())


def test_the_wait_ends_on_stop_without_cancelling_the_check():
    async def main():
        changes = StateChanges()
        check = _Check(changes=changes)
        check.block = asyncio.Event()
        runner = StartupRecoveryRunner(check)
        stop = asyncio.Event()
        waiting = asyncio.create_task(runner.run_once(stop=stop))
        await changes.until(lambda: check.calls == 1, within=PROMPT_S)
        stop.set()
        await asyncio.wait_for(waiting, PROMPT_S)  # well inside the bound
        assert runner.running and not check.cancelled  # stop() owns the cancel
        await asyncio.wait_for(runner.stop(), PROMPT_S)
        assert check.cancelled

    virtual_clock.run_timing(main())


# --- dispatcher wiring ----------------------------------------------------


def test_run_starts_the_check_after_initial_recover_expired():
    """Recovery steps defer claimed deliveries to recover_expired, so the
    check must see the claims it released."""
    async def main():
        changes = StateChanges()
        check = _Check(changes=changes)
        dispatcher = _dispatcher(check, changes)
        task = asyncio.create_task(dispatcher.run())
        await _entered(dispatcher, changes, 0)
        await _stop(dispatcher, task)
        return check.events

    assert virtual_clock.run_timing(main())[:3] == [
        "recover_expired", "startup_check", "recover_expired"]


def test_a_crash_reentry_of_run_never_reruns_the_check():
    """run() is re-entered after a crash under the same leadership: that is
    not a startup."""
    async def main():
        changes = StateChanges()
        check = _Check(changes=changes)
        dispatcher = _dispatcher(check, changes)
        for _ in range(2):
            dispatcher._stop_event.clear()
            task = asyncio.create_task(dispatcher.run())
            await _entered(dispatcher, changes, dispatcher.store.recover_expired_calls)
            await _stop(dispatcher, task)
        return dispatcher.store.recover_expired_calls, check.calls

    # Two entries, each its initial recover_expired and one woken sweep.
    assert virtual_clock.run_timing(main()) == (4, 1)


def test_the_check_never_reruns_on_a_timer_or_a_wake():
    """No periodic pass: many loop iterations, each a full delivery sweep, and
    the check ran exactly once."""
    sweeps = 20

    async def main():
        changes = StateChanges()
        check = _Check(changes=changes)
        dispatcher = _dispatcher(check, changes)
        task = asyncio.create_task(dispatcher.run())
        await _entered(dispatcher, changes, 0)
        await _iterate(dispatcher, changes, 3)
        # The rest are the loop's own timed sweeps, one per interval.
        await changes.until(lambda: dispatcher.store.recover_expired_calls >= sweeps,
                            within=sweeps * SWEEP_S)
        await _stop(dispatcher, task)
        return check.calls

    assert virtual_clock.run_timing(main()) == 1


def test_a_hung_check_is_bounded_dispatch_starts_and_it_stops_before_unlock():
    async def main():
        loop = asyncio.get_running_loop()
        changes = StateChanges()
        check = _Check(changes=changes)
        check.block = asyncio.Event()
        dispatcher = _dispatcher(check, changes)
        started = loop.time()
        dispatcher._task = asyncio.create_task(dispatcher.run())
        # Dispatch starts once the bound passes, with the check still running.
        await _entered(dispatcher, changes, 0, within=WAIT_S + PROMPT_S)
        assert loop.time() - started >= WAIT_S
        await _iterate(dispatcher, changes, 2)
        assert dispatcher.store.recover_expired_calls == 4
        assert not check.cancelled  # still running, not cut off
        await _stop_dispatcher(dispatcher)
        assert check.cancelled  # shutdown is the only cancel
        assert dispatcher.store.released_with_check_running is False

    virtual_clock.run_timing(main())


def test_a_check_that_outlives_the_wait_wakes_dispatch_when_it_ends():
    async def main():
        changes = StateChanges()
        check = _Check(changes=changes)
        check.block = asyncio.Event()
        dispatcher = _dispatcher(check, changes)
        dispatcher.recovery_interval_seconds = 3600.0  # only a wake can iterate the loop
        task = asyncio.create_task(dispatcher.run())
        await _entered(dispatcher, changes, 0, within=WAIT_S + PROMPT_S)
        await asyncio.sleep(SWEEP_S)  # a production interval: still no sweep without a wake
        assert dispatcher.store.recover_expired_calls == 2
        check.block.set()  # its end wakes the loop (on_done)
        await changes.until(lambda: dispatcher.store.recover_expired_calls == 3, within=PROMPT_S)
        await _stop(dispatcher, task)

    virtual_clock.run_timing(main())


def test_stop_during_the_wait_is_not_held_by_the_bound(caplog):
    """Shutdown within the bound (a redeploy right after boot) must not wait
    the bound out: stop() cancels the check right after anyway."""
    caplog.set_level(logging.WARNING)
    assert RELEASE_S + PROMPT_S < WAIT_S  # so a bounded stop() never waited the bound out

    async def main():
        changes = StateChanges()
        check = _Check(changes=changes)
        check.block = asyncio.Event()
        dispatcher = _dispatcher(check, changes)
        dispatcher._task = asyncio.create_task(dispatcher.run())
        await changes.until(lambda: check.calls == 1, within=PROMPT_S)
        await _stop_dispatcher(dispatcher)
        assert check.cancelled
        assert dispatcher.store.recover_expired_calls == 1  # no sweep after shutdown

    virtual_clock.run_timing(main())
    assert not any("still running after" in r.getMessage() for r in caplog.records)


# --- leadership read ---------------------------------------------------


@pytest.mark.asyncio
async def test_is_leader_follows_the_dispatcher_lock():
    dispatcher = _dispatcher(_Check())
    assert dispatcher.is_leader is True
    await _stop_dispatcher(dispatcher)
    assert dispatcher.is_leader is False


def test_passive_dispatcher_is_not_leader():
    dispatcher = _dispatcher(_Check())
    dispatcher._dispatcher_lock = None
    assert dispatcher.is_leader is False


class _Sweep:
    """Records the passes ``StartupRecoveryCheck`` asks the recovery sweep for."""

    def __init__(self):
        self.calls: list[dict] = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        return {"busy": False, "sweep_run_id": "run-1", "steps": []}


def test_the_wired_check_runs_one_full_mutating_sweep_per_process():
    """What api.py wires: every default step, mutating, once, however long
    the dispatcher keeps sweeping deliveries."""
    from llm_bawt.service.recovery.sweep import StartupRecoveryCheck

    async def main():
        changes = StateChanges()
        sweep = _Sweep()
        dispatcher = _dispatcher(_Check(), changes)

        def on_done() -> None:
            dispatcher._wake_event.set()
            changes.poke()

        dispatcher._recovery = StartupRecoveryRunner(StartupRecoveryCheck(sweep), on_done=on_done)
        task = asyncio.create_task(dispatcher.run())
        await _entered(dispatcher, changes, 0)
        await _iterate(dispatcher, changes, 18)
        await _stop(dispatcher, task)
        return sweep.calls

    assert virtual_clock.run_timing(main()) == [{"dry_run": False, "actor": "recovery_startup"}]
