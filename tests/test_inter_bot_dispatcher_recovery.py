"""The dispatcher's startup recovery check (TASK-1232).

The check runs once per leadership term: at the first ``run()`` entry, after
the initial ``recover_expired``. A crash re-entry of ``run()`` never reruns it
and nothing reruns it on a timer; only a re-acquired leadership term does
(TASK-1311, ``tests/test_dispatcher_leadership.py``). Dispatch waits for it
only up to a bound; a slow check is never cancelled for being slow, only
``stop()`` cancels it, and its end wakes the dispatcher.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

from llm_bawt.service.inter_bot_dispatcher import InterBotDeliveryDispatcher
from llm_bawt.service.inter_bot_dispatcher_recovery import StartupRecoveryRunner


class _Check:
    def __init__(self, events: list[str] | None = None):
        self.events = events if events is not None else []
        self.calls = 0
        self.raises: Exception | None = None
        self.block: asyncio.Event | None = None
        self.cancelled = False

    async def run_at_startup(self) -> None:
        self.calls += 1
        self.events.append("startup_check")
        if self.block is not None:
            try:
                await self.block.wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        if self.raises is not None:
            raise self.raises


class _Store:
    def __init__(self, events: list[str]):
        self.events = events
        self.recover_expired_calls = 0
        self.released_with_check_running: bool | None = None
        self.check: _Check | None = None

    def recover_expired(self, *, live_owner=None):
        self.recover_expired_calls += 1
        self.events.append("recover_expired")

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


def _dispatcher(check: _Check):
    events = check.events
    dispatcher = InterBotDeliveryDispatcher.__new__(InterBotDeliveryDispatcher)
    dispatcher.service = SimpleNamespace(
        config=SimpleNamespace(DEFAULT_USER="nick"),
        _redis_subscriber=SimpleNamespace(connected=True),
    )
    dispatcher.store = _Store(events)
    dispatcher.store.check = check
    # The loop's wake timeout; tests drive iterations with explicit wakes.
    dispatcher.recovery_interval_seconds = 5.0
    dispatcher._wake_event = asyncio.Event()
    dispatcher._recovery = StartupRecoveryRunner(check, on_done=dispatcher._wake_event.set)
    dispatcher._stop_event = asyncio.Event()
    dispatcher._task = None
    dispatcher._leader_task = None
    dispatcher._loop = None
    dispatcher._target_tasks = {}
    dispatcher._dispatch_tasks = set()
    dispatcher._talk_reply_task = None
    dispatcher.talk_replies = None
    dispatcher.claim_owner = "test-dispatcher"
    dispatcher._dispatcher_lock = object()
    dispatcher._subscriber_gap_logged = False
    return dispatcher


async def _iterate(dispatcher, sweeps_before: int, count: int) -> None:
    """Drive ``count`` loop iterations by wake, waiting for each sweep."""
    for index in range(count):
        dispatcher._wake_event.set()
        for _ in range(200):
            if dispatcher.store.recover_expired_calls >= sweeps_before + index + 1:
                break
            await asyncio.sleep(0.005)
        else:  # pragma: no cover - diagnostic
            raise AssertionError("dispatcher loop did not iterate")


async def _stop(dispatcher, task) -> None:
    dispatcher._stop_event.set()
    dispatcher._wake_event.set()
    await asyncio.wait_for(task, timeout=2.0)


# --- runner unit contract -------------------------------------------------


async def _until(predicate, *, polls: int = 200) -> None:
    for _ in range(polls):
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition not reached")  # pragma: no cover - diagnostic


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


@pytest.mark.asyncio
async def test_a_slow_check_is_waited_on_boundedly_never_cancelled_and_never_restarted(monkeypatch, caplog):
    """Cancelling the await would not stop a check's worker-thread write, and
    the sweep would release its lock with that write in flight."""
    check, woke = _Check(), []
    check.block = asyncio.Event()
    monkeypatch.setattr(StartupRecoveryRunner, "WAIT_SECONDS", 0.05)
    runner = StartupRecoveryRunner(check, on_done=lambda: woke.append(True))
    caplog.set_level(logging.WARNING)

    await asyncio.wait_for(runner.run_once(), timeout=1.0)
    assert any("still running after" in r.getMessage() and r.levelno == logging.WARNING
               for r in caplog.records)
    assert runner.running and not check.cancelled and woke == []

    await asyncio.wait_for(runner.run_once(), timeout=1.0)  # crash re-entry
    assert check.calls == 1

    check.block.set()
    await _until(lambda: not runner.running)
    assert not check.cancelled and woke == [True]
    await runner.run_once()  # a finished check is not rerun either
    assert check.calls == 1


@pytest.mark.asyncio
async def test_stop_cancels_a_running_check_without_waking_dispatch(monkeypatch):
    check, woke = _Check(), []
    check.block = asyncio.Event()
    monkeypatch.setattr(StartupRecoveryRunner, "WAIT_SECONDS", 0.01)
    runner = StartupRecoveryRunner(check, on_done=lambda: woke.append(True))
    await runner.run_once()
    await asyncio.wait_for(runner.stop(), timeout=1.0)
    assert check.cancelled and not runner.running and woke == []


@pytest.mark.asyncio
async def test_the_wait_ends_on_stop_without_cancelling_the_check():
    check = _Check()
    check.block = asyncio.Event()
    runner = StartupRecoveryRunner(check)
    stop = asyncio.Event()
    waiting = asyncio.create_task(runner.run_once(stop=stop))
    await _until(lambda: check.calls == 1)
    stop.set()
    await asyncio.wait_for(waiting, timeout=1.0)
    assert runner.running and not check.cancelled  # stop() owns the cancel
    await asyncio.wait_for(runner.stop(), timeout=1.0)
    assert check.cancelled


# --- dispatcher wiring ----------------------------------------------------


@pytest.mark.asyncio
async def test_run_starts_the_check_after_initial_recover_expired():
    """Recovery steps defer claimed deliveries to recover_expired, so the
    check must see the claims it released."""
    check = _Check()
    dispatcher = _dispatcher(check)
    task = asyncio.create_task(dispatcher.run())
    await _iterate(dispatcher, 0, 1)
    await _stop(dispatcher, task)
    assert check.events[:2] == ["recover_expired", "startup_check"]


@pytest.mark.asyncio
async def test_a_crash_reentry_of_run_never_reruns_the_check():
    """run() is re-entered after a crash under the same leadership: that is
    not a startup."""
    check = _Check()
    dispatcher = _dispatcher(check)
    for _ in range(2):
        dispatcher._stop_event.clear()
        task = asyncio.create_task(dispatcher.run())
        await _iterate(dispatcher, dispatcher.store.recover_expired_calls, 1)
        await _stop(dispatcher, task)
    assert dispatcher.store.recover_expired_calls >= 4 and check.calls == 1


@pytest.mark.asyncio
async def test_the_check_never_reruns_on_a_timer_or_a_wake(monkeypatch):
    """No periodic pass: many loop iterations, each a full delivery sweep, and
    the check ran exactly once."""
    check = _Check()
    dispatcher = _dispatcher(check)
    dispatcher.recovery_interval_seconds = 0.001  # the loop's timed sweep fires constantly
    task = asyncio.create_task(dispatcher.run())
    await _iterate(dispatcher, 0, 1)
    await _until(lambda: dispatcher.store.recover_expired_calls >= 60)
    await _stop(dispatcher, task)
    assert check.calls == 1


@pytest.mark.asyncio
async def test_a_hung_check_is_bounded_dispatch_starts_and_it_stops_before_unlock(monkeypatch):
    check = _Check()
    check.block = asyncio.Event()
    monkeypatch.setattr(StartupRecoveryRunner, "WAIT_SECONDS", 0.05)
    dispatcher = _dispatcher(check)
    dispatcher._task = asyncio.create_task(dispatcher.run())
    await _iterate(dispatcher, 0, 3)
    assert dispatcher.store.recover_expired_calls >= 3
    assert not check.cancelled  # still running, not cut off
    await asyncio.wait_for(dispatcher.stop(), timeout=2.0)
    assert check.cancelled  # shutdown is the only cancel
    assert dispatcher.store.released_with_check_running is False


@pytest.mark.asyncio
async def test_a_check_that_outlives_the_wait_wakes_dispatch_when_it_ends(monkeypatch):
    check = _Check()
    check.block = asyncio.Event()
    monkeypatch.setattr(StartupRecoveryRunner, "WAIT_SECONDS", 0.01)
    dispatcher = _dispatcher(check)
    dispatcher.recovery_interval_seconds = 60.0  # only a wake can iterate the loop
    task = asyncio.create_task(dispatcher.run())
    await _until(lambda: dispatcher.store.recover_expired_calls == 2)  # entry + first iteration
    await asyncio.sleep(0.02)
    assert dispatcher.store.recover_expired_calls == 2
    check.block.set()
    await _until(lambda: dispatcher.store.recover_expired_calls == 3)
    await _stop(dispatcher, task)


@pytest.mark.asyncio
async def test_stop_during_the_wait_is_not_held_by_the_bound(caplog):
    """Shutdown within the bound (a redeploy right after boot) must not wait
    the bound out: stop() cancels the check right after anyway."""
    check = _Check()
    check.block = asyncio.Event()
    assert StartupRecoveryRunner.WAIT_SECONDS >= 30.0  # the real bound
    dispatcher = _dispatcher(check)
    dispatcher._task = asyncio.create_task(dispatcher.run())
    await _until(lambda: check.calls == 1)
    caplog.set_level(logging.WARNING)

    await asyncio.wait_for(asyncio.shield(dispatcher.stop()), timeout=2.0)

    assert check.cancelled
    assert dispatcher.store.recover_expired_calls == 1  # no sweep after shutdown
    assert not any("still running after" in r.getMessage() for r in caplog.records)


# --- leadership read ---------------------------------------------------


@pytest.mark.asyncio
async def test_is_leader_follows_the_dispatcher_lock():
    dispatcher = _dispatcher(_Check())
    assert dispatcher.is_leader is True
    await asyncio.wait_for(dispatcher.stop(), timeout=2.0)
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


@pytest.mark.asyncio
async def test_the_wired_check_runs_one_full_mutating_sweep_per_process():
    """What api.py wires: every default step, mutating, once, however long
    the dispatcher keeps sweeping deliveries."""
    from llm_bawt.service.recovery.sweep import StartupRecoveryCheck

    sweep = _Sweep()
    dispatcher = _dispatcher(_Check())
    dispatcher._recovery = StartupRecoveryRunner(StartupRecoveryCheck(sweep), on_done=dispatcher._wake_event.set)
    task = asyncio.create_task(dispatcher.run())
    await _iterate(dispatcher, 0, 20)
    await _stop(dispatcher, task)
    assert sweep.calls == [{"dry_run": False, "actor": "recovery_startup"}]
