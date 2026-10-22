"""Leader-only recovery cadence in the inter-bot dispatcher (TASK-1232 N1/N2).

N1: ``on_leadership_acquired`` runs at every ``run()`` entry (crash restarts
included), after the initial ``recover_expired``, and never blocks dispatch past
its bound. A slow pass is never cancelled for being slow: only ``stop()`` does.
N2: ``periodic`` is gated on monotonic elapsed time, not on loop wakes, and
shares one single-flight slot with the leadership pass, so passes never overlap
and a hung pass never stalls the delivery sweep.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

from llm_bawt.service.inter_bot_dispatcher import InterBotDeliveryDispatcher
from llm_bawt.service.inter_bot_dispatcher_recovery import RecoveryCadence


class _Clock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


class _Hooks:
    def __init__(self, events: list[str] | None = None):
        self.events = events if events is not None else []
        self.leadership_calls = 0
        self.periodic_calls = 0
        self.leadership_raises: Exception | None = None
        self.periodic_raises: Exception | None = None
        self.leadership_block: asyncio.Event | None = None
        self.periodic_block: asyncio.Event | None = None
        self.periodic_cancelled = False
        self.leadership_cancelled = False

    async def on_leadership_acquired(self) -> None:
        self.leadership_calls += 1
        self.events.append("leadership")
        if self.leadership_block is not None:
            try:
                await self.leadership_block.wait()
            except asyncio.CancelledError:
                self.leadership_cancelled = True
                raise
        if self.leadership_raises is not None:
            raise self.leadership_raises

    async def periodic(self) -> None:
        self.periodic_calls += 1
        self.events.append("periodic")
        if self.periodic_block is not None:
            try:
                await self.periodic_block.wait()
            except asyncio.CancelledError:
                self.periodic_cancelled = True
                raise
        if self.periodic_raises is not None:
            raise self.periodic_raises


class _Store:
    def __init__(self, events: list[str]):
        self.events = events
        self.recover_expired_calls = 0
        self.released_with_periodic_running: bool | None = None
        self.hooks: _Hooks | None = None

    def recover_expired(self):
        self.recover_expired_calls += 1
        self.events.append("recover_expired")

    def eligible_targets(self):
        return []

    def release_dispatcher_lock(self, *_a, **_k):
        hooks = self.hooks
        self.released_with_periodic_running = bool(
            hooks is not None and hooks.periodic_calls and not hooks.periodic_cancelled
        )


def _dispatcher(hooks: _Hooks, clock: _Clock, *, interval: float = 15.0):
    events = hooks.events
    dispatcher = InterBotDeliveryDispatcher.__new__(InterBotDeliveryDispatcher)
    dispatcher.service = SimpleNamespace(
        config=SimpleNamespace(DEFAULT_USER="nick"),
        _redis_subscriber=SimpleNamespace(connected=True),
    )
    dispatcher.store = _Store(events)
    dispatcher.store.hooks = hooks
    # The loop's wake timeout; tests drive iterations with explicit wakes.
    dispatcher.recovery_interval_seconds = 5.0
    dispatcher._recovery = RecoveryCadence(hooks, interval_seconds=interval, clock=clock)
    dispatcher._wake_event = asyncio.Event()
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


# --- cadence unit contract ------------------------------------------------


@pytest.mark.asyncio
async def test_periodic_is_gated_on_elapsed_time_not_on_calls():
    clock, hooks = _Clock(), _Hooks()
    cadence = RecoveryCadence(hooks, interval_seconds=15.0, clock=clock)
    await cadence.on_leadership()

    for second in range(15):
        clock.now = 1000.0 + second
        assert cadence.maybe_start_periodic() is None
    clock.now = 1015.0
    first = cadence.maybe_start_periodic()
    assert first is not None
    await first
    for second in range(16, 30):
        clock.now = 1000.0 + second
        assert cadence.maybe_start_periodic() is None
    clock.now = 1030.0
    second_pass = cadence.maybe_start_periodic()
    assert second_pass is not None
    await second_pass
    assert hooks.periodic_calls == 2


@pytest.mark.asyncio
async def test_periodic_is_single_flight_while_a_pass_hangs():
    clock, hooks = _Clock(), _Hooks()
    hooks.periodic_block = asyncio.Event()
    cadence = RecoveryCadence(hooks, interval_seconds=15.0, clock=clock)
    await cadence.on_leadership()
    clock.now += 15.0
    running = cadence.maybe_start_periodic()
    await asyncio.sleep(0)
    clock.now += 60.0
    assert cadence.maybe_start_periodic() is None
    assert hooks.periodic_calls == 1

    await cadence.stop()
    assert running.cancelled() and hooks.periodic_cancelled


@pytest.mark.asyncio
async def test_failed_periodic_retries_next_interval_not_next_wake(caplog):
    clock, hooks = _Clock(), _Hooks()
    hooks.periodic_raises = RuntimeError("db blip")
    cadence = RecoveryCadence(hooks, interval_seconds=15.0, clock=clock)
    await cadence.on_leadership()
    clock.now += 15.0
    caplog.set_level(logging.ERROR)
    await asyncio.gather(cadence.maybe_start_periodic(), return_exceptions=True)
    await asyncio.sleep(0)
    clock.now += 1.0
    assert cadence.maybe_start_periodic() is None
    assert any("inter-bot-recovery-periodic failed" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_leadership_hook_failure_is_contained(caplog):
    clock, hooks = _Clock(), _Hooks()
    cadence = RecoveryCadence(hooks, interval_seconds=15.0, clock=clock)
    caplog.set_level(logging.ERROR)
    hooks.leadership_raises = RuntimeError("boom")
    await cadence.on_leadership()
    await asyncio.sleep(0)
    assert any("inter-bot-recovery-leadership failed" in r.getMessage() for r in caplog.records)
    assert not cadence.running


@pytest.mark.asyncio
async def test_slow_leadership_pass_is_waited_on_boundedly_and_never_cancelled(monkeypatch, caplog):
    """Cancelling the await would not stop a pass's worker-thread write, and
    the hook would release its lock with that write in flight."""
    clock, hooks = _Clock(), _Hooks()
    hooks.leadership_block = asyncio.Event()
    monkeypatch.setattr(RecoveryCadence, "LEADERSHIP_WAIT_SECONDS", 0.05)
    cadence = RecoveryCadence(hooks, interval_seconds=15.0, clock=clock)
    caplog.set_level(logging.ERROR)

    await asyncio.wait_for(cadence.on_leadership(), timeout=1.0)
    assert any("still running after" in r.getMessage() for r in caplog.records)
    assert cadence.running and not hooks.leadership_cancelled

    clock.now += 100.0  # periodic is due, but shares the slot: it waits
    assert cadence.maybe_start_periodic() is None
    await asyncio.wait_for(cadence.on_leadership(), timeout=1.0)  # crash re-entry
    assert hooks.leadership_calls == 1 and hooks.periodic_calls == 0

    hooks.leadership_block.set()
    for _ in range(100):
        if not cadence.running:
            break
        await asyncio.sleep(0.005)
    assert not hooks.leadership_cancelled
    clock.now += 15.0
    periodic = cadence.maybe_start_periodic()
    assert periodic is not None
    await periodic
    assert hooks.periodic_calls == 1


@pytest.mark.asyncio
async def test_stop_cancels_a_running_leadership_pass(monkeypatch):
    clock, hooks = _Clock(), _Hooks()
    hooks.leadership_block = asyncio.Event()
    monkeypatch.setattr(RecoveryCadence, "LEADERSHIP_WAIT_SECONDS", 0.01)
    cadence = RecoveryCadence(hooks, interval_seconds=15.0, clock=clock)
    await cadence.on_leadership()
    await asyncio.wait_for(cadence.stop(), timeout=1.0)
    assert hooks.leadership_cancelled and not cadence.running


# --- dispatcher wiring ----------------------------------------------------


@pytest.mark.asyncio
async def test_run_calls_leadership_hook_after_initial_recover_expired():
    """Recovery steps defer claimed deliveries to recover_expired, so the
    leadership pass must see the claims it released."""
    clock, hooks = _Clock(), _Hooks()
    dispatcher = _dispatcher(hooks, clock)
    task = asyncio.create_task(dispatcher.run())
    await _iterate(dispatcher, 0, 1)
    await _stop(dispatcher, task)
    assert hooks.events[:2] == ["recover_expired", "leadership"]


@pytest.mark.asyncio
async def test_every_run_entry_reruns_the_leadership_hook():
    """run() is re-entered after a crash under the same leadership."""
    clock, hooks = _Clock(), _Hooks()
    dispatcher = _dispatcher(hooks, clock)
    for _ in range(2):
        dispatcher._stop_event.clear()
        task = asyncio.create_task(dispatcher.run())
        await _iterate(dispatcher, dispatcher.store.recover_expired_calls, 1)
        await _stop(dispatcher, task)
    assert hooks.leadership_calls == 2


@pytest.mark.asyncio
async def test_wake_storm_does_not_multiply_periodic_recovery():
    clock, hooks = _Clock(), _Hooks()
    dispatcher = _dispatcher(hooks, clock)
    task = asyncio.create_task(dispatcher.run())
    await _iterate(dispatcher, 0, 1)
    swept = dispatcher.store.recover_expired_calls
    await _iterate(dispatcher, swept, 40)  # announces/enqueues, no time passes
    assert hooks.periodic_calls == 0

    clock.now += 15.0
    swept = dispatcher.store.recover_expired_calls
    await _iterate(dispatcher, swept, 10)
    await asyncio.sleep(0)
    assert hooks.periodic_calls == 1
    await _stop(dispatcher, task)


@pytest.mark.asyncio
async def test_hung_periodic_never_stalls_the_delivery_sweep_and_stops_before_unlock():
    clock, hooks = _Clock(), _Hooks()
    hooks.periodic_block = asyncio.Event()
    dispatcher = _dispatcher(hooks, clock)
    dispatcher._task = asyncio.create_task(dispatcher.run())
    await _iterate(dispatcher, 0, 1)
    clock.now += 15.0
    swept = dispatcher.store.recover_expired_calls
    await _iterate(dispatcher, swept, 1)
    assert hooks.periodic_calls == 1
    swept = dispatcher.store.recover_expired_calls
    await _iterate(dispatcher, swept, 5)  # sweep keeps running while periodic hangs

    await asyncio.wait_for(dispatcher.stop(), timeout=2.0)
    assert hooks.periodic_cancelled
    assert dispatcher.store.released_with_periodic_running is False


@pytest.mark.asyncio
async def test_hung_leadership_hook_is_bounded_and_dispatch_starts(monkeypatch):
    clock, hooks = _Clock(), _Hooks()
    hooks.leadership_block = asyncio.Event()
    monkeypatch.setattr(RecoveryCadence, "LEADERSHIP_WAIT_SECONDS", 0.05)
    dispatcher = _dispatcher(hooks, clock)
    dispatcher._task = asyncio.create_task(dispatcher.run())
    await _iterate(dispatcher, 0, 3)
    assert dispatcher.store.recover_expired_calls >= 3
    assert not hooks.leadership_cancelled  # still running, not cut off
    await asyncio.wait_for(dispatcher.stop(), timeout=2.0)
    assert hooks.leadership_cancelled  # shutdown is the only cancel


async def _until(predicate, *, polls: int = 200) -> None:
    for _ in range(polls):
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition not reached")  # pragma: no cover - diagnostic


@pytest.mark.asyncio
async def test_stop_during_the_leadership_wait_is_not_held_by_the_bound(caplog):
    """Shutdown within the bound of leadership (crash re-entry, redeploy) must
    not wait the bound out: stop() cancels the pass right after anyway."""
    clock, hooks = _Clock(), _Hooks()
    hooks.leadership_block = asyncio.Event()
    assert RecoveryCadence.LEADERSHIP_WAIT_SECONDS >= 30.0  # the real bound
    dispatcher = _dispatcher(hooks, clock)
    dispatcher._task = asyncio.create_task(dispatcher.run())
    await _until(lambda: hooks.leadership_calls == 1)
    caplog.set_level(logging.ERROR)

    await asyncio.wait_for(asyncio.shield(dispatcher.stop()), timeout=2.0)

    assert hooks.leadership_cancelled
    assert dispatcher.store.recover_expired_calls == 1  # no sweep after shutdown
    assert not any("still running after" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_leadership_wait_ends_on_stop_without_cancelling_the_pass():
    clock, hooks = _Clock(), _Hooks()
    hooks.leadership_block = asyncio.Event()
    cadence = RecoveryCadence(hooks, interval_seconds=15.0, clock=clock)
    stop = asyncio.Event()
    waiting = asyncio.create_task(cadence.on_leadership(stop=stop))
    await _until(lambda: hooks.leadership_calls == 1)
    stop.set()
    await asyncio.wait_for(waiting, timeout=1.0)
    assert cadence.running and not hooks.leadership_cancelled  # stop() owns the cancel
    await asyncio.wait_for(cadence.stop(), timeout=1.0)
    assert hooks.leadership_cancelled



# --- leadership read ---------------------------------------------------


@pytest.mark.asyncio
async def test_is_leader_follows_the_dispatcher_lock():
    dispatcher = _dispatcher(_Hooks(), _Clock())
    assert dispatcher.is_leader is True
    await asyncio.wait_for(dispatcher.stop(), timeout=2.0)
    assert dispatcher.is_leader is False


def test_passive_dispatcher_is_not_leader():
    dispatcher = _dispatcher(_Hooks(), _Clock())
    dispatcher._dispatcher_lock = None
    assert dispatcher.is_leader is False


class _Sweep:
    """Records the passes ``SweepRecoveryHooks`` asks the recovery sweep for."""

    def __init__(self):
        self.calls: list[dict] = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        return {"busy": False, "steps": []}


@pytest.mark.asyncio
async def test_sweep_recovery_hooks_run_full_sweep_on_leadership_and_periodic_steps_after():
    """The hooks api.py wires satisfy the cadence: a full mutating sweep at
    leadership, then only the periodic steps once the interval elapses."""
    from llm_bawt.service.recovery.sweep import RecoverySweep, SweepRecoveryHooks

    clock, sweep = _Clock(), _Sweep()
    dispatcher = _dispatcher(_Hooks(), clock)
    dispatcher._recovery = RecoveryCadence(SweepRecoveryHooks(sweep), interval_seconds=15.0, clock=clock)
    task = asyncio.create_task(dispatcher.run())
    await _iterate(dispatcher, 0, 1)
    assert sweep.calls == [{"dry_run": False, "actor": "recovery_on_leadership"}]

    clock.now += 15.0
    await _iterate(dispatcher, dispatcher.store.recover_expired_calls, 1)
    for _ in range(200):
        if len(sweep.calls) == 2:
            break
        await asyncio.sleep(0.005)
    await _stop(dispatcher, task)
    assert sweep.calls[1] == {
        "dry_run": False, "steps": list(RecoverySweep.PERIODIC_STEPS), "actor": "recovery_periodic",
    }


