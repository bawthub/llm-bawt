"""TASK-1308: the dispatcher wakes when a delayed retry is due, not 15 s later.

Driven on the virtual clock: the fake store's DB clock is the loop clock, a
"scan" claims what's due and claimable, and side work can take time.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from llm_bawt.service import inter_bot_dispatcher as dispatcher_module
from llm_bawt.service.inter_bot_dispatcher import (
    MIN_DEADLINE_WAIT_S,
    InterBotDeliveryDispatcher,
    next_wait_seconds,
)
from tests.virtual_clock import run as run_virtual

INTERVAL = 15.0
EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


class _Clock:
    def __init__(self):
        self.start = asyncio.get_running_loop().time()

    def now(self) -> datetime:
        return EPOCH + timedelta(seconds=asyncio.get_running_loop().time() - self.start)

    def t(self) -> float:
        return asyncio.get_running_loop().time() - self.start


class _Store:
    """Rows are (due_seconds, claimable). A scan claims due claimable rows."""

    def __init__(self, clock: _Clock, rows=(), fail_query: bool = False):
        self.clock = clock
        self.rows = list(rows)
        self.fail_query = fail_query
        self.last_scan_at = None
        self.scans: list[float] = []
        self.claimed: list[float] = []

    def recover_expired(self, *, live_owner=None):
        return []

    def scan(self):
        now = self.clock.now()
        self.scans.append(self.clock.t())
        keep = []
        for due, claimable in self.rows:
            if EPOCH + timedelta(seconds=due) <= now and claimable:
                self.claimed.append(self.clock.t())
            else:
                keep.append((due, claimable))
        self.rows = keep
        self.last_scan_at = now

    def next_due_after(self, watermark):
        if self.fail_query:
            raise RuntimeError("pool invalidated")
        dues = [EPOCH + timedelta(seconds=d) for d, _ in self.rows]
        future = [d for d in dues if d > watermark]
        return (min(future) if future else None), self.clock.now()


class _SidePumps:
    def __init__(self, durations):
        self.durations = list(durations)

    async def run_pass(self, _emit):
        if self.durations:
            await asyncio.sleep(self.durations.pop(0))


def _dispatcher(store, *, side=()):
    d = InterBotDeliveryDispatcher.__new__(InterBotDeliveryDispatcher)
    d.service = SimpleNamespace(config=SimpleNamespace(DEFAULT_USER="operator"),
                                _redis_subscriber=SimpleNamespace(connected=True))
    d.store = store
    d.recovery_interval_seconds = INTERVAL
    d._wake_event = asyncio.Event()
    d._stop_event = asyncio.Event()
    d._task = d._leader_task = d._loop = None
    d._target_tasks = {}
    d._dispatch_tasks = set()
    d.claim_owner = "test-dispatcher"
    d._dispatcher_lock = object()
    d._subscriber_gap_logged = False
    d.side_pumps = _SidePumps(side)
    d._schedule_eligible_targets = store.scan

    async def confirmed():
        return True

    d._leadership_confirmed = confirmed

    async def inline(fn, *args):  # the virtual loop refuses threads
        return fn(*args)

    d._off_loop = inline
    return d


def _drive(body):
    return run_virtual(body(), follow=[dispatcher_module])


async def _run_for(d, seconds: float):
    task = asyncio.create_task(d.run())
    await asyncio.sleep(seconds)
    d._stop_event.set()
    d._wake_event.set()
    await asyncio.wait_for(task, timeout=1.0)


def test_wait_math_floors_caps_and_subtracts_query_time():
    now = EPOCH
    assert next_wait_seconds(None, now, 0.0, INTERVAL) == INTERVAL
    assert next_wait_seconds(now + timedelta(seconds=1), now, 0.0, INTERVAL) == 1.0
    assert next_wait_seconds(now + timedelta(seconds=1), now, 0.4, INTERVAL) == 0.6
    assert next_wait_seconds(now - timedelta(seconds=3), now, 0.0, INTERVAL) == MIN_DEADLINE_WAIT_S
    assert next_wait_seconds(now + timedelta(hours=1), now, 0.0, INTERVAL) == INTERVAL


def test_retry_due_in_one_second_runs_at_one_second():
    async def body():
        store = _Store(_Clock(), rows=[(1.0, True)])
        await _run_for(_dispatcher(store), 5.0)
        return store.claimed

    assert _drive(body) == [1.0]


def test_deadline_crossed_during_side_work_runs_right_after_it():
    async def body():
        store = _Store(_Clock(), rows=[(1.0, True)])
        await _run_for(_dispatcher(store, side=[1.2]), 5.0)
        return store.claimed

    [claimed] = _drive(body)
    assert claimed == 1.2 + MIN_DEADLINE_WAIT_S


def test_overdue_unclaimable_head_does_not_spin():
    async def body():
        store = _Store(_Clock(), rows=[(0.0, False), (1.0, False)])
        await _run_for(_dispatcher(store), 60.0)
        return store.scans

    scans = _drive(body)
    # The first scan, one at the 1 s deadline, then one per recovery interval.
    assert scans[:2] == [0.0, 1.0]
    assert len(scans) <= 2 + int(60 / INTERVAL) + 1


def test_empty_queue_and_query_failure_fall_back_to_the_interval():
    async def body(fail):
        store = _Store(_Clock(), fail_query=fail)
        await _run_for(_dispatcher(store), 31.0)
        return store.scans

    for fail in (False, True):
        assert _drive(lambda: body(fail))[:3] == [0.0, INTERVAL, 2 * INTERVAL]


def test_explicit_wake_still_runs_before_a_later_deadline():
    async def body():
        store = _Store(_Clock(), rows=[(10.0, True)])
        d = _dispatcher(store)
        task = asyncio.create_task(d.run())
        await asyncio.sleep(2.0)
        d.wake()
        await asyncio.sleep(0)
        await asyncio.sleep(12.0)
        d._stop_event.set()
        d._wake_event.set()
        await task
        return store.scans, store.claimed

    scans, claimed = _drive(body)
    assert 2.0 in scans
    assert claimed == [10.0]


def test_leader_loss_returns_without_another_claim():
    async def body():
        store = _Store(_Clock(), rows=[(1.0, True)])
        d = _dispatcher(store)

        async def lost():
            return False

        d._leadership_confirmed = lost
        await asyncio.wait_for(d.run(), timeout=30.0)
        return store.scans, store.claimed

    scans, claimed = _drive(body)
    assert scans == [] and claimed == []


def test_stop_during_the_deadline_query_claims_nothing_after():
    async def body():
        store = _Store(_Clock(), rows=[(1.0, True)])
        d = _dispatcher(store)

        async def stopping(fn, *args):
            d._stop_event.set()
            d._wake_event.set()
            return fn(*args)

        d._off_loop = stopping
        await asyncio.wait_for(d.run(), timeout=30.0)
        return store.scans, store.claimed

    scans, claimed = _drive(body)
    assert scans == [0.0] and claimed == []
