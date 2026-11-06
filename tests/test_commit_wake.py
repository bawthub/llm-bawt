"""Commit wakes replace the app's short polls (TASK-2108)."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine

from llm_bawt.service import commit_wake
from llm_bawt.service.commit_wake import WakeHub, notify_in_transaction, wait_for_wake


def run(coro):
    return asyncio.run(coro)


def test_hub_fire_from_another_thread_wakes_every_subscriber():
    async def main():
        hub = WakeHub()
        a, b = hub.subscribe(), hub.subscribe()
        threading.Thread(target=hub.fire).start()
        started = time.monotonic()
        assert await wait_for_wake(a, 5) and await wait_for_wake(b, 5)
        assert time.monotonic() - started < 1
        assert not a.is_set()  # cleared for the next wait
        assert await wait_for_wake(a, 0.05) is False
    run(main())


def test_no_hub_is_the_plain_backstop_sleep():
    started = time.monotonic()
    assert run(wait_for_wake(None, 0.05)) is False
    assert time.monotonic() - started >= 0.05
    assert commit_wake.backstop(SimpleNamespace(), 2.0) == 2.0
    assert commit_wake.backstop(SimpleNamespace(_commit_wake_hub=WakeHub()), 2.0) == commit_wake.BACKSTOP_SECONDS


def test_notify_is_a_noop_off_postgres():
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        notify_in_transaction(conn)  # no pg_notify on sqlite, never raises


def test_commit_hook_notifies_only_when_something_was_staged(monkeypatch):
    from llm_bawt import system_events

    calls = []
    monkeypatch.setattr(commit_wake, "notify_in_transaction", lambda conn, payload="": calls.append(conn))
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        system_events.CommitStagedAppends.stage(conn, lambda c: None)
    assert len(calls) == 1
    with engine.begin() as conn:
        pass
    assert len(calls) == 1
    try:
        with engine.begin() as conn:
            system_events.CommitStagedAppends.stage(conn, lambda c: None)
            raise RuntimeError("rolled back")
    except RuntimeError:
        pass
    assert len(calls) == 1


def test_system_event_outbox_publishes_on_wake_not_on_the_backstop(monkeypatch):
    from llm_bawt.service import system_event_outbox

    class Emitter:
        def __init__(self):
            self.calls = []

        def publish_batch(self, client, *, limit, report_only):
            self.calls.append(time.monotonic())
            return 0

    class Client:
        def close(self):
            pass

    monkeypatch.setattr(system_event_outbox.redis.Redis, "from_url", staticmethod(lambda *a, **k: Client()))

    async def main():
        hub, emitter = WakeHub(), Emitter()
        task = asyncio.create_task(system_event_outbox.run_system_event_outbox(emitter, "redis://x", hub=hub))
        while not emitter.calls:
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.1)
        assert len(emitter.calls) == 1  # idle: backstop is 60s
        hub.fire()
        for _ in range(100):
            if len(emitter.calls) == 2:
                break
            await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert len(emitter.calls) == 2
    run(main())


def test_continuation_loop_sleeps_until_the_next_retry_is_due(monkeypatch):
    from llm_bawt.service import approval_continuations

    passes = []

    async def once(service, store, **_):
        passes.append(time.monotonic())
        return set()

    monkeypatch.setattr(approval_continuations, "dispatch_due_continuations_once", once)
    store = SimpleNamespace(seconds_until_next_continuation=lambda **_: 0.1)

    async def main():
        service = SimpleNamespace(_commit_wake_hub=WakeHub())
        task = asyncio.create_task(approval_continuations.run_mcp_continuation_outbox(service, store))
        await asyncio.sleep(0.5)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    run(main())
    assert 2 <= len(passes) <= 8  # each due retry wakes it, not a 60s backstop


def test_home_audio_worker_claims_the_next_clip_without_waiting():
    from llm_bawt.integrations.home_audio_worker import HomeAudioWorker

    claims = []
    jobs = [{"id": "a"}, {"id": "b"}]

    class Store:
        def claim(self, owner):
            claims.append(time.monotonic())
            return jobs.pop(0) if jobs else None

    async def main():
        worker = HomeAudioWorker(None, Store(), hub=WakeHub())

        async def done(owner, job):
            return None
        worker._run_claim = done
        task = asyncio.create_task(worker.run())
        await asyncio.sleep(0.2)
        assert len(claims) == 3  # a, b, then idle on the backstop
        worker.hub.fire()
        await asyncio.sleep(0.1)
        assert len(claims) == 4
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    run(main())


class _RunRedis:
    """Just enough async Redis for ``wait_any_run_done``."""

    def __init__(self):
        self.terminal: set[str] = set()
        self.streams: dict[str, list] = {}
        self.appended = asyncio.Event()

    async def exists(self, key):
        return int(key in self.terminal)

    async def xrevrange(self, key, count=1):
        return list(reversed(self.streams.get(key, [])))[:count]

    async def xread(self, cursors, count=50, block=0):
        deadline = time.monotonic() + block / 1000
        while time.monotonic() < deadline:
            out = []
            for key, after in cursors.items():
                fresh = [(i, f) for i, f in self.streams.get(key, []) if i > after]
                if fresh:
                    out.append((key, fresh))
            if out:
                return out
            await asyncio.sleep(0.01)
        return []

    def add(self, key, fields):
        entries = self.streams.setdefault(key, [])
        entries.append((f"{len(entries) + 1}-0", fields))


def _subscriber(redis):
    from agent_bridge.subscriber import RedisSubscriber as AgentSubscriber

    sub = AgentSubscriber.__new__(AgentSubscriber)
    sub._redis = sub._pub_redis = redis
    return sub


def test_wait_any_run_done_returns_when_a_run_publishes_done():
    async def main():
        redis = _RunRedis()
        redis.add("agent:run:r1", {"type": "delta"})
        sub = _subscriber(redis)

        async def finish():
            await asyncio.sleep(0.1)
            redis.add("agent:run:r1", {"done": "1"})
        asyncio.create_task(finish())
        started = time.monotonic()
        assert await sub.wait_any_run_done(["r1", "r2"], 5) is True
        assert time.monotonic() - started < 1
        assert await sub.wait_any_run_done(["r2"], 0.1) is False
        redis.terminal.add("agent:run:r2:terminal")
    run(main())


def test_wait_any_run_done_sees_an_existing_terminal_record(monkeypatch):
    from agent_bridge import subscriber as mod

    monkeypatch.setattr(mod, "run_terminal_key", lambda rid: f"term:{rid}")

    async def main():
        redis = _RunRedis()
        redis.terminal.add("term:r9")
        assert await _subscriber(redis).wait_any_run_done(["r9"], 5) is True
    run(main())


def test_follow_up_wakes_on_run_end_with_a_minimum_gap():
    from llm_bawt.service.inter_bot_dispatcher_recovery import StartupRecoveryRunner

    class Check:
        def __init__(self):
            self.waits = []

        async def wait_for_runs(self, result, timeout):
            self.waits.append(timeout)

    async def main():
        runner = StartupRecoveryRunner(Check())
        runner.FOLLOW_UP_MIN_GAP_SECONDS = 0.1
        started = time.monotonic()
        await runner._wait_for_next_pass({"steps": []})
        assert 0.1 <= time.monotonic() - started < 1
        assert runner.check.waits == [runner.FOLLOW_UP_SECONDS]
    run(main())


@pytest.mark.parametrize("last", [None])
def test_follow_up_after_a_failed_pass_sleeps_the_interval(last):
    from llm_bawt.service.inter_bot_dispatcher_recovery import StartupRecoveryRunner

    async def main():
        runner = StartupRecoveryRunner(SimpleNamespace(wait_for_runs=None))
        runner.FOLLOW_UP_SECONDS = 0.05
        started = time.monotonic()
        await runner._wait_for_next_pass(last)
        assert time.monotonic() - started >= 0.05
    run(main())


class _ContinuationStore:
    """Rows ``r00..r20`` all due; ``parked`` ids wait on an active ops job."""

    def __init__(self, n, parked):
        self.rows = {f"r{i:02d}": "pending" for i in range(n)}
        self.parked = set(parked)
        self.finds = 0
        self.dispatched = []

    def find_pending_continuations(self, *, limit, exclude_ids=()):
        self.finds += 1
        due = [rid for rid, state in sorted(self.rows.items()) if state == "pending" and rid not in exclude_ids]
        return [SimpleNamespace(id=rid) for rid in due[:limit]]

    def claim_continuation(self, rid, lease_seconds):
        self.rows[rid] = "dispatching"
        return SimpleNamespace(id=rid)

    def seconds_until_next_continuation(self, *, exclude_ids=()):
        due = [rid for rid, state in self.rows.items() if state == "pending" and rid not in exclude_ids]
        return 0.0 if due else None


def _continuation_loop(monkeypatch, store, seconds):
    from llm_bawt.service import approval_continuations as mod

    async def no_recovery(_store):
        return None

    async def dispatch(service, store_, claimed, result_override=None):
        store_.dispatched.append(claimed.id)
        store_.rows[claimed.id] = "delivered"

    monkeypatch.setattr(mod, "recover_approval_requests", no_recovery)
    monkeypatch.setattr(mod, "_terminal_ops_result",
                        lambda service, pending: None if pending.id in store.parked else {})
    monkeypatch.setattr(mod, "_admission_card_result", lambda pending: None)
    monkeypatch.setattr(mod, "dispatch_mcp_result_continuation", dispatch)

    async def main():
        service = SimpleNamespace(_commit_wake_hub=WakeHub())
        task = asyncio.create_task(mod.run_mcp_continuation_outbox(service, store))
        await asyncio.sleep(seconds)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    run(main())


def test_a_full_parked_batch_does_not_hide_a_runnable_row(monkeypatch):
    store = _ContinuationStore(21, parked=[f"r{i:02d}" for i in range(20)])
    _continuation_loop(monkeypatch, store, 0.5)
    assert store.dispatched == ["r20"]
    assert store.finds <= 4  # head batch, the tail, then idle on the backstop


def test_all_parked_rows_do_not_spin_the_loop(monkeypatch):
    store = _ContinuationStore(5, parked=[f"r{i:02d}" for i in range(5)])
    _continuation_loop(monkeypatch, store, 0.5)
    assert store.dispatched == [] and store.finds == 1
