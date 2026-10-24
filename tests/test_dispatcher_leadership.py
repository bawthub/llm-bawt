"""The dispatcher's leadership lock session dies with PostgreSQL (TASK-1311).

The lock is session-level: the nightly PostgreSQL restart ends the session
and the lock with it. Before this, the leader never noticed (it only
re-acquired when its lock attribute was None) and shutdown's unlock on the
dead connection raised out of the lifespan. These tests pin:

A. ``release_lock_session`` never raises on a dead session (WARNING only),
   and still raises on anything that is not a session error.
B. The leader probes its session off the loop, boundedly; a dead or hung
   probe drops leadership, no sweep (so no new claim) runs while leaderless,
   in-flight dispatch tasks are left alone, acquisition retries on the
   passive cadence (no hot loop while PostgreSQL is down), and a re-acquired
   term reruns the startup recovery check.

Pure asyncio unit tests against stubbed collaborators; no DB.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from types import SimpleNamespace

import pytest
from sqlalchemy.exc import OperationalError

from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
from llm_bawt.service import dispatcher_leadership as leadership
from llm_bawt.service.inter_bot_dispatcher import InterBotDeliveryDispatcher
from llm_bawt.service.inter_bot_dispatcher_recovery import StartupRecoveryRunner


def _session_gone() -> OperationalError:
    return OperationalError(
        "SELECT 1", {}, Exception("server closed the connection unexpectedly")
    )


class _Conn:
    """A lock connection whose server side may be gone."""

    def __init__(self, *, dead: bool = False, error: Exception | None = None):
        self.info: dict = {}
        self.dead = dead
        self.error = error
        self.executed: list[str] = []
        self.commits = 0
        self.closes = 0

    def execute(self, statement, params=None):
        self.executed.append(str(statement))
        if self.error is not None:
            raise self.error
        if self.dead:
            raise _session_gone()
        return None

    def commit(self):
        self.commits += 1

    def close(self):
        self.closes += 1
        if self.dead:
            raise _session_gone()


async def _until(predicate, *, polls: int = 400) -> None:
    for _ in range(polls):
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition not reached")  # pragma: no cover - diagnostic


# --- A: release on a dead session ------------------------------------------


def test_release_on_a_dead_session_warns_and_does_not_raise(caplog):
    conn = _Conn(dead=True)
    conn.info[leadership.LOCK_NAME_INFO_KEY] = "inter-bot-dispatcher"
    caplog.set_level(logging.WARNING, logger=leadership.logger.name)

    leadership.release_lock_session(conn)  # must not raise

    assert conn.closes == 1
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "already gone at release" in warnings[0].getMessage()
    assert "'inter-bot-dispatcher'" in warnings[0].getMessage()


def test_release_still_raises_on_a_non_session_error():
    """The catch is narrow: a bug is not a dead session."""
    conn = _Conn(error=RuntimeError("not a connection problem"))
    with pytest.raises(RuntimeError):
        leadership.release_lock_session(conn)
    assert conn.closes == 1  # still closed


def test_release_on_a_live_session_unlocks_commits_and_closes():
    conn = _Conn()
    leadership.release_lock_session(conn)
    assert any("pg_advisory_unlock" in s for s in conn.executed)
    assert conn.commits == 1 and conn.closes == 1


def test_store_release_tolerates_a_dead_session_and_none():
    """What ``stop()`` calls: the store method, not the module function."""
    store = InterBotDeliveryStore.__new__(InterBotDeliveryStore)
    store.release_dispatcher_lock(_Conn(dead=True))
    store.release_dispatcher_lock(None)


# --- probe ---------------------------------------------------------------------


def test_probe_reports_a_live_session_and_leaves_it_idle():
    conn = _Conn()
    assert leadership.probe_lock_session(conn) is True
    assert conn.commits == 1 and conn.closes == 0


def test_probe_reports_a_dead_session_and_closes_it(caplog):
    conn = _Conn(dead=True)
    caplog.set_level(logging.WARNING, logger=leadership.logger.name)
    assert leadership.probe_lock_session(conn) is False
    assert conn.closes == 1
    assert any("no longer answers" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_probe_within_returns_the_probe_result():
    abandoned: list[bool] = []
    assert await leadership.probe_within(lambda: True, timeout=1.0, abandon=lambda: abandoned.append(True))
    assert not await leadership.probe_within(lambda: False, timeout=1.0, abandon=lambda: abandoned.append(True))
    assert abandoned == []


@pytest.mark.asyncio
async def test_a_hung_probe_is_lost_and_its_late_live_answer_closes_the_session(caplog):
    """The thread cannot be interrupted; a session it later reports alive is
    closed anyway, or it would keep the lock this process gave up on."""
    release = threading.Event()
    abandoned = threading.Event()

    def probe() -> bool:
        release.wait(timeout=5.0)
        return True

    caplog.set_level(logging.WARNING, logger=leadership.logger.name)
    assert await leadership.probe_within(probe, timeout=0.05, abandon=abandoned.set) is False
    assert any("did not answer within" in r.getMessage() for r in caplog.records)
    assert not abandoned.is_set()

    release.set()
    await asyncio.to_thread(abandoned.wait, 2.0)
    assert abandoned.is_set()


@pytest.mark.asyncio
async def test_a_hung_probe_that_later_reports_dead_is_not_abandoned_twice():
    release = threading.Event()
    abandoned: list[bool] = []

    def probe() -> bool:
        release.wait(timeout=5.0)
        return False  # probe_lock_session already closed it

    assert await leadership.probe_within(probe, timeout=0.05, abandon=lambda: abandoned.append(True)) is False
    release.set()
    await asyncio.sleep(0.1)
    assert abandoned == []


# --- B: dispatcher ---------------------------------------------------------------


class _Check:
    def __init__(self):
        self.calls = 0

    async def run_at_startup(self) -> None:
        self.calls += 1


class _Store:
    """PostgreSQL as the lock session sees it: up, or restarting."""

    def __init__(self):
        self.pg_up = True
        self.acquires = 0
        self.sessions: list[_Conn] = []
        self.probes = 0
        self.probe_block: threading.Event | None = None
        self.recover_expired_calls = 0
        self.released: list = []

    def acquire_dispatcher_lock(self):
        self.acquires += 1
        if not self.pg_up:
            raise _session_gone()
        conn = _Conn()
        self.sessions.append(conn)
        return conn

    def probe_dispatcher_lock(self, conn):
        self.probes += 1
        if self.probe_block is not None:
            self.probe_block.wait(timeout=5.0)
            return True
        conn.dead = not self.pg_up
        return leadership.probe_lock_session(conn)

    def release_dispatcher_lock(self, conn):
        self.released.append(conn)

    def recover_expired(self, *, live_owner=None):
        self.recover_expired_calls += 1

    def eligible_targets(self):
        return []


def _dispatcher(check: _Check | None = None) -> InterBotDeliveryDispatcher:
    dispatcher = InterBotDeliveryDispatcher.__new__(InterBotDeliveryDispatcher)
    dispatcher.service = SimpleNamespace(
        config=SimpleNamespace(DEFAULT_USER="nick"),
        _redis_subscriber=SimpleNamespace(connected=True),
    )
    dispatcher.store = _Store()
    dispatcher.recovery_interval_seconds = 60.0  # only wakes iterate the loop
    dispatcher.leadership_retry_delay_seconds = 0.0
    dispatcher.LOCK_PROBE_INTERVAL_SECONDS = 0.0  # probe on every sweep
    dispatcher.LOCK_PROBE_TIMEOUT_SECONDS = 1.0
    dispatcher.LEADERSHIP_RETRY_SECONDS = 0.02
    dispatcher.RELEASE_TIMEOUT_SECONDS = 0.05
    dispatcher._wake_event = asyncio.Event()
    dispatcher._recovery = (
        StartupRecoveryRunner(
            check, on_done=dispatcher._wake_event.set, is_leader=lambda: dispatcher.is_leader,
        )
        if check else None
    )
    dispatcher._stop_event = asyncio.Event()
    dispatcher._task = None
    dispatcher._leader_task = None
    dispatcher._loop = None
    dispatcher._target_tasks = {}
    dispatcher._dispatch_tasks = set()
    dispatcher._announce_tasks = set()
    dispatcher.side_pumps = None
    dispatcher.claim_owner = "test-dispatcher"
    dispatcher._dispatcher_lock = None
    dispatcher._lock_probed_at = None
    dispatcher._subscriber_gap_logged = False
    return dispatcher


async def _sweep(dispatcher) -> None:
    before = dispatcher.store.recover_expired_calls
    dispatcher._wake_event.set()
    await _until(lambda: dispatcher.store.recover_expired_calls > before)


@pytest.mark.asyncio
async def test_a_dead_lock_session_drops_leadership_stops_sweeping_and_is_reacquired(caplog):
    check = _Check()
    dispatcher = _dispatcher(check)
    store = dispatcher.store
    caplog.set_level(logging.WARNING)
    dispatcher.start()
    await _until(lambda: store.acquires == 1 and store.recover_expired_calls >= 1)
    await _sweep(dispatcher)
    assert dispatcher.is_leader and check.calls == 1
    first_session = dispatcher._dispatcher_lock

    # In-flight work under its own claim lease: leadership loss must not touch it.
    in_flight_gate = asyncio.Event()
    in_flight = asyncio.create_task(in_flight_gate.wait())
    dispatcher._dispatch_tasks.add(in_flight)

    store.pg_up = False  # the nightly restart
    dispatcher._wake_event.set()
    await _until(lambda: not dispatcher.is_leader)
    assert first_session.closes == 1
    assert any("lost its leadership lock session" in r.getMessage() and r.levelno == logging.ERROR
               for r in caplog.records)
    await _until(lambda: store.acquires >= 3)  # retrying, PostgreSQL still down
    sweeps_while_leaderless = store.recover_expired_calls
    for _ in range(3):
        dispatcher._wake_event.set()  # wakes land on no run loop: no sweep, no claim
        await asyncio.sleep(0.01)
    assert store.recover_expired_calls == sweeps_while_leaderless
    assert not in_flight.done() and not in_flight.cancelled()
    assert not any("is passive" in r.getMessage() for r in caplog.records)

    store.pg_up = True  # PostgreSQL is back
    await _until(lambda: dispatcher.is_leader)
    assert dispatcher._dispatcher_lock is not first_session
    await _until(lambda: check.calls == 2)  # the new term reruns the startup check
    await _until(lambda: store.recover_expired_calls > sweeps_while_leaderless)
    await _sweep(dispatcher)

    in_flight_gate.set()
    await asyncio.wait_for(dispatcher.stop(), timeout=2.0)
    assert store.released == [dispatcher.store.sessions[-1]]


@pytest.mark.asyncio
async def test_acquisition_while_postgres_is_down_is_paced_not_a_hot_loop(caplog):
    dispatcher = _dispatcher()
    dispatcher.LEADERSHIP_RETRY_SECONDS = 0.05
    dispatcher.store.pg_up = False
    caplog.set_level(logging.INFO)
    dispatcher.start()
    await asyncio.sleep(0.3)
    attempts = dispatcher.store.acquires
    assert 2 <= attempts <= 8, attempts
    assert not dispatcher.is_leader
    assert any("cannot reach PostgreSQL for leadership" in r.getMessage() for r in caplog.records)
    assert not any("is passive" in r.getMessage() for r in caplog.records)
    await asyncio.wait_for(dispatcher.stop(), timeout=2.0)


@pytest.mark.asyncio
async def test_the_probe_is_rate_limited_by_the_interval():
    dispatcher = _dispatcher()
    dispatcher.LOCK_PROBE_INTERVAL_SECONDS = 60.0
    assert await dispatcher._acquire_leadership() is True
    assert await dispatcher._leadership_confirmed() is True
    assert await dispatcher._leadership_confirmed() is True
    assert dispatcher.store.probes == 0  # just acquired: nothing to probe yet
    dispatcher._lock_probed_at = None
    assert await dispatcher._leadership_confirmed() is True
    assert dispatcher.store.probes == 1


@pytest.mark.asyncio
async def test_a_hung_probe_drops_leadership_and_closes_the_session_once_it_answers(caplog):
    dispatcher = _dispatcher()
    store = dispatcher.store
    store.probe_block = threading.Event()
    dispatcher.LOCK_PROBE_TIMEOUT_SECONDS = 0.05
    caplog.set_level(logging.WARNING)
    assert await dispatcher._acquire_leadership() is True
    session = dispatcher._dispatcher_lock
    dispatcher._lock_probed_at = None

    assert await dispatcher._leadership_confirmed() is False
    assert not dispatcher.is_leader
    assert session.closes == 0  # the thread is still stuck in the probe

    store.probe_block.set()
    await _until(lambda: session.closes == 1)


@pytest.mark.asyncio
async def test_stop_during_the_probe_never_sweeps_after_the_workers_were_cancelled():
    """Codex r1 blocker 2: stop() cancels the workers while the probe is out;
    a probe that then answers "alive" must not start one more sweep."""
    dispatcher = _dispatcher()
    store = dispatcher.store
    store.probe_block = threading.Event()
    dispatcher.start()
    await _until(lambda: store.recover_expired_calls >= 1)
    dispatcher._wake_event.set()
    await _until(lambda: store.probes >= 1)  # the probe thread is now blocked
    sweeps_before_stop = store.recover_expired_calls

    stop_task = asyncio.create_task(dispatcher.stop())
    await _until(lambda: dispatcher._stop_event.is_set())
    store.probe_block.set()  # the session answers "alive" after stop() began
    await asyncio.wait_for(stop_task, timeout=2.0)
    assert store.recover_expired_calls == sweeps_before_stop


@pytest.mark.asyncio
async def test_a_new_term_gets_its_own_recovery_pass_once_the_old_one_settles():
    """Codex r1 blocker 1: a check still running at re-acquisition may be on
    the dead term's connections and fail; the new term needs its own pass."""
    finish_first = asyncio.Event()
    woke: list[bool] = []

    class _SlowFailingCheck:
        def __init__(self):
            self.calls = 0

        async def run_at_startup(self):
            self.calls += 1
            if self.calls == 1:
                await finish_first.wait()
                raise RuntimeError("first term's connection died")

    check = _SlowFailingCheck()
    runner = StartupRecoveryRunner(check, on_done=lambda: woke.append(True))
    runner.WAIT_SECONDS = 0.01
    await runner.run_once()  # first term: bound passes, the check runs on
    assert runner.running and check.calls == 1

    runner.rearm()  # PostgreSQL is back, a new term is acquired
    await runner.run_once()
    assert check.calls == 1  # never doubled while the old pass runs

    finish_first.set()
    await _until(lambda: check.calls == 2)
    await _until(lambda: not runner.running)
    assert woke == [True, True]  # both ends woke dispatch
    await runner.run_once()  # the new term's pass is not repeated either
    assert check.calls == 2


@pytest.mark.asyncio
async def test_a_pending_pass_never_starts_leaderless_and_the_next_term_still_gets_one(caplog):
    """Codex r2 blocker: the lock dies twice. The first term's check settles
    while nobody holds the lock: no pass (leader-only work). The term that
    then acquires the lock runs its own pass."""
    finish_first = asyncio.Event()

    class _LongCheck:
        def __init__(self):
            self.calls = 0

        async def run_at_startup(self):
            self.calls += 1
            if self.calls == 1:
                await finish_first.wait()

    check = _LongCheck()
    dispatcher = _dispatcher(check)
    dispatcher._recovery.WAIT_SECONDS = 0.01
    store = dispatcher.store
    caplog.set_level(logging.INFO, logger="llm_bawt.service.inter_bot_dispatcher_recovery")
    dispatcher.start()
    await _until(lambda: check.calls == 1 and dispatcher.is_leader)

    store.pg_up = False  # first loss
    dispatcher._wake_event.set()
    await _until(lambda: not dispatcher.is_leader)
    store.pg_up = True
    await _until(lambda: dispatcher.is_leader and dispatcher._recovery._term_pending)
    store.pg_up = False  # second loss, the old check still running
    dispatcher._wake_event.set()
    await _until(lambda: not dispatcher.is_leader)

    finish_first.set()  # the old check settles while nobody holds the lock
    await _until(lambda: not dispatcher._recovery.running)
    await asyncio.sleep(0.05)
    assert check.calls == 1 and not dispatcher.is_leader
    assert any("not rerun: leadership was lost again" in r.getMessage() for r in caplog.records)

    store.pg_up = True  # a valid term again: it starts like a startup
    await _until(lambda: check.calls == 2)
    assert dispatcher.is_leader
    await asyncio.wait_for(dispatcher.stop(), timeout=2.0)


@pytest.mark.asyncio
async def test_the_runner_starts_a_pending_pass_only_under_leadership():
    gate = asyncio.Event()
    leader = [True]

    class _Check:
        def __init__(self):
            self.calls = 0

        async def run_at_startup(self):
            self.calls += 1
            if self.calls == 1:
                await gate.wait()

    check = _Check()
    runner = StartupRecoveryRunner(check, is_leader=lambda: leader[0])
    runner.WAIT_SECONDS = 0.01
    await runner.run_once()
    runner.rearm()
    assert runner._term_pending
    leader[0] = False
    gate.set()
    await _until(lambda: not runner.running)
    await asyncio.sleep(0.01)
    assert check.calls == 1 and not runner._term_pending
    leader[0] = True
    runner.rearm()  # the next acquired term
    await runner.run_once()
    assert check.calls == 2


@pytest.mark.asyncio
async def test_the_dispatcher_wires_its_leadership_into_the_runner(monkeypatch):
    monkeypatch.setattr(
        "llm_bawt.service.inter_bot_dispatcher.InterBotDeliveryStore",
        lambda config: SimpleNamespace(engine=None),
    )
    dispatcher = InterBotDeliveryDispatcher(
        SimpleNamespace(config=SimpleNamespace()), startup_recovery=_Check(),
    )
    assert dispatcher._recovery._is_leader() is False
    dispatcher._dispatcher_lock = object()
    assert dispatcher._recovery._is_leader() is True


@pytest.mark.asyncio
async def test_rearm_after_the_old_pass_ended_starts_one_new_pass_and_stop_restarts_nothing():
    gate = asyncio.Event()

    class _Check:
        def __init__(self):
            self.calls = 0

        async def run_at_startup(self):
            self.calls += 1
            await gate.wait()

    check = _Check()
    runner = StartupRecoveryRunner(check)
    runner.WAIT_SECONDS = 0.01
    gate.set()
    await runner.run_once()
    await _until(lambda: not runner.running)
    runner.rearm()
    await runner.run_once()
    assert check.calls == 2

    gate.clear()
    runner.rearm()  # nothing running: no pending term either
    await runner.run_once()
    assert check.calls == 3 and runner.running
    runner.rearm()  # pending while the third pass runs
    await runner.stop()  # shutdown cancels it; the pending term never starts
    await asyncio.sleep(0.01)
    assert check.calls == 3 and not runner.running


@pytest.mark.asyncio
async def test_leaderless_is_confirmed_false_without_probing():
    dispatcher = _dispatcher()
    assert await dispatcher._leadership_confirmed() is False
    assert dispatcher.store.probes == 0


@pytest.mark.asyncio
async def test_a_held_lock_is_passive_and_logged_once(caplog):
    dispatcher = _dispatcher()
    dispatcher.store.acquire_dispatcher_lock = lambda: None
    caplog.set_level(logging.INFO)
    dispatcher.start()
    await _until(lambda: any("is passive" in r.getMessage() for r in caplog.records))
    await asyncio.sleep(0.1)
    assert sum("is passive" in r.getMessage() for r in caplog.records) == 1
    await asyncio.wait_for(dispatcher.stop(), timeout=2.0)


@pytest.mark.asyncio
async def test_stop_bounds_a_hung_release_and_still_completes(caplog):
    dispatcher = _dispatcher()
    release = threading.Event()
    dispatcher.store.release_dispatcher_lock = lambda conn: release.wait(timeout=5.0)
    assert await dispatcher._acquire_leadership() is True
    caplog.set_level(logging.WARNING)
    await asyncio.wait_for(dispatcher.stop(), timeout=1.0)
    assert not dispatcher.is_leader
    assert any("release still pending" in r.getMessage() for r in caplog.records)
    release.set()


# --- the real store builds a dedicated keepalive session ----------------------


def test_lock_engine_is_a_dedicated_nullpool_session_with_keepalives(monkeypatch):
    from sqlalchemy.pool import NullPool

    captured: dict = {}

    def fake_create_engine(url, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(url=url)

    monkeypatch.setattr("sqlalchemy.create_engine", fake_create_engine)
    monkeypatch.setattr("llm_bawt.utils.db.build_postgres_url", lambda config: "postgresql://x")
    monkeypatch.setattr("llm_bawt.utils.db.set_utc_on_connect", lambda engine: None)

    engine = leadership.build_lock_engine(SimpleNamespace())
    assert engine is not None
    assert captured["poolclass"] is NullPool
    args = captured["connect_args"]
    assert args["application_name"] == "llm-bawt:dispatcher-lock"
    assert args["keepalives"] == 1 and args["tcp_user_timeout"] == 15_000
    assert args["connect_timeout"] == 10


def test_lock_engine_is_none_without_postgres(monkeypatch):
    monkeypatch.setattr("llm_bawt.utils.db.build_postgres_url", lambda config: None)
    assert leadership.build_lock_engine(SimpleNamespace()) is None
