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

Pure asyncio unit tests against stubbed collaborators; no DB. Anything timed
runs on ``tests/virtual_clock.run_timing`` at the production intervals and
timeouts (TASK-1457): the clock jumps between timers, a held worker thread's
deadline is passed with ``loop.advance()``, and waits are driven by the fakes
(``tests/state_changes``), never by polling.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
from types import SimpleNamespace

import pytest
from sqlalchemy.exc import OperationalError

from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
from llm_bawt.service import dispatcher_leadership as leadership
from llm_bawt.service.inter_bot_dispatcher import InterBotDeliveryDispatcher
from llm_bawt.service.inter_bot_dispatcher_recovery import StartupRecoveryRunner
from tests import virtual_clock
from tests.state_changes import StateChanges

_DEFAULTS = inspect.signature(InterBotDeliveryDispatcher.__init__).parameters
SWEEP_S = _DEFAULTS["recovery_interval_seconds"].default
RESTART_DELAY_S = _DEFAULTS["leadership_retry_delay_seconds"].default
PROBE_EVERY_S = InterBotDeliveryDispatcher.LOCK_PROBE_INTERVAL_SECONDS
PROBE_S = InterBotDeliveryDispatcher.LOCK_PROBE_TIMEOUT_SECONDS
RETRY_S = InterBotDeliveryDispatcher.LEADERSHIP_RETRY_SECONDS
RELEASE_S = InterBotDeliveryDispatcher.RELEASE_TIMEOUT_SECONDS
WAIT_S = StartupRecoveryRunner.WAIT_SECONDS
PROMPT_S = RETRY_S / 2
"""Virtual seconds for a step that waits out no timer: a few scheduler turns
(20ms each under the timing report) and worker threads that answer at once.
Half the acquisition cadence, so meeting it proves no retry was waited out."""
HANG_GUARD_S = 10.0
"""Real seconds a held fake waits to be released before it fails instead of hanging."""


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
        self.closed = threading.Event()

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
        self.closed.set()
        if self.dead:
            raise _session_gone()


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
    assert await leadership.probe_within(
        lambda: True, timeout=PROBE_S, abandon=lambda: abandoned.append(True))
    assert not await leadership.probe_within(
        lambda: False, timeout=PROBE_S, abandon=lambda: abandoned.append(True))
    assert abandoned == []


def _held_probe(entered: threading.Event, release: threading.Event, answer: bool):
    """A probe stuck on a half-open socket until the test releases it."""
    def probe() -> bool:
        entered.set()
        if not release.wait(HANG_GUARD_S):
            raise AssertionError(f"held probe was never released within {HANG_GUARD_S:g}s")
        return answer
    return probe


async def _timed_out(entered: threading.Event, probing: asyncio.Task) -> bool:
    """Pass the probe's deadline while its thread is held, and return its verdict."""
    loop = asyncio.get_running_loop()
    started = loop.time()
    assert await asyncio.to_thread(entered.wait, HANG_GUARD_S)
    loop.advance(PROBE_S)  # the probe never answers within its bound
    verdict = await probing
    assert loop.time() - started >= PROBE_S
    return verdict


def test_a_hung_probe_is_lost_and_its_late_live_answer_closes_the_session(caplog):
    """The thread cannot be interrupted; a session it later reports alive is
    closed anyway, or it would keep the lock this process gave up on."""
    caplog.set_level(logging.WARNING, logger=leadership.logger.name)

    async def main():
        entered, release, abandoned = threading.Event(), threading.Event(), threading.Event()
        probing = asyncio.create_task(leadership.probe_within(
            _held_probe(entered, release, True), timeout=PROBE_S, abandon=abandoned.set,
        ))
        try:
            assert await _timed_out(entered, probing) is False
            assert not abandoned.is_set()
        finally:
            release.set()
        assert await asyncio.to_thread(abandoned.wait, HANG_GUARD_S)

    virtual_clock.run_timing(main())
    assert any("did not answer within" in r.getMessage() for r in caplog.records)


def test_a_hung_probe_that_later_reports_dead_is_not_abandoned_twice():
    abandoned: list[bool] = []

    async def main():
        entered, release = threading.Event(), threading.Event()
        probing = asyncio.create_task(leadership.probe_within(
            _held_probe(entered, release, False),  # probe_lock_session already closed it
            timeout=PROBE_S, abandon=lambda: abandoned.append(True),
        ))
        try:
            assert await _timed_out(entered, probing) is False
            # The probe's own task outlives probe_within; its done callback
            # decides on abandon when the late answer arrives.
            late = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
            assert len(late) == 1
        finally:
            release.set()
        await asyncio.wait(late)  # done callbacks run in order: the late-answer check first
        assert late[0].result() is False

    # asyncio.run joins the default executor: an abandon queued there has run.
    virtual_clock.run_timing(main())
    assert abandoned == []


# --- B: dispatcher ---------------------------------------------------------------


class _Check:
    def __init__(self, changes: StateChanges | None = None):
        self.calls = 0
        self.changes = changes

    async def run_at_startup(self) -> None:
        self.calls += 1
        if self.changes is not None:
            self.changes.poke()


class _Store:
    """PostgreSQL as the lock session sees it: up, restarting, or locked by
    another process. Every call pokes ``changes``."""

    def __init__(self, changes: StateChanges | None = None):
        self.changes = changes
        self.pg_up = True
        self.held_elsewhere = False
        self.acquires = 0
        self.sessions: list[_Conn] = []
        self.probes = 0
        self.probe_block: threading.Event | None = None
        self.probe_entered = threading.Event()
        self.recover_expired_calls = 0
        self.released: list = []

    def _poke(self) -> None:
        if self.changes is not None:
            self.changes.poke()

    def acquire_dispatcher_lock(self):
        self.acquires += 1
        self._poke()
        if not self.pg_up:
            raise _session_gone()
        if self.held_elsewhere:
            return None
        conn = _Conn()
        self.sessions.append(conn)
        return conn

    def probe_dispatcher_lock(self, conn):
        self.probes += 1
        self._poke()
        if self.probe_block is not None:
            self.probe_entered.set()
            if not self.probe_block.wait(HANG_GUARD_S):
                raise AssertionError(f"held probe was never released within {HANG_GUARD_S:g}s")
            return True
        conn.dead = not self.pg_up
        return leadership.probe_lock_session(conn)

    def release_dispatcher_lock(self, conn):
        self.released.append(conn)
        self._poke()

    def recover_expired(self, *, live_owner=None):
        self.recover_expired_calls += 1
        self._poke()

    def eligible_targets(self):
        return []


def _dispatcher(check=None, changes: StateChanges | None = None) -> InterBotDeliveryDispatcher:
    """A dispatcher on the fakes, at every production interval and timeout."""
    dispatcher = InterBotDeliveryDispatcher.__new__(InterBotDeliveryDispatcher)
    dispatcher.service = SimpleNamespace(
        config=SimpleNamespace(DEFAULT_USER="nick"),
        _redis_subscriber=SimpleNamespace(connected=True),
    )
    dispatcher.store = _Store(changes)
    dispatcher.recovery_interval_seconds = SWEEP_S
    dispatcher.leadership_retry_delay_seconds = RESTART_DELAY_S
    dispatcher._wake_event = asyncio.Event()

    def on_done() -> None:
        dispatcher._wake_event.set()
        if changes is not None:
            changes.poke()

    dispatcher._recovery = (
        StartupRecoveryRunner(check, on_done=on_done, is_leader=lambda: dispatcher.is_leader)
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


async def _sweep(dispatcher, changes: StateChanges) -> None:
    """One woken sweep, well before the timed one (recover_expired pokes)."""
    before = dispatcher.store.recover_expired_calls
    dispatcher._wake_event.set()
    await changes.until(lambda: dispatcher.store.recover_expired_calls > before, within=PROMPT_S)


async def _stop(dispatcher) -> None:
    """stop() is bounded by the release timeout, whatever is still running."""
    await asyncio.wait_for(dispatcher.stop(), RELEASE_S + PROMPT_S)


def test_a_dead_lock_session_drops_leadership_stops_sweeping_and_is_reacquired(caplog):
    caplog.set_level(logging.WARNING)

    async def main():
        changes = StateChanges()
        check = _Check(changes)
        dispatcher = _dispatcher(check, changes)
        store = dispatcher.store
        dispatcher.start()
        # Acquired on the first attempt; the term's run() then calls
        # recover_expired and the startup check, which both poke.
        await changes.until(
            lambda: dispatcher.is_leader and check.calls == 1 and store.recover_expired_calls >= 1,
            within=PROMPT_S,
        )
        await _sweep(dispatcher, changes)
        assert dispatcher.is_leader and check.calls == 1
        first_session = dispatcher._dispatcher_lock

        # In-flight work under its own claim lease: leadership loss must not touch it.
        in_flight_gate = asyncio.Event()
        in_flight = asyncio.create_task(in_flight_gate.wait())
        dispatcher._dispatch_tasks.add(in_flight)

        store.pg_up = False  # the nightly restart
        # No wake: a timed sweep probes once the probe interval has passed.
        # Leadership drops, and the acquisition attempt right after pokes.
        await changes.until(lambda: not dispatcher.is_leader,
                            within=PROBE_EVERY_S + SWEEP_S + PROMPT_S)
        assert first_session.closes == 1
        assert any("lost its leadership lock session" in r.getMessage() and r.levelno == logging.ERROR
                   for r in caplog.records)

        # PostgreSQL still down: acquisition retries on its cadence, and over
        # more than a sweep interval of it neither a wake nor the timer sweeps.
        sweeps_while_leaderless = store.recover_expired_calls
        attempts = store.acquires
        retries = 4
        assert retries * RETRY_S > SWEEP_S
        dispatcher._wake_event.set()  # lands on no run loop: no sweep, no claim
        await changes.until(lambda: store.acquires >= attempts + retries,
                            within=retries * RETRY_S + PROMPT_S)
        assert store.recover_expired_calls == sweeps_while_leaderless
        assert not in_flight.done() and not in_flight.cancelled()
        assert not any("is passive" in r.getMessage() for r in caplog.records)

        store.pg_up = True  # PostgreSQL is back
        # Re-acquired by the next attempt; the new term reruns the startup check.
        await changes.until(lambda: dispatcher.is_leader and check.calls == 2,
                            within=RETRY_S + PROMPT_S)
        assert dispatcher._dispatcher_lock is not first_session
        await changes.until(lambda: store.recover_expired_calls > sweeps_while_leaderless,
                            within=PROMPT_S)
        await _sweep(dispatcher, changes)

        in_flight_gate.set()
        await _stop(dispatcher)
        assert store.released == [store.sessions[-1]]

    virtual_clock.run_timing(main())


def test_acquisition_while_postgres_is_down_is_paced_not_a_hot_loop(caplog):
    caplog.set_level(logging.INFO)

    async def main():
        dispatcher = _dispatcher()
        dispatcher.store.pg_up = False
        dispatcher.start()
        # Attempts at 0, RETRY_S, 2 RETRY_S and 3 RETRY_S; the window closes
        # half a cadence after the last one.
        await asyncio.sleep(3.5 * RETRY_S)
        attempts = dispatcher.store.acquires
        assert not dispatcher.is_leader
        await _stop(dispatcher)
        return attempts

    assert virtual_clock.run_timing(main()) == 4
    assert any("cannot reach PostgreSQL for leadership" in r.getMessage() for r in caplog.records)
    assert not any("is passive" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_the_probe_is_rate_limited_by_the_interval():
    dispatcher = _dispatcher()
    assert await dispatcher._acquire_leadership() is True
    assert await dispatcher._leadership_confirmed() is True
    assert await dispatcher._leadership_confirmed() is True
    assert dispatcher.store.probes == 0  # just acquired: nothing to probe yet
    dispatcher._lock_probed_at = None
    assert await dispatcher._leadership_confirmed() is True
    assert dispatcher.store.probes == 1


def test_a_hung_probe_drops_leadership_and_closes_the_session_once_it_answers(caplog):
    caplog.set_level(logging.WARNING)

    async def main():
        loop = asyncio.get_running_loop()
        dispatcher = _dispatcher()
        store = dispatcher.store
        store.probe_block = threading.Event()
        assert await dispatcher._acquire_leadership() is True
        session = dispatcher._dispatcher_lock
        dispatcher._lock_probed_at = None
        started = loop.time()
        confirming = asyncio.create_task(dispatcher._leadership_confirmed())
        try:
            assert await asyncio.to_thread(store.probe_entered.wait, HANG_GUARD_S)
            loop.advance(PROBE_S)  # the probe never answers within its bound
            assert await confirming is False
            assert loop.time() - started >= PROBE_S
            assert not dispatcher.is_leader
            assert session.closes == 0  # the thread is still stuck in the probe
        finally:
            store.probe_block.set()
        # It answers "alive" late: the session is closed anyway.
        assert await asyncio.to_thread(session.closed.wait, HANG_GUARD_S)
        assert session.closes == 1

    virtual_clock.run_timing(main())


def test_stop_during_the_probe_never_sweeps_after_the_workers_were_cancelled():
    """Codex r1 blocker 2: stop() cancels the workers while the probe is out;
    a probe that then answers "alive" must not start one more sweep."""
    async def main():
        changes = StateChanges()
        dispatcher = _dispatcher(changes=changes)
        store = dispatcher.store
        store.probe_block = threading.Event()
        try:
            dispatcher.start()
            # A timed sweep probes once the interval has passed; the probe
            # pokes, then its thread is held.
            await changes.until(lambda: store.probes >= 1, within=PROBE_EVERY_S + SWEEP_S + PROMPT_S)
            assert await asyncio.to_thread(store.probe_entered.wait, HANG_GUARD_S)
            sweeps_before_stop = store.recover_expired_calls

            stop_task = asyncio.create_task(dispatcher.stop())
            await asyncio.sleep(0)  # stop() sets the stop event before its first await
            assert dispatcher._stop_event.is_set()
            store.probe_block.set()  # the session answers "alive" after stop() began
            await asyncio.wait_for(stop_task, RELEASE_S + PROMPT_S)
        finally:
            store.probe_block.set()
        assert store.recover_expired_calls == sweeps_before_stop

    virtual_clock.run_timing(main())


def test_a_new_term_gets_its_own_recovery_pass_once_the_old_one_settles():
    """Codex r1 blocker 1: a check still running at re-acquisition may be on
    the dead term's connections and fail; the new term needs its own pass."""
    async def main():
        loop = asyncio.get_running_loop()
        changes = StateChanges()
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

        def on_done() -> None:
            woke.append(True)
            changes.poke()

        check = _SlowFailingCheck()
        runner = StartupRecoveryRunner(check, on_done=on_done)
        started = loop.time()
        await runner.run_once()  # first term: the bound passes, the check runs on
        assert loop.time() - started >= WAIT_S
        assert runner.running and check.calls == 1

        runner.rearm()  # PostgreSQL is back, a new term is acquired
        await runner.run_once()
        assert check.calls == 1  # never doubled while the old pass runs

        finish_first.set()
        # The old pass fails (on_done pokes); the new term's starts and ends (pokes).
        await changes.until(lambda: check.calls == 2 and not runner.running and len(woke) == 2,
                            within=PROMPT_S)
        assert woke == [True, True]  # both ends woke dispatch
        await runner.run_once()  # the new term's pass is not repeated either
        assert check.calls == 2

    virtual_clock.run_timing(main())


def test_a_pending_pass_never_starts_leaderless_and_the_next_term_still_gets_one(caplog):
    """Codex r2 blocker: the lock dies twice. The first term's check settles
    while nobody holds the lock: no pass (leader-only work). The term that
    then acquires the lock runs its own pass."""
    caplog.set_level(logging.INFO, logger="llm_bawt.service.inter_bot_dispatcher_recovery")

    async def main():
        changes = StateChanges()
        finish_first = asyncio.Event()

        class _LongCheck:
            def __init__(self):
                self.calls = 0

            async def run_at_startup(self):
                self.calls += 1
                changes.poke()
                if self.calls == 1:
                    await finish_first.wait()

        check = _LongCheck()
        dispatcher = _dispatcher(check, changes)
        store = dispatcher.store
        dispatcher.start()
        await changes.until(lambda: check.calls == 1 and dispatcher.is_leader, within=PROMPT_S)

        store.pg_up = False  # first loss
        # Dispatch waits out the check's bound, then a sweep probes the dead
        # session; the acquisition attempt right after pokes.
        await changes.until(lambda: not dispatcher.is_leader,
                            within=WAIT_S + PROBE_EVERY_S + SWEEP_S + PROMPT_S)
        store.pg_up = True
        # Re-acquired: rearm finds the old check running; run() then pokes.
        await changes.until(lambda: dispatcher.is_leader and dispatcher._recovery._term_pending,
                            within=RETRY_S + PROMPT_S)
        store.pg_up = False  # second loss, the old check still running
        await changes.until(lambda: not dispatcher.is_leader,
                            within=PROBE_EVERY_S + SWEEP_S + PROMPT_S)

        finish_first.set()  # the old check settles while nobody holds the lock
        await changes.until(lambda: not dispatcher._recovery.running, within=PROMPT_S)  # on_done pokes
        assert check.calls == 1 and not dispatcher.is_leader
        assert any("not rerun: leadership was lost again" in r.getMessage() for r in caplog.records)

        store.pg_up = True  # a valid term again: it starts like a startup
        await changes.until(lambda: check.calls == 2 and dispatcher.is_leader,
                            within=RETRY_S + PROMPT_S)
        await _stop(dispatcher)

    virtual_clock.run_timing(main())


def test_the_runner_starts_a_pending_pass_only_under_leadership():
    async def main():
        changes = StateChanges()
        gate = asyncio.Event()
        leader = [True]

        class _GatedCheck:
            def __init__(self):
                self.calls = 0

            async def run_at_startup(self):
                self.calls += 1
                if self.calls == 1:
                    await gate.wait()

        check = _GatedCheck()
        runner = StartupRecoveryRunner(check, on_done=changes.poke, is_leader=lambda: leader[0])
        await runner.run_once()  # the bound passes, the check runs on
        runner.rearm()
        assert runner._term_pending
        leader[0] = False
        gate.set()
        await changes.until(lambda: not runner.running, within=PROMPT_S)  # on_done pokes
        assert check.calls == 1 and not runner._term_pending
        leader[0] = True
        runner.rearm()  # the next acquired term
        await runner.run_once()
        assert check.calls == 2

    virtual_clock.run_timing(main())


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


def test_rearm_after_the_old_pass_ended_starts_one_new_pass_and_stop_restarts_nothing():
    async def main():
        gate = asyncio.Event()

        class _GatedCheck:
            def __init__(self):
                self.calls = 0

            async def run_at_startup(self):
                self.calls += 1
                await gate.wait()

        check = _GatedCheck()
        runner = StartupRecoveryRunner(check)
        gate.set()
        await runner.run_once()  # the check ended, so the wait did too
        assert not runner.running
        runner.rearm()
        await runner.run_once()
        assert check.calls == 2

        gate.clear()
        runner.rearm()  # nothing running: no pending term either
        await runner.run_once()  # the bound passes, the third pass runs on
        assert check.calls == 3 and runner.running
        runner.rearm()  # pending while the third pass runs
        # Shutdown cancels it. The runner's done callback was added before
        # stop()'s wait, so it has already run by now and started nothing.
        await runner.stop()
        assert check.calls == 3 and not runner.running

    virtual_clock.run_timing(main())


@pytest.mark.asyncio
async def test_leaderless_is_confirmed_false_without_probing():
    dispatcher = _dispatcher()
    assert await dispatcher._leadership_confirmed() is False
    assert dispatcher.store.probes == 0


def test_a_held_lock_is_passive_and_logged_once(caplog):
    caplog.set_level(logging.INFO)

    async def main():
        changes = StateChanges()
        dispatcher = _dispatcher(changes=changes)
        dispatcher.store.held_elsewhere = True
        dispatcher.start()
        await changes.until(lambda: dispatcher.store.acquires >= 4, within=3 * RETRY_S + PROMPT_S)
        assert not dispatcher.is_leader
        await _stop(dispatcher)

    virtual_clock.run_timing(main())
    assert sum("is passive" in r.getMessage() for r in caplog.records) == 1


def test_stop_bounds_a_hung_release_and_still_completes(caplog):
    caplog.set_level(logging.WARNING)
    released_in_time: list[bool] = []

    async def main():
        loop = asyncio.get_running_loop()
        dispatcher = _dispatcher()
        entered, release, returned = threading.Event(), threading.Event(), threading.Event()

        def hung_release(conn):
            entered.set()
            released_in_time.append(release.wait(HANG_GUARD_S))
            returned.set()

        dispatcher.store.release_dispatcher_lock = hung_release
        assert await dispatcher._acquire_leadership() is True
        started = loop.time()
        stopping = asyncio.create_task(dispatcher.stop())
        try:
            assert await asyncio.to_thread(entered.wait, HANG_GUARD_S)
            loop.advance(RELEASE_S)  # the unlock never answers within its bound
            await stopping  # completes with the unlock still held
            assert loop.time() - started >= RELEASE_S
            assert not dispatcher.is_leader
        finally:
            release.set()
        assert await asyncio.to_thread(returned.wait, HANG_GUARD_S)

    virtual_clock.run_timing(main())
    assert released_in_time == [True]
    assert any("release still pending" in r.getMessage() for r in caplog.records)


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
