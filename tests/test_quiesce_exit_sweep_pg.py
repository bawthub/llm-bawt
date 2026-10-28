"""PostgreSQL tests for the TASK-1233 B1 (b)/(c) quiesce exit sweep.

The real driver opens a real hold (``rig`` from :mod:`tests.test_ops_quiesce_pg`);
only the bridge is faked: ``run.status`` replies and terminal records per
request id, and ``chat.abort`` over a real :class:`agent_bridge.session_queue.SessionQueue`
so the abort's two orderings run the bridge's own admission rule. Each test checks the driver's own verdict (``gate_drain_status``)
and, where it drains, that the job dispatches.
"""
from __future__ import annotations

import asyncio
import threading
import time
import uuid
from threading import Event
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from agent_bridge.run_contract import RunAccepted, RunTerminal, TeardownReport
from agent_bridge.session_queue import Admission, SessionQueue
from llm_bawt.ops import JOB_ACCEPTED
from llm_bawt.ops.models import JOB_QUIESCING
from llm_bawt.ops.quiesce_exit import ensure_sighting_schema, purge_sightings
from llm_bawt.service import quiesce_exit_sweep
from llm_bawt.service.quiesce_exit_sweep import WRITE_DEADLINE, QuiesceExitSweep
from llm_bawt.service.run_accepts import purge_accepts, record_accept
from llm_bawt.service.turn_execution import TurnExecution
from llm_bawt.service.turn_gates import gate_drain_status
from tests import test_ops_quiesce_pg as _quiesce

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]

rig = _quiesce.rig


BOOT = "boot-a"


class _Bridge:
    """Answers ``run.status`` for the requests it knows; times out for every other.

    ``chat.abort`` stops the request in a real ``SessionQueue``. A send in
    ``ahead`` was read off the command stream before the abort, so it is
    accepted first and the abort cancels it. A run's terminal goes to the
    stream ``worker`` when one is attached (source (a)), and to the terminal
    record unless ``records`` is off.
    """

    def __init__(self):
        self.replies: dict[str, dict] = {}
        self.terminals: dict[str, RunTerminal] = {}
        self.asked: list[str] = []
        self.aborted: list[str] = []
        self.queue = SessionQueue()
        self.ahead: set[str] = set()
        self.worker: TurnExecution | None = None
        self.records = True
        self._runs: dict[str, asyncio.Task] = {}

    def publish(self, terminal):
        if self.worker is not None:
            self.worker.record_exit(terminal.to_dict())
        if self.records:
            self.terminals[terminal.request_id] = terminal

    async def _send(self, session_key, request_id):
        if self.queue.accept(session_key, request_id) is not Admission.ACCEPTED:
            self.publish(RunTerminal(request_id, "claude-code", BOOT,
                                     self.queue.refusal_teardown(session_key, request_id)))
            return
        try:
            if self.worker is not None:
                self.worker.record_accept(RunAccepted(request_id, "claude-code", BOOT, True).to_dict())
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.publish(RunTerminal(request_id, "claude-code", BOOT, TeardownReport("ok", "none", True, 3)))
        finally:
            self.queue.release(session_key, request_id)

    async def deliver(self, session_key, request_id):
        """The send command reaches the bridge; returns its run task."""
        task = self._runs[request_id] = asyncio.create_task(self._send(session_key, request_id))
        await asyncio.sleep(0)
        return task

    async def _abort(self, session_key, request_id):
        self.aborted.append(request_id)
        if request_id in self.ahead:
            self.ahead.discard(request_id)
            await self.deliver(session_key, request_id)
        cancelled = self.queue.stop_request(session_key, request_id)
        if cancelled:  # a bridge replies after the cancelled run's teardown
            await asyncio.gather(self._runs[request_id], return_exceptions=True)
        state = "cancelled" if cancelled else "tombstoned"
        return {"backend": "claude-code", "boot_id": BOOT, "request_id": request_id, "cancelled": cancelled,
                "state": state, "detail": ""}

    def answer(self, request_id, state, *, boot, backend="claude-code", pid1=True):
        self.replies[request_id] = {"backend": backend, "boot_id": boot, "request_id": request_id,
                                    "state": state, "detail": "", "pid1": pid1}

    async def send_rpc(self, method, params, request_id, *, timeout_s, backend):
        if method == "chat.abort":
            return await self._abort(params["sessionKey"], params["requestId"])
        self.asked.append(params["requestId"])
        reply = self.replies.get(params["requestId"])
        if reply is None:
            raise TimeoutError("no reply")
        return reply

    async def read_run_terminal(self, request_id):
        return self.terminals.get(request_id)


@pytest.fixture
def sweep(rig):
    ensure_sighting_schema(rig.engine)
    rig.bridge = _Bridge()
    rig.sweep = QuiesceExitSweep(rig.engine, subscriber=lambda: rig.bridge)

    def turn(backend="claude-code"):
        turn_id = f"turn-{uuid.uuid4().hex}"
        rig.sql("""
            INSERT INTO turn_logs (id, created_at, path, stream, bot_id, user_id, status, user_prompt,
                                   response_text, agent_request_id, agent_backend, agent_session_key)
            VALUES (:id, now(), '/test', true, :bot, 'nick', 'streaming', '', '', :req, :backend, :session)
        """, id=turn_id, bot=rig.bot, req=f"req-{turn_id}", backend=backend, session=f"{rig.bot}:nick")
        return turn_id, f"req-{turn_id}"

    def held(*turns):
        job_id = rig.job()
        if rig.row(job_id)["quiesce_gate_id"] is None:
            assert rig.driver.step(job_id).outcome == "opened"
        gate_id = rig.row(job_id)["quiesce_gate_id"]
        assert gate_id is not None and rig.row(job_id)["state"] == JOB_QUIESCING
        return job_id, gate_id

    def drain(gate_id):
        with rig.engine.connect() as conn:
            return {s.turn_id: (s.drained, s.reason) for s in gate_drain_status(conn, gate_id)}

    def run_once():
        async def one_pass():
            counts = await rig.sweep.run_once()
            await rig.sweep.settle_aborts()
            return counts

        return asyncio.run(one_pass())

    rig.turn_with, rig.hold, rig.drain, rig.run_once = turn, held, drain, run_once
    yield rig
    rig.sql("DELETE FROM quiesce_run_sightings WHERE request_id LIKE 'req-turn-%'")
    rig.sql("DELETE FROM bridge_run_accepts WHERE request_id LIKE 'req-turn-%'")


def _dispatched(rig, job_id):
    return rig.service.get_job_status(job_id)["state"] == JOB_ACCEPTED


def test_a_pid1_restart_after_a_sighting_drains_claude_code(sweep):
    turn_id, req = sweep.turn_with()
    job_id, gate_id = sweep.hold(turn_id)
    sweep.bridge.answer(req, "running", boot="boot-a")
    sweep.run_once()
    assert sweep.drain(gate_id)[turn_id] == (False, "no_execution_exit")
    assert sweep.sql("SELECT boot_id, pid1 FROM quiesce_run_sightings WHERE request_id=:r", r=req).one() == (
        "boot-a", True)

    sweep.bridge.answer(req, "not_found", boot="boot-b")
    sweep.run_once()
    assert sweep.drain(gate_id)[turn_id] == (True, "execution_exit:b")
    assert _dispatched(sweep, job_id)


def test_a_local_bridge_restart_never_drains(sweep):
    sweep.sql("UPDATE bot_profiles SET agent_backend='local' WHERE slug=:b", b=sweep.bot)
    turn_id, req = sweep.turn_with("local")
    _job, gate_id = sweep.hold(turn_id)
    sweep.bridge.answer(req, "running", boot="boot-a", backend="local")
    sweep.run_once()
    sweep.bridge.answer(req, "not_found", boot="boot-b", backend="local")
    sweep.run_once()
    assert sweep.drain(gate_id)[turn_id] == (False, "no_execution_exit")


@pytest.mark.parametrize("sighted_pid1", [None, False])
def test_no_restart_proof_without_a_pid1_sighting(sweep, sighted_pid1):
    turn_id, req = sweep.turn_with()
    _job, gate_id = sweep.hold(turn_id)
    if sighted_pid1 is not None:
        sweep.bridge.answer(req, "running", boot="boot-a", pid1=sighted_pid1)
        sweep.run_once()
    sweep.bridge.answer(req, "not_found", boot="boot-b")
    sweep.run_once()
    assert sweep.drain(gate_id)[turn_id] == (False, "no_execution_exit")


def test_same_boot_not_found_drains_only_with_a_terminal_record(sweep):
    turn_id, req = sweep.turn_with()
    job_id, gate_id = sweep.hold(turn_id)
    sweep.bridge.answer(req, "not_found", boot="boot-a")
    sweep.run_once()
    assert sweep.drain(gate_id)[turn_id] == (False, "no_execution_exit")

    sweep.bridge.terminals[req] = RunTerminal(req, "claude-code", "boot-a", TeardownReport(exited=True))
    sweep.run_once()
    assert sweep.drain(gate_id)[turn_id] == (True, "execution_exit:c")
    assert _dispatched(sweep, job_id)


def test_tombstoned_on_the_sighted_boot_is_not_evidence(sweep):
    # Vex r1 B1: released in ``finally`` while a teardown may have failed.
    turn_id, req = sweep.turn_with()
    _job, gate_id = sweep.hold(turn_id)
    sweep.bridge.answer(req, "stopping", boot="boot-a", pid1=False)
    sweep.run_once()
    sweep.bridge.answer(req, "tombstoned", boot="boot-a", pid1=False)
    sweep.run_once()
    assert sweep.drain(gate_id)[turn_id] == (False, "no_execution_exit")
    sweep.bridge.terminals[req] = RunTerminal(req, "claude-code", "boot-a", TeardownReport())  # exited null
    sweep.run_once()
    assert sweep.drain(gate_id)[turn_id] == (False, "no_execution_exit")
    sweep.bridge.terminals[req] = RunTerminal(req, "claude-code", "boot-a",
                                              TeardownReport(disconnect="refused", exited=True))
    sweep.run_once()
    assert sweep.drain(gate_id)[turn_id] == (True, "execution_exit:c")


def test_a_survivor_terminal_is_exit_unconfirmed_until_a_restart_proves_it(sweep):
    turn_id, req = sweep.turn_with()
    _job, gate_id = sweep.hold(turn_id)
    sweep.bridge.answer(req, "running", boot="boot-a")
    sweep.run_once()
    sweep.bridge.terminals[req] = RunTerminal(req, "claude-code", "boot-a", TeardownReport(exited=False))
    sweep.bridge.answer(req, "not_found", boot="boot-a")
    sweep.run_once()
    assert sweep.drain(gate_id)[turn_id] == (False, "exit_unconfirmed")
    stamped = sweep.sql("SELECT request_json FROM turn_logs WHERE id=:t", t=turn_id).scalar_one()
    sweep.bridge.asked.clear()
    sweep.run_once()  # re-probed every pass, still unconfirmed, and not rewritten
    assert req in sweep.bridge.asked
    assert sweep.drain(gate_id)[turn_id] == (False, "exit_unconfirmed")
    assert sweep.sql("SELECT request_json FROM turn_logs WHERE id=:t", t=turn_id).scalar_one() == stamped

    sweep.bridge.answer(req, "not_found", boot="boot-b")
    sweep.run_once()
    assert sweep.drain(gate_id)[turn_id] == (True, "execution_exit:b")


def test_a_deleted_turn_is_stamped_on_its_member_row(sweep):
    turn_id, req = sweep.turn_with()
    job_id, gate_id = sweep.hold(turn_id)
    sweep.bridge.answer(req, "running", boot="boot-a")
    sweep.run_once()
    sweep.sql("UPDATE gate_members SET agent_request_id=:r WHERE turn_id=:t", r=req, t=turn_id)
    sweep.sql("DELETE FROM turn_logs WHERE id=:t", t=turn_id)
    assert sweep.drain(gate_id)[turn_id] == (False, "turn_missing:no_execution_exit")

    sweep.bridge.answer(req, "not_found", boot="boot-b")
    sweep.run_once()
    assert sweep.drain(gate_id)[turn_id] == (True, "execution_exit:b")
    member_exit = sweep.sql("SELECT execution_exit FROM gate_members WHERE turn_id=:t", t=turn_id).scalar_one()
    assert (member_exit["source"], member_exit["boot_id"], member_exit["exited"]) == ("b", "boot-b", True)
    assert _dispatched(sweep, job_id)


class _TurnLock:
    """Holds the turn row ``FOR UPDATE`` on its own connection and thread until released."""

    HOLD_GUARD_S = 30.0

    def __init__(self, engine, turn_id):
        self._locked, self._release = Event(), Event()
        self._released_in_time: bool | None = None
        self._thread = threading.Thread(target=self._hold, args=(engine, turn_id), daemon=True)

    def _hold(self, engine, turn_id):
        with engine.begin() as conn:
            conn.execute(text("SELECT 1 FROM turn_logs WHERE id=:t FOR UPDATE"), {"t": turn_id})
            self._hold_until_released()

    def _hold_until_released(self):
        """Hold the lock until released. Checked in __exit__: a lock that let
        go on its hang guard instead of the test's release fails the test."""
        self._locked.set()
        self._released_in_time = self._release.wait(self.HOLD_GUARD_S)

    def __enter__(self):
        self._thread.start()
        assert self._locked.wait(10)
        return self

    def release(self, after=0.0):
        threading.Timer(after, self._release.set).start()

    def __exit__(self, *exc):
        self._release.set()
        self._thread.join(10)
        if exc[0] is None:  # never mask the test's own failure
            assert not self._thread.is_alive(), "the lock holder never let go"
            assert self._released_in_time, "the lock let go on its hang guard, not the test's release"


def _deadlines(sweep, monkeypatch, *, probe=0.2, budget=1.5, reserve=0.5):
    monkeypatch.setattr(quiesce_exit_sweep, "PROBE_TIMEOUT_SECONDS", probe)
    sweep.sweep.budget_seconds, sweep.sweep.write_reserve_seconds = budget, reserve


def test_a_write_blocked_past_the_pass_deadline_ends_and_never_lands_later(sweep, monkeypatch):
    """Loopy r1 A2: a cut write committed execution_exit:c after run_once returned."""
    turn_id, req = sweep.turn_with()
    _job, gate_id = sweep.hold(turn_id)
    sweep.bridge.answer(req, "not_found", boot=BOOT)
    sweep.bridge.terminals[req] = RunTerminal(req, "claude-code", BOOT, TeardownReport(exited=True))
    _deadlines(sweep, monkeypatch)
    with _TurnLock(sweep.engine, turn_id):
        started = time.monotonic()
        assert sweep.run_once() == {WRITE_DEADLINE: 1}
        # timing-ok: real PG lock: the 1.5s budget vs the 30s lock hold
        assert time.monotonic() - started < 3.0  # budget 1.5s; the lock is held for 30s
    time.sleep(0.5)  # the lock is gone: nothing still in flight may now commit
    assert sweep.drain(gate_id)[turn_id] == (False, "no_execution_exit")
    assert "execution_exit" not in (sweep.sql("SELECT request_json FROM turn_logs WHERE id=:t",
                                              t=turn_id).scalar_one() or {})

    sweep.run_once()  # the next pass writes it
    assert sweep.drain(gate_id)[turn_id] == (True, "execution_exit:c")


def test_a_write_unblocked_before_the_deadline_lands_within_the_pass(sweep, monkeypatch):
    turn_id, req = sweep.turn_with()
    _job, gate_id = sweep.hold(turn_id)
    sweep.bridge.answer(req, "not_found", boot=BOOT)
    sweep.bridge.terminals[req] = RunTerminal(req, "claude-code", BOOT, TeardownReport(exited=True))
    _deadlines(sweep, monkeypatch, budget=0.6, reserve=0.2)
    with _TurnLock(sweep.engine, turn_id) as lock:
        lock.release(after=0.4)  # after the probes' start-by (0.2s), before the write deadline (0.6s)
        assert sweep.run_once() == {"terminal_exited": 1}
        # run_once returned only after the write committed
        assert sweep.drain(gate_id)[turn_id] == (True, "execution_exit:c")


@pytest.mark.parametrize("cancels", [1, 2])
def test_a_cancelled_pass_waits_for_its_write_through_every_cancel(sweep, monkeypatch, cancels):
    """Loopy r2 A2(1): a second cancel interrupted the join and run_once returned before the commit."""
    turn_id, req = sweep.turn_with()
    _job, gate_id = sweep.hold(turn_id)
    sweep.bridge.answer(req, "not_found", boot=BOOT)
    sweep.bridge.terminals[req] = RunTerminal(req, "claude-code", BOOT, TeardownReport(exited=True))
    _deadlines(sweep, monkeypatch, probe=0.02, budget=4.0, reserve=0.1)
    entered = Event()
    original_stamp = quiesce_exit_sweep.stamp

    def observed_stamp(*args, **kwargs):
        entered.set()
        return original_stamp(*args, **kwargs)

    monkeypatch.setattr(quiesce_exit_sweep, "stamp", observed_stamp)

    async def exercise(lock):
        task = asyncio.create_task(sweep.sweep.run_once())
        assert await asyncio.to_thread(entered.wait, 2)
        writers = list(sweep.sweep._writing)
        assert len(writers) == 1
        for _ in range(cancels):
            task.cancel()
            await asyncio.sleep(0.05)
        returned_while_writing = task.done()
        lock.release()
        await asyncio.wait(writers, timeout=3)
        with pytest.raises(asyncio.CancelledError):
            await task
        return returned_while_writing

    with _TurnLock(sweep.engine, turn_id) as lock:
        assert not asyncio.run(exercise(lock)), "run_once returned while its PostgreSQL write was in flight"
    assert sweep.drain(gate_id)[turn_id] == (True, "execution_exit:c")


class _ShareTableLock(_TurnLock):
    """SHARE lets the stamp's SELECT ... FOR UPDATE through but blocks its UPDATE."""

    def _hold(self, engine, turn_id):
        with engine.begin() as conn:
            conn.execute(text("LOCK TABLE turn_logs IN SHARE MODE"))
            self._hold_until_released()


def test_two_lock_waits_share_one_absolute_write_deadline(sweep, monkeypatch):
    """Loopy r2 A2(2): each wait fit its own statement_timeout; together they ran past the deadline."""
    turn_id, req = sweep.turn_with()
    _job, gate_id = sweep.hold(turn_id)
    sweep.bridge.answer(req, "not_found", boot=BOOT)
    sweep.bridge.terminals[req] = RunTerminal(req, "claude-code", BOOT, TeardownReport(exited=True))
    _deadlines(sweep, monkeypatch, probe=0.02, budget=1.0, reserve=0.1)
    with _TurnLock(sweep.engine, turn_id) as row_lock, _ShareTableLock(sweep.engine, turn_id) as table_lock:
        row_lock.release(after=0.6)
        table_lock.release(after=1.3)
        started = time.monotonic()
        counts = sweep.run_once()
        elapsed = time.monotonic() - started
    drained = sweep.drain(gate_id)[turn_id]
    assert counts == {WRITE_DEADLINE: 1}, (counts, elapsed, drained)
    # The UPDATE's lock wait gets only the time left (~0.4s), not a fresh 1s.
    # timing-ok: real PG lock_timeout: the remaining ~0.4s vs a fresh 1s (>=1.4s total)
    assert elapsed < 1.2, elapsed
    assert drained == (False, "no_execution_exit")


def test_a_drained_member_is_not_probed_again(sweep):
    turn_id, req = sweep.turn_with()
    _job, gate_id = sweep.hold(turn_id)
    sweep.bridge.terminals[req] = RunTerminal(req, "claude-code", "boot-a", TeardownReport(exited=True))
    sweep.bridge.answer(req, "not_found", boot="boot-a")
    sweep.run_once()
    assert sweep.drain(gate_id)[turn_id] == (True, "execution_exit:c")
    sweep.bridge.asked.clear()
    sweep.run_once()
    assert req not in sweep.bridge.asked


def test_old_sightings_are_purged(sweep):
    req = f"req-turn-{uuid.uuid4().hex}"
    sweep.sql("INSERT INTO quiesce_run_sightings (backend, request_id, boot_id, last_seen_at)"
              " VALUES ('codex', :r, 'boot-a', now() - interval '2 days')", r=req)
    assert purge_sightings(sweep.engine) >= 1
    assert sweep.sql("SELECT count(*) FROM quiesce_run_sightings WHERE request_id=:r", r=req).scalar_one() == 0


def test_bootstrap_is_idempotent(sweep):
    ensure_sighting_schema(sweep.engine)
    with sweep.engine.connect() as conn:
        assert conn.execute(text("SELECT to_regclass('quiesce_run_sightings')")).scalar_one() is not None


# -- (b) from the accept record ------------------------------------------------

def _accept(sweep, req, *, boot=BOOT, pid1=True, backend="claude-code"):
    return record_accept(sweep.engine, RunAccepted(req, backend, boot, pid1).to_dict())


def test_the_boot_changes_before_any_sighting_and_the_accept_drains_it(sweep):
    """Snark: the accepting bridge died before the sweep ever saw the run live."""
    turn_id, req = sweep.turn_with()
    job_id, gate_id = sweep.hold(turn_id)
    assert _accept(sweep, req) is True
    sweep.bridge.answer(req, "not_found", boot="boot-b")
    sweep.run_once()
    assert sweep.sql("SELECT count(*) FROM quiesce_run_sightings WHERE request_id=:r", r=req).scalar_one() == 0
    assert sweep.drain(gate_id)[turn_id] == (True, "execution_exit:b")
    assert sweep.bridge.aborted == []
    assert _dispatched(sweep, job_id)


@pytest.mark.parametrize("pid1", [None, False])
def test_an_accept_without_pid1_proves_nothing(sweep, pid1):
    turn_id, req = sweep.turn_with()
    _job, gate_id = sweep.hold(turn_id)
    _accept(sweep, req, pid1=pid1)
    sweep.bridge.answer(req, "not_found", boot="boot-b")
    sweep.run_once()
    assert sweep.drain(gate_id)[turn_id] == (False, "no_execution_exit")
    assert sweep.bridge.aborted == []  # the accept is a bridge record: nothing to abort


def test_the_accept_and_the_sighting_must_both_prove_it(sweep):
    turn_id, req = sweep.turn_with()
    _job, gate_id = sweep.hold(turn_id)
    _accept(sweep, req)
    sweep.bridge.answer(req, "running", boot="boot-b", pid1=False)
    sweep.run_once()
    sweep.bridge.answer(req, "not_found", boot="boot-c")
    sweep.run_once()
    assert sweep.drain(gate_id)[turn_id] == (False, "no_execution_exit")


def test_record_accept_is_idempotent_per_request(sweep):
    req = f"req-turn-{uuid.uuid4().hex}"
    assert _accept(sweep, req) is True
    assert _accept(sweep, req, boot="boot-b") is False
    assert _accept(sweep, req, backend="Claude-Code") is False  # the backend key is lowercased
    assert sweep.sql("SELECT boot_id, pid1 FROM bridge_run_accepts WHERE request_id=:r", r=req).all() == [
        ("boot-a", True)]


def test_old_accepts_are_purged(sweep):
    req = f"req-turn-{uuid.uuid4().hex}"
    sweep.sql("INSERT INTO bridge_run_accepts (backend, request_id, boot_id, accepted_at)"
              " VALUES ('codex', :r, 'boot-a', now() - interval '2 days')", r=req)
    assert purge_accepts(sweep.engine) >= 1
    assert sweep.sql("SELECT count(*) FROM bridge_run_accepts WHERE request_id=:r", r=req).scalar_one() == 0


# -- the sweep abort: both orderings -------------------------------------------

def test_an_abort_ahead_of_the_send_refuses_it_and_c_drains(sweep):
    turn_id, req = sweep.turn_with()
    job_id, gate_id = sweep.hold(turn_id)
    sweep.bridge.answer(req, "not_found", boot=BOOT)  # the send is still on the command stream
    sweep.run_once()
    assert sweep.bridge.aborted == [req]
    assert sweep.drain(gate_id)[turn_id] == (False, "no_execution_exit")

    async def deliver():
        await (await sweep.bridge.deliver(f"{sweep.bot}:nick", req))

    asyncio.run(deliver())
    assert sweep.bridge.terminals[req].teardown == TeardownReport(disconnect="refused", exited=True)
    sweep.bridge.answer(req, "tombstoned", boot=BOOT)
    sweep.run_once()
    assert sweep.drain(gate_id)[turn_id] == (True, "execution_exit:c")
    assert sweep.bridge.aborted == [req]  # once per member and boot
    assert _dispatched(sweep, job_id)


def test_a_send_queued_ahead_of_the_abort_is_cancelled_and_a_drains(sweep):
    turn_id, req = sweep.turn_with()
    job_id, gate_id = sweep.hold(turn_id)
    worker = TurnExecution(turn_id, Event(), is_agent=True, store=SimpleNamespace(engine=sweep.engine))
    worker.request_id = req
    sweep.bridge.worker, sweep.bridge.records = worker, False  # only the run's own terminal can drain it
    sweep.bridge.ahead.add(req)
    sweep.bridge.answer(req, "not_found", boot=BOOT)
    sweep.run_once()
    assert sweep.bridge.aborted == [req]
    assert sweep.drain(gate_id)[turn_id] == (True, "execution_exit:a")
    assert sweep.sql("SELECT boot_id FROM bridge_run_accepts WHERE request_id=:r", r=req).scalar_one() == BOOT
    assert _dispatched(sweep, job_id)
