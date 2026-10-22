"""PostgreSQL tests for the TASK-1233 B1 (b)/(c) quiesce exit sweep.

The real driver opens a real hold (``rig`` from :mod:`tests.test_ops_quiesce_pg`);
only the bridge is faked: ``run.status`` replies and terminal records per
request id, and ``chat.abort`` over a real :class:`agent_bridge.session_queue.SessionQueue`
so the abort's two orderings run the bridge's own admission rule. Each test checks the driver's own verdict (``gate_drain_status``)
and, where it drains, that the job dispatches.
"""
from __future__ import annotations

import asyncio
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
from llm_bawt.service.quiesce_exit_sweep import QuiesceExitSweep
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


def test_an_openclaw_restart_never_drains(sweep):
    sweep.sql("UPDATE bot_profiles SET agent_backend='openclaw' WHERE slug=:b", b=sweep.bot)
    turn_id, req = sweep.turn_with("openclaw")
    _job, gate_id = sweep.hold(turn_id)
    sweep.bridge.answer(req, "running", boot="boot-a", backend="openclaw")
    sweep.run_once()
    sweep.bridge.answer(req, "not_found", boot="boot-b", backend="openclaw")
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
