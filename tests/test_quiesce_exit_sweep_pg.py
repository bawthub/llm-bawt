"""PostgreSQL tests for the TASK-1233 B1 (b)/(c) quiesce exit sweep.

The real driver opens a real hold (``rig`` from :mod:`tests.test_ops_quiesce_pg`);
only the bridge is faked: ``run.status`` replies and terminal records per
request id. Each test checks the driver's own verdict (``gate_drain_status``)
and, where it drains, that the job dispatches.
"""
from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy import text

from agent_bridge.run_contract import RunTerminal, TeardownReport
from llm_bawt.ops import JOB_ACCEPTED
from llm_bawt.ops.models import JOB_QUIESCING
from llm_bawt.ops.quiesce_exit import ensure_sighting_schema, purge_sightings
from llm_bawt.service.quiesce_exit_sweep import QuiesceExitSweep
from llm_bawt.service.turn_gates import gate_drain_status
from tests import test_ops_quiesce_pg as _quiesce

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]

rig = _quiesce.rig


class _Bridge:
    """Answers ``run.status`` for the requests it knows; times out for every other."""

    def __init__(self):
        self.replies: dict[str, dict] = {}
        self.terminals: dict[str, RunTerminal] = {}
        self.asked: list[str] = []

    def answer(self, request_id, state, *, boot, backend="claude-code", pid1=True):
        self.replies[request_id] = {"backend": backend, "boot_id": boot, "request_id": request_id,
                                    "state": state, "detail": "", "pid1": pid1}

    async def send_rpc(self, method, params, request_id, *, timeout_s, backend):
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
        return asyncio.run(rig.sweep.run_once())

    rig.turn_with, rig.hold, rig.drain, rig.run_once = turn, held, drain, run_once
    yield rig
    rig.sql("DELETE FROM quiesce_run_sightings WHERE request_id LIKE 'req-turn-%'")


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
