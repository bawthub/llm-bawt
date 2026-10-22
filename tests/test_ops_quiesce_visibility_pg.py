"""PostgreSQL tests for the TASK-1233 (f) visibility slice.

Roster rows, job phase and deadlines, phase-transition logs, the live-holds
view, the operator cancel before dispatch (including its race with a step
that is mid-open), and the app-side receipt loop that announces each receipt
(Vex r2 N1). Reuses the quiesce driver rig: each test scopes its operation to
its own agent bot, so no other test's turns are held.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from datetime import timedelta
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import text

from llm_bawt.ops.models import JOB_CANCELLED, JOB_DISPATCHING, JOB_QUIESCING, _iso_utc
from llm_bawt.ops.quiesce import cancel_quiesce_job
from llm_bawt.ops.quiesce_status import attach_members, live_holds
from llm_bawt.service.turn_gates import _bootstrap
from tests.test_ops_quiesce_pg import _as_caller, _receipt_deliveries, rig  # noqa: F401  (fixture)

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]

PHASE_LOG = "llm_bawt.ops.quiesce"


def _status(rig, job_id):
    return rig.service.get_job_status(job_id, reconcile_if_active=False)


def _phase_lines(caplog, job_id):
    return [r.getMessage().split(" phase ", 1)[1].split(" ", 1)[0]
            for r in caplog.records if r.name == PHASE_LOG and f"job={job_id} " in r.getMessage()]


# -- (1) roster rows ----------------------------------------------------------

def test_roster_rows_carry_identity_control_state_and_timestamps(rig):
    turn_id = rig.turn()
    job_id = rig.job()
    assert rig.driver.step(job_id).outcome == "opened"
    [member] = _status(rig, job_id)["quiesce"]["members"]
    assert (member["bot_id"], member["turn_id"], member["disposition"]) == (rig.bot, turn_id, "running")
    assert (member["control_state"], member["control_attempts"], member["ack_status"]) == ("pending", 0, None)
    assert member["enrolled_at"] and member["control_changed_at"] == member["enrolled_at"]
    assert member["drained"] is None  # enrolled, not yet judged by a hold step

    assert rig.driver.step(job_id).outcome == "holding"
    [member] = _status(rig, job_id)["quiesce"]["members"]
    assert (member["drained"], member["drain_reason"]) == (False, "no_execution_exit")

    # A row whose column is NULL (pre-dates the column) reads its enrolment time.
    rig.sql("UPDATE gate_members SET control_changed_at=NULL,"
            " created_at = created_at - interval '1 hour' WHERE turn_id=:t", t=turn_id)
    [member] = _status(rig, job_id)["quiesce"]["members"]
    assert member["control_changed_at"] == member["enrolled_at"]

    listed = attach_members(rig.engine, [rig.store.get_job(job_id).to_api(include_output=False),
                                         {"id": "plain", "quiesce": None}])
    assert [m["turn_id"] for m in listed[0]["quiesce"]["members"]] == [turn_id]


def test_the_column_migration_never_stamps_pre_existing_rows(rig):
    """M2: ADD COLUMN then SET DEFAULT leaves old rows NULL (read as created_at)."""
    turn_id = rig.turn()
    job_id = rig.job()
    rig.driver.step(job_id)
    with rig.engine.connect() as conn:
        tx = conn.begin()
        try:
            conn.execute(text("ALTER TABLE gate_members DROP COLUMN control_changed_at"))
            conn.execute(text("UPDATE gate_members SET created_at = now() - interval '1 day'"
                              " WHERE turn_id=:t"), {"t": turn_id})
            _bootstrap(conn)  # the production DDL, against a row that pre-dates the column
            old = conn.execute(text("SELECT control_changed_at, created_at FROM gate_members"
                                    " WHERE turn_id=:t"), {"t": turn_id}).one()
            assert old[0] is None
            gate = conn.execute(text("SELECT gate_id FROM gate_members WHERE turn_id=:t"),
                                {"t": turn_id}).scalar()
            conn.execute(text("INSERT INTO gate_members (gate_id, bot_id, turn_id, turn_status, disposition)"
                              " VALUES (:g, :b, 'turn-new-row', 'streaming', 'running')"),
                         {"g": gate, "b": rig.bot})
            new = conn.execute(text("SELECT control_changed_at FROM gate_members"
                                    " WHERE turn_id='turn-new-row'")).scalar()
            assert new is not None
            read = conn.execute(text("SELECT COALESCE(control_changed_at, created_at) FROM gate_members"
                                     " WHERE turn_id=:t"), {"t": turn_id}).scalar()
            assert read == old[1]
        finally:
            tx.rollback()


def test_cancelling_pending_steers_stamps_control_changed_at(rig):
    turn_id = rig.turn()
    job_id = rig.job()
    rig.driver.step(job_id)
    rig.sql("UPDATE gate_members SET control_changed_at = now() - interval '1 hour' WHERE turn_id=:t",
            t=turn_id)
    assert cancel_quiesce_job(rig.engine, job_id, actor="nick") == ("cancelled", JOB_CANCELLED)
    state, fresh = rig.sql("SELECT control_state, control_changed_at > now() - interval '1 minute'"
                           " FROM gate_members WHERE turn_id=:t", t=turn_id).one()
    assert (state, fresh) == ("cancelled", True)


# -- (2) phase + deadlines, (5) one log line per transition --------------------

def test_phases_and_deadlines_follow_the_driver_and_log_once_per_change(rig, caplog):
    caplog.set_level(logging.INFO, logger=PHASE_LOG)
    turn_id = rig.turn()
    first, second = rig.job(), rig.job()
    unopened = _status(rig, first)["quiesce"]
    submitted = rig.row(first)["submitted_at"]
    assert unopened["open_deadline_at"] == _iso_utc(submitted + timedelta(seconds=300))
    assert unopened["phase"]["name"] == "opening"  # no step yet

    assert rig.driver.step(first).outcome == "opened"
    assert rig.driver.step(second).outcome == "waiting_slot"  # max_concurrent=1
    waiting = _status(rig, second)["quiesce"]
    assert waiting["phase"]["name"] == "waiting_slot" and waiting["open_deadline_at"] is not None

    for _ in range(3):  # steady state logs nothing
        assert rig.driver.step(first).outcome == "holding"
    held = _status(rig, first)["quiesce"]
    assert held["open_deadline_at"] is None and held["deadline_at"] is not None
    phase = held["phase"]
    assert (phase["name"], phase["drained"], phase["total"]) == ("draining", 0, 1)
    assert phase["since"] and phase["observed_at"] == held["receipt"]["observed_at"]

    rig.end(turn_id)
    margin = rig.driver.margin_seconds
    rig.driver.margin_seconds = 10 ** 7  # the dispatch CAS cannot take this gate
    assert rig.driver.step(first).outcome == "cas_retry"
    phase = _status(rig, first)["quiesce"]["phase"]
    assert (phase["name"], phase["drained"], phase["total"]) == ("dispatch_pending", 1, 1)
    rig.driver.margin_seconds = margin
    assert rig.driver.step(first).outcome == "dispatched"
    dispatched = _status(rig, first)
    assert dispatched["state"] == JOB_DISPATCHING
    assert dispatched["quiesce"]["phase"]["name"] == "dispatched"
    assert dispatched["quiesce"]["phase"]["since"] == dispatched["dispatched_at"]

    assert _phase_lines(caplog, first) == ["submitted->draining", "draining->dispatch_pending",
                                           "dispatch_pending->dispatched"]
    assert _phase_lines(caplog, second) == ["submitted->waiting_slot"]
    [line] = [r.getMessage() for r in caplog.records if f"job={first} " in r.getMessage()
              and "->draining" in r.getMessage()]
    assert f"op={rig.slug} " in line and "drained=0/1" in line and "deadline=" in line


def test_a_preflight_note_and_a_failure_are_phases_with_their_detail_logged(rig, caplog):
    caplog.set_level(logging.INFO, logger=PHASE_LOG)
    job_id = rig.job()
    assert rig.driver.note_unopened(job_id, "target unhealthy").outcome == "preflight_blocked"
    assert _status(rig, job_id)["quiesce"]["phase"]["name"] == "preflight_blocked"
    rig.sql("UPDATE ops_jobs SET submitted_at = now() - interval '1 hour' WHERE id=:id", id=job_id)
    assert rig.driver.note_unopened(job_id, "target unhealthy").outcome == "failed"
    status = _status(rig, job_id)
    assert status["quiesce"]["phase"]["name"] == "failed" and status["quiesce"]["open_deadline_at"] is None
    assert _phase_lines(caplog, job_id) == ["submitted->preflight_blocked", "preflight_blocked->failed"]
    assert any("(quiesce_timeout)" in r.getMessage() for r in caplog.records if f"job={job_id} " in r.getMessage())


# -- (3) live holds -----------------------------------------------------------

def test_live_holds_lists_job_and_operator_gates(rig):
    rig.turn()
    job_id = rig.job()
    rig.driver.step(job_id)
    gate_id = rig.row(job_id)["quiesce_gate_id"]
    operator = rig.sql(
        "INSERT INTO system_gates (owner_kind, owner_id, mode, scope, reason, expires_at)"
        " VALUES ('operator', :o, 'deny', '{\"bots\": []}', 'maintenance', now() + interval '5 minutes')"
        " RETURNING id", o=f"op-{rig.bot}").scalar()
    try:
        holds = {h["gate_id"]: h for h in live_holds(rig.engine)}
        job_hold = holds[gate_id]
        assert (job_hold["owner_kind"], job_hold["mode"]) == ("job", "hold")
        assert job_hold["members"] == {"total": 1, "by_disposition": {"running": 1},
                                       "by_control_state": {"pending": 1}}
        assert job_hold["job"] == {"id": job_id, "operation": rig.slug, "state": JOB_QUIESCING,
                                   "phase": "draining"}
        op_hold = holds[operator]
        assert op_hold["job"] is None and op_hold["reason"] == "maintenance"
        assert op_hold["members"]["total"] == 0 and op_hold["scope"] == {"bots": []}

        cancel_quiesce_job(rig.engine, job_id, actor="nick")
        assert gate_id not in {h["gate_id"] for h in live_holds(rig.engine)}
    finally:
        rig.sql("DELETE FROM system_gates WHERE id=:id", id=operator)


# -- (4) operator cancel ------------------------------------------------------

def _route(rig, monkeypatch):
    from llm_bawt.service.routes import ops as routes

    monkeypatch.setattr(routes, "_service", lambda: rig.service)
    return routes


def test_cancel_before_dispatch_releases_every_generation_and_is_idempotent(rig, monkeypatch):
    routes = _route(rig, monkeypatch)
    turn_id = rig.turn()
    job_id = rig.job()
    rig.driver.step(job_id)
    first = rig.row(job_id)["quiesce_gate_id"]
    rig.sql("UPDATE system_gates SET expires_at = now() - interval '1 second' WHERE id=:id", id=first)
    rig.driver.step(job_id)  # rebuild: a second generation
    rig.sql("UPDATE system_gates SET expires_at = now() + interval '1 minute' WHERE id=:id", id=first)
    assert len(rig.gates_of(job_id)) == 2 and rig.held()

    body = routes.CancelRequest(actor="nick-ui", reason="wrong target")
    result = routes.cancel_job(job_id, body)
    assert result["cancelled"] is True and result["job"]["state"] == JOB_CANCELLED
    row = rig.row(job_id)
    assert row["error_text"] == ("cancelled by operator (claimed actor: nick-ui) before dispatch;"
                                 " no action submitted: wrong target")
    assert row["finished_at"] is not None and row["quiesce_driver_id"] is None
    receipt = json.loads(row["quiesce_json"])
    assert receipt["outcome"] == "cancelled"
    assert receipt["cancelled_by"] == {"actor_claimed": "nick-ui", "reason": "wrong target"}
    assert result["job"]["quiesce"]["phase"]["name"] == "cancelled"
    assert all(g["released_at"] is not None for g in rig.gates_of(job_id)) and not rig.held()
    assert rig.sql("SELECT control_state FROM gate_members WHERE turn_id=:t", t=turn_id).scalars().all() == [
        "cancelled", "cancelled"]

    again = routes.cancel_job(job_id, body)
    assert (again["cancelled"], again["already"]) == (False, True)
    assert rig.driver.step(job_id).outcome == "not_owner" and rig.executor.dispatch_calls == []
    with pytest.raises(HTTPException) as missing:
        routes.cancel_job("no-such-job", body)
    assert missing.value.status_code == 404


def test_a_dispatched_job_is_not_cancellable(rig, monkeypatch):
    routes = _route(rig, monkeypatch)
    turn_id = rig.turn()
    job_id = rig.job()
    rig.driver.step(job_id)
    rig.end(turn_id)
    assert rig.driver.step(job_id).outcome == "dispatched"
    with pytest.raises(HTTPException) as refused:
        routes.cancel_job(job_id, routes.CancelRequest(actor="nick"))
    assert refused.value.status_code == 409
    assert refused.value.detail["code"] == "not_cancellable"
    assert refused.value.detail["state"] == JOB_DISPATCHING
    assert rig.row(job_id)["state"] == JOB_DISPATCHING


def test_a_cancelled_unopened_job_ignores_a_late_preflight_note(rig):
    job_id = rig.job()
    assert cancel_quiesce_job(rig.engine, job_id, actor="nick") == ("cancelled", JOB_CANCELLED)
    error = rig.row(job_id)["error_text"]
    assert rig.driver.note_unopened(job_id, "target unhealthy").outcome == "not_owner"
    assert rig.driver.step(job_id).outcome == "not_owner"
    row = rig.row(job_id)
    assert (row["state"], row["error_text"]) == (JOB_CANCELLED, error)
    assert rig.gates_of(job_id) == []


def test_a_cancel_waits_for_a_step_that_is_mid_open_then_releases_its_new_gate(rig):
    rig.turn()
    job_id = rig.job()
    opened, proceed = threading.Event(), threading.Event()
    activate = rig.gates.activate

    def paused(conn, spec, predecessor):
        activated = activate(conn, spec, predecessor)
        opened.set()  # the gate row exists, uncommitted, inside the step's transaction
        assert proceed.wait(10)
        return activated

    rig.gates.activate = paused
    results: dict[str, object] = {}
    step = threading.Thread(target=lambda: results.setdefault("step", rig.driver.step(job_id).outcome))
    cancel = threading.Thread(target=lambda: results.setdefault(
        "cancel", cancel_quiesce_job(rig.engine, job_id, actor="nick")))
    step.start()
    assert opened.wait(10)
    cancel.start()
    cancel.join(0.5)
    assert cancel.is_alive()  # blocked behind the step's locks, not racing past it
    proceed.set()
    step.join(10)
    cancel.join(10)
    assert results == {"step": "opened", "cancel": ("cancelled", JOB_CANCELLED)}
    [gate] = rig.gates_of(job_id)
    assert gate["released_at"] is not None and not rig.held()
    assert rig.row(job_id)["state"] == JOB_CANCELLED and rig.executor.dispatch_calls == []


# -- N1: the app enqueues the receipt and announces it ---------------------------

def test_the_app_receipt_loop_announces_a_cancelled_jobs_receipt(rig, monkeypatch):
    import llm_bawt.service.ops_receipt_outbox as outbox
    from llm_bawt.service.inter_bot_dispatcher import InterBotDeliveryDispatcher

    ops_tools = _as_caller(rig, monkeypatch, turn_id=rig.turn())
    job_id = asyncio.run(ops_tools.ops_run(rig.slug, {"target": "a"}))["job_id"]
    rig.driver.step(job_id)
    assert cancel_quiesce_job(rig.engine, job_id, actor="nick")[0] == "cancelled"

    dispatcher = InterBotDeliveryDispatcher(SimpleNamespace(config=rig.store.config))
    emitted = []

    async def emit(record):
        emitted.append((record.id, record.status))
    dispatcher._emit = emit
    monkeypatch.setattr(outbox, "get_ops_service", lambda _config: rig.service)
    app = SimpleNamespace(_inter_bot_dispatcher=dispatcher, config=rig.store.config)

    async def one_pass():
        dispatcher._loop = asyncio.get_running_loop()
        loop_task = asyncio.create_task(outbox.run_ops_receipt_outbox(app, idle_seconds=0.05))
        for _ in range(200):
            if emitted:
                break
            await asyncio.sleep(0.05)
        loop_task.cancel()
        await asyncio.gather(loop_task, return_exceptions=True)
        return dispatcher._wake_event.is_set()

    woke = asyncio.run(one_pass())
    [delivery] = _receipt_deliveries(rig)
    assert emitted == [(delivery["id"], "QUEUED")] and woke
    assert rig.row(job_id)["receipt_callback_delivery_id"] == delivery["id"]
    sent = json.loads(delivery["message"].split("\n", 1)[1])
    assert (sent["id"], sent["state"]) == (job_id, JOB_CANCELLED)
