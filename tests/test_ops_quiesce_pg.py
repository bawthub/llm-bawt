"""PostgreSQL tests for the TASK-1233 quiesce driver (llm_bawt.ops.quiesce).

Each test scopes its operation to its own agent bot, so no other test's turns
are ever held. Every test drives the production ``TurnGates`` seam: the real
``activate_hold`` (with ``predecessor=`` on a rebuild) and the real
``gate_drain_status``, which drains a member only on a confirmed
``execution_exit`` for its pinned request.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from llm_bawt.ops import JOB_ACCEPTED, JOB_DISPATCHING, JOB_FAILED, OpsService
from llm_bawt.ops.models import JOB_QUIESCING
from llm_bawt.ops.quiesce import MARGIN_SECONDS, QuiesceDriver, TurnGates, record_ack
from llm_bawt.service.turn_admission import lock_target
from llm_bawt.service.turn_gates import EXECUTION_EXIT_KEY, EXIT_SOURCES, GateActor, gate_blocks
from tests.test_ops_service import FakeExecutor, _op_data

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


class RecordingGates(TurnGates):
    """The production gates, recording each activation's ``predecessor``."""

    def __init__(self):
        self.predecessors: list[int | None] = []

    def activate(self, conn, spec, predecessor):
        self.predecessors.append(predecessor)
        return super().activate(conn, spec, predecessor)


@pytest.fixture
def rig(private_app_pg):
    from llm_bawt.ops.store import OpsStore
    from llm_bawt.runtime_settings import BotProfileStore

    store = OpsStore(private_app_pg)
    BotProfileStore(private_app_pg)
    engine = store.engine
    tag = uuid.uuid4().hex[:12]
    bot = f"test-quiesce-{tag}"
    now = datetime.now(timezone.utc)
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO bot_profiles (slug, name, description, system_prompt, requires_memory,
                voice_optimized, uses_tools, uses_search, uses_home_assistant, bot_type,
                agent_backend, created_at, updated_at)
            VALUES (:slug, :slug, '', '', false, false, false, false, false, 'agent',
                'claude-code', :now, :now)
        """), {"slug": bot, "now": now})
    slug = f"test.quiesce-{tag}"
    store.create_operation(_op_data(slug=slug, quiesce_enabled=True, quiesce_timeout_seconds=300,
                                    quiesce_scope_json=json.dumps({"bots": [bot]}), max_concurrent=1))
    executor = FakeExecutor()
    service = OpsService(store, executor=executor)
    gates = RecordingGates()
    service.quiesce_driver = QuiesceDriver(engine, driver_id="driver-a", gates=gates)
    ctx = SimpleNamespace(store=store, engine=engine, bot=bot, slug=slug, executor=executor,
                          service=service, gates=gates, driver=service.quiesce_driver)

    def job(key=None):
        return service.dispatch_job(operation_slug=slug, args={"target": "a"},
                                    idempotency_key=key or uuid.uuid4().hex)["id"]

    def turn():
        turn_id = f"turn-{uuid.uuid4().hex}"
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO turn_logs (id, created_at, path, stream, bot_id, user_id, status,
                                       user_prompt, response_text, ended_at, agent_request_id)
                VALUES (:id, now(), '/test', true, :bot, 'nick', 'streaming', '', '', NULL, :req)
            """), {"id": turn_id, "bot": bot, "req": f"req-{turn_id}"})
        return turn_id

    def end(turn_id, *, exited=True):
        """End the row; ``exited`` stamps the bridge's confirmed execution_exit for its request."""
        record = {"request_id": f"req-{turn_id}", "backend": None, "source": sorted(EXIT_SOURCES)[0],
                  "exited": exited}
        with engine.begin() as conn:
            conn.execute(text("UPDATE turn_logs SET status='completed', ended_at=now(), request_json=:j"
                              " WHERE id=:id"), {"id": turn_id, "j": json.dumps({EXECUTION_EXIT_KEY: record})})

    def sql(statement, **params):
        with engine.begin() as conn:
            return conn.execute(text(statement), params)

    def row(job_id):
        return sql("SELECT * FROM ops_jobs WHERE id=:id", id=job_id).mappings().one()

    def gates_of(job_id):
        return sql("SELECT id, released_at, expires_at > now() AS live FROM system_gates"
                   " WHERE owner_id=:o ORDER BY id", o=job_id).mappings().all()

    def held():
        with engine.begin() as conn:
            lock_target(conn, bot)
            return gate_blocks(conn, bot, GateActor("save_turn")) is not None

    ctx.job, ctx.turn, ctx.end, ctx.sql, ctx.row, ctx.gates_of, ctx.held = (
        job, turn, end, sql, row, gates_of, held)
    yield ctx
    with engine.begin() as conn:
        owners = [r[0] for r in conn.execute(text("SELECT id FROM ops_jobs WHERE operation_slug=:s"),
                                             {"s": slug}).all()]
        conn.execute(text("DELETE FROM gate_members WHERE gate_id IN"
                          " (SELECT id FROM system_gates WHERE owner_id = ANY(:o))"), {"o": owners})
        conn.execute(text("DELETE FROM system_gates WHERE owner_id = ANY(:o)"), {"o": owners})
        conn.execute(text("DELETE FROM ops_jobs WHERE operation_slug=:s"), {"s": slug})
        conn.execute(text("DELETE FROM turn_logs WHERE bot_id=:b"), {"b": bot})
        conn.execute(text("DELETE FROM bot_profiles WHERE slug=:b"), {"b": bot})


def test_job_is_created_quiescing_and_never_claimed(rig):
    job_id = rig.job()
    assert rig.row(job_id)["state"] == JOB_QUIESCING
    assert rig.store.claim_job(job_id) is False
    assert rig.gates_of(job_id) == [] and rig.executor.dispatch_calls == []


def test_holds_until_drained_then_dispatches_inside_the_fence(rig):
    turn_id = rig.turn()
    job_id = rig.job()
    assert rig.service.get_job_status(job_id)["state"] == JOB_QUIESCING
    [gate] = rig.gates_of(job_id)
    assert rig.held() and rig.row(job_id)["quiesce_gate_id"] == gate["id"]
    roster = json.loads(rig.row(job_id)["quiesce_json"])["roster"]
    assert [m["turn_id"] for m in roster] == [turn_id]

    assert rig.driver.step(job_id).outcome == "holding"
    assert rig.executor.dispatch_calls == []

    rig.end(turn_id)
    status = rig.service.get_job_status(job_id)
    job = rig.row(job_id)
    assert status["state"] == JOB_ACCEPTED and job["dispatched_at"] is not None
    fence = json.loads(job["quiesce_fence_json"])
    assert fence["generation"] == gate["id"]
    [call] = rig.executor.dispatch_calls
    assert call["fence"] == fence
    snapshot = json.loads(job["invocation_snapshot_json"])
    timeout = snapshot["execution"]["timeout_seconds"]
    cover = snapshot["execution"]["start_delay_seconds"] + timeout + 2 * MARGIN_SECONDS
    expires, extended = rig.sql(
        "SELECT expires_at, expires_at >= dispatched_at + (:cover * interval '1 second') - interval '2 seconds'"
        " FROM system_gates g JOIN ops_jobs j ON j.quiesce_gate_id=g.id WHERE j.id=:id",
        id=job_id, cover=cover).one()
    assert extended
    assert datetime.fromisoformat(fence["start_not_after"]) == expires - timedelta(
        seconds=timeout + MARGIN_SECONDS)
    assert rig.held()  # the hold covers the action, released only once terminal


def test_a_driver_without_the_lease_writes_nothing(rig):
    job_id = rig.job()
    rig.turn()
    assert rig.driver.step(job_id).outcome == "opened"
    before = dict(rig.row(job_id))
    other = QuiesceDriver(rig.engine, driver_id="driver-b", gates=rig.gates)
    assert other.step(job_id).outcome == "not_owner"
    assert dict(rig.row(job_id)) == before
    assert len(rig.gates_of(job_id)) == 1

    rig.sql("UPDATE ops_jobs SET quiesce_driver_lease_expires_at = now() - interval '1 second'"
            " WHERE id=:id", id=job_id)
    assert other.step(job_id).outcome == "holding"
    assert rig.row(job_id)["quiesce_driver_id"] == "driver-b"
    assert rig.driver.step(job_id).outcome == "not_owner"


def test_a_job_that_left_quiescing_never_opens_a_gate(rig):
    job_id = rig.job()
    rig.store.mark_terminal(job_id, state=JOB_FAILED, error_text="cancelled")
    assert rig.driver.step(job_id).outcome == "not_owner"
    assert rig.gates_of(job_id) == []


def test_a_lapsed_gate_is_rebuilt_then_lost_past_the_deadline(rig):
    rig.turn()
    job_id = rig.job()
    rig.driver.step(job_id)
    [first] = rig.gates_of(job_id)
    rig.sql("UPDATE system_gates SET expires_at = now() - interval '1 second' WHERE id=:id", id=first["id"])
    assert rig.driver.step(job_id).outcome == "holding"
    second = rig.row(job_id)["quiesce_gate_id"]
    assert second != first["id"] and rig.gates.predecessors == [None, first["id"]]
    assert json.loads(rig.row(job_id)["quiesce_json"])["generations"] == [first["id"], second]
    assert rig.held()

    rig.sql("UPDATE system_gates SET expires_at = now() - interval '1 second' WHERE id=:id", id=second)
    rig.sql("UPDATE ops_jobs SET quiesce_deadline_at = now() - interval '1 second' WHERE id=:id", id=job_id)
    assert rig.driver.step(job_id).outcome == "failed"
    job = rig.row(job_id)
    assert job["state"] == JOB_FAILED and job["error_text"].startswith("quiesce_lost")
    assert all(g["released_at"] is not None for g in rig.gates_of(job_id))
    assert len(rig.gates.predecessors) == 2  # no third generation
    assert rig.executor.dispatch_calls == []


def test_timeout_fails_releases_and_cancels_pending_steers(rig):
    turn_id = rig.turn()
    job_id = rig.job()
    rig.driver.step(job_id)
    assert rig.sql("SELECT control_state FROM gate_members WHERE turn_id=:t", t=turn_id).scalar() == "pending"
    rig.sql("UPDATE ops_jobs SET quiesce_deadline_at = now() - interval '1 second' WHERE id=:id", id=job_id)
    assert rig.driver.step(job_id).outcome == "failed"
    job = rig.row(job_id)
    assert job["state"] == JOB_FAILED and job["error_text"].startswith("quiesce_timeout")
    assert json.loads(job["quiesce_json"])["outcome"] == "quiesce_timeout"
    assert rig.sql("SELECT control_state FROM gate_members WHERE turn_id=:t", t=turn_id).scalar() == "cancelled"
    assert not rig.held() and rig.executor.dispatch_calls == []


def test_the_deadline_is_set_once(rig):
    rig.turn()
    job_id = rig.job()
    rig.driver.step(job_id)
    first = rig.row(job_id)["quiesce_deadline_at"]
    gate = rig.row(job_id)["quiesce_gate_id"]
    rig.sql("UPDATE system_gates SET expires_at = now() - interval '1 second' WHERE id=:id", id=gate)
    rig.driver.step(job_id)
    assert rig.row(job_id)["quiesce_deadline_at"] == first


def test_terminal_jobs_gates_are_swept(rig):
    turn_id = rig.turn()
    job_id = rig.job()
    rig.driver.step(job_id)
    rig.store.mark_terminal(job_id, state=JOB_FAILED, error_text="operator cancelled")
    assert rig.held()
    rig.service.reconcile_active_jobs()
    assert all(g["released_at"] is not None for g in rig.gates_of(job_id))
    assert rig.sql("SELECT control_state FROM gate_members WHERE turn_id=:t", t=turn_id).scalar() == "cancelled"
    assert not rig.held()
    assert rig.driver.release_terminal() == []


def test_disabled_operation_fails_before_and_after_the_hold(rig):
    rig.turn()
    late = rig.job()
    rig.driver.step(late)
    rig.store.update_operation(rig.slug, {"enabled": False})
    assert rig.driver.step(late).outcome == "failed"
    assert "disabled" in rig.row(late)["error_text"]
    assert all(g["released_at"] is not None for g in rig.gates_of(late))

    rig.store.update_operation(rig.slug, {"enabled": True})
    early = rig.job()
    rig.store.update_operation(rig.slug, {"enabled": False})
    assert rig.driver.step(early).outcome == "failed"
    assert rig.gates_of(early) == [] and rig.executor.dispatch_calls == []


def test_a_reached_cap_waits_without_holding(rig):
    rig.turn()
    first = rig.job()
    second = rig.job()
    assert rig.driver.step(first).outcome == "opened"
    assert rig.driver.step(second).outcome == "waiting_slot"
    job = rig.row(second)
    assert job["state"] == JOB_QUIESCING and "slot" in job["error_text"]
    assert rig.gates_of(second) == []
    rig.store.mark_terminal(first, state=JOB_FAILED, error_text="done")
    assert rig.driver.step(second).outcome == "opened"


def test_a_dispatching_quiesce_job_holds_the_slot(rig):
    first = rig.job()
    assert rig.driver.step(first).outcome == "opened"
    assert rig.driver.step(first).outcome == "dispatched"
    assert rig.row(first)["state"] == JOB_DISPATCHING
    second = rig.job()
    assert rig.driver.step(second).outcome == "waiting_slot"


def test_ack_lands_only_on_a_live_member(rig):
    turn_id = rig.turn()
    job_id = rig.job()
    with pytest.raises(ValueError):
        record_ack(rig.engine, bot_id=rig.bot, turn_id=turn_id, status="done")
    assert record_ack(rig.engine, bot_id=rig.bot, turn_id=turn_id, status="stopping") is None
    rig.driver.step(job_id)
    gate = rig.row(job_id)["quiesce_gate_id"]
    assert record_ack(rig.engine, bot_id=rig.bot, turn_id=turn_id, status="cannot_stop",
                      note="mid-migration") == gate
    status, note = rig.sql("SELECT ack_status, ack_note FROM gate_members WHERE turn_id=:t",
                           t=turn_id).one()
    assert (status, note) == ("cannot_stop", "mid-migration")
    assert record_ack(rig.engine, bot_id=rig.bot, turn_id="turn-other", status="stopping") is None
    assert record_ack(rig.engine, bot_id="someone-else", turn_id=turn_id, status="stopping") is None
    rig.store.mark_terminal(job_id, state=JOB_FAILED, error_text="x")
    rig.driver.release_terminal()
    assert record_ack(rig.engine, bot_id=rig.bot, turn_id=turn_id, status="stopping") is None


def test_the_production_gates_hold_rebuild_with_carry_forward_and_release(rig):
    """The real ``TurnGates()`` seam end to end: no stand-in, no recording subclass."""
    driver = rig.service.quiesce_driver = QuiesceDriver(rig.engine, driver_id="driver-real", gates=TurnGates())
    turn_id = rig.turn()
    job_id = rig.job()
    assert driver.step(job_id).outcome == "opened"
    [first] = rig.gates_of(job_id)
    with rig.engine.connect() as conn:
        [status] = TurnGates().drain_status(conn, first["id"])
    assert (status.turn_id, status.drained, status.reason) == (turn_id, False, "no_execution_exit")

    # The row ends without a confirmed exit: the run may still be alive, so no dispatch.
    rig.end(turn_id, exited=False)
    assert driver.step(job_id).outcome == "holding"
    assert rig.executor.dispatch_calls == []

    # Gate loss: the rebuild is a new generation that carries the undrained member forward.
    rig.sql("UPDATE system_gates SET expires_at = now() - interval '1 second' WHERE id=:id", id=first["id"])
    assert driver.step(job_id).outcome == "holding"
    second = rig.row(job_id)["quiesce_gate_id"]
    assert second != first["id"]
    assert rig.sql("SELECT turn_id, disposition FROM gate_members WHERE gate_id=:g", g=second).all() == [
        (turn_id, "running")]
    assert rig.held() and rig.executor.dispatch_calls == []

    rig.end(turn_id)  # the bridge confirms the exit
    assert rig.service.get_job_status(job_id)["state"] == JOB_ACCEPTED
    assert json.loads(rig.row(job_id)["quiesce_fence_json"])["generation"] == second
    rig.store.mark_terminal(job_id, state=JOB_FAILED, error_text="done")
    rig.service.reconcile_active_jobs()
    assert all(g["released_at"] is not None for g in rig.gates_of(job_id))
    assert not rig.held()


def test_ack_lands_only_on_the_current_generation(rig):
    turn_id = rig.turn()
    job_id = rig.job()
    rig.driver.step(job_id)
    first = rig.row(job_id)["quiesce_gate_id"]
    rig.sql("UPDATE system_gates SET expires_at = now() - interval '1 second' WHERE id=:id", id=first)
    rig.driver.step(job_id)
    second = rig.row(job_id)["quiesce_gate_id"]
    # A superseded generation that is still live (an unreleased rebuild race) never takes the ack.
    rig.sql("UPDATE system_gates SET expires_at = now() + interval '1 minute' WHERE id=:id", id=first)
    assert record_ack(rig.engine, bot_id=rig.bot, turn_id=turn_id, status="stopping") == second
    acks = dict(rig.sql("SELECT gate_id, ack_status FROM gate_members WHERE turn_id=:t", t=turn_id).all())
    assert acks == {first: None, second: "stopping"}


def test_ack_is_refused_once_the_job_dispatches(rig):
    turn_id = rig.turn()
    job_id = rig.job()
    rig.driver.step(job_id)
    rig.end(turn_id)
    assert rig.driver.step(job_id).outcome == "dispatched"
    assert rig.row(job_id)["state"] == JOB_DISPATCHING
    assert record_ack(rig.engine, bot_id=rig.bot, turn_id=turn_id, status="stopping") is None
    assert rig.sql("SELECT ack_status FROM gate_members WHERE turn_id=:t", t=turn_id).scalar() is None
