"""PostgreSQL tests for the TASK-1233 B3 quiesce control pump.

A real quiescing job opens a real gate over this test's own bot (the ``rig``
of test_ops_quiesce_pg); the pump claims its members and steers each through
a recorder that asks ``check_steer`` with the claim's grant, exactly as
``steer_active_turn`` does, then answers with a scripted outcome.
"""
from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy import text

from llm_bawt.ops.quiesce import release_job_gates
from llm_bawt.service.quiesce_control import (
    GONE,
    RETRY,
    STEERED,
    QuiesceControlPump,
    SteerOutcome,
    control_request_id,
)
from llm_bawt.service.system_gate_admission import (
    ControlGrant,
    ControlGrantRefused,
    ControlTarget,
    SteerGateRefused,
    check_steer,
)
from tests.test_ops_quiesce_pg import rig  # noqa: F401

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


class Steers:
    """The pump's RPC seam: admit the grant as steer_active_turn would, then answer."""

    def __init__(self, engine, *outcomes: SteerOutcome):
        self.engine = engine
        self.outcomes = list(outcomes)
        self.calls: list[tuple] = []

    async def __call__(self, claim):
        target = check_steer(self.engine, claim.grant.bot_id, turn_id=claim.grant.turn_id, grant=claim.grant)
        self.calls.append((claim, target))
        return self.outcomes.pop(0) if self.outcomes else SteerOutcome(STEERED)


def _turn(rig, *, backend="claude-code", session="sess-held", request="auto", status="streaming"):
    turn_id = f"turn-{uuid.uuid4().hex}"
    rig.sql("""
        INSERT INTO turn_logs (id, created_at, path, stream, bot_id, user_id, status, user_prompt,
                               response_text, ended_at, agent_request_id, agent_backend, agent_session_key)
        VALUES (:id, now(), '/test', true, :bot, 'nick', :status, '', '', NULL, :req, :backend, :session)
    """, id=turn_id, bot=rig.bot, status=status, backend=backend, session=session,
            req=f"req-{turn_id}" if request == "auto" else request)
    return turn_id


def _open(rig):
    job_id = rig.job()
    assert rig.driver.step(job_id).outcome == "opened"
    return job_id


def _member(rig, turn_id, gate_id=None):
    clause = " AND gate_id=:g" if gate_id is not None else ""
    return rig.sql(
        "SELECT *, control_next_attempt_at > now() AS backing_off FROM gate_members"
        f" WHERE turn_id=:t{clause} ORDER BY id DESC LIMIT 1", t=turn_id, g=gate_id,
    ).mappings().one()


def _pump(rig, *outcomes):
    steers = Steers(rig.engine, *outcomes)
    return QuiesceControlPump(rig.engine, steers), steers


def _pass(pump):
    return asyncio.run(pump.run_pass())


def _due_now(rig, turn_id):
    rig.sql("UPDATE gate_members SET control_next_attempt_at = now() - interval '1 second'"
            " WHERE turn_id=:t", t=turn_id)


def test_a_held_running_turn_is_steered_once_and_no_delivery_is_written(rig):
    turn_id = _turn(rig)
    job_id = _open(rig)
    record, _ = rig.deliveries.enqueue(
        sender_bot_id=f"{rig.bot}-sender", target_bot_id=rig.bot, message="callback",
        payload={"messages": [{"role": "user", "content": "callback"}], "bot_id": rig.bot,
                 "user": "nick", "stream": True}, on_gate="hold",
    )
    before = _member(rig, turn_id)
    deliveries = lambda: rig.sql(  # noqa: E731
        "SELECT id, status, attempt_count FROM inter_bot_deliveries WHERE target_bot_id=:b", b=rig.bot,
    ).all()
    assert deliveries() == [(record.id, "QUEUED", 0)]

    pump, steers = _pump(rig)
    assert _pass(pump) == 1
    [(claim, target)] = steers.calls
    rid = control_request_id(turn_id, f"req-{turn_id}")
    assert claim.request_id == rid and claim.user_id == "nick" and claim.attempts == 1
    assert claim.grant.gate_id == rig.row(job_id)["quiesce_gate_id"] and claim.grant.turn_id == turn_id
    assert target == ControlTarget(f"req-{turn_id}", "claude-code", "sess-held")
    after = _member(rig, turn_id)
    assert (after["control_state"], after["control_request_id"], after["control_attempts"]) == ("steered", rid, 1)
    assert after["control_claim_token"] is None and after["control_lease_expires_at"] is None
    assert after["control_changed_at"] > before["control_changed_at"]
    # The member row is the message: the held FIFO head is untouched and nothing was enqueued.
    assert deliveries() == [(record.id, "QUEUED", 0)]
    assert rig.sql("SELECT count(*) FROM turn_logs WHERE bot_id=:b", b=rig.bot).scalar() == 1

    assert _pass(pump) == 0 and len(steers.calls) == 1
    assert not pump.has_open_work()


def test_a_public_steer_is_still_refused_under_the_hold(rig):
    turn_id = _turn(rig)
    _open(rig)
    with pytest.raises(SteerGateRefused) as refused:
        check_steer(rig.engine, rig.bot, turn_id=turn_id)
    assert refused.value.status_code == 409


def test_an_ended_member_is_not_needed_and_never_steered(rig):
    turn_id = _turn(rig)
    _open(rig)
    rig.end(turn_id, exited=False)
    pump, steers = _pump(rig)
    assert _pass(pump) == 0 and steers.calls == []
    member = _member(rig, turn_id)
    assert (member["control_state"], member["control_error"]) == ("not_needed", "turn_not_running")


def test_a_pinned_run_on_another_backend_is_unsupported(rig):
    turn_id = _turn(rig, backend="codex")
    _open(rig)
    pump, steers = _pump(rig)
    assert _pass(pump) == 0 and steers.calls == []
    member = _member(rig, turn_id)
    assert (member["control_state"], member["control_error"]) == ("unsupported", "backend:codex")


def test_a_pinned_run_with_no_known_session_is_unconfirmed(rig):
    turn_id = _turn(rig, session=None)
    _open(rig)
    pump, steers = _pump(rig)
    assert _pass(pump) == 0 and steers.calls == []
    member = _member(rig, turn_id)
    assert (member["control_state"], member["control_error"]) == ("unconfirmed", "pinned_run_unaddressable")


def test_the_callers_own_turn_is_never_steered(rig):
    turn_id = _turn(rig)
    job_id = _open(rig)
    rig.sql("UPDATE ops_jobs SET caller_turn_id=:t WHERE id=:id", t=turn_id, id=job_id)
    pump, steers = _pump(rig)
    assert _pass(pump) == 0 and steers.calls == []
    assert _member(rig, turn_id)["control_state"] == "self"


def test_an_unpinned_member_waits_without_changing_state(rig):
    turn_id = _turn(rig, request=None)
    _open(rig)
    before = _member(rig, turn_id)
    pump, steers = _pump(rig)
    assert _pass(pump) == 0 and steers.calls == []
    after = _member(rig, turn_id)
    assert after["control_state"] == "pending" and after["backing_off"]
    assert after["control_changed_at"] == before["control_changed_at"]
    assert after["control_attempts"] == 0

    _due_now(rig, turn_id)
    rig.sql("UPDATE turn_logs SET agent_request_id='req-late' WHERE id=:t", t=turn_id)
    rig.sql("UPDATE gate_members SET agent_request_id='req-late' WHERE turn_id=:t", t=turn_id)
    assert _pass(pump) == 1
    assert steers.calls[0][0].request_id == control_request_id(turn_id, "req-late")


def test_an_ambiguous_result_retries_with_the_same_id(rig):
    turn_id = _turn(rig)
    _open(rig)
    pump, steers = _pump(rig, SteerOutcome(RETRY, "bridge timeout"))
    assert _pass(pump) == 1
    first = _member(rig, turn_id)
    assert (first["control_state"], first["control_error"], first["control_attempts"]) == (
        "pending", "bridge timeout", 1)
    assert first["backing_off"] and first["control_claim_token"] is None

    assert _pass(pump) == 0  # backing off
    _due_now(rig, turn_id)
    assert _pass(pump) == 1
    second = _member(rig, turn_id)
    assert (second["control_state"], second["control_attempts"], second["control_error"]) == ("steered", 2, None)
    assert second["control_changed_at"] > first["control_changed_at"]
    assert [claim.request_id for claim, _ in steers.calls] == [second["control_request_id"]] * 2


def test_a_gone_run_is_not_needed(rig):
    turn_id = _turn(rig)
    _open(rig)
    pump, _ = _pump(rig, SteerOutcome(GONE, "no_active_run"))
    assert _pass(pump) == 1
    member = _member(rig, turn_id)
    assert (member["control_state"], member["control_error"]) == ("not_needed", "no_active_run")


def test_attempts_run_out_to_unconfirmed(rig):
    turn_id = _turn(rig)
    _open(rig)
    pump, steers = _pump(rig, *[SteerOutcome(RETRY, "not ready")] * 3)
    pump.MAX_ATTEMPTS = 2
    assert _pass(pump) == 1
    _due_now(rig, turn_id)
    assert _pass(pump) == 1
    member = _member(rig, turn_id)
    assert (member["control_state"], member["control_attempts"]) == ("unconfirmed", 2)
    assert member["control_next_attempt_at"] is None
    assert _pass(pump) == 0 and len(steers.calls) == 2


def test_an_exhausted_member_is_unconfirmed_without_a_steer(rig):
    turn_id = _turn(rig)
    _open(rig)
    rig.sql("UPDATE gate_members SET control_attempts=:n WHERE turn_id=:t",
            n=QuiesceControlPump.MAX_ATTEMPTS, t=turn_id)
    pump, steers = _pump(rig)
    assert _pass(pump) == 0 and steers.calls == []
    member = _member(rig, turn_id)
    assert (member["control_state"], member["control_error"]) == ("unconfirmed", "attempts_exhausted")


def test_a_lapsed_claim_is_reclaimed_with_a_new_token_and_a_new_stamp(rig):
    turn_id = _turn(rig)
    _open(rig)
    pump, _ = _pump(rig)
    first = pump.claim(_member(rig, turn_id)["id"], rig.bot)
    claimed = _member(rig, turn_id)
    assert claimed["control_state"] == "steering"
    assert pump.claim(claimed["id"], rig.bot) is None  # lease live: not due

    rig.sql("UPDATE gate_members SET control_lease_expires_at = now() - interval '1 second' WHERE id=:id",
            id=claimed["id"])
    second = pump.claim(claimed["id"], rig.bot)
    reclaimed = _member(rig, turn_id)
    assert second.grant.claim_token != first.grant.claim_token and second.attempts == 2
    assert reclaimed["control_changed_at"] > claimed["control_changed_at"]
    assert second.request_id == first.request_id
    # The lapsed claim can no longer settle, and its grant is refused.
    assert pump.settle(first, SteerOutcome(STEERED)) is False
    with pytest.raises(ControlGrantRefused):
        check_steer(rig.engine, rig.bot, turn_id=turn_id, grant=first.grant)
    assert pump.settle(second, SteerOutcome(STEERED)) is True
    assert _member(rig, turn_id)["control_changed_at"] > reclaimed["control_changed_at"]


@pytest.mark.parametrize("forge", ["token", "bot", "turn", "gate", "lease", "released", "settled", "stale"])
def test_a_grant_naming_no_live_claim_is_refused(rig, forge):
    turn_id = _turn(rig)
    job_id = _open(rig)
    pump, _ = _pump(rig)
    claim = pump.claim(_member(rig, turn_id)["id"], rig.bot)
    grant = claim.grant
    bot = rig.bot
    if forge == "token":
        grant = ControlGrant(grant.gate_id, grant.bot_id, grant.turn_id, uuid.uuid4().hex)
    elif forge == "bot":
        bot = "someone-else"
        grant = ControlGrant(grant.gate_id, bot, grant.turn_id, grant.claim_token)
    elif forge == "turn":
        grant = ControlGrant(grant.gate_id, grant.bot_id, "turn-other", grant.claim_token)
    elif forge == "gate":
        grant = ControlGrant(grant.gate_id + 1, grant.bot_id, grant.turn_id, grant.claim_token)
    elif forge == "lease":
        rig.sql("UPDATE gate_members SET control_lease_expires_at = now() - interval '1 second'"
                " WHERE turn_id=:t", t=turn_id)
    elif forge == "released":
        with rig.engine.begin() as conn:
            release_job_gates(conn, job_id)
    elif forge == "settled":
        assert pump.settle(claim, SteerOutcome(STEERED))
    elif forge == "stale":
        # The job moved to a new generation; the old gate, even live again, is not its current one.
        rig.sql("UPDATE system_gates SET expires_at = now() - interval '1 second' WHERE id=:g", g=grant.gate_id)
        assert rig.driver.step(job_id).outcome == "holding"
        assert rig.row(job_id)["quiesce_gate_id"] != grant.gate_id
        rig.sql("UPDATE system_gates SET expires_at = now() + interval '10 minutes' WHERE id=:g", g=grant.gate_id)
    with pytest.raises(ControlGrantRefused):
        check_steer(rig.engine, bot, turn_id=grant.turn_id, grant=grant)
    with pytest.raises(ControlGrantRefused):
        check_steer(None, bot, turn_id=grant.turn_id, grant=grant)  # off PostgreSQL there is no claim


def test_a_release_cancels_a_claim_and_its_late_result_cannot_settle(rig):
    turn_id = _turn(rig)
    job_id = _open(rig)
    pump, _ = _pump(rig)
    claim = pump.claim(_member(rig, turn_id)["id"], rig.bot)
    with rig.engine.begin() as conn:
        release_job_gates(conn, job_id)
    cancelled = _member(rig, turn_id)
    assert cancelled["control_state"] == "cancelled"
    assert pump.settle(claim, SteerOutcome(STEERED)) is False
    after = _member(rig, turn_id)
    assert after["control_state"] == "cancelled" and after["control_changed_at"] == cancelled["control_changed_at"]
    assert pump.due() == [] and not pump.has_open_work()


def test_a_run_already_steered_by_an_earlier_generation_is_not_steered_again(rig):
    turn_id = _turn(rig)
    job_id = _open(rig)
    pump, steers = _pump(rig)
    assert _pass(pump) == 1
    first_gate = rig.row(job_id)["quiesce_gate_id"]
    rig.sql("UPDATE system_gates SET expires_at = now() - interval '1 second' WHERE id=:g", g=first_gate)
    assert rig.driver.step(job_id).outcome == "holding"
    second_gate = rig.row(job_id)["quiesce_gate_id"]
    fresh = _member(rig, turn_id, second_gate)
    assert fresh["control_state"] == "pending"

    assert _pass(pump) == 0 and len(steers.calls) == 1
    deduped = _member(rig, turn_id, second_gate)
    assert deduped["control_state"] == "steered"
    assert deduped["control_request_id"] == _member(rig, turn_id, first_gate)["control_request_id"]
    assert deduped["control_attempts"] == 0
    assert deduped["control_changed_at"] > fresh["control_changed_at"]


def test_a_run_being_steered_for_an_earlier_generation_waits_for_it(rig):
    turn_id = _turn(rig)
    job_id = _open(rig)
    pump, steers = _pump(rig)
    first_gate = rig.row(job_id)["quiesce_gate_id"]
    in_flight = pump.claim(_member(rig, turn_id, first_gate)["id"], rig.bot)
    rig.sql("UPDATE system_gates SET expires_at = now() - interval '1 second' WHERE id=:g", g=first_gate)
    assert rig.driver.step(job_id).outcome == "holding"
    second_gate = rig.row(job_id)["quiesce_gate_id"]
    fresh = _member(rig, turn_id, second_gate)

    assert _pass(pump) == 0 and steers.calls == []
    waiting = _member(rig, turn_id, second_gate)
    assert waiting["control_state"] == "pending" and waiting["backing_off"]
    assert waiting["control_changed_at"] == fresh["control_changed_at"]

    assert pump.settle(in_flight, SteerOutcome(STEERED))  # the old claim's result still lands
    _due_now(rig, turn_id)
    assert _pass(pump) == 0 and steers.calls == []
    assert _member(rig, turn_id, second_gate)["control_state"] == "steered"


def test_a_lapsed_claim_of_an_earlier_generation_does_not_hold_the_steer(rig):
    """The pump died mid-RPC on gen1 and the job rolled: gen1's row is never reclaimed (Loopy R1)."""
    turn_id = _turn(rig)
    job_id = _open(rig)
    pump, steers = _pump(rig)
    first_gate = rig.row(job_id)["quiesce_gate_id"]
    dead = pump.claim(_member(rig, turn_id, first_gate)["id"], rig.bot)
    rig.sql("UPDATE gate_members SET control_lease_expires_at = now() - interval '1 second' WHERE id=:id",
            id=dead.member_id)
    rig.sql("UPDATE system_gates SET expires_at = now() - interval '1 second' WHERE id=:g", g=first_gate)
    assert rig.driver.step(job_id).outcome == "holding"
    second_gate = rig.row(job_id)["quiesce_gate_id"]

    assert _pass(pump) == 1
    [(claim, _)] = steers.calls
    assert claim.request_id == dead.request_id and claim.member_id != dead.member_id
    assert _member(rig, turn_id, second_gate)["control_state"] == "steered"
    assert _member(rig, turn_id, first_gate)["control_state"] == "steering"  # not live: left as it was


def test_a_lapsed_claim_whose_run_another_claim_holds_goes_back_to_pending(rig):
    """Two live claims on one run, this one lapsed (e.g. two jobs on one bot) (Loopy R2).

    The gen1 claim stands in for the other job's live claim; the gen2 row is
    put in the lapsed-claim state directly.
    """
    turn_id = _turn(rig)
    job_id = _open(rig)
    pump, steers = _pump(rig)
    first_gate = rig.row(job_id)["quiesce_gate_id"]
    live = pump.claim(_member(rig, turn_id, first_gate)["id"], rig.bot)
    rig.sql("UPDATE system_gates SET expires_at = now() - interval '1 second' WHERE id=:g", g=first_gate)
    assert rig.driver.step(job_id).outcome == "holding"
    second_gate = rig.row(job_id)["quiesce_gate_id"]
    rig.sql("""
        UPDATE gate_members SET control_state = 'steering', control_request_id = :rid,
            control_claim_token = 'lapsed', control_attempts = 1,
            control_lease_expires_at = now() - interval '1 second',
            control_changed_at = now() - interval '1 minute'
        WHERE turn_id = :t AND gate_id = :g
    """, rid=live.request_id, t=turn_id, g=second_gate)
    lapsed = _member(rig, turn_id, second_gate)

    assert pump.claim(lapsed["id"], rig.bot) is None and steers.calls == []
    back = _member(rig, turn_id, second_gate)
    assert back["control_state"] == "pending" and back["backing_off"]
    assert back["control_claim_token"] is None and back["control_lease_expires_at"] is None
    assert back["control_attempts"] == 1
    assert back["control_changed_at"] > lapsed["control_changed_at"]


def test_only_the_pinned_run_is_addressed_never_the_bots_current_one(rig):
    """A carried member whose bot moved backend and started a new run (Snark)."""
    turn_id = _turn(rig, session="sess-pinned")
    _open(rig)
    pinned = f"req-{turn_id}"
    # The member's turn row was rebound to another request on another backend, the
    # bot's profile moved to codex, and a new turn is running there.
    rig.sql("UPDATE turn_logs SET agent_request_id='req-new', agent_backend='codex',"
            " agent_session_key='sess-new' WHERE id=:t", t=turn_id)
    rig.sql("UPDATE bot_profiles SET agent_backend='codex' WHERE slug=:b", b=rig.bot)
    other = _turn(rig, backend="codex", session="sess-other")
    pump, steers = _pump(rig)
    assert _pass(pump) == 1
    [(claim, target)] = steers.calls
    assert claim.grant.turn_id == turn_id and claim.request_id == control_request_id(turn_id, pinned)
    assert target == ControlTarget(pinned, "claude-code", "sess-pinned")
    assert rig.sql("SELECT count(*) FROM gate_members WHERE turn_id=:t", t=other).scalar() == 0


def test_a_member_bound_after_capture_takes_the_binding_of_its_pinned_request(rig):
    """Captured before bind (NULL backend): the turn row's binding for that same request is used."""
    turn_id = _turn(rig, backend=None, session=None)
    _open(rig)
    rig.sql("UPDATE turn_logs SET agent_backend='claude-code', agent_session_key='sess-bound' WHERE id=:t",
            t=turn_id)
    pump, steers = _pump(rig)
    assert _pass(pump) == 1
    assert steers.calls[0][1] == ControlTarget(f"req-{turn_id}", "claude-code", "sess-bound")


def test_a_steer_that_raises_is_retried(rig):
    turn_id = _turn(rig)
    _open(rig)

    async def boom(claim):
        raise RuntimeError("redis down")

    pump = QuiesceControlPump(rig.engine, boom)
    assert _pass(pump) == 1
    member = _member(rig, turn_id)
    assert (member["control_state"], member["control_error"]) == ("pending", "redis down")
    assert pump.has_open_work()


def test_a_job_that_left_quiescing_steers_no_more(rig):
    turn_id = _turn(rig)
    job_id = _open(rig)
    rig.sql("UPDATE ops_jobs SET state='accepted' WHERE id=:id", id=job_id)
    pump, steers = _pump(rig)
    assert pump.due() == [] and _pass(pump) == 0 and steers.calls == []
    assert _member(rig, turn_id)["control_state"] == "pending" and not pump.has_open_work()
