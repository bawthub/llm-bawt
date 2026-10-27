"""PostgreSQL tests for the TASK-1431 S2 control lane.

A real quiescing job opens a real gate over this test's own bot (the ``rig`` of
test_ops_quiesce_pg). A control row is enqueued through ``enqueue_control`` and
linked to the member the way the S3 producer will; the claim, the grant and the
steer-failure settlement then run against the real store.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest

from llm_bawt.inter_bot_control_lane import (
    CONTROL_LANE,
    GRANT_REFUSED,
    ControlSteerFailure,
    DeliveryControlGrant,
    enqueue_control,
)
from llm_bawt.ops.quiesce import release_job_gates
from llm_bawt.service.inter_bot_dispatcher_steer import SteerGeneration
from llm_bawt.service.system_gate_admission import (
    ControlGrantRefused,
    ControlTarget,
    SteerGateRefused,
    check_steer,
)
from llm_bawt.service.turn_admission import lock_target
from llm_bawt.steer_capabilities import ALL_STEER_CAPS
from tests.test_ops_quiesce_pg import rig  # noqa: F401

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]

USER = "nick"  # the rig's turn rows are owned by this fixture user


def _turn(rig, *, backend="claude-code", session="sess-held", status="streaming", caps=None):
    """``caps`` None is a row bound before TASK-1243 recorded steer capabilities."""
    turn_id = f"turn-{uuid.uuid4().hex}"
    rig.sql("""
        INSERT INTO turn_logs (id, created_at, path, stream, bot_id, user_id, status, user_prompt,
                               response_text, ended_at, agent_request_id, agent_backend, agent_session_key,
                               agent_steer_caps)
        VALUES (:id, now(), '/test', true, :bot, :user, :status, '', '', NULL, :req, :backend, :session,
                :caps)
    """, id=turn_id, bot=rig.bot, user=USER, status=status, backend=backend, session=session,
            req=f"req-{turn_id}", caps=caps)
    return turn_id


def _open(rig):
    job_id = rig.job()
    assert rig.driver.step(job_id).outcome == "opened"
    return job_id


def _pin(turn_id, *, backend="claude-code", session="sess-held", **caps):
    """A pin without ``steer_caps`` is one stored before TASK-1243."""
    return {"turn_id": turn_id, "agent_request_id": f"req-{turn_id}", "backend": backend,
            "session_key": session, **caps}


def _run_key(turn_id):
    return f"steer_quiesce_{turn_id}_req-{turn_id}"


def _control(rig, turn_id, *, key=None, user=USER, **pin):
    with rig.deliveries.engine.begin() as conn:
        lock_target(conn, rig.bot)
        return enqueue_control(
            rig.deliveries, conn, target_bot_id=rig.bot, target_run=_pin(turn_id, **pin),
            run_key=_run_key(turn_id), idempotency_key=key or _run_key(turn_id),
            message="Please stop at the next safe point.", user=user,
        )


def _link(rig, job_id, turn_id, delivery_id):
    gate_id = rig.row(job_id)["quiesce_gate_id"]
    assert rig.sql("""
        UPDATE gate_members SET control_delivery_id=:d, control_state='steering', control_changed_at=now()
        WHERE turn_id=:t AND gate_id=:g
    """, d=delivery_id, t=turn_id, g=gate_id).rowcount == 1


def _normal(rig, message="callback"):
    record, _ = rig.deliveries.enqueue(
        sender_bot_id=f"{rig.bot}-sender", target_bot_id=rig.bot, message=message,
        payload={"messages": [{"role": "user", "content": message}], "bot_id": rig.bot,
                 "user": USER, "stream": True}, on_gate="hold",
    )
    return record


def _claim(rig):
    return rig.deliveries.claim_next(rig.bot, claim_owner="test-control", steer_capable=True)


def _row(rig, delivery_id):
    return rig.sql("SELECT * FROM inter_bot_deliveries WHERE id=:id", id=delivery_id).mappings().one()


def _turn_count(rig):
    return rig.sql("SELECT count(*) FROM turn_logs WHERE bot_id=:b", b=rig.bot).scalar()


def _ready(rig, delivery_id):
    rig.sql("UPDATE inter_bot_deliveries SET available_at = now() - interval '1 second' WHERE id=:id",
            id=delivery_id)


def test_a_held_head_is_passed_by_the_control_row_and_normal_order_survives(rig):
    """(a) The control row steers under the hold; normal rows wait, then go in order."""
    turn_id = _turn(rig)
    job_id = _open(rig)
    first, second = _normal(rig, "first"), _normal(rig, "second")
    control, duplicate = _control(rig, turn_id)
    assert not duplicate and control.lane == CONTROL_LANE
    _link(rig, job_id, turn_id, control.id)

    claimed = _claim(rig)
    assert claimed.id == control.id and claimed.status == "STEERING"
    assert claimed.delivery_mode == "steer" and claimed.target_turn_id == turn_id
    grant = DeliveryControlGrant.of(claimed)
    assert grant == DeliveryControlGrant(rig.bot, turn_id, control.id, claimed.claim_token)
    assert check_steer(rig.engine, rig.bot, turn_id=turn_id, grant=grant) == ControlTarget(
        f"req-{turn_id}", "claude-code", "sess-held", ALL_STEER_CAPS)  # a legacy row decodes
    # A normal steer is still refused, and the held head never moved.
    with pytest.raises(SteerGateRefused):
        check_steer(rig.engine, rig.bot, turn_id=turn_id)
    assert _claim(rig) is None
    assert [_row(rig, d)["status"] for d in (first.id, second.id)] == ["QUEUED", "QUEUED"]
    # A STEERING control row does not head or fence the normal FIFO.
    with rig.engine.begin() as conn:
        lock_target(conn, rig.bot)
        from llm_bawt.inter_bot_preclaim import load_claim_head
        assert load_claim_head(conn, rig.bot, rig.deliveries.config)["id"] == first.id

    assert rig.deliveries.mark_delivered(control.id, claimed.claim_token, response_model=None,
                                         response_chars=0)
    with rig.engine.begin() as conn:
        release_job_gates(conn, job_id)
    after = _claim(rig)
    assert after.id == first.id and after.status == "STEERING"
    assert _row(rig, second.id)["status"] == "QUEUED"
    assert _turn_count(rig) == 1


def test_the_same_run_key_makes_one_row_and_a_rebuilt_member_inherits_it(rig):
    """(b) One delivery per run; the next generation's member authorizes it."""
    turn_id = _turn(rig)
    job_id = _open(rig)
    control, _ = _control(rig, turn_id)
    again, duplicate = _control(rig, turn_id)
    assert duplicate and again.id == control.id
    assert rig.sql("SELECT count(*) FROM inter_bot_deliveries WHERE target_bot_id=:b AND lane='control'",
                   b=rig.bot).scalar() == 1
    _link(rig, job_id, turn_id, control.id)
    first_gate = rig.row(job_id)["quiesce_gate_id"]
    rig.sql("UPDATE system_gates SET expires_at = now() - interval '1 second' WHERE id=:g", g=first_gate)
    assert rig.driver.step(job_id).outcome == "holding"
    assert rig.row(job_id)["quiesce_gate_id"] != first_gate
    _link(rig, job_id, turn_id, control.id)

    claimed = _claim(rig)
    assert claimed.id == control.id and claimed.status == "STEERING"
    assert check_steer(rig.engine, rig.bot, turn_id=turn_id, grant=DeliveryControlGrant.of(claimed))


@pytest.mark.parametrize("end", ["ended", "gone"])
def test_an_ended_pinned_run_is_not_needed_and_never_reserves(rig, end):
    """(c) Settled at claim with no reservation; the drain continues."""
    turn_id = _turn(rig)
    job_id = _open(rig)
    control, _ = _control(rig, turn_id)
    _link(rig, job_id, turn_id, control.id)
    if end == "ended":
        rig.end(turn_id, exited=False)
    else:
        rig.sql("DELETE FROM turn_logs WHERE id=:id", id=turn_id)
    before = _turn_count(rig)
    settled = _claim(rig)
    assert settled.id == control.id
    assert (settled.status, settled.delivery_mode, settled.last_error) == (
        "CANCELLED", "skipped", "not_needed:run_ended")
    assert settled.attempt_count == 0 and settled.claim_token is None
    assert _turn_count(rig) == before
    assert rig.sql("SELECT count(*) FROM turn_logs WHERE id=:t", t=control.turn_id).scalar() == 0


def test_a_row_no_live_member_links_is_revoked(rig):
    """The linking member's gate lapsed (a release cancels the row itself: S3 reference-counted release)."""
    turn_id = _turn(rig)
    job_id = _open(rig)
    control, _ = _control(rig, turn_id)
    _link(rig, job_id, turn_id, control.id)
    rig.sql("UPDATE system_gates SET expires_at = now() - interval '1 second' WHERE id=:g",
            g=rig.row(job_id)["quiesce_gate_id"])
    settled = _claim(rig)
    assert (settled.status, settled.last_error) == ("CANCELLED", "not_needed:control_revoked")


def test_a_release_cancels_the_queued_row_it_alone_linked(rig):
    turn_id = _turn(rig)
    job_id = _open(rig)
    control, _ = _control(rig, turn_id)
    _link(rig, job_id, turn_id, control.id)
    with rig.engine.begin() as conn:
        release_job_gates(conn, job_id)
    row = _row(rig, control.id)
    assert (row["status"], row["delivery_mode"], row["last_error"]) == (
        "CANCELLED", "skipped", "not_needed:control_released")
    assert _claim(rig) is None


def test_an_unlinked_row_is_revoked_even_without_a_gate(rig):
    turn_id = _turn(rig)
    control, _ = _control(rig, turn_id)
    settled = _claim(rig)
    assert settled.id == control.id and settled.last_error == "not_needed:control_revoked"


def test_a_pin_on_another_backend_is_unsupported(rig):
    turn_id = _turn(rig, backend="codex")
    job_id = _open(rig)
    control, _ = _control(rig, turn_id, backend="codex")
    _link(rig, job_id, turn_id, control.id)
    settled = _claim(rig)
    assert (settled.status, settled.last_error) == ("CANCELLED", "not_needed:unsupported_backend")


@pytest.mark.parametrize("backend", ["codex", "claude-code"])
def test_a_pin_whose_run_steers_is_claimed_on_any_backend(rig, backend):
    """TASK-1243: the pinned run's own caps decide, not its backend's name."""
    turn_id = _turn(rig, backend=backend, caps="next")
    job_id = _open(rig)
    control, _ = _control(rig, turn_id, backend=backend, steer_caps="next")
    _link(rig, job_id, turn_id, control.id)
    claimed = _claim(rig)
    assert (claimed.id, claimed.status) == (control.id, "STEERING")
    assert check_steer(rig.engine, rig.bot, turn_id=turn_id, grant=DeliveryControlGrant.of(claimed)) == (
        ControlTarget(f"req-{turn_id}", backend, "sess-held", frozenset({"next"})))


def test_a_claude_pin_whose_run_cannot_steer_is_unsupported(rig):
    turn_id = _turn(rig, caps="")
    job_id = _open(rig)
    control, _ = _control(rig, turn_id, steer_caps="")
    _link(rig, job_id, turn_id, control.id)
    settled = _claim(rig)
    assert (settled.status, settled.last_error) == ("CANCELLED", "not_needed:unsupported_backend")


def test_a_pin_whose_caps_differ_from_the_members_run_is_unconfirmed(rig):
    turn_id = _turn(rig, caps="next")
    job_id = _open(rig)
    control, _ = _control(rig, turn_id, steer_caps="")
    _link(rig, job_id, turn_id, control.id)
    settled = _claim(rig)
    assert (settled.status, settled.last_error) == ("FAILED", "unconfirmed:pinned_run_unaddressable")


def test_a_payload_user_that_is_not_the_turns_is_unconfirmed(rig):
    turn_id = _turn(rig)
    job_id = _open(rig)
    control, _ = _control(rig, turn_id, user="someone-else")
    _link(rig, job_id, turn_id, control.id)
    settled = _claim(rig)
    assert (settled.status, settled.last_error) == ("FAILED", "unconfirmed:user_mismatch")


def test_a_pin_that_differs_from_the_members_binding_is_unconfirmed(rig):
    turn_id = _turn(rig)
    job_id = _open(rig)
    control, _ = _control(rig, turn_id, session="sess-other")
    _link(rig, job_id, turn_id, control.id)
    settled = _claim(rig)
    assert (settled.status, settled.last_error) == ("FAILED", "unconfirmed:pinned_run_unaddressable")


@pytest.mark.parametrize("case, expected", [
    ("user", ("FAILED", "unconfirmed:user_mismatch")),
    ("pin", ("FAILED", "unconfirmed:pinned_run_unaddressable")),
    ("unlinked", ("CANCELLED", "not_needed:control_revoked")),
])
def test_the_drain_emits_a_settled_control_row_and_never_dispatches_it(rig, monkeypatch, case, expected):
    """A control row the claim settles terminal is emitted and the drain moves
    on; it must never reach ``_dispatch`` as a fallback turn."""
    from unittest.mock import AsyncMock

    import llm_bawt.bots as bots

    turn_id = _turn(rig)
    job_id = _open(rig)
    control, _ = _control(rig, turn_id, **({"user": "someone-else"} if case == "user" else
                                           {"session": "sess-other"} if case == "pin" else {}))
    if case != "unlinked":
        _link(rig, job_id, turn_id, control.id)
    monkeypatch.setattr(bots, "BotManager", lambda _config: SimpleNamespace(
        get_bot=lambda _bot: SimpleNamespace(agent_backend="claude-code")))
    monkeypatch.setattr("llm_bawt.service.maintenance_queue.MaintenanceQueue.publish_ready", AsyncMock())
    dispatcher, route, emitted = _dispatcher(rig, monkeypatch)
    dispatcher.claim_owner = "test-drain"
    dispatched = []

    async def _dispatch(record):
        dispatched.append(record)

    dispatcher._dispatch = _dispatch
    before = _turn_count(rig)
    assert asyncio.run(dispatcher._drain_target(control.target_bot_id)) is True
    assert [(r.id, r.status, r.last_error) for r in emitted] == [(control.id, *expected)]
    assert dispatched == [] and route.calls == [] and _turn_count(rig) == before


@pytest.mark.parametrize("forge", ["token", "bot", "turn", "delivery", "lease", "released", "settled",
                                   "unlinked"])
def test_a_delivery_grant_naming_no_live_claim_is_refused(rig, forge):
    turn_id = _turn(rig)
    job_id = _open(rig)
    control, _ = _control(rig, turn_id)
    _link(rig, job_id, turn_id, control.id)
    grant = DeliveryControlGrant.of(_claim(rig))
    bot = rig.bot
    if forge == "token":
        grant = DeliveryControlGrant(grant.bot_id, grant.turn_id, grant.delivery_id, "claim-forged")
    elif forge == "bot":
        bot = "someone-else"
        grant = DeliveryControlGrant(bot, grant.turn_id, grant.delivery_id, grant.claim_token)
    elif forge == "turn":
        grant = DeliveryControlGrant(grant.bot_id, "turn-other", grant.delivery_id, grant.claim_token)
    elif forge == "delivery":
        other, _ = _control(rig, turn_id, key=f"{_run_key(turn_id)}:r1")
        grant = DeliveryControlGrant(grant.bot_id, grant.turn_id, other.id, grant.claim_token)
    elif forge == "lease":
        rig.sql("UPDATE inter_bot_deliveries SET lease_expires_at = now() - interval '1 second'"
                " WHERE id=:id", id=control.id)
    elif forge == "released":
        with rig.engine.begin() as conn:
            release_job_gates(conn, job_id)
    elif forge == "settled":
        assert rig.deliveries.mark_delivered(control.id, grant.claim_token, response_model=None,
                                             response_chars=0)
    elif forge == "unlinked":
        rig.sql("UPDATE gate_members SET control_delivery_id=NULL WHERE turn_id=:t", t=turn_id)
    with pytest.raises(ControlGrantRefused):
        check_steer(rig.engine, bot, turn_id=grant.turn_id, grant=grant)


def test_a_normal_row_carries_no_grant(rig):
    record = _normal(rig)
    assert record.lane == "normal" and DeliveryControlGrant.of(record) is None


def _claimed(rig):
    turn_id = _turn(rig)
    job_id = _open(rig)
    control, _ = _control(rig, turn_id)
    _link(rig, job_id, turn_id, control.id)
    return turn_id, job_id, _claim(rig)


def test_a_definitive_rejection_of_an_open_run_spends_an_attempt_and_advances(rig):
    turn_id, _, claimed = _claimed(rig)
    queued = ControlSteerFailure.settle(rig.deliveries, claimed, "no_active_run")
    assert queued.status == "QUEUED" and queued.attempt_count == 1
    assert SteerGeneration.of(queued).generation == 1
    assert queued.available_at > claimed.dispatch_started_at
    _ready(rig, claimed.id)
    again = _claim(rig)
    assert again.id == claimed.id and again.attempt_count == 2 and again.target_turn_id == turn_id
    assert SteerGeneration.of(again).request_id.endswith("_g1")


def test_an_accepted_ask_is_retried_under_the_same_generation(rig):
    _, _, claimed = _claimed(rig)
    assert rig.deliveries.mark_transport_accepted(claimed.id, claimed.claim_token)
    queued = ControlSteerFailure.settle(rig.deliveries, claimed, "no_active_run")
    assert queued.status == "QUEUED" and SteerGeneration.of(queued).generation == 0


def test_attempts_run_out_to_unconfirmed(rig):
    _, _, claimed = _claimed(rig)
    rig.sql("UPDATE inter_bot_deliveries SET attempt_count=max_attempts WHERE id=:id", id=claimed.id)
    failed = ControlSteerFailure.settle(rig.deliveries, claimed, "no_active_run")
    assert (failed.status, failed.last_error, failed.claim_token) == (
        "FAILED", "unconfirmed:no_active_run", None)


@pytest.mark.parametrize("error", ["steer_unsupported_transport", "steer_priority_unsupported"])
def test_a_capability_refusal_settles_at_once_without_retries(rig, error):
    """TASK-1243: the run cannot take this ask and never will; no budget is spent on it."""
    _, _, claimed = _claimed(rig)
    settled = ControlSteerFailure.settle(rig.deliveries, claimed, error)
    assert (settled.status, settled.last_error, settled.claim_token) == (
        "CANCELLED", f"not_needed:{error}", None)
    assert _row(rig, claimed.id)["status"] == "CANCELLED"


@pytest.mark.parametrize("error", ["steer_unsupported_transport", "steer_priority_unsupported"])
def test_a_capability_refusal_after_acceptance_retries_the_same_generation(rig, error):
    """TASK-1243: an accepted ask may have landed; only an unaccepted refusal settles."""
    _, _, claimed = _claimed(rig)
    assert rig.deliveries.mark_transport_accepted(claimed.id, claimed.claim_token)
    queued = ControlSteerFailure.settle(rig.deliveries, claimed, error)
    assert (queued.status, queued.attempt_count, queued.last_error) == ("QUEUED", 1, error)
    assert SteerGeneration.of(queued).generation == 0


def test_a_failure_after_the_run_ended_is_not_needed(rig):
    turn_id, _, claimed = _claimed(rig)
    rig.end(turn_id, exited=True)
    settled = ControlSteerFailure.settle(rig.deliveries, claimed, "no_active_run")
    assert (settled.status, settled.delivery_mode, settled.last_error) == (
        "CANCELLED", "skipped", "not_needed:run_ended")


def test_a_refused_grant_requeues_without_spending_and_the_next_claim_revokes(rig):
    _, job_id, claimed = _claimed(rig)
    with rig.engine.begin() as conn:
        release_job_gates(conn, job_id)
    queued = ControlSteerFailure.settle(rig.deliveries, claimed, GRANT_REFUSED)
    assert queued.status == "QUEUED" and queued.attempt_count == 0
    _ready(rig, claimed.id)
    assert _claim(rig).last_error == "not_needed:control_revoked"


def test_a_stale_claim_cannot_settle(rig):
    _, _, claimed = _claimed(rig)
    assert ControlSteerFailure.settle(rig.deliveries, claimed, "no_active_run")
    assert ControlSteerFailure.settle(rig.deliveries, claimed, "no_active_run") is None


class _Route:
    """``steer_active_turn``'s control seam: admit the grant as the route does, then answer."""

    def __init__(self, engine, *errors):
        self.engine = engine
        self.errors = list(errors)
        self.calls = []

    async def __call__(self, service, request, *, steer_request_id, author, wire_suffix=None, control=None):
        from fastapi import HTTPException

        self.calls.append((request, steer_request_id, author, control))
        check_steer(self.engine, request.bot_id, turn_id=request.turn_id, grant=control)
        error = self.errors.pop(0) if self.errors else None
        if error:
            raise HTTPException(status_code=409, detail=error)
        return SimpleNamespace(ok=True, persisted=True, detail=None)


def _dispatcher(rig, monkeypatch, *errors):
    from llm_bawt.service.inter_bot_dispatcher import InterBotDeliveryDispatcher
    from llm_bawt.service.turn_logs import TurnLogStore

    route = _Route(rig.engine, *errors)
    monkeypatch.setattr("llm_bawt.service.routes.chat.steer_active_turn", route)
    dispatcher = InterBotDeliveryDispatcher.__new__(InterBotDeliveryDispatcher)
    dispatcher.service = SimpleNamespace(config=rig.deliveries.config,
                                         _turn_log_store=TurnLogStore(rig.deliveries.config))
    dispatcher.store = rig.deliveries
    emitted = []

    async def _emit(record):
        emitted.append(record)

    dispatcher._emit = _emit
    return dispatcher, route, emitted


def test_the_dispatcher_steers_a_control_row_under_its_grant_from_system(rig, monkeypatch):
    turn_id, _, claimed = _claimed(rig)
    dispatcher, route, emitted = _dispatcher(rig, monkeypatch)
    asyncio.run(dispatcher._steer(claimed))
    [(request, rpc_id, author, control)] = route.calls
    assert control == DeliveryControlGrant.of(claimed) and request.turn_id == turn_id
    assert rpc_id == SteerGeneration.of(claimed).request_id and author.entity_id == "system"
    assert request.priority == "next" and request.user_id == USER
    assert request.message_id == claimed.user_message_id
    assert emitted[-1].status == "DELIVERED" and _row(rig, claimed.id)["status"] == "DELIVERED"


def test_a_rejected_control_steer_requeues_and_never_falls_back(rig, monkeypatch):
    turn_id, _, claimed = _claimed(rig)
    dispatcher, _, _ = _dispatcher(rig, monkeypatch, "no_active_run")
    before = _turn_count(rig)
    asyncio.run(dispatcher._steer(claimed))
    row = _row(rig, claimed.id)
    assert (row["status"], row["delivery_mode"], row["attempt_count"]) == ("QUEUED", "steer", 1)
    assert row["metadata_json"]["steer_generation"] == 1
    assert row["metadata_json"]["control"]["target_run"] == _pin(turn_id)  # never rewritten
    assert _turn_count(rig) == before


def test_a_control_steer_after_release_is_refused_then_revoked(rig, monkeypatch):
    _, job_id, claimed = _claimed(rig)
    with rig.engine.begin() as conn:
        release_job_gates(conn, job_id)
    dispatcher, _, _ = _dispatcher(rig, monkeypatch)
    asyncio.run(dispatcher._steer(claimed))
    row = _row(rig, claimed.id)
    assert (row["status"], row["attempt_count"], row["last_error"]) == ("QUEUED", 0, GRANT_REFUSED)
    _ready(rig, claimed.id)
    assert _claim(rig).last_error == "not_needed:control_revoked"
