"""PostgreSQL tests for the TASK-1431 S3 quiesce control producer.

A real quiescing job opens a real gate over this test's own bot (the ``rig``
of test_ops_quiesce_pg). The producer links each member that needs asking to
its run's control-lane delivery; the real claim and the dispatcher's control
steer (through the recording route of test_inter_bot_control_lane_pg) then
ask the run. "One ask" is counted at that route.
"""
from __future__ import annotations

import asyncio
import uuid

import pytest

from llm_bawt.bot_identities import SYSTEM_BOT_ID
from llm_bawt.message_authorship import AuthorReference
from llm_bawt.ops.quiesce import release_job_gates
from llm_bawt.service.quiesce_control import (
    ENQUEUED,
    LINKED,
    UNPINNED,
    QuiesceControlProducer,
    control_request_id,
    successor_key,
)
from tests.test_inter_bot_control_lane_pg import _claim, _dispatcher, _row
from tests.test_ops_quiesce_pg import rig  # noqa: F401

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]

USER = "nick"  # the rig's fixture user
MESSAGE = "Please stop at the next safe point."


@pytest.fixture(autouse=True)
def _message(monkeypatch):
    monkeypatch.setattr("llm_bawt.ops.quiesce_prompts.resolve", lambda key, bot: MESSAGE)


def _turn(rig, *, backend="claude-code", session="sess-held", request="auto", user=USER):
    turn_id = f"turn-{uuid.uuid4().hex}"
    rig.sql("""
        INSERT INTO turn_logs (id, created_at, path, stream, bot_id, user_id, status, user_prompt,
                               response_text, ended_at, agent_request_id, agent_backend, agent_session_key)
        VALUES (:id, now(), '/test', true, :bot, :user, 'streaming', '', '', NULL, :req, :backend, :session)
    """, id=turn_id, bot=rig.bot, user=user, backend=backend, session=session,
            req=f"req-{turn_id}" if request == "auto" else request)
    return turn_id


def _open(rig):
    job_id = rig.job()
    assert rig.driver.step(job_id).outcome == "opened"
    return job_id


@pytest.fixture
def second(rig):
    """A second quiesce operation over the rig's bot: its own concurrency slot, so two jobs hold at once."""
    import json

    from tests.test_ops_service import _op_data

    slug = f"{rig.slug}-b"
    rig.store.create_operation(_op_data(slug=slug, quiesce_enabled=True, quiesce_timeout_seconds=300,
                                        quiesce_scope_json=json.dumps({"bots": [rig.bot]}), max_concurrent=1))

    def open_job():
        job_id = rig.service.dispatch_job(operation_slug=slug, args={"target": "a"},
                                          idempotency_key=uuid.uuid4().hex)["id"]
        assert rig.driver.step(job_id).outcome == "opened"
        return job_id

    yield open_job
    owners = [r[0] for r in rig.sql("SELECT id FROM ops_jobs WHERE operation_slug=:s", s=slug).all()]
    rig.sql("DELETE FROM gate_members WHERE gate_id IN (SELECT id FROM system_gates WHERE owner_id = ANY(:o))",
            o=owners)
    rig.sql("DELETE FROM system_gates WHERE owner_id = ANY(:o)", o=owners)
    rig.sql("DELETE FROM ops_jobs WHERE operation_slug=:s", s=slug)


def _gate(rig, job_id):
    return rig.row(job_id)["quiesce_gate_id"]


def _member(rig, turn_id, gate_id=None):
    clause = " AND gate_id=:g" if gate_id is not None else ""
    return rig.sql(f"SELECT * FROM gate_members WHERE turn_id=:t{clause} ORDER BY id DESC LIMIT 1",
                   t=turn_id, g=gate_id).mappings().one()


def _controls(rig):
    return rig.sql("SELECT * FROM inter_bot_deliveries WHERE target_bot_id=:b AND lane='control'"
                   " ORDER BY created_at, id", b=rig.bot).mappings().all()


def _producer(rig):
    return QuiesceControlProducer(rig.deliveries)


def _rebuild(rig, job_id):
    """Lapse the job's gate; the next driver step activates the next generation."""
    first = _gate(rig, job_id)
    rig.sql("UPDATE system_gates SET expires_at = now() - interval '1 second' WHERE id=:g", g=first)
    assert rig.driver.step(job_id).outcome == "holding"
    second = _gate(rig, job_id)
    assert second != first
    return first, second


def _steer(rig, monkeypatch):
    """Claim the bot's next row and run the dispatcher's control steer; the route's calls."""
    dispatcher, route, _ = _dispatcher(rig, monkeypatch)
    claimed = _claim(rig)
    if claimed is not None and claimed.status == "STEERING":
        asyncio.run(dispatcher._steer(claimed))
    return claimed, route.calls


def _run_key(turn_id, request=None):
    return control_request_id(turn_id, request or f"req-{turn_id}")


# --- member-local decisions --------------------------------------------------


def test_a_held_running_turn_gets_one_control_row_and_is_asked_once(rig, monkeypatch):
    turn_id = _turn(rig)
    _open(rig)
    before = _member(rig, turn_id)
    producer = _producer(rig)
    result = producer.run_pass()
    assert (result.linked, result.enqueued, result.retry_in) == (1, 1, None)
    [control] = _controls(rig)
    run_key = _run_key(turn_id)
    assert control["idempotency_key"] == run_key and control["sender_bot_id"] == SYSTEM_BOT_ID
    assert control["metadata_json"]["control"] == {
        "run_key": run_key,
        "target_run": {"turn_id": turn_id, "agent_request_id": f"req-{turn_id}",
                       "backend": "claude-code", "session_key": "sess-held"}}
    assert control["payload_json"]["user"] == USER and control["message"] == MESSAGE
    member = _member(rig, turn_id)
    assert (member["control_state"], member["control_delivery_id"]) == ("steering", control["id"])
    assert member["control_changed_at"] > before["control_changed_at"]

    assert producer.run_pass().linked == 0  # linked to a live row: nothing to do
    claimed, calls = _steer(rig, monkeypatch)
    assert claimed.id == control["id"] and len(calls) == 1
    assert _row(rig, control["id"])["status"] == "DELIVERED"
    assert producer.run_pass().linked == 0 and len(_controls(rig)) == 1


def test_an_ended_member_is_not_needed_and_gets_no_row(rig):
    turn_id = _turn(rig)
    _open(rig)
    rig.end(turn_id, exited=False)
    assert _producer(rig).run_pass().linked == 0 and _controls(rig) == []
    member = _member(rig, turn_id)
    assert (member["control_state"], member["control_error"]) == ("not_needed", "turn_not_running")


@pytest.mark.parametrize("turn, expected", [
    ({"backend": "codex"}, ("unsupported", "backend:codex")),
    ({"session": None}, ("unconfirmed", "pinned_run_unaddressable")),
    ({"user": None}, ("unconfirmed", "turn_user_missing")),
], ids=["other-backend", "no-session", "no-user"])
def test_a_run_that_cannot_be_asked_is_settled_without_a_row(rig, turn, expected):
    turn_id = _turn(rig, **turn)
    _open(rig)
    assert _producer(rig).run_pass().linked == 0 and _controls(rig) == []
    member = _member(rig, turn_id)
    assert (member["control_state"], member["control_error"], member["control_delivery_id"]) == (
        *expected, None)


def test_the_callers_own_turn_is_never_asked(rig):
    turn_id = _turn(rig)
    job_id = _open(rig)
    rig.sql("UPDATE ops_jobs SET caller_turn_id=:t WHERE id=:id", t=turn_id, id=job_id)
    assert _producer(rig).run_pass().linked == 0 and _controls(rig) == []
    assert _member(rig, turn_id)["control_state"] == "self"


def test_an_unpinned_member_writes_nothing_and_asks_for_a_recheck(rig):
    turn_id = _turn(rig, request=None)
    _open(rig)
    before = _member(rig, turn_id)
    producer = _producer(rig)
    assert producer.produce(before["id"], rig.bot, MESSAGE) == UNPINNED
    result = producer.run_pass()
    assert (result.linked, result.retry_in) == (0, QuiesceControlProducer.UNPINNED_RECHECK_SECONDS)
    after = _member(rig, turn_id)
    assert (after["control_state"], after["control_changed_at"]) == ("pending", before["control_changed_at"])
    assert _controls(rig) == []

    rig.sql("UPDATE turn_logs SET agent_request_id='req-late' WHERE id=:t", t=turn_id)
    rig.sql("UPDATE gate_members SET agent_request_id='req-late' WHERE turn_id=:t", t=turn_id)
    assert producer.run_pass().enqueued == 1
    assert _controls(rig)[0]["idempotency_key"] == _run_key(turn_id, "req-late")


def test_a_job_that_left_quiescing_produces_nothing(rig):
    turn_id = _turn(rig)
    job_id = _open(rig)
    rig.sql("UPDATE ops_jobs SET state='accepted' WHERE id=:id", id=job_id)
    producer = _producer(rig)
    assert producer.due() == [] and producer.run_pass().linked == 0 and _controls(rig) == []
    assert _member(rig, turn_id)["control_state"] == "pending"


def test_only_the_pinned_run_is_addressed_never_the_bots_current_one(rig, monkeypatch):
    """The turn row was rebound to another request on another backend (Snark)."""
    turn_id = _turn(rig, session="sess-pinned")
    _open(rig)
    rig.sql("UPDATE turn_logs SET agent_request_id='req-new', agent_backend='codex',"
            " agent_session_key='sess-new' WHERE id=:t", t=turn_id)
    rig.sql("UPDATE bot_profiles SET agent_backend='codex' WHERE slug=:b", b=rig.bot)
    other = _turn(rig, backend="codex", session="sess-other")
    assert _producer(rig).run_pass().enqueued == 1
    [control] = _controls(rig)
    assert control["metadata_json"]["control"]["target_run"] == {
        "turn_id": turn_id, "agent_request_id": f"req-{turn_id}", "backend": "claude-code",
        "session_key": "sess-pinned"}
    assert rig.sql("SELECT count(*) FROM gate_members WHERE turn_id=:t", t=other).scalar() == 0


def test_a_member_bound_after_capture_pins_the_binding_of_its_request(rig):
    turn_id = _turn(rig, backend=None, session=None)
    _open(rig)
    rig.sql("UPDATE turn_logs SET agent_backend='claude-code', agent_session_key='sess-bound' WHERE id=:t",
            t=turn_id)
    assert _producer(rig).run_pass().enqueued == 1
    pin = _controls(rig)[0]["metadata_json"]["control"]["target_run"]
    assert (pin["backend"], pin["session_key"]) == ("claude-code", "sess-bound")


# --- design §3 tests (e)–(j) -------------------------------------------------


def test_e_a_lapsed_gate_rebuild_inherits_the_queued_row_and_asks_once(rig, monkeypatch):
    turn_id = _turn(rig)
    job_id = _open(rig)
    producer = _producer(rig)
    assert producer.run_pass().enqueued == 1
    [control] = _controls(rig)
    first, second = _rebuild(rig, job_id)
    carried = _member(rig, turn_id, second)
    assert (carried["control_state"], carried["control_delivery_id"]) == ("steering", control["id"])

    assert producer.run_pass().linked == 0
    claimed, calls = _steer(rig, monkeypatch)
    assert claimed.id == control["id"] and len(calls) == 1
    assert _row(rig, control["id"])["status"] == "DELIVERED" and len(_controls(rig)) == 1


def test_f_a_rebuild_after_the_row_was_revoked_gets_successor_r1_and_asks_once(rig, monkeypatch):
    turn_id = _turn(rig)
    job_id = _open(rig)
    producer = _producer(rig)
    assert producer.run_pass().enqueued == 1
    [control] = _controls(rig)
    rig.sql("UPDATE system_gates SET expires_at = now() - interval '1 second' WHERE id=:g",
            g=_gate(rig, job_id))
    revoked, calls = _steer(rig, monkeypatch)  # no live member links it: revoked at claim
    assert (revoked.id, revoked.last_error, calls) == (control["id"], "not_needed:control_revoked", [])
    assert rig.driver.step(job_id).outcome == "holding"
    fresh = _member(rig, turn_id, _gate(rig, job_id))
    assert (fresh["control_state"], fresh["control_delivery_id"]) == ("pending", None)  # dead: not carried

    assert producer.run_pass().enqueued == 1
    successor = _controls(rig)[-1]
    assert successor["idempotency_key"] == successor_key(_run_key(turn_id), 1)
    assert _member(rig, turn_id, _gate(rig, job_id))["control_delivery_id"] == successor["id"]
    claimed, calls = _steer(rig, monkeypatch)
    assert claimed.id == successor["id"] and len(calls) == 1


@pytest.mark.parametrize("then", ["steer", "release"])
def test_g_two_jobs_share_one_row_and_release_is_reference_counted(rig, second, monkeypatch, then):
    turn_id = _turn(rig)
    job_a, job_b = _open(rig), second()
    result = _producer(rig).run_pass()
    assert (result.linked, result.enqueued) == (2, 1)
    [control] = _controls(rig)
    assert {_member(rig, turn_id, _gate(rig, j))["control_delivery_id"] for j in (job_a, job_b)} == {
        control["id"]}

    with rig.engine.begin() as conn:
        release_job_gates(conn, job_a)
    assert _member(rig, turn_id, _gate(rig, job_a))["control_state"] == "cancelled"
    assert _row(rig, control["id"])["status"] == "QUEUED"  # job B still links it
    if then == "steer":
        claimed, calls = _steer(rig, monkeypatch)
        assert claimed.id == control["id"] and len(calls) == 1
        assert _row(rig, control["id"])["status"] == "DELIVERED"
    else:
        with rig.engine.begin() as conn:
            release_job_gates(conn, job_b)
        row = _row(rig, control["id"])
        assert (row["status"], row["delivery_mode"], row["last_error"]) == (
            "CANCELLED", "skipped", "not_needed:control_released")
        assert row["claim_token"] is None


def test_h_an_accepted_row_absorbs_a_new_job_and_is_never_asked_again(rig, second, monkeypatch):
    turn_id = _turn(rig)
    _open(rig)
    producer = _producer(rig)
    assert producer.run_pass().enqueued == 1
    [control] = _controls(rig)
    claimed = _claim(rig)
    assert rig.deliveries.mark_transport_accepted(claimed.id, claimed.claim_token)
    job_b = second()

    result = producer.run_pass()
    assert (result.linked, result.enqueued) == (1, 0)
    member = _member(rig, turn_id, _gate(rig, job_b))
    assert (member["control_state"], member["control_delivery_id"]) == ("steering", control["id"])
    assert len(_controls(rig)) == 1 and _claim(rig) is None  # nothing else to ask


def test_i_a_key_collision_with_a_foreign_row_is_unconfirmed_and_not_linked(rig):
    turn_id = _turn(rig)
    _open(rig)
    foreign, duplicate = rig.deliveries.enqueue(
        sender_bot_id=SYSTEM_BOT_ID, target_bot_id=rig.bot, message="not a control row",
        payload={"messages": [{"role": "user", "content": "x"}], "bot_id": rig.bot, "user": USER,
                 "stream": True},
        author=AuthorReference.bot(SYSTEM_BOT_ID), idempotency_key=_run_key(turn_id), on_gate="hold",
    )
    assert not duplicate and foreign.lane == "normal"
    assert _producer(rig).run_pass().linked == 0
    member = _member(rig, turn_id)
    assert (member["control_state"], member["control_error"], member["control_delivery_id"]) == (
        "unconfirmed", "control_key_collision", None)
    assert _controls(rig) == [] and _row(rig, foreign.id)["lane"] == "normal"


def test_j_a_carried_link_to_a_row_revoked_in_the_race_is_relinked_to_r1(rig, monkeypatch):
    """R1: carry_forward copies D's link onto G2 while a claim settles D revoked; both commit.

    The claim's half of the race is played after the carry, as the SQL its
    no-live-member settle writes: the claim read D's authority while G1 had
    lapsed and G2 was not yet committed, so it never saw G2's member.
    """
    turn_id = _turn(rig)
    job_id = _open(rig)
    producer = _producer(rig)
    assert producer.run_pass().enqueued == 1
    [control] = _controls(rig)
    _, second = _rebuild(rig, job_id)
    assert _member(rig, turn_id, second)["control_delivery_id"] == control["id"]
    rig.sql("""
        UPDATE inter_bot_deliveries SET status='CANCELLED', delivery_mode='skipped',
            last_error='not_needed:control_revoked', updated_at=now()
        WHERE id=:id AND status='QUEUED'
    """, id=control["id"])
    dead = _member(rig, turn_id, second)
    assert dead["control_state"] == "steering"  # linked to a dead row

    result = producer.run_pass()
    assert (result.linked, result.enqueued) == (1, 1)
    successor = _controls(rig)[-1]
    assert successor["idempotency_key"] == successor_key(_run_key(turn_id), 1)
    relinked = _member(rig, turn_id, second)
    assert (relinked["control_state"], relinked["control_delivery_id"]) == ("steering", successor["id"])
    assert relinked["control_changed_at"] > dead["control_changed_at"]
    claimed, calls = _steer(rig, monkeypatch)
    assert claimed.id == successor["id"] and len(calls) == 1
    assert producer.run_pass().linked == 0 and len(_controls(rig)) == 2


def test_the_newest_absorbing_successor_is_linked(rig, second):
    """A run with a dead :r0 and a live :r1 links :r1, never a new :r2."""
    turn_id = _turn(rig)
    job_a = _open(rig)
    producer = _producer(rig)
    assert producer.run_pass().enqueued == 1
    first = _controls(rig)[0]
    rig.sql("UPDATE inter_bot_deliveries SET status='FAILED', last_error='unconfirmed:x' WHERE id=:id",
            id=first["id"])
    assert producer.run_pass().enqueued == 1  # dead link: successor :r1
    successor = _controls(rig)[-1]
    job_b = second()
    assert producer.run_pass().linked == 1
    assert _member(rig, turn_id, _gate(rig, job_b))["control_delivery_id"] == successor["id"]
    assert _member(rig, turn_id, _gate(rig, job_a))["control_delivery_id"] == successor["id"]
    assert [r["idempotency_key"] for r in _controls(rig)] == [
        _run_key(turn_id), successor_key(_run_key(turn_id), 1)]


# --- the operator view and the retry timer -----------------------------------


@pytest.mark.parametrize("delivery, expected", [
    ("UPDATE inter_bot_deliveries SET attempt_count=2, last_error='bridge timeout' WHERE id=:id",
     ("steering", 2, "bridge timeout")),
    ("UPDATE inter_bot_deliveries SET status='DELIVERED', attempt_count=1 WHERE id=:id",
     ("steered", 1, None)),
    ("UPDATE inter_bot_deliveries SET status='CANCELLED', last_error='not_needed:run_ended' WHERE id=:id",
     ("not_needed", 0, "not_needed:run_ended")),
    ("UPDATE inter_bot_deliveries SET status='CANCELLED', last_error='cancelled_by_operator' WHERE id=:id",
     ("cancelled", 0, "cancelled_by_operator")),
    ("UPDATE inter_bot_deliveries SET status='FAILED', attempt_count=12,"
     " last_error='unconfirmed:no_active_run' WHERE id=:id",
     ("unconfirmed", 12, "unconfirmed:no_active_run")),
], ids=["queued", "delivered", "not-needed", "cancelled", "failed"])
def test_the_operator_view_reports_the_linked_rows_state(rig, delivery, expected):
    turn_id = _turn(rig)
    job_id = _open(rig)
    assert _producer(rig).run_pass().enqueued == 1
    rig.sql(delivery, id=_controls(rig)[0]["id"])
    [member] = rig.service.get_job_status(job_id, reconcile_if_active=False)["quiesce"]["members"]
    assert member["turn_id"] == turn_id
    assert (member["control_state"], member["control_attempts"], member["control_error"]) == expected


def test_a_backing_off_control_row_sets_the_recheck(rig):
    _turn(rig)
    _open(rig)
    producer = _producer(rig)
    assert producer.run_pass().enqueued == 1
    rig.sql("UPDATE inter_bot_deliveries SET available_at = clock_timestamp() + interval '30 seconds'"
            " WHERE id=:id", id=_controls(rig)[0]["id"])
    assert 25 < producer.run_pass().retry_in <= 30


def test_produce_is_a_no_op_for_a_member_already_linked(rig):
    turn_id = _turn(rig)
    _open(rig)
    producer = _producer(rig)
    member = _member(rig, turn_id)
    assert producer.produce(member["id"], rig.bot, MESSAGE) == ENQUEUED
    assert producer.produce(member["id"], rig.bot, MESSAGE) is None
    rig.sql("UPDATE gate_members SET control_state='pending', control_delivery_id=NULL WHERE id=:id",
            id=member["id"])
    assert producer.produce(member["id"], rig.bot, MESSAGE) == LINKED  # finds the run's live row
    assert len(_controls(rig)) == 1
