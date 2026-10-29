"""PostgreSQL tests for TASK-1749: the user's Send now past a system hold.

Only the user's own held chat row is admitted, past only the gate generations
its stamp names; its idle turn is rostered ``accepted`` with an audit record,
and bot deliveries stay held.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text as sa_text

from llm_bawt.service.gate_steer_escalation import (
    KIND_USER_SEND_NOW,
    SendNowRefused,
    gate_escalations,
    stamp_send_now,
)
from llm_bawt.service.turn_gates import GateScope, release_gates
from tests.test_delivery_claim_gate_pg import _claim, _enqueue, _row
from tests.test_turn_gates_pg import _activate, _open, _spec, gated  # noqa: F401  (gated is a fixture)
from tests.test_user_stop_hold_pg import _user_chat as _shared_user_chat

pytestmark = pytest.mark.usefixtures("private_app_pg")


def _user_chat(gated, deliveries, bot):
    """The user's chat row with the payload ``ordinary_turn_queue`` writes (prefer_steer=False)."""
    record = _shared_user_chat(gated, deliveries, bot)
    with deliveries.engine.begin() as conn:
        conn.execute(sa_text(
            "UPDATE inter_bot_deliveries SET payload_json=payload_json || '{\"prefer_steer\": false}'::jsonb"
            " WHERE id=:id"), {"id": record.id})
    return record


def _quiesce(gated, bot):
    owner = gated.owner()
    return owner, _activate(gated.store, _spec(owner, GateScope.for_bots(bot))).gate


def _members(gated, gate_id):
    with gated.store.engine.connect() as conn:
        return conn.execute(sa_text(
            "SELECT turn_id, disposition, producer_id FROM gate_members WHERE gate_id=:g"
        ), {"g": gate_id}).mappings().all()


@pytest.mark.integration
def test_send_now_refuses_bot_rows_other_users_and_unheld_rows(gated):
    bot = gated.bot()
    deliveries, bot_row = _enqueue(gated, bot, "bot callback")
    user = _user_chat(gated, deliveries, bot)
    engine = gated.store.engine

    with pytest.raises(SendNowRefused) as unheld:
        stamp_send_now(engine, user.id, "nick")
    assert unheld.value.status_code == 409 and unheld.value.detail == "not_held"

    _quiesce(gated, bot)
    with pytest.raises(SendNowRefused) as foreign:
        stamp_send_now(engine, bot_row.id, "nick")
    assert foreign.value.status_code == 403
    with pytest.raises(SendNowRefused) as other_user:
        stamp_send_now(engine, user.id, "someone-else")
    assert other_user.value.status_code == 403


@pytest.mark.integration
def test_stamped_user_row_runs_under_a_live_quiesce_and_is_rostered(gated):
    bot = gated.bot()
    deliveries, bot_row = _enqueue(gated, bot, "bot callback")
    user = _user_chat(gated, deliveries, bot)
    _owner, gate = _quiesce(gated, bot)
    assert _claim(deliveries, bot) is None  # everything is held

    result = stamp_send_now(gated.store.engine, user.id, "nick")
    assert result["gate_ids"] == [gate.id]

    claimed = _claim(deliveries, bot)
    assert claimed is not None and claimed.id == user.id and claimed.status == "DISPATCHING"
    assert _row(deliveries, bot_row.id)["status"] == "QUEUED"  # the bot delivery is still held
    members = [m for m in _members(gated, gate.id) if m["producer_id"] == user.id]
    assert len(members) == 1
    assert members[0]["disposition"] == "accepted" and members[0]["turn_id"] == claimed.turn_id
    with gated.store.engine.connect() as conn:
        audit = gate_escalations(conn, gate.id)
    assert [(a["kind"], a["subject"], a["actor"], a["outcome"], a["turn_id"]) for a in audit] == [
        (KIND_USER_SEND_NOW, user.id, "user:nick", "turn_started", claimed.turn_id)]


@pytest.mark.integration
def test_a_revoked_unstarted_capture_of_the_same_turn_becomes_accepted(gated):
    """Codex r1 B1: a requeued row's earlier revoked reservation shares its turn id."""
    from llm_bawt.service.turn_gates import REVOKED_UNSTARTED, GateMember, insert_member

    bot = gated.bot()
    deliveries, _ = _enqueue(gated, bot, "bot callback")
    user = _user_chat(gated, deliveries, bot)
    _owner, gate = _quiesce(gated, bot)
    with gated.store.engine.begin() as conn:
        insert_member(conn, GateMember(
            gate_id=gate.id, bot_id=bot, turn_id=user.turn_id, turn_status="reserved",
            disposition=REVOKED_UNSTARTED, producer_kind="delivery", producer_id=user.id,
            control_state="not_needed",
        ))
    stamp_send_now(gated.store.engine, user.id, "nick")

    claimed = _claim(deliveries, bot)

    assert claimed is not None and claimed.id == user.id
    [member] = [m for m in _members(gated, gate.id) if m["turn_id"] == user.turn_id]
    assert member["disposition"] == "accepted"


@pytest.mark.integration
def test_a_run_without_now_is_never_next_steered_and_runs_after_its_turn(gated):
    """Codex r1 B2: Send now never silently degrades to a `next` steer."""
    bot = gated.bot()
    deliveries, _ = _enqueue(gated, bot, "bot callback")
    user = _user_chat(gated, deliveries, bot)
    turn = _open(gated.store, bot)
    _quiesce(gated, bot)
    stamp_send_now(gated.store.engine, user.id, "nick")

    assert deliveries.claim_next(bot, claim_owner="worker", steer_capable=frozenset({"next"})) is None
    with gated.store.engine.begin() as conn:
        conn.execute(sa_text("UPDATE turn_logs SET status='ok', ended_at=now() WHERE id=:id"), {"id": turn})
    claimed = deliveries.claim_next(bot, claim_owner="worker", steer_capable=frozenset({"next"}))
    assert claimed is not None and claimed.id == user.id and claimed.status == "DISPATCHING"


@pytest.mark.integration
def test_busy_target_takes_the_stamped_row_as_a_steer(gated):
    bot = gated.bot()
    deliveries, _ = _enqueue(gated, bot, "bot callback")
    user = _user_chat(gated, deliveries, bot)
    _open(gated.store, bot)
    _quiesce(gated, bot)
    stamp_send_now(gated.store.engine, user.id, "nick")

    claimed = deliveries.claim_next(bot, claim_owner="worker", steer_capable=frozenset({"next", "now"}))

    assert claimed is not None and claimed.id == user.id and claimed.status == "STEERING"


@pytest.mark.integration
def test_steer_check_passes_only_the_stamped_generations(gated):
    from llm_bawt.service.system_gate_admission import SteerGateRefused, check_steer

    bot = gated.bot()
    deliveries, bot_row = _enqueue(gated, bot, "bot callback")
    user = _user_chat(gated, deliveries, bot)
    turn = _open(gated.store, bot)
    owner, gate = _quiesce(gated, bot)
    stamp_send_now(gated.store.engine, user.id, "nick")
    engine = gated.store.engine

    assert check_steer(engine, bot, turn_id=turn, send_now=user.id) is None
    with pytest.raises(SteerGateRefused):
        check_steer(engine, bot, turn_id=turn, send_now=bot_row.id)  # unstamped: still held

    with engine.begin() as conn:
        release_gates(conn, owner_id=owner)
    _quiesce(gated, bot)  # a later generation holds again
    with pytest.raises(SteerGateRefused):
        check_steer(engine, bot, turn_id=turn, send_now=user.id)
    with engine.connect() as conn:
        assert gate_escalations(conn, gate.id)[0]["outcome"] == "steer_admitted"
