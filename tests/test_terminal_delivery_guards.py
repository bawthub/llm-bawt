"""Private-PG delivery retries never replay a protocol-owned execution."""
from __future__ import annotations

import json

import pytest
from sqlalchemy import text

from tests.test_inter_bot_delivery import _enqueue, delivery_store  # noqa: F401

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


def _owned_turn(store, sender, target, *, ended=False):
    delivery, _ = _enqueue(store, sender, target, "one logical message")
    claim = store.claim_next(target, claim_owner="original", steer_capable=False)
    assert claim is not None
    with store.engine.begin() as conn:
        conn.execute(text("""
            UPDATE turn_logs SET terminal_protocol_state=:state, status=:status,
                ended_at=CASE WHEN :ended THEN now() ELSE NULL END,
                response_text='preserved partial', assistant_message_id=:assistant,
                terminal_snapshot_json=:snapshot
            WHERE id=:id
        """), {"id": delivery.turn_id, "state": "pending" if ended else "open",
                "status": "completed" if ended else "streaming", "ended": ended,
                "assistant": delivery.turn_id,
                "snapshot": json.dumps({"immutable": True}) if ended else None})
        conn.execute(text("""
            UPDATE inter_bot_deliveries SET lease_expires_at=now() - interval '1 second'
            WHERE id=:id
        """), {"id": delivery.id})
    return delivery


@pytest.mark.parametrize("accepted", [False, True])
def test_expired_delivery_waits_for_open_protocol_owner(delivery_store, accepted):  # noqa: F811
    store, sender, target = delivery_store
    delivery = _owned_turn(store, sender, target)
    with store.engine.begin() as conn:
        if accepted:
            conn.execute(text("UPDATE inter_bot_deliveries SET transport_accepted_at=now() WHERE id=:id"),
                         {"id": delivery.id})
        before = dict(conn.execute(text("SELECT * FROM turn_logs WHERE id=:id"),
                                   {"id": delivery.turn_id}).mappings().one())
    assert store.recover_expired() == []
    with store.engine.connect() as conn:
        row = conn.execute(text("SELECT status, attempt_count, claim_owner FROM inter_bot_deliveries WHERE id=:id"),
                           {"id": delivery.id}).mappings().one()
        assert dict(row) == {"status": "DISPATCHING", "attempt_count": 1, "claim_owner": "original"}
        assert dict(conn.execute(text("SELECT * FROM turn_logs WHERE id=:id"),
                                 {"id": delivery.turn_id}).mappings().one()) == before


def test_delivery_settles_after_protocol_owner_commits(delivery_store):  # noqa: F811
    store, sender, target = delivery_store
    delivery = _owned_turn(store, sender, target, ended=True)
    recovered = store.recover_expired()
    assert len(recovered) == 1 and recovered[0].id == delivery.id
    assert recovered[0].status == "DELIVERED" and recovered[0].attempt_count == 1
    with store.engine.connect() as conn:
        row = conn.execute(text("SELECT terminal_snapshot_json, response_text FROM turn_logs WHERE id=:id"),
                           {"id": delivery.turn_id}).mappings().one()
        assert json.loads(row["terminal_snapshot_json"]) == {"immutable": True}
        assert row["response_text"] == "preserved partial"


@pytest.mark.parametrize("ended", [False, True])
def test_refused_reservation_rolls_back_preparation_and_settles_head(delivery_store, monkeypatch, ended):  # noqa: F811
    store, sender, target = delivery_store
    delivery = _owned_turn(store, sender, target, ended=ended)
    with store.engine.begin() as conn:
        conn.execute(text("UPDATE inter_bot_deliveries SET status='QUEUED', claim_token=NULL, claim_owner=NULL WHERE id=:id"),
                     {"id": delivery.id})
        before = dict(conn.execute(text("SELECT * FROM inter_bot_deliveries WHERE id=:id"),
                                   {"id": delivery.id}).mappings().one())
    from llm_bawt.service.session_lifecycle.lifecycle import NewSessionLifecycle
    original = NewSessionLifecycle.rotate_delivery_claim
    prepared = []

    def prepare(conn, **kwargs):
        prepared.append(True)
        conn.execute(text("UPDATE inter_bot_deliveries SET last_error='must roll back' WHERE id=:id"),
                     {"id": delivery.id})
        return original(conn, **kwargs)

    monkeypatch.setattr(NewSessionLifecycle, "rotate_delivery_claim", prepare)
    result = store.claim_next(target, claim_owner="replacement", steer_capable=False)
    assert result.status == ("DELIVERED" if ended else "FAILED")
    assert prepared == [True]
    with store.engine.connect() as conn:
        after = conn.execute(text("SELECT * FROM inter_bot_deliveries WHERE id=:id"),
                             {"id": delivery.id}).mappings().one()
        assert after["attempt_count"] == before["attempt_count"]
        assert after["last_error"] != "must roll back"
        assert after["claim_token"] is None and after["claim_owner"] is None
        assert after["payload_json"] == before["payload_json"]
        row = conn.execute(text("SELECT status, terminal_snapshot_json FROM turn_logs WHERE id=:id"),
                           {"id": delivery.turn_id}).mappings().one()
        assert row["status"] == ("completed" if ended else "streaming")
        assert (json.loads(row["terminal_snapshot_json"]) if ended else row["terminal_snapshot_json"]) == (
            {"immutable": True} if ended else None)
    if ended:
        _enqueue(store, sender, target, "later message")
        assert store.claim_next(target, claim_owner="next", steer_capable=False).status == "DISPATCHING"


@pytest.mark.parametrize("ended", [False, True])
def test_live_requeue_keeps_consumed_protocol_claim_for_recovery(delivery_store, ended):  # noqa: F811
    store, sender, target = delivery_store
    delivery = _owned_turn(store, sender, target, ended=ended)
    with store.engine.connect() as conn:
        token = conn.execute(text("SELECT claim_token FROM inter_bot_deliveries WHERE id=:id"),
                             {"id": delivery.id}).scalar_one()
    assert store.requeue(delivery.id, "stale", "wrong owner", delay_seconds=0) is None
    result = store.requeue(delivery.id, token, "transient drain failure", delay_seconds=0,
                           clear_delivery_mode=True, refund_attempt=True)
    assert result.status == "DISPATCHING" and result.claim_token == token
    assert result.delivery_mode == "turn" and result.attempt_count == 1
    recovered = store.recover_expired()
    if ended:
        assert len(recovered) == 1 and recovered[0].status == "DELIVERED"
    else:
        assert recovered == []


@pytest.fixture
def protocol_delivery(delivery_store, monkeypatch):  # noqa: F811
    from llm_bawt.service import turn_terminal_admission

    monkeypatch.setattr(turn_terminal_admission, "resolve_bot_runtime_setting", lambda *args: True)
    monkeypatch.setattr(turn_terminal_admission, "resolve_global_runtime_setting", lambda *args: "recover")
    store, sender, target = delivery_store
    monkeypatch.setattr(store.config, "REDIS_URL", "redis://fixture-unused")
    delivery, _ = _enqueue(store, sender, target, "protocol reservation")
    claim = store.claim_next(target, claim_owner="first", steer_capable=False)
    return store, delivery, claim, target


def test_hold_requeue_reclaim_and_consume_preserves_producer_assistant_id(protocol_delivery):
    from llm_bawt.service.turn_admission import AdmissionClaim, revoke_reservation
    from llm_bawt.service.turn_logs import TurnLogStore
    from tests.test_turn_terminal_protocol import read

    store, delivery, claim, target = protocol_delivery
    assistant = store.payload(delivery.id)["assistant_message_id"]
    assert read(store.engine, claim.turn_id).assistant_message_id == assistant
    held = revoke_reservation(store.engine, claim.turn_id, disposition="hold", audit={"reason": "test hold"})
    assert held["held"] and read(store.engine, claim.turn_id) is None
    reclaimed = store.claim_next(target, claim_owner="second", steer_capable=False)
    assert reclaimed.turn_id == claim.turn_id
    assert read(store.engine, claim.turn_id).assistant_message_id == assistant
    # Exercise another pre-start retry through the central requeue path.
    store.requeue(delivery.id, reclaimed.claim_token, "pre-start retry", delay_seconds=0)
    reclaimed = store.claim_next(target, claim_owner="third", steer_capable=False)
    admission = AdmissionClaim(
        owner=reclaimed.claim_token, allow_unowned_reservation=True,
        require_reservation=True, reservation_delivery_id=delivery.id,
        fence=lambda conn: bool(conn.execute(text(
            "SELECT 1 FROM inter_bot_deliveries WHERE id=:id AND claim_token=:token AND status='DISPATCHING'"
        ), {"id": delivery.id, "token": reclaimed.claim_token}).first()),
    )
    lease = TurnLogStore(store.config).save_turn(
        turn_id=claim.turn_id, request_id="request", path="/test", stream=False,
        model="test", bot_id=target, user_id="nick", status="pending", latency_ms=None,
        user_prompt="protocol reservation", request_payload={}, response_text="", tool_calls=[],
        assistant_message_id="replacement-must-not-win", admission_claim=admission,
    )
    assert lease.assistant_message_id == assistant
    assert read(store.engine, claim.turn_id).assistant_message_id == assistant
    assert read(store.engine, claim.turn_id).terminal_protocol_state == "binding"


@pytest.mark.parametrize("action", ["stop", "fail", "withdraw"])
def test_terminal_producer_outcome_freezes_unconsumed_reservation(protocol_delivery, action):
    from llm_bawt.inter_bot_delivery_recovery import DeliveryLeaseRecovery
    from llm_bawt.service.turn_admission import revoke_reservation
    from tests.test_turn_terminal_protocol import read

    store, delivery, claim, _ = protocol_delivery
    if action == "stop":
        revoke_reservation(store.engine, claim.turn_id, disposition="user_stop", audit={"by": "test"})
    elif action == "fail":
        store.fail_claim(delivery.id, claim.claim_token, "pre-start failure")
    else:
        with store.engine.begin() as conn:
            assert DeliveryLeaseRecovery.settle_withdrawn_turn(conn, claim.turn_id, cause="withdrawn")
    row = read(store.engine, claim.turn_id)
    assert row.ended_at is not None and row.terminal_protocol_state == "pending"
    frozen = json.loads(row.terminal_snapshot_json)
    assert frozen["history_disposition"] == "unbound" and frozen["response_text"] == ""
    assert frozen["assistant_message_id"] == store.payload(delivery.id)["assistant_message_id"]


def test_preclaim_dead_letter_cannot_freeze_a_claimed_reservation(protocol_delivery):
    from tests.test_turn_terminal_protocol import read

    store, delivery, claim, _ = protocol_delivery
    before = read(store.engine, claim.turn_id).model_dump()
    assert store.mark_failed(delivery.id, "late validation failure") is None
    assert read(store.engine, claim.turn_id).model_dump() == before
    current = store.get(delivery.id)
    assert current.status == "DISPATCHING" and current.claim_token == claim.claim_token


def test_legacy_overflow_retry_cannot_delete_a_protocol_outcome(delivery_store, monkeypatch):  # noqa: F811
    from unittest.mock import Mock

    store, sender, target = delivery_store
    delivery = _owned_turn(store, sender, target, ended=True)
    with store.engine.connect() as conn:
        token = conn.execute(text("SELECT claim_token FROM inter_bot_deliveries WHERE id=:id"),
                             {"id": delivery.id}).scalar_one()
        before = dict(conn.execute(text("SELECT * FROM turn_logs WHERE id=:id"),
                                   {"id": delivery.turn_id}).mappings().one())
    forbidden = Mock(side_effect=AssertionError("must not rotate/delete/reinfer"))
    monkeypatch.setattr("llm_bawt.agent_context.recover_delivery_after_overflow", forbidden)
    assert store.recover_context_overflow(delivery.id, token, backend="claude-code", error="overflow") is None
    forbidden.assert_not_called()
    with store.engine.connect() as conn:
        assert dict(conn.execute(text("SELECT * FROM turn_logs WHERE id=:id"),
                                 {"id": delivery.turn_id}).mappings().one()) == before
