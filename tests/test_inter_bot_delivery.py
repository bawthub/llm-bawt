"""Durable inter-bot delivery contract (TASK-710)."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import text as sa_text

from llm_bawt.agent_context import SessionPolicy
from llm_bawt.inter_bot_delivery import (
    DELIVERED,
    DISPATCHING,
    FAILED,
    QUEUED,
    STEERING,
    InterBotDeliveryStore,
)
from llm_bawt.message_authorship import AuthorReference
from llm_bawt.service.turn_logs import DeliveryTargetBusy, TurnLogStore
from llm_bawt.utils.config import Config

pytestmark = pytest.mark.integration


@pytest.fixture
def delivery_store():
    store = InterBotDeliveryStore(Config())
    if store.engine is None:
        pytest.skip("PostgreSQL unavailable")
    token = uuid.uuid4().hex
    sender = f"test-sender-{token}"
    target = f"test-target-{token}"
    yield store, sender, target
    with store.engine.begin() as conn:
        turn_ids = conn.execute(sa_text(
            "SELECT turn_id FROM inter_bot_deliveries WHERE sender_bot_id=:sender"
        ), {"sender": sender}).scalars().all()
        conn.execute(sa_text(
            "DELETE FROM inter_bot_deliveries WHERE sender_bot_id=:sender"
        ), {"sender": sender})
        if turn_ids:
            conn.execute(sa_text(
                "DELETE FROM tool_call_records WHERE turn_id = ANY(:ids)"
            ), {"ids": list(turn_ids)})
            conn.execute(sa_text(
                "DELETE FROM turn_logs WHERE id = ANY(:ids)"
            ), {"ids": list(turn_ids)})
        conn.execute(sa_text(
            "DELETE FROM turn_logs WHERE bot_id=:target"
        ), {"target": target})
        from llm_bawt.memory.postgresql import _sanitize_table_name
        conn.execute(sa_text(
            "DELETE FROM messages WHERE bot_id=:target"
        ), {"target": _sanitize_table_name(target)})
        conn.execute(sa_text(
            "DELETE FROM sessions WHERE bot_id=:target"
        ), {"target": target})


def _enqueue(
    store, sender, target, message, *, key=None, max_attempts=5, author=None
):
    return store.enqueue(
        sender_bot_id=sender,
        target_bot_id=target,
        message=message,
        payload={
            "messages": [{"role": "user", "content": message}],
            "bot_id": target,
            "stream": False,
        },
        author=author,
        idempotency_key=key,
        task_id="TASK-700",
        project_id="project-1",
        message_kind="READY",
        metadata={"source": "test"},
        max_attempts=max_attempts,
    )


def test_delivery_persists_explicit_user_author(delivery_store):
    store, sender, target = delivery_store
    record, duplicate = _enqueue(
        store,
        sender,
        target,
        "UI commit request",
        author=AuthorReference.user("Nick"),
    )

    assert duplicate is False
    assert record.author == AuthorReference.user("nick")
    assert record.to_api()["author"] == {
        "entity_type": "user",
        "entity_id": "nick",
        "status": "canonical",
    }


def test_delivery_defaults_to_bot_sender_author(delivery_store):
    store, sender, target = delivery_store
    record, _ = _enqueue(store, sender, target, "bot callback")

    assert record.author == AuthorReference.bot(sender)


def _finish_reserved_turn(store, turn_id, *, status="ok"):
    with store.engine.begin() as conn:
        conn.execute(sa_text("""
            UPDATE turn_logs SET status=:status, end_reason='stop', ended_at=now()
            WHERE id=:id
        """), {"id": turn_id, "status": status})


def test_duplicate_idempotency_key_returns_one_stable_delivery(delivery_store):
    store, sender, target = delivery_store
    first, duplicate1 = _enqueue(store, sender, target, "TASK-700 READY", key="ready-700")
    second, duplicate2 = _enqueue(store, sender, target, "different retry body", key="ready-700")

    assert duplicate1 is False
    assert duplicate2 is True
    assert second.id == first.id
    assert second.user_message_id == first.user_message_id
    assert second.turn_id == first.turn_id
    assert second.message == "TASK-700 READY"
    first_payload = store.payload(first.id)
    second_payload = store.payload(second.id)
    assert first_payload["assistant_message_id"] == second_payload["assistant_message_id"]
    assert str(uuid.UUID(first_payload["assistant_message_id"])) == first_payload["assistant_message_id"]
    assert str(uuid.UUID(first.user_message_id)) == first.user_message_id
    assert len(first.user_message_id) == 36
    rows = store.list(sender_bot_id=sender)
    assert [row.id for row in rows] == [first.id]


def test_fifo_claims_one_target_delivery_without_overlap(delivery_store):
    store, sender, target = delivery_store
    first, _ = _enqueue(store, sender, target, "first")
    second, _ = _enqueue(store, sender, target, "second")

    claimed_first = store.claim_next(target, claim_owner="test-owner", steer_capable=False)
    assert claimed_first and claimed_first.id == first.id
    assert claimed_first.status == DISPATCHING
    assert store.claim_next(target, claim_owner="test-owner", steer_capable=False) is None

    _finish_reserved_turn(store, first.turn_id)
    delivered = store.mark_delivered(first.id, claimed_first.claim_token or "", response_model="test", response_chars=2)
    assert delivered and delivered.status == DELIVERED

    claimed_second = store.claim_next(target, claim_owner="test-owner", steer_capable=False)
    assert claimed_second and claimed_second.id == second.id
    assert claimed_second.attempt_count == 1


def test_busy_target_and_dispatch_race_requeue_safely(delivery_store):
    store, sender, target = delivery_store
    queued, _ = _enqueue(store, sender, target, "callback")
    active_id = f"turn-active-{uuid.uuid4().hex}"
    with store.engine.begin() as conn:
        conn.execute(sa_text("""
            INSERT INTO turn_logs (
                id, created_at, path, stream, bot_id, user_id, status,
                user_prompt, response_text, ended_at
            ) VALUES (:id, now(), '/test', true, :target, 'nick', 'streaming', '', '', NULL)
        """), {"id": active_id, "target": target})
    assert store.claim_next(target, claim_owner="test-owner", steer_capable=False) is None

    with store.engine.begin() as conn:
        conn.execute(sa_text(
            "UPDATE turn_logs SET status='ok', end_reason='stop', ended_at=now() WHERE id=:id"
        ), {"id": active_id})
    claimed = store.claim_next(target, claim_owner="test-owner", steer_capable=False)
    assert claimed and claimed.id == queued.id

    # A normal target turn trying to start after the reservation must lose the
    # race instead of overlapping the queued callback.
    turn_store = TurnLogStore(Config())
    with pytest.raises(DeliveryTargetBusy):
        turn_store.save_turn(
            turn_id=f"turn-user-{uuid.uuid4().hex}",
            request_id="req-user",
            path="/v1/chat/completions",
            stream=True,
            model="test",
            bot_id=target,
            user_id="nick",
            status="streaming",
            latency_ms=None,
            user_prompt="user turn",
            request_payload={"messages": []},
            response_text="",
            tool_calls=None,
        )

    requeued = store.requeue(claimed.id, claimed.claim_token or "", "target became busy", delay_seconds=0)
    assert requeued and requeued.status == QUEUED
    with store.engine.connect() as conn:
        assert conn.execute(sa_text(
            "SELECT count(*) FROM turn_logs WHERE id=:id"
        ), {"id": claimed.turn_id}).scalar_one() == 0


def test_active_claude_turn_claims_steering_without_new_turn(delivery_store):
    store, sender, target = delivery_store
    original, _ = _enqueue(store, sender, target, "interrupt active turn")
    active_id = f"turn-active-{uuid.uuid4().hex}"
    with store.engine.begin() as conn:
        conn.execute(sa_text("""
            INSERT INTO turn_logs (
                id, created_at, path, stream, bot_id, user_id, status,
                user_prompt, response_text, agent_session_key,
                agent_request_id, ended_at
            ) VALUES (
                :id, now(), '/test', true, :target, 'nick', 'streaming',
                'working', '', 'snark:nick', 'req-active', NULL
            )
        """), {"id": active_id, "target": target})

    claimed = store.claim_next(
        target, claim_owner="test-owner", steer_capable=True
    )
    assert claimed.status == STEERING
    assert claimed.target_turn_id == active_id
    assert claimed.delivery_mode == "steer"
    with store.engine.connect() as conn:
        assert conn.execute(sa_text(
            "SELECT count(*) FROM turn_logs WHERE id=:id"
        ), {"id": original.turn_id}).scalar_one() == 0


def test_user_delivery_does_not_steer_another_users_active_turn(delivery_store):
    store, sender, target = delivery_store
    original, _ = store.enqueue(
        sender_bot_id=sender,
        target_bot_id=target,
        message="UI request",
        payload={
            "messages": [{"role": "user", "content": "UI request"}],
            "bot_id": target,
            "user": "nick",
            "prefer_steer": True,
            "stream": False,
        },
        author=AuthorReference.user("nick"),
    )
    active_id = f"turn-other-user-{uuid.uuid4().hex}"
    with store.engine.begin() as conn:
        conn.execute(sa_text("""
            INSERT INTO turn_logs (
                id, created_at, path, stream, bot_id, user_id, status,
                user_prompt, response_text, agent_session_key,
                agent_request_id, ended_at
            ) VALUES (
                :id, now(), '/test', true, :target, 'alex', 'streaming',
                'working', '', 'snark:alex', 'req-active', NULL
            )
        """), {"id": active_id, "target": target})

    assert store.claim_next(
        target, claim_owner="test-owner", steer_capable=True
    ) is None
    assert store.get(original.id).status == QUEUED


def test_active_claude_not_ready_still_claims_steering_for_retry(delivery_store):
    store, sender, target = delivery_store
    _enqueue(store, sender, target, "steer when ready")
    active_id = f"turn-not-ready-{uuid.uuid4().hex}"
    with store.engine.begin() as conn:
        conn.execute(sa_text("""
            INSERT INTO turn_logs (
                id, created_at, path, stream, bot_id, user_id, status,
                user_prompt, response_text, ended_at
            ) VALUES (:id, now(), '/test', true, :target, 'nick',
                      'streaming', '', '', NULL)
        """), {"id": active_id, "target": target})

    claimed = store.claim_next(
        target, claim_owner="test-owner", steer_capable=True
    )
    assert claimed.status == STEERING
    assert claimed.target_turn_id == active_id


def test_active_unsupported_backend_stays_queued_until_idle(delivery_store):
    store, sender, target = delivery_store
    original, _ = _enqueue(store, sender, target, "wait safely")
    active_id = f"turn-unsupported-{uuid.uuid4().hex}"
    with store.engine.begin() as conn:
        conn.execute(sa_text("""
            INSERT INTO turn_logs (
                id, created_at, path, stream, bot_id, user_id, status,
                user_prompt, response_text, ended_at
            ) VALUES (:id, now(), '/test', true, :target, 'nick',
                      'streaming', '', '', NULL)
        """), {"id": active_id, "target": target})
    assert store.claim_next(
        target, claim_owner="test-owner", steer_capable=False
    ) is None
    assert store.get(original.id).status == QUEUED


def test_accepted_steer_retry_never_converts_to_new_turn(delivery_store):
    store, sender, target = delivery_store
    original, _ = _enqueue(store, sender, target, "accepted steer")
    active_id = f"turn-steer-{uuid.uuid4().hex}"
    with store.engine.begin() as conn:
        conn.execute(sa_text("""
            INSERT INTO turn_logs (
                id, created_at, path, stream, bot_id, user_id, status,
                user_prompt, response_text, agent_session_key,
                agent_request_id, ended_at
            ) VALUES (:id, now(), '/test', true, :target, 'nick',
                      'streaming', '', '', 'snark:nick', 'req-active', NULL)
        """), {"id": active_id, "target": target})
    first = store.claim_next(
        target, claim_owner="owner-one", steer_capable=True
    )
    assert store.mark_transport_accepted(first.id, first.claim_token or "")
    queued = store.requeue(
        first.id, first.claim_token or "", "persist retry", delay_seconds=0
    )
    with store.engine.begin() as conn:
        conn.execute(sa_text(
            "UPDATE turn_logs SET status='ok', ended_at=now() WHERE id=:id"
        ), {"id": active_id})
    second = store.claim_next(
        target, claim_owner="owner-two", steer_capable=True
    )
    assert queued.status == QUEUED
    assert second.status == STEERING
    assert second.target_turn_id == active_id
    assert second.delivery_mode == "steer"
    with store.engine.connect() as conn:
        assert conn.execute(sa_text(
            "SELECT count(*) FROM turn_logs WHERE id=:id"
        ), {"id": original.turn_id}).scalar_one() == 0


def test_not_ready_retry_refunds_attempt_and_keeps_steer_target(delivery_store):
    store, sender, target = delivery_store
    original, _ = _enqueue(store, sender, target, "wait for metadata", max_attempts=1)
    active_id = f"turn-not-ready-refund-{uuid.uuid4().hex}"
    with store.engine.begin() as conn:
        conn.execute(sa_text("""
            INSERT INTO turn_logs (
                id, created_at, path, stream, bot_id, user_id, status,
                user_prompt, response_text, ended_at
            ) VALUES (:id, now(), '/test', true, :target, 'nick',
                      'streaming', '', '', NULL)
        """), {"id": active_id, "target": target})

    claimed = store.claim_next(target, claim_owner="owner-one", steer_capable=True)
    assert claimed.attempt_count == 1
    queued = store.requeue(
        original.id,
        claimed.claim_token or "",
        "Active bridge run is not ready",
        delay_seconds=0,
        refund_attempt=True,
    )
    assert queued.status == QUEUED
    assert queued.attempt_count == 0
    assert queued.delivery_mode == "steer"
    assert queued.target_turn_id == active_id


def test_definitive_unaccepted_steer_waits_for_rejected_turn_then_falls_back_once(delivery_store):
    store, sender, target = delivery_store
    original, _ = _enqueue(store, sender, target, "fallback once", max_attempts=1)
    active_id = f"turn-stale-steer-{uuid.uuid4().hex}"
    with store.engine.begin() as conn:
        conn.execute(sa_text("""
            INSERT INTO turn_logs (
                id, created_at, path, stream, bot_id, user_id, status,
                user_prompt, response_text, agent_session_key,
                agent_request_id, ended_at
            ) VALUES (:id, now(), '/test', true, :target, 'nick',
                      'streaming', '', '', 'snark:nick', 'req-old', NULL)
        """), {"id": active_id, "target": target})

    claimed = store.claim_next(target, claim_owner="owner-one", steer_capable=True)
    queued = store.requeue(
        original.id,
        claimed.claim_token or "",
        "no_active_run",
        delay_seconds=0,
        reject_steer_target=True,
        refund_attempt=True,
    )
    assert queued.status == QUEUED
    assert queued.attempt_count == 0
    assert queued.delivery_mode == "steer_rejected"
    assert queued.target_turn_id == active_id
    assert queued.transport_accepted_at is None
    assert store.claim_next(target, claim_owner="owner-two", steer_capable=True) is None

    with store.engine.begin() as conn:
        conn.execute(sa_text(
            "UPDATE turn_logs SET status='ok', ended_at=now() WHERE id=:id"
        ), {"id": active_id})
    fallback = store.claim_next(target, claim_owner="owner-two", steer_capable=True)
    assert fallback.status == DISPATCHING
    assert fallback.delivery_mode == "turn"
    assert fallback.target_turn_id == original.turn_id
    assert fallback.attempt_count == 1


def test_restart_recovery_reuses_same_delivery_turn_and_message_ids(delivery_store):
    store, sender, target = delivery_store
    original, _ = _enqueue(store, sender, target, "restart me")
    claimed = store.claim_next(target, claim_owner="old-owner", steer_capable=False, lease_seconds=60)
    assert claimed and claimed.id == original.id

    # Simulate app death after target acceptance began: deterministic turn exists
    # and the lease expires. Recovery queues reattachment instead of minting IDs.
    with store.engine.begin() as conn:
        conn.execute(sa_text("""
            UPDATE turn_logs SET status='pending' WHERE id=:turn_id
        """), {"turn_id": original.turn_id})
        conn.execute(sa_text("""
            UPDATE inter_bot_deliveries
            SET lease_expires_at=now() - interval '1 second'
            WHERE id=:id
        """), {"id": original.id})

    recovered = store.recover_expired()
    assert [row.id for row in recovered] == [original.id]
    assert recovered[0].status == QUEUED
    assert recovered[0].user_message_id == original.user_message_id
    assert recovered[0].turn_id == original.turn_id
    original_payload = store.payload(original.id)

    reclaimed = store.claim_next(target, claim_owner="test-owner", steer_capable=False)
    assert reclaimed and reclaimed.id == original.id
    assert reclaimed.turn_id == original.turn_id
    assert reclaimed.user_message_id == original.user_message_id
    assert reclaimed.attempt_count == 2
    payload = store.payload(original.id)
    assert payload["inter_bot_bridge_request_id"].startswith("req_delivery_")
    assert payload["assistant_message_id"] == original_payload["assistant_message_id"]


def test_restart_recovery_does_not_deliver_ok_row_with_error_text(delivery_store):
    store, sender, target = delivery_store
    original, _ = _enqueue(store, sender, target, "partial failure")
    claimed = store.claim_next(
        target, claim_owner="old-owner", steer_capable=False, lease_seconds=60
    )
    assert claimed

    with store.engine.begin() as conn:
        conn.execute(sa_text("""
            UPDATE turn_logs SET status='ok', end_reason='stop', ended_at=now(),
                error_text='upstream exploded'
            WHERE id=:turn_id
        """), {"turn_id": original.turn_id})
        conn.execute(sa_text("""
            UPDATE inter_bot_deliveries
            SET lease_expires_at=now() - interval '1 second'
            WHERE id=:id
        """), {"id": original.id})

    recovered = store.recover_expired()[0]
    assert recovered.status == FAILED
    assert recovered.last_error == "target turn ended unsuccessfully"


def test_retry_then_dead_letter_is_inspectable(delivery_store):
    store, sender, target = delivery_store
    original, _ = _enqueue(store, sender, target, "retry me", max_attempts=2)

    first = store.claim_next(target, claim_owner="test-owner", steer_capable=False)
    requeued = store.requeue(first.id, first.claim_token or "", "transient bridge outage", delay_seconds=0)
    assert requeued.status == QUEUED
    assert requeued.next_retry_at is not None
    assert requeued.last_error == "transient bridge outage"

    second = store.claim_next(target, claim_owner="test-owner", steer_capable=False)
    assert second.attempt_count == 2
    failed = store.fail_claim(second.id, second.claim_token or "", "attempts exhausted")
    assert failed.status == FAILED
    assert failed.last_error == "attempts exhausted"
    inspected = store.get(original.id)
    assert inspected.status == FAILED
    assert inspected.attempt_count == 2


def test_active_turn_older_than_thirty_minutes_still_blocks(delivery_store):
    store, sender, target = delivery_store
    queued, _ = _enqueue(store, sender, target, "wait for old turn")
    active_id = f"turn-old-{uuid.uuid4().hex}"
    with store.engine.begin() as conn:
        conn.execute(sa_text("""
            INSERT INTO turn_logs (
                id, created_at, path, stream, bot_id, user_id, status,
                user_prompt, response_text, ended_at
            ) VALUES (
                :id, now() - interval '31 minutes', '/test', true, :target,
                'nick', 'streaming', '', '', NULL
            )
        """), {"id": active_id, "target": target})
    assert store.claim_next(target, claim_owner="test-owner", steer_capable=False) is None
    assert store.get(queued.id).status == QUEUED


def test_stale_claim_cannot_complete_or_requeue_new_owner(delivery_store):
    store, sender, target = delivery_store
    original, _ = _enqueue(store, sender, target, "stale claim")
    first = store.claim_next(target, claim_owner="owner-one", steer_capable=False, lease_seconds=60)
    with store.engine.begin() as conn:
        conn.execute(sa_text("""
            UPDATE inter_bot_deliveries SET lease_expires_at=now() - interval '1 second'
            WHERE id=:id
        """), {"id": original.id})
    store.recover_expired()
    second = store.claim_next(target, claim_owner="owner-two", steer_capable=False)

    assert store.mark_delivered(
        original.id,
        first.claim_token or "",
        response_model="stale",
        response_chars=1,
    ) is None
    assert store.requeue(
        original.id,
        first.claim_token or "",
        "stale",
        delay_seconds=0,
    ) is None
    assert store.get(original.id).claim_token == second.claim_token


def test_recovered_queued_cancel_removes_reservation(delivery_store):
    store, sender, target = delivery_store
    original, _ = _enqueue(store, sender, target, "cancel after restart")
    claimed = store.claim_next(target, claim_owner="owner-one", steer_capable=False, lease_seconds=60)
    with store.engine.begin() as conn:
        conn.execute(sa_text("""
            UPDATE inter_bot_deliveries SET lease_expires_at=now() - interval '1 second'
            WHERE id=:id
        """), {"id": original.id})
    recovered = store.recover_expired()[0]
    assert recovered.status == QUEUED

    cancelled = store.cancel(original.id)
    assert cancelled and cancelled.status == "CANCELLED"
    with store.engine.connect() as conn:
        count = conn.execute(sa_text(
            "SELECT count(*) FROM turn_logs WHERE id=:id AND ended_at IS NULL"
        ), {"id": claimed.turn_id}).scalar_one()
    assert count == 0


def _request_stop_for_delivery(store, turn_id):
    from llm_bawt.service.turn_termination import TurnAbortCoordinator

    turn_store = TurnLogStore(Config())
    with store.engine.begin() as conn:
        conn.execute(sa_text("UPDATE turn_logs SET status='streaming' WHERE id=:id"), {"id": turn_id})
    assert TurnAbortCoordinator(turn_store).request(turn_id, source="chat_stop", peer=None)


def test_stopped_delivery_cannot_requeue_or_be_reclaimed(delivery_store):
    store, sender, target = delivery_store
    original, _ = _enqueue(store, sender, target, "stop this delivery")
    claim = store.claim_next(target, claim_owner="old-owner", steer_capable=True)
    assert store.mark_transport_accepted(original.id, claim.claim_token or "")
    _request_stop_for_delivery(store, original.turn_id)

    result = store.requeue(original.id, claim.claim_token or "", "detached request ended", delay_seconds=0)
    assert result and result.status == "CANCELLED"
    assert result.next_retry_at is None
    assert store.claim_next(target, claim_owner="new-owner", steer_capable=True) is None


def test_explicit_stop_cancels_only_the_claimed_delivery(delivery_store):
    store, sender, target = delivery_store
    original, _ = _enqueue(store, sender, target, "stop this")
    following, _ = _enqueue(store, sender, target, "keep this")
    claim = store.claim_next(target, claim_owner="old-owner", steer_capable=True)
    _request_stop_for_delivery(store, original.turn_id)
    cancelled = store.cancel_stopped_claim(original.id, claim.claim_token or "")
    assert cancelled and cancelled.status == "CANCELLED"
    assert store.requeue(original.id, claim.claim_token or "", "late", delay_seconds=0) is None
    next_claim = store.claim_next(target, claim_owner="new-owner", steer_capable=True)
    assert next_claim and next_claim.id == following.id


def test_stopped_delivery_lease_recovery_does_not_requeue(delivery_store):
    store, sender, target = delivery_store
    original, _ = _enqueue(store, sender, target, "stop during app failure")
    claim = store.claim_next(target, claim_owner="old-owner", steer_capable=True)
    assert claim
    _request_stop_for_delivery(store, original.turn_id)
    with store.engine.begin() as conn:
        conn.execute(sa_text("UPDATE inter_bot_deliveries SET lease_expires_at=now() - interval '1 second' WHERE id=:id"), {"id": original.id})
    recovered = store.recover_expired()
    assert len(recovered) == 1 and recovered[0].status == "CANCELLED"
    assert store.claim_next(target, claim_owner="new-owner", steer_capable=True) is None


def test_queued_stopped_delivery_cannot_resurrect_when_claimed(delivery_store):
    store, sender, target = delivery_store
    original, _ = _enqueue(store, sender, target, "stop before reclaim")
    claim = store.claim_next(target, claim_owner="old-owner", steer_capable=True)
    with store.engine.begin() as conn:
        conn.execute(sa_text("UPDATE turn_logs SET status='streaming' WHERE id=:id"), {"id": original.turn_id})
    assert store.mark_transport_accepted(original.id, claim.claim_token or "")
    queued = store.requeue(original.id, claim.claim_token or "", "transient", delay_seconds=0)
    assert queued.status == QUEUED
    _request_stop_for_delivery(store, original.turn_id)
    assert store.claim_next(target, claim_owner="new-owner", steer_capable=True) is None
    assert store.get(original.id).status == "CANCELLED"


def test_accepted_run_beyond_retention_dead_letters_without_replay(delivery_store):
    store, sender, target = delivery_store
    original, _ = _enqueue(store, sender, target, "expired accepted run")
    claimed = store.claim_next(target, claim_owner="owner-one", steer_capable=False, lease_seconds=60)
    assert store.mark_transport_accepted(original.id, claimed.claim_token or "")
    with store.engine.begin() as conn:
        conn.execute(sa_text("""
            UPDATE inter_bot_deliveries
            SET lease_expires_at=now() - interval '1 second',
                transport_accepted_at=now() - interval '8 days'
            WHERE id=:id
        """), {"id": original.id})
        conn.execute(sa_text(
            "UPDATE turn_logs SET status='pending' WHERE id=:id"
        ), {"id": original.turn_id})

    recovered = store.recover_expired()[0]
    assert recovered.status == FAILED
    assert "seven-day recovery window" in (recovered.last_error or "")
    assert store.claim_next(target, claim_owner="owner-two", steer_capable=False) is None


def test_accepted_queued_run_beyond_retention_is_not_eligible(delivery_store):
    store, sender, target = delivery_store
    original, _ = _enqueue(store, sender, target, "queued accepted expiry")
    with store.engine.begin() as conn:
        conn.execute(sa_text("""
            UPDATE inter_bot_deliveries SET
                transport_accepted_at=now() - interval '8 days'
            WHERE id=:id
        """), {"id": original.id})

    assert target not in store.eligible_targets()
    failed = store.get(original.id)
    assert failed.status == FAILED
    assert "seven-day recovery window" in (failed.last_error or "")


def test_dispatcher_singleton_advisory_lock(delivery_store):
    store, _sender, _target = delivery_store
    lock_name = f"inter-bot-dispatcher-test-{uuid.uuid4().hex}"
    first = store.acquire_dispatcher_lock(lock_name)
    try:
        assert first is not None
        assert not first.in_transaction()
        assert store.acquire_dispatcher_lock(lock_name) is None
    finally:
        store.release_dispatcher_lock(first)
    second = store.acquire_dispatcher_lock(lock_name)
    assert second is not None
    assert not second.in_transaction()
    store.release_dispatcher_lock(second)


def test_delivery_api_shape_exposes_correlation_and_lifecycle(delivery_store):
    store, sender, target = delivery_store
    record, _ = _enqueue(store, sender, target, "shape", key="shape-key")
    api = record.to_api()

    assert api["delivery_id"] == record.id
    assert api["status"] == QUEUED
    assert api["task_id"] == "TASK-700"
    assert api["project_id"] == "project-1"
    assert api["message_kind"] == "READY"
    assert api["metadata"] == {"source": "test"}
    assert api["attempt_count"] == 0
    assert api["next_retry_at"] is None



def test_reset_without_history_rotates_once_and_reserves_new_thread(delivery_store):
    store, sender, target = delivery_store
    old_session = str(uuid.uuid4())
    with store.engine.begin() as conn:
        conn.execute(sa_text("""
            INSERT INTO sessions (id, bot_id, user_id, status, started_at)
            VALUES (:id, :bot, 'nick', 'active', now())
        """), {"id": old_session, "bot": target})
    record, _ = store.enqueue(
        sender_bot_id=sender,
        target_bot_id=target,
        message="clean callback",
        payload={
            "messages": [{"role": "user", "content": "clean callback"}],
            "bot_id": target,
            "user": "nick",
            "stream": False,
        },
        idempotency_key="clean-once",
        session_policy=SessionPolicy.RESET_WITHOUT_HISTORY,
        reset_reason="critical headroom",
    )

    claimed = store.claim_next(target, claim_owner="owner-one", steer_capable=True)
    assert claimed.status == DISPATCHING
    assert claimed.reset_status == "COMPLETED"
    assert claimed.old_session_id == old_session
    assert claimed.new_session_id
    assert claimed.new_session_id != old_session
    payload = store.payload(record.id)
    assert payload["session_id"] == claimed.new_session_id
    assert payload["inter_bot_session_policy"] == "reset_without_history"
    assert payload["inter_bot_seed_session_id"] == old_session

    requeued = store.requeue(
        record.id, claimed.claim_token or "", "transient", delay_seconds=0
    )
    assert requeued.status == QUEUED
    assert requeued.reset_status == "COMPLETED"
    reclaimed = store.claim_next(target, claim_owner="owner-two", steer_capable=True)
    assert reclaimed.new_session_id == claimed.new_session_id
    with store.engine.connect() as conn:
        sessions = conn.execute(sa_text("""
            SELECT id, status FROM sessions
            WHERE bot_id=:bot AND user_id='nick' ORDER BY started_at
        """), {"bot": target}).all()
        assert sessions == [(old_session, "archived"), (claimed.new_session_id, "active")]


def test_reset_policy_waits_for_active_turn_before_rotation(delivery_store):
    store, sender, target = delivery_store
    old_session = str(uuid.uuid4())
    active_turn = f"turn-active-reset-{uuid.uuid4().hex}"
    with store.engine.begin() as conn:
        conn.execute(sa_text("""
            INSERT INTO sessions (id, bot_id, user_id, status, started_at)
            VALUES (:id, :bot, 'nick', 'active', now())
        """), {"id": old_session, "bot": target})
        conn.execute(sa_text("""
            INSERT INTO turn_logs (
                id, created_at, path, stream, bot_id, user_id, status,
                user_prompt, response_text, ended_at
            ) VALUES (:id, now(), '/test', true, :bot, 'nick', 'streaming', '', '', NULL)
        """), {"id": active_turn, "bot": target})
    record, _ = store.enqueue(
        sender_bot_id=sender,
        target_bot_id=target,
        message="wait then reset",
        payload={"messages": [{"role": "user", "content": "wait"}], "bot_id": target},
        session_policy=SessionPolicy.RESET_RETAIN_HISTORY,
    )

    assert store.claim_next(target, claim_owner="owner", steer_capable=True) is None
    queued = store.get(record.id)
    assert queued.reset_status == "PENDING"
    assert queued.new_session_id is None
    with store.engine.begin() as conn:
        conn.execute(sa_text("UPDATE turn_logs SET status='ok', ended_at=now() WHERE id=:id"), {"id": active_turn})
    claimed = store.claim_next(target, claim_owner="owner", steer_capable=True)
    assert claimed.status == DISPATCHING
    assert claimed.reset_status == "COMPLETED"


def test_duplicate_idempotency_key_rejects_different_session_policy(delivery_store):
    store, sender, target = delivery_store
    store.enqueue(
        sender_bot_id=sender,
        target_bot_id=target,
        message="same logical send",
        payload={"messages": [{"role": "user", "content": "same"}], "bot_id": target},
        idempotency_key="same-reset-key",
        session_policy=SessionPolicy.CONTINUE,
    )
    with pytest.raises(ValueError, match="session_policy"):
        store.enqueue(
            sender_bot_id=sender,
            target_bot_id=target,
            message="retry with reset",
            payload={"messages": [{"role": "user", "content": "retry"}], "bot_id": target},
            idempotency_key="same-reset-key",
            session_policy=SessionPolicy.RESET_WITHOUT_HISTORY,
        )


def test_confirmed_overflow_rotates_same_delivery_once(delivery_store):
    store, sender, target = delivery_store
    old_session = str(uuid.uuid4())
    from llm_bawt.memory.postgresql import ensure_bot_partitions, partition_name, MESSAGES_PARENT
    table = partition_name(MESSAGES_PARENT, target)
    with store.engine.begin() as conn:
        ensure_bot_partitions(conn, target)
        conn.execute(sa_text("""
            INSERT INTO sessions (id, bot_id, user_id, status, started_at)
            VALUES (:id, :bot, 'nick', 'active', now())
        """), {"id": old_session, "bot": target})
    original, _ = _enqueue(store, sender, target, "overflow recovery")
    from llm_bawt.memory.postgresql import _sanitize_table_name
    with store.engine.begin() as conn:
        conn.execute(sa_text("""
            INSERT INTO messages (
                bot_id, id, role, content, timestamp, session_id
            ) VALUES (:bot, :id, 'user', 'overflow recovery', 1, :session)
        """), {
            "bot": _sanitize_table_name(target),
            "id": original.user_message_id,
            "session": old_session,
        })
    claimed = store.claim_next(target, claim_owner="owner-one", steer_capable=False)
    recovered = store.recover_context_overflow(
        original.id,
        claimed.claim_token or "",
        backend="claude-code",
        error="Your input exceeds the context window of this model",
    )

    assert recovered.status == QUEUED
    assert recovered.id == original.id
    assert recovered.user_message_id == original.user_message_id
    assert recovered.turn_id == original.turn_id
    assert recovered.overflow_recovery_count == 1
    assert recovered.session_policy == "reset_retain_history"
    assert recovered.old_session_id == old_session
    assert recovered.new_session_id
    payload = store.payload(original.id)
    assert payload["inter_bot_bridge_request_id"].endswith("_recovery_1")
    assert payload["inter_bot_seed_session_id"] == old_session
    with store.engine.connect() as conn:
        rows = conn.execute(sa_text("""
            SELECT id, session_id FROM messages
            WHERE bot_id=:bot AND id=:id
        """), {
            "bot": _sanitize_table_name(target),
            "id": original.user_message_id,
        }).all()
    assert rows == [(original.user_message_id, recovered.new_session_id)]

    reclaimed = store.claim_next(target, claim_owner="owner-two", steer_capable=False)
    assert reclaimed.id == original.id
    assert store.recover_context_overflow(
        original.id,
        reclaimed.claim_token or "",
        backend="claude-code",
        error="maximum context length exceeded again",
    ) is None
    current = store.get(original.id)
    assert current.overflow_recovery_count == 1
    assert current.new_session_id == recovered.new_session_id


def test_overflow_after_tool_side_effect_does_not_replay(delivery_store):
    store, sender, target = delivery_store
    original, _ = _enqueue(store, sender, target, "unsafe overflow")
    claimed = store.claim_next(target, claim_owner="owner", steer_capable=False)
    with store.engine.begin() as conn:
        conn.execute(sa_text("""
            INSERT INTO tool_call_records (
                turn_id, tool_name, arguments_json, result_text,
                iteration, created_at
            ) VALUES (
                :turn_id, 'Bash', '{}', 'ran', 1, now()
            )
        """), {"turn_id": original.turn_id})
    assert store.recover_context_overflow(
        original.id,
        claimed.claim_token or "",
        backend="claude-code",
        error="Your input exceeds the context window",
    ) is None
    current = store.get(original.id)
    assert current.status == DISPATCHING
    assert current.overflow_recovery_count == 0
    assert current.new_session_id is None


def test_submission_result_reports_requested_durable_mode(delivery_store):
    from llm_bawt.inter_bot_delivery import submission_result

    store, sender, target = delivery_store
    record, _ = _enqueue(store, sender, target, "shape mode")

    _status, default_result = submission_result(
        record,
        duplicate=False,
        target_exists=True,
        requested_delivery="steer_or_idle",
    )
    _status, idle_result = submission_result(
        record,
        duplicate=False,
        target_exists=True,
        requested_delivery="when_idle",
    )

    assert default_result["delivery"] == "steer_or_idle"
    assert "steer an active Claude turn" in default_result["note"]
    assert idle_result["delivery"] == "when_idle"
    assert "when that bot is idle" in idle_result["note"]
