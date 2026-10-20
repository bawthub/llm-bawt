"""Independent bot-policy receipts: rotation is not provider completion."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import text

from llm_bawt.message_authorship import AuthorReference
from llm_bawt.service.inter_bot_dispatcher import InterBotDeliveryDispatcher
from llm_bawt.service.session_lifecycle.receipt import NewSessionReceipt, RESET_CONFIRMATION

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]
POLICIES = ["reset_retain_history", "reset_without_history"]
TASK_TEXT = "Perform this supplied task exactly once."
TASK_REPLY = "The actual task answer must survive reset confirmation."


def claim_policy(rig, lifecycle, policy):
    record, _ = lifecycle.store.enqueue(
        sender_bot_id="test-sender", target_bot_id=rig.bot,
        author=AuthorReference.bot("test-sender"), message=TASK_TEXT,
        session_policy=policy, idempotency_key=f"policy-review:{rig.bot}",
        payload={"user": rig.user, "messages": [{"role": "user", "content": TASK_TEXT}]},
    )
    claim = lifecycle.store.claim_next(rig.bot, claim_owner="review", steer_capable=False)
    assert claim is not None and claim.id == record.id
    assert claim.old_session_id == rig.session_id and claim.new_session_id != rig.session_id
    return claim


def finish_follow_up(rig, lifecycle, claim, *, backend="claude-code", provider_id="sdk-fresh"):
    assert lifecycle.store.mark_transport_accepted(claim.id, claim.claim_token)
    payload = lifecycle.store.payload(claim.id)
    with rig.engine.begin() as conn:
        conn.execute(text("""
            UPDATE sessions SET session_metadata=jsonb_set(
                session_metadata, '{agent_session_keys}', CAST(:keys AS jsonb))
            WHERE id=:id
        """), {"id": claim.new_session_id, "keys": json.dumps({backend.replace("-", "_"): provider_id})})
        conn.execute(text("""
            INSERT INTO messages (bot_id,id,role,content,timestamp,session_id,
                                  author_entity_type,author_entity_id)
            VALUES (:bot,:id,'user',:task,2,:session,'bot','test-sender')
        """), {"bot": rig.bot, "id": claim.user_message_id,
               "task": TASK_TEXT, "session": claim.new_session_id})
        conn.execute(text("""
            INSERT INTO messages (bot_id,id,role,content,timestamp,session_id)
            VALUES (:bot,:id,'assistant',:reply,3,:session)
        """), {"bot": rig.bot, "id": payload["assistant_message_id"],
               "reply": TASK_REPLY, "session": claim.new_session_id})
        conn.execute(text("""
            UPDATE turn_logs SET status='ok',ended_at=now(),response_text=:reply,
                assistant_message_id=:assistant,trigger_message_id=:trigger WHERE id=:turn
        """), {"reply": TASK_REPLY, "assistant": payload["assistant_message_id"],
               "trigger": claim.user_message_id, "turn": claim.turn_id})
    event = SimpleNamespace(
        run_id=payload["inter_bot_bridge_request_id"], provider=backend,
        session_key=rig.bot, trigger_message_id=claim.user_message_id,
        raw={"new_session_ack": {"thread_session_id": claim.new_session_id,
                                 "provider_session_id": provider_id, "clean_start": False}},
    )
    return payload, event


def settle(lifecycle, claim):
    delivered = lifecycle.store.mark_delivered(
        claim.id, claim.claim_token, response_model="fixture", response_chars=len(TASK_REPLY),
    )
    dispatcher = InterBotDeliveryDispatcher.__new__(InterBotDeliveryDispatcher)
    dispatcher.store = lifecycle.store
    dispatcher._emit = AsyncMock()
    asyncio.run(dispatcher._finish_delivered(delivered))
    return lifecycle.store.get(claim.id), dispatcher


@pytest.mark.parametrize("policy", POLICIES)
def test_policy_rotation_is_not_yet_completed_reset(rig, real_lifecycle, policy):
    claim = claim_policy(rig, real_lifecycle, policy)
    assert claim.reset_status != "COMPLETED"
    assert "new_session_receipt" not in claim.metadata


@pytest.mark.parametrize("policy", POLICIES)
def test_policy_receipt_is_separate_and_task_reply_is_unchanged(rig, real_lifecycle, policy):
    claim = claim_policy(rig, real_lifecycle, policy)
    payload, event = finish_follow_up(rig, real_lifecycle, claim)
    assert NewSessionReceipt.record_provider_ack(
        rig.engine, request_id=event.run_id, bot_id=rig.bot, event=event,
    )
    completed, dispatcher = settle(real_lifecycle, claim)
    assert completed.status == "DELIVERED" and completed.reset_status == "COMPLETED"
    receipt = completed.metadata["new_session_receipt"]
    assert receipt["message_id"] != payload["assistant_message_id"]
    assert receipt["new_session_id"] == claim.new_session_id
    # Re-observation must not duplicate either the supplied task or confirmation.
    asyncio.run(dispatcher._finish_delivered(completed))
    with rig.engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id,role,content,author_entity_type,author_entity_id
            FROM messages WHERE bot_id=:bot AND session_id=:session
        """), {"bot": rig.bot, "session": claim.new_session_id}).mappings().all()
        assert len(rows) == 3
        by_id = {row["id"]: row for row in rows}
        assert by_id[claim.user_message_id]["content"] == TASK_TEXT
        assert by_id[payload["assistant_message_id"]]["content"] == TASK_REPLY
        confirmation = by_id[receipt["message_id"]]
        assert confirmation["role"] == "assistant" and confirmation["content"] == RESET_CONFIRMATION
        assert (confirmation["author_entity_type"], confirmation["author_entity_id"]) == ("bot", rig.bot)
        assert conn.execute(text("SELECT response_text FROM turn_logs WHERE id=:id"),
                            {"id": claim.turn_id}).scalar_one() == TASK_REPLY
        assert conn.execute(text("SELECT count(*) FROM new_session_event_outbox WHERE delivery_id=:id"),
                            {"id": claim.id}).scalar_one() == 1
    assert real_lifecycle.store.claim_next(rig.bot, claim_owner="replay", steer_capable=False) is None


@pytest.mark.parametrize("policy", POLICIES)
def test_policy_missing_provider_ack_is_not_completed_or_replayed(rig, real_lifecycle, policy):
    claim = claim_policy(rig, real_lifecycle, policy)
    payload, _event = finish_follow_up(rig, real_lifecycle, claim)
    settled, dispatcher = settle(real_lifecycle, claim)
    assert settled.status == "FAILED" and settled.reset_status != "COMPLETED"
    assert settled.metadata.get("new_session_ambiguous") is True
    assert "new_session_receipt" not in settled.metadata
    assert dispatcher._emit.await_args.args[0].status == "FAILED"
    assert real_lifecycle.store.claim_next(rig.bot, claim_owner="replay", steer_capable=False) is None
    with rig.engine.connect() as conn:
        assert conn.execute(text("SELECT content FROM messages WHERE bot_id=:bot AND id=:id"),
                            {"bot": rig.bot, "id": payload["assistant_message_id"]}).scalar_one() == TASK_REPLY
