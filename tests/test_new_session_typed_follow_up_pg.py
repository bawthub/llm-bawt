"""Typed /new with a task keeps its visible command and original task answer."""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
from llm_bawt.memory.session_reset_lineage import SessionResetLineage
from llm_bawt.service.ordinary_turn_queue import OrdinaryTurnQueue
from llm_bawt.service.schemas import ChatCompletionRequest, ChatMessage
from llm_bawt.service.session_lifecycle.receipt import NewSessionReceipt, RESET_CONFIRMATION
from tests.test_new_session_typed_pg import _dispatcher

pytest_plugins = ["tests.session_lifecycle_pg_fixtures"]
pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


def _finished_follow_up(rig, provider, command):
    """Prepare a claimed, rotated turn before its provider ack and receipt."""
    deliveries = InterBotDeliveryStore(rig.config)
    queue = OrdinaryTurnQueue(SimpleNamespace(engine=rig.engine), deliveries)
    request = ChatCompletionRequest(
        bot_id=rig.bot, user=rig.user, model="fixture", stream=True,
        messages=[ChatMessage(role="user", content=command)],
        user_message_id=str(uuid.uuid4()), assistant_message_id=str(uuid.uuid4()),
    )
    decision = queue.decide(request=request, bot_id=rig.bot, user_id=rig.user, user_prompt=command)
    record = deliveries.get(decision.delivery_id)
    assert record.metadata["new_session_intent"]["follow_up"] == "Keep this task"
    claim = deliveries.claim_next(rig.bot, claim_owner="typed-follow-up", steer_capable=False)
    assert claim.id == record.id
    assert deliveries.mark_transport_accepted(claim.id, claim.claim_token)
    payload = deliveries.payload(claim.id)
    assert payload["inter_bot_typed_follow_up"] is True
    successor = SessionResetLineage.delivery_successor_id(claim.id)
    with rig.engine.begin() as conn:
        SessionResetLineage.rotate(
            None, connection=conn, bot_id=rig.bot, user_id=rig.user,
            new_id=successor, idempotency_key=claim.id, accept_pointer=True,
        )
        conn.execute(text("""
            UPDATE sessions SET session_metadata=jsonb_set(session_metadata,
                '{agent_session_keys}', CAST(:keys AS jsonb)) WHERE id=:id
        """), {"id": successor, "keys": '{"' + provider.replace("-", "_") + '":"fresh-sdk"}'})
        conn.execute(text("""
            INSERT INTO messages (bot_id,id,role,content,timestamp,session_id,
                                  author_entity_type,author_entity_id)
            VALUES (:bot,:id,'user',:command,2,:session,'user',:user)
        """), {"bot": rig.bot, "id": claim.user_message_id,
               "command": command, "session": successor, "user": rig.user})
        conn.execute(text("""
            INSERT INTO messages (bot_id,id,role,content,timestamp,session_id)
            VALUES (:bot,:id,'assistant','task answer',3,:session)
        """), {"bot": rig.bot, "id": payload["assistant_message_id"], "session": successor})
        conn.execute(text("""
            UPDATE turn_logs SET status='ok',response_text='task answer',ended_at=now(),
                assistant_message_id=:assistant,trigger_message_id=:trigger WHERE id=:turn
        """), {"turn": claim.turn_id, "assistant": payload["assistant_message_id"],
               "trigger": claim.user_message_id})
    event = SimpleNamespace(
        run_id=payload["inter_bot_bridge_request_id"], provider=provider,
        trigger_message_id=claim.user_message_id,
        raw={"new_session_ack": {"thread_session_id": successor,
                                 "provider_session_id": "fresh-sdk", "clean_start": False}},
    )
    return deliveries, claim, payload, successor, event


@pytest.mark.parametrize("provider", ["claude-code", "codex"])
@pytest.mark.parametrize("command", ["/new Keep this task", "  /new\nKeep this task  "])
def test_typed_follow_up_receipt_preserves_visible_command_and_answer(rig, provider, command):
    deliveries, claim, payload, successor, event = _finished_follow_up(rig, provider, command)
    assert NewSessionReceipt.record_provider_ack(
        rig.engine, request_id=event.run_id, bot_id=rig.bot, event=event,
    )
    delivered = deliveries.mark_delivered(
        claim.id, claim.claim_token, response_model="fixture", response_chars=11,
    )
    dispatcher = _dispatcher(deliveries)
    asyncio.run(dispatcher._finish_delivered(delivered))
    settled = deliveries.get(claim.id)
    assert settled.status == "DELIVERED"
    receipt = settled.metadata["new_session_receipt"]
    assert receipt["message_id"] != payload["assistant_message_id"]
    asyncio.run(dispatcher._finish_delivered(settled))
    with rig.engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT role, content FROM messages WHERE bot_id=:bot AND session_id=:session
        """), {"bot": rig.bot, "session": successor}).all()
        assert sorted(rows) == sorted([
            ("user", command), ("assistant", "task answer"),
            ("assistant", RESET_CONFIRMATION),
        ])
        assert conn.execute(text("""
            SELECT count(*) FROM new_session_event_outbox WHERE delivery_id=:id
        """), {"id": claim.id}).scalar_one() == 1
    assert deliveries.claim_next(rig.bot, claim_owner="retry", steer_capable=False) is None


@pytest.mark.parametrize("fault", [
    "missing_ack", "wrong_provider_key", "wrong_answer", "stale_claim_predecessor",
    "wrong_payload_command",
])
def test_typed_follow_up_rejects_unverified_receipt(rig, fault):
    deliveries, claim, payload, successor, event = _finished_follow_up(
        rig, "claude-code", "/new Keep this task",
    )
    if fault == "wrong_provider_key":
        event.raw["new_session_ack"]["provider_session_id"] = "different-sdk"
    elif fault == "wrong_answer":
        with rig.engine.begin() as conn:
            conn.execute(text("""
                UPDATE turn_logs SET response_text='different answer' WHERE id=:turn
            """), {"turn": claim.turn_id})
    elif fault == "stale_claim_predecessor":
        with rig.engine.begin() as conn:
            conn.execute(text("""
                UPDATE inter_bot_deliveries SET metadata_json=jsonb_set(
                    metadata_json, '{new_session_claim_predecessor_id}',
                    to_jsonb(CAST(:prior AS text))) WHERE id=:id
            """), {"prior": str(uuid.uuid4()), "id": claim.id})
    elif fault == "wrong_payload_command":
        with rig.engine.begin() as conn:
            conn.execute(text("""
                UPDATE inter_bot_deliveries SET payload_json=jsonb_set(
                    payload_json, '{messages,0,content}',
                    to_jsonb(CAST(:command AS text))) WHERE id=:id
            """), {"command": "/new A different task", "id": claim.id})
    if fault != "missing_ack":
        assert NewSessionReceipt.record_provider_ack(
            rig.engine, request_id=event.run_id, bot_id=rig.bot, event=event,
        )
    delivered = deliveries.mark_delivered(
        claim.id, claim.claim_token, response_model="fixture", response_chars=11,
    )
    asyncio.run(_dispatcher(deliveries)._finish_delivered(delivered))
    failed = deliveries.get(claim.id)
    assert failed.status == "FAILED"
    assert failed.metadata["new_session_ambiguous"] is True
    assert "new_session_receipt" not in failed.metadata
    assert deliveries.claim_next(rig.bot, claim_owner="retry", steer_capable=False) is None
    with rig.engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT content FROM messages WHERE bot_id=:bot AND session_id=:session
        """), {"bot": rig.bot, "session": successor}).scalars().all()
        assert sorted(rows) == sorted(["/new Keep this task", "task answer"])
        assert conn.execute(text("""
            SELECT count(*) FROM new_session_event_outbox WHERE delivery_id=:id
        """), {"id": claim.id}).scalar_one() == 0


def test_newspaper_is_not_stamped_as_typed_reset(rig):
    deliveries = InterBotDeliveryStore(rig.config)
    queue = OrdinaryTurnQueue(SimpleNamespace(engine=rig.engine), deliveries)
    command = "/newspaper is literal task text"
    request = ChatCompletionRequest(
        bot_id=rig.bot, user=rig.user, model="fixture",
        messages=[ChatMessage(role="user", content=command)],
    )
    decision = queue.decide(request=request, bot_id=rig.bot, user_id=rig.user, user_prompt=command)
    record = deliveries.get(decision.delivery_id)
    assert "new_session_intent" not in record.metadata
    assert not deliveries.payload(record.id).get("inter_bot_typed_follow_up")
