"""Typed /new must commit the same exact bridge-backed receipt as scheduled /new."""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import text

from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
from llm_bawt.memory.session_reset_lineage import SessionResetLineage
from llm_bawt.service.inter_bot_dispatcher import InterBotDeliveryDispatcher
from llm_bawt.service.ordinary_turn_queue import OrdinaryTurnQueue
from llm_bawt.service.schemas import ChatCompletionRequest, ChatMessage
from llm_bawt.service.session_lifecycle.receipt import NewSessionReceipt, RESET_CONFIRMATION

pytest_plugins = ["tests.session_lifecycle_pg_fixtures"]
pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


def _admit(rig, *, explicit: bool = False):
    deliveries = InterBotDeliveryStore(rig.config)
    queue = OrdinaryTurnQueue(SimpleNamespace(engine=rig.engine), deliveries)
    request = ChatCompletionRequest(
        bot_id=rig.bot, user=rig.user, model="fixture", stream=True,
        session_id=rig.session_id if explicit else None,
        messages=[ChatMessage(role="user", content="/new")],
        user_message_id=str(uuid.uuid4()), assistant_message_id=str(uuid.uuid4()),
    )
    decision = queue.decide(request=request, bot_id=rig.bot, user_id=rig.user,
                            user_prompt="/new")
    return deliveries, deliveries.get(decision.delivery_id)


def _finish_bridge(rig, deliveries, record):
    claim = deliveries.claim_next(rig.bot, claim_owner="fixture", steer_capable=False)
    assert claim is not None and claim.id == record.id
    assert deliveries.mark_transport_accepted(claim.id, claim.claim_token)
    payload = deliveries.payload(claim.id)
    new_id = SessionResetLineage.delivery_successor_id(claim.id)
    with rig.engine.begin() as conn:
        SessionResetLineage.rotate(
            None, connection=conn, bot_id=rig.bot, user_id=rig.user,
            new_id=new_id, idempotency_key=claim.id, accept_pointer=True,
            working_memory_mode="pinned",
        )
        conn.execute(text("""
            INSERT INTO messages (bot_id, id, role, content, timestamp, session_id,
                                  author_entity_type, author_entity_id)
            VALUES (:bot, :id, 'user', '/new', 2, :session, 'user', :user)
        """), {"bot": rig.bot, "id": claim.user_message_id,
               "session": new_id, "user": rig.user})
        conn.execute(text("""
            INSERT INTO messages (bot_id, id, role, content, timestamp, session_id)
            VALUES (:bot, :id, 'assistant', 'bridge prose', 3, :session)
        """), {"bot": rig.bot, "id": payload["assistant_message_id"], "session": new_id})
        conn.execute(text("""
            UPDATE turn_logs SET status='ok', response_text='bridge prose',
                assistant_message_id=:assistant, ended_at=now(),
                trigger_message_id=:trigger WHERE id=:turn
        """), {"assistant": payload["assistant_message_id"],
               "trigger": claim.user_message_id, "turn": claim.turn_id})
    event = SimpleNamespace(
        run_id=payload["inter_bot_bridge_request_id"], provider="claude-code",
        trigger_message_id=claim.user_message_id,
        raw={"new_session_ack": {"thread_session_id": new_id,
                                 "clean_start": True, "provider_session_id": None}},
    )
    return claim, payload, new_id, event


def _dispatcher(deliveries):
    dispatcher = InterBotDeliveryDispatcher.__new__(InterBotDeliveryDispatcher)
    dispatcher.store = deliveries
    dispatcher._emit = AsyncMock()
    return dispatcher


def test_typed_command_admission_to_dispatcher_receipt(rig):
    deliveries, record = _admit(rig)
    raw = record.metadata["new_session_intent"]
    assert raw["origin"] == "chat_command"
    assert raw["expected_session_id"] == rig.session_id
    assert raw["author"] == {"entity_type": "user", "entity_id": rig.user,
                             "status": "canonical"}
    claim, payload, new_id, event = _finish_bridge(rig, deliveries, record)
    assert NewSessionReceipt.record_provider_ack(
        rig.engine, request_id=event.run_id, bot_id=rig.bot, event=event,
    )
    delivered = deliveries.mark_delivered(claim.id, claim.claim_token,
                                          response_model="fixture", response_chars=12)
    dispatcher = _dispatcher(deliveries)
    asyncio.run(dispatcher._finish_delivered(delivered))
    emitted = dispatcher._emit.await_args.args[0]
    assert emitted.status == "DELIVERED"
    receipt = deliveries.get(record.id).metadata["new_session_receipt"]
    assert receipt["new_session_id"] == new_id
    assert receipt["message_id"] == payload["assistant_message_id"]
    with rig.engine.connect() as conn:
        reply = conn.execute(text("""
            SELECT content FROM messages WHERE bot_id=:bot AND id=:id
        """), {"bot": rig.bot, "id": payload["assistant_message_id"]}).scalar_one()
        assert reply == RESET_CONFIRMATION
        assert conn.execute(text("""
            SELECT count(*) FROM new_session_event_outbox WHERE delivery_id=:id
        """), {"id": record.id}).scalar_one() == 1
    # A repeated observation is stable and does not write another outbox event.
    result = NewSessionReceipt.ensure(rig.engine, deliveries.get(record.id),
                                      NewSessionReceipt.typed_intent(deliveries.get(record.id)))
    assert result.receipt_message_id == receipt["message_id"]


def test_typed_command_missing_bridge_ack_fails_closed(rig):
    deliveries, record = _admit(rig)
    claim, payload, _new_id, _event = _finish_bridge(rig, deliveries, record)
    delivered = deliveries.mark_delivered(claim.id, claim.claim_token,
                                          response_model="fixture", response_chars=12)
    dispatcher = _dispatcher(deliveries)
    asyncio.run(dispatcher._finish_delivered(delivered))
    failed = deliveries.get(record.id)
    assert failed.status == "FAILED"
    assert failed.metadata["new_session_ambiguous"] is True
    assert "new_session_receipt" not in failed.metadata
    assert dispatcher._emit.await_args.args[0].status == "FAILED"
    with rig.engine.connect() as conn:
        assert conn.execute(text("""
            SELECT content FROM messages WHERE bot_id=:bot AND id=:id
        """), {"bot": rig.bot, "id": payload["assistant_message_id"]}).scalar_one() == "bridge prose"


@pytest.mark.parametrize("acknowledged", [True, False])
def test_crash_after_delivered_reconciles_without_replaying_command(rig, acknowledged):
    deliveries, record = _admit(rig)
    claim, _payload, _new_id, event = _finish_bridge(rig, deliveries, record)
    if acknowledged:
        assert NewSessionReceipt.record_provider_ack(
            rig.engine, request_id=event.run_id, bot_id=rig.bot, event=event,
        )
    deliveries.mark_delivered(claim.id, claim.claim_token,
                              response_model="fixture", response_chars=12)
    settled = NewSessionReceipt.reconcile_typed(rig.engine)
    assert len(settled) == 1 and settled[0].id == record.id
    assert settled[0].status == ("DELIVERED" if acknowledged else "FAILED")
    assert NewSessionReceipt.reconcile_typed(rig.engine) == []
    assert deliveries.claim_next(rig.bot, claim_owner="replay", steer_capable=False) is None


def test_explicit_archived_thread_command_has_no_reset_receipt_intent(rig):
    deliveries, record = _admit(rig, explicit=True)
    assert "new_session_intent" not in record.metadata
    assert NewSessionReceipt.typed_intent(record) is None
