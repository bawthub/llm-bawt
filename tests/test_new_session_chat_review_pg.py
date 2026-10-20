"""Independent negative evidence matrix for stateless-chat completion."""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
from llm_bawt.service.ordinary_turn_queue import OrdinaryTurnQueue
from llm_bawt.service.schemas import ChatCompletionRequest, ChatMessage
from llm_bawt.service.session_lifecycle.receipt import NewSessionReceipt, RESET_CONFIRMATION
from tests.test_new_session_typed_pg import _dispatcher, _finish_bridge

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.fixture(params=["/new", "/new Keep this task"])
def chat_completed(rig, request):
    command = request.param
    deliveries = InterBotDeliveryStore(rig.config)
    queue = OrdinaryTurnQueue(SimpleNamespace(engine=rig.engine), deliveries)
    req = ChatCompletionRequest(
        bot_id=rig.bot, user=rig.user, model="fixture", stream=True,
        messages=[ChatMessage(role="user", content=command)],
        user_message_id=str(uuid.uuid4()), assistant_message_id=str(uuid.uuid4()),
    )
    decision = queue.decide(request=req, bot_id=rig.bot, user_id=rig.user,
                            user_prompt=command, chat_reset=True)
    record = deliveries.get(decision.delivery_id)
    # This fixture supplies persisted output only, not any provider event.
    claim, payload, successor, _unused_ack = _finish_bridge(rig, deliveries, record)
    answer = RESET_CONFIRMATION if command == "/new" else "unchanged task answer"
    with rig.engine.begin() as conn:
        conn.execute(text("UPDATE messages SET content=:command WHERE bot_id=:bot AND id=:id"),
                     {"bot": rig.bot, "id": claim.user_message_id, "command": command})
        conn.execute(text("""
            UPDATE messages SET content=:answer, author_entity_type='bot', author_entity_id=:bot
            WHERE bot_id=:bot AND id=:id
        """), {"bot": rig.bot, "id": payload["assistant_message_id"], "answer": answer})
        conn.execute(text("UPDATE turn_logs SET response_text=:answer WHERE id=:id"),
                     {"id": claim.turn_id, "answer": answer})
    delivered = deliveries.mark_delivered(claim.id, claim.claim_token,
                                          response_model="fixture", response_chars=len(answer))
    return SimpleNamespace(rig=rig, store=deliveries, record=delivered, successor=successor,
                           assistant=payload["assistant_message_id"], answer=answer)


@pytest.mark.parametrize("fault", [
    "missing_answer", "wrong_author", "wrong_role", "wrong_session", "failed_turn",
    "partial_turn", "wrong_user", "wrong_trigger", "stale_provider_key", "wrong_lineage",
])
def test_chat_receipt_rejects_incomplete_or_cross_thread_evidence(chat_completed, fault):
    state = chat_completed
    rig = state.rig
    sql = {
        "missing_answer": "DELETE FROM messages WHERE bot_id=:bot AND id=:assistant",
        "wrong_author": "UPDATE messages SET author_entity_id='someone-else' WHERE bot_id=:bot AND id=:assistant",
        "wrong_role": "UPDATE messages SET role='user' WHERE bot_id=:bot AND id=:assistant",
        "wrong_session": "UPDATE messages SET session_id=:old WHERE bot_id=:bot AND id=:assistant",
        "failed_turn": "UPDATE turn_logs SET status='error', error_text='failed' WHERE id=:turn",
        "partial_turn": "UPDATE turn_logs SET ended_at=NULL WHERE id=:turn",
        "wrong_user": "UPDATE turn_logs SET user_id='someone-else' WHERE id=:turn",
        "wrong_trigger": "UPDATE turn_logs SET trigger_message_id='wrong' WHERE id=:turn",
        "stale_provider_key": "UPDATE sessions SET session_metadata=session_metadata || '{\"agent_session_keys\":{\"codex\":\"stale\"}}'::jsonb WHERE id=:new",
        "wrong_lineage": "UPDATE sessions SET session_metadata=session_metadata || '{\"rotation_delivery_id\":\"wrong\"}'::jsonb WHERE id=:new",
    }[fault]
    with rig.engine.begin() as conn:
        conn.execute(text(sql), {"bot": rig.bot, "assistant": state.assistant,
                                "old": rig.session_id, "new": state.successor,
                                "turn": state.record.turn_id})
    asyncio.run(_dispatcher(state.store)._finish_delivered(state.record))
    current = state.store.get(state.record.id)
    assert current.status == "FAILED"
    assert current.metadata["new_session_ambiguous"] is True
    assert "new_session_receipt" not in current.metadata
    assert state.store.claim_next(rig.bot, claim_owner="must-not-replay", steer_capable=False) is None
    with rig.engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM new_session_event_outbox WHERE delivery_id=:id"),
                            {"id": current.id}).scalar_one() == 0


def test_chat_crash_reconciliation_is_idempotent_and_keeps_task_answer(chat_completed):
    state = chat_completed
    rig = state.rig
    # The dispatcher died after DELIVERED but before writing the receipt.
    repaired = NewSessionReceipt.reconcile_typed(rig.engine)
    assert any(row.id == state.record.id for row in repaired)
    current = state.store.get(state.record.id)
    assert current.status == "DELIVERED"
    receipt = current.metadata["new_session_receipt"]
    assert receipt["provider_session_id"] is None
    assert "new_session_provider_ack" not in current.metadata
    assert NewSessionReceipt.reconcile_typed(rig.engine) == []
    with rig.engine.connect() as conn:
        assert conn.execute(text("SELECT content FROM messages WHERE bot_id=:bot AND id=:id"),
                            {"bot": rig.bot, "id": state.assistant}).scalar_one() == state.answer
        assert conn.execute(text("SELECT count(*) FROM new_session_event_outbox WHERE delivery_id=:id"),
                            {"id": current.id}).scalar_one() == 1
