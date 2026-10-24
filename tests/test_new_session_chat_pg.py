"""Stateless chat /new needs a claimed rotation and a transcript-backed receipt."""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sqlalchemy import text

from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
from llm_bawt.memory.session_reset_lineage import SessionResetLineage
from llm_bawt.models.message import Message
from llm_bawt.service.ordinary_turn_queue import OrdinaryTurnQueue
from llm_bawt.service.schemas import ChatCompletionRequest, ChatMessage
from llm_bawt.service.session_lifecycle.execution import NewSessionExecution
from llm_bawt.service.session_lifecycle.receipt import RESET_CONFIRMATION
from tests.test_new_session_typed_pg import _dispatcher

pytest_plugins = ["tests.session_lifecycle_pg_fixtures"]
pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.mark.parametrize("command,answer", [
    ("/new", RESET_CONFIRMATION),
    ("  /new\nKeep this task  ", "task answer"),
])
def test_chat_claim_rotation_and_exact_receipt(rig, command, answer):
    deliveries = InterBotDeliveryStore(rig.config)
    queue = OrdinaryTurnQueue(SimpleNamespace(engine=rig.engine), deliveries)
    request = ChatCompletionRequest(
        bot_id=rig.bot, user=rig.user, model="fixture", stream=True,
        messages=[ChatMessage(role="user", content=command)],
        user_message_id=str(uuid.uuid4()), assistant_message_id=str(uuid.uuid4()),
    )
    decision = queue.decide(
        request=request, bot_id=rig.bot, user_id=rig.user,
        user_prompt=command, chat_reset=True,
    )
    claim = deliveries.claim_next(rig.bot, claim_owner="chat-reset", steer_capable=False)
    assert claim and claim.id == decision.delivery_id
    assert claim.metadata["new_session_execution"] == "chat"
    assert claim.metadata["new_session_claim_predecessor_id"] == rig.session_id
    payload = deliveries.payload(claim.id)
    claimed_request = ChatCompletionRequest.model_validate({
        **payload, "inter_bot_claim_token": claim.claim_token,
    })

    def get_session(sid):
        with rig.engine.connect() as conn:
            row = conn.execute(text("""
                SELECT bot_id, user_id, status, session_metadata
                FROM sessions WHERE id=:id
            """), {"id": sid}).mappings().one_or_none()
            return dict(row) if row is not None else None

    def get_active_session(**kwargs):
        with rig.engine.connect() as conn:
            sid = conn.execute(text("""
                SELECT id FROM sessions WHERE bot_id=:bot AND user_id=:user
                  AND status='active' AND ended_at IS NULL
            """), {"bot": kwargs["bot_id"], "user": kwargs["user_id"]}).scalar_one_or_none()
            return {"id": sid} if sid else None

    backend = SimpleNamespace(get_session=get_session, get_active_session=get_active_session)
    bawt = SimpleNamespace(history_manager=SimpleNamespace(_db_backend=backend))

    def rotate(_bawt, _bot_id, *, idempotency_key, working_memory_mode):
        successor = SessionResetLineage.delivery_successor_id(idempotency_key)
        SessionResetLineage.rotate(
            rig.engine, bot_id=rig.bot, user_id=rig.user, new_id=successor,
            idempotency_key=idempotency_key, accept_pointer=True,
            working_memory_mode=working_memory_mode,
        )
        return successor

    service = SimpleNamespace(
        _maybe_summarize_on_new=Mock(), _rotate_chat_session=Mock(side_effect=rotate),
    )
    plan = NewSessionExecution.prepare_chat(
        service, bawt, claimed_request, bot_id=rig.bot,
        user_id=rig.user, user_prompt=command,
    )
    assert plan.session_id == SessionResetLineage.delivery_successor_id(claim.id)
    assert plan.follow_up == ("Keep this task" if answer != RESET_CONFIRMATION else None)
    assert service._maybe_summarize_on_new.call_args.kwargs["thread_binding"] == {
        "thread_session_id": rig.session_id,
    }
    if plan.follow_up:
        original = Message(role="user", content=command)
        assert NewSessionExecution.bridge_messages(
            [original], plan.follow_up, command,
        )[-1].content == "Keep this task"
        assert original.content == command
    # A retry reuses the same transition; it cannot summarize the successor.
    assert NewSessionExecution.prepare_chat(
        service, bawt, claimed_request, bot_id=rig.bot,
        user_id=rig.user, user_prompt=command,
    ) == plan
    service._maybe_summarize_on_new.assert_called_once()

    assert deliveries.mark_transport_accepted(claim.id, claim.claim_token)
    with rig.engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO messages (bot_id, id, role, content, timestamp, session_id,
                                  author_entity_type, author_entity_id)
            VALUES (:bot, :id, 'user', :command, 2, :session, 'user', :user)
        """), {"bot": rig.bot, "id": claim.user_message_id,
               "command": command, "session": plan.session_id, "user": rig.user})
        conn.execute(text("""
            INSERT INTO messages (bot_id, id, role, content, timestamp, session_id,
                                  author_entity_type, author_entity_id)
            VALUES (:bot, :id, 'assistant', :answer, 3, :session, 'bot', :bot)
        """), {"bot": rig.bot, "id": payload["assistant_message_id"],
               "answer": answer, "session": plan.session_id})
        conn.execute(text("""
            UPDATE turn_logs SET status='ok', response_text=:answer, ended_at=now(),
                assistant_message_id=:assistant, trigger_message_id=:trigger
            WHERE id=:turn
        """), {"answer": answer, "assistant": payload["assistant_message_id"],
               "trigger": claim.user_message_id, "turn": claim.turn_id})
    delivered = deliveries.mark_delivered(
        claim.id, claim.claim_token, response_model="fixture", response_chars=len(answer),
    )
    dispatcher = _dispatcher(deliveries)
    asyncio.run(dispatcher._finish_delivered(delivered))
    settled = deliveries.get(claim.id)
    assert settled.status == "DELIVERED", settled.last_error
    receipt = settled.metadata["new_session_receipt"]
    assert receipt["provider_session_id"] is None
    assert receipt["new_session_id"] == plan.session_id
    assert receipt["message_id"] != payload["assistant_message_id"] if plan.follow_up else (
        receipt["message_id"] == payload["assistant_message_id"]
    )
    asyncio.run(dispatcher._finish_delivered(settled))
    with rig.engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT content FROM messages WHERE bot_id=:bot AND session_id=:session
            ORDER BY timestamp
        """), {"bot": rig.bot, "session": plan.session_id}).scalars().all()
        assert rows.count(command) == 1
        assert rows.count(answer) == 1
        if plan.follow_up:
            assert rows.count(RESET_CONFIRMATION) == 1
        assert conn.execute(text("""
            SELECT count(*) FROM new_session_event_outbox WHERE delivery_id=:id
        """), {"id": claim.id}).scalar_one() == 1
    assert deliveries.claim_next(rig.bot, claim_owner="retry", steer_capable=False) is None


@pytest.mark.parametrize("fault", ["wrong_answer", "wrong_confirm", "fake_provider_ack"])
def test_chat_receipt_refuses_missing_chat_proof(rig, fault):
    deliveries = InterBotDeliveryStore(rig.config)
    queue = OrdinaryTurnQueue(SimpleNamespace(engine=rig.engine), deliveries)
    command = "/new"
    request = ChatCompletionRequest(
        bot_id=rig.bot, user=rig.user, model="fixture", stream=True,
        messages=[ChatMessage(role="user", content=command)],
        user_message_id=str(uuid.uuid4()), assistant_message_id=str(uuid.uuid4()),
    )
    decision = queue.decide(
        request=request, bot_id=rig.bot, user_id=rig.user,
        user_prompt=command, chat_reset=True,
    )
    claim = deliveries.claim_next(rig.bot, claim_owner="chat-negative", steer_capable=False)
    assert claim and claim.id == decision.delivery_id
    payload = deliveries.payload(claim.id)
    successor = SessionResetLineage.delivery_successor_id(claim.id)
    SessionResetLineage.rotate(
        rig.engine, bot_id=rig.bot, user_id=rig.user, new_id=successor,
        idempotency_key=claim.id, accept_pointer=True, working_memory_mode="pinned",
    )
    answer ="wrong confirmation" if fault == "wrong_confirm" else RESET_CONFIRMATION
    with rig.engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO messages (bot_id, id, role, content, timestamp, session_id,
                                  author_entity_type, author_entity_id)
            VALUES (:bot, :id, 'user', :command, 2, :session, 'user', :user)
        """), {"bot": rig.bot, "id": claim.user_message_id,
               "command": command, "session": successor, "user": rig.user})
        conn.execute(text("""
            INSERT INTO messages (bot_id, id, role, content, timestamp, session_id,
                                  author_entity_type, author_entity_id)
            VALUES (:bot, :id, 'assistant', :answer, 3, :session, 'bot', :bot)
        """), {"bot": rig.bot, "id": payload["assistant_message_id"],
               "answer": answer, "session": successor})
        conn.execute(text("""
            UPDATE turn_logs SET status='ok', response_text=:answer, ended_at=now(),
                assistant_message_id=:assistant, trigger_message_id=:trigger
            WHERE id=:turn
        """), {"answer": "different turn answer" if fault == "wrong_answer" else answer,
               "assistant": payload["assistant_message_id"],
               "trigger": claim.user_message_id, "turn": claim.turn_id})
        if fault == "fake_provider_ack":
            conn.execute(text("""
                UPDATE inter_bot_deliveries SET metadata_json=jsonb_set(
                    metadata_json, '{new_session_provider_ack}',
                    '{"provider":"claude-code","provider_session_id":"fake"}'::jsonb)
                WHERE id=:id
            """), {"id": claim.id})
    assert deliveries.mark_transport_accepted(claim.id, claim.claim_token)
    delivered = deliveries.mark_delivered(claim.id, claim.claim_token,
                                          response_model="fixture", response_chars=len(answer))
    asyncio.run(_dispatcher(deliveries)._finish_delivered(delivered))
    settled = deliveries.get(claim.id)
    assert settled.status == "FAILED"
    assert settled.metadata["new_session_ambiguous"] is True
    assert "new_session_receipt" not in settled.metadata
    with rig.engine.connect() as conn:
        assert conn.execute(text("""
            SELECT count(*) FROM new_session_event_outbox WHERE delivery_id=:id
        """), {"id": claim.id}).scalar_one() == 0
