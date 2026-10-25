"""Policy reset receipts survive dispatch crashes and reject mismatched binding."""
from __future__ import annotations

from sqlalchemy import text
import pytest

from llm_bawt.message_authorship import AuthorReference
from llm_bawt.service.session_lifecycle.intent import Admission, Continuity, NewSessionIntent, Origin
from llm_bawt.service.session_lifecycle.policy_receipt import BotPolicyReceipt
from llm_bawt.service.session_lifecycle.receipt import NewSessionReceipt
from tests.test_new_session_policy_review_pg import claim_policy, finish_follow_up, TASK_REPLY

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.mark.parametrize("continuity", [Continuity.RETAIN, Continuity.CLEAN])
def test_bot_follow_up_admits_through_shared_lifecycle(rig, real_lifecycle, continuity):
    intent = NewSessionIntent(
        bot_id=rig.bot, user_id=rig.user, author=AuthorReference.bot("test-sender"),
        continuity=continuity, admission=Admission.WAIT,
        origin=Origin.BOT_DELIVERY, idempotency_key=f"lifecycle:{rig.bot}",
        follow_up="Perform one task",
    )
    queued, reason = real_lifecycle._admit(intent)
    assert reason is None and queued.status == "QUEUED"
    assert real_lifecycle._admit(intent)[0].id == queued.id
    assert real_lifecycle.store.payload(queued.id)["messages"][-1]["content"] == "Perform one task"
    claimed = real_lifecycle.store.claim_next(rig.bot, claim_owner="policy", steer_capable=False)
    assert claimed.id == queued.id and claimed.reset_status == "ROTATED"
    assert claimed.old_session_id == rig.session_id
    assert "new_session_receipt" not in claimed.metadata


@pytest.mark.parametrize("policy", ["reset_retain_history", "reset_without_history"])
def test_crashed_dispatch_reconciles_only_verified_ack(rig, real_lifecycle, policy):
    claim = claim_policy(rig, real_lifecycle, policy)
    payload, event = finish_follow_up(rig, real_lifecycle, claim)
    assert NewSessionReceipt.record_provider_ack(
        rig.engine, request_id=event.run_id, bot_id=rig.bot, event=event,
    )
    delivered = real_lifecycle.store.mark_delivered(
        claim.id, claim.claim_token, response_model="fixture", response_chars=len(TASK_REPLY),
    )
    assert delivered.reset_status == "ROTATED"
    settled = BotPolicyReceipt.reconcile(rig.engine)
    assert len(settled) == 1
    assert settled[0].id == claim.id and settled[0].reset_status == "COMPLETED"
    assert settled[0].metadata["new_session_receipt"]["message_id"] != payload["assistant_message_id"]
    assert BotPolicyReceipt.reconcile(rig.engine) == []


@pytest.mark.parametrize("policy", ["reset_retain_history", "reset_without_history"])
def test_mismatched_task_answer_fails_closed(rig, real_lifecycle, policy):
    claim = claim_policy(rig, real_lifecycle, policy)
    payload, event = finish_follow_up(rig, real_lifecycle, claim)
    assert NewSessionReceipt.record_provider_ack(
        rig.engine, request_id=event.run_id, bot_id=rig.bot, event=event,
    )
    with rig.engine.begin() as conn:
        conn.execute(text("""
            UPDATE turn_logs SET response_text='different answer' WHERE id=:turn
        """), {"turn": claim.turn_id})
    real_lifecycle.store.mark_delivered(
        claim.id, claim.claim_token, response_model="fixture", response_chars=len(TASK_REPLY),
    )
    settled = BotPolicyReceipt.reconcile(rig.engine)
    assert len(settled) == 1 and settled[0].status == "FAILED"
    assert settled[0].metadata["new_session_ambiguous"] is True
    assert "new_session_receipt" not in settled[0].metadata
    with rig.engine.connect() as conn:
        assert conn.execute(text("""
            SELECT content FROM messages WHERE bot_id=:bot AND id=:id
        """), {"bot": rig.bot, "id": payload["assistant_message_id"]}).scalar_one() == TASK_REPLY


@pytest.mark.parametrize("policy", ["reset_retain_history", "reset_without_history"])
def test_wrong_provider_key_fails_closed_without_receipt(rig, real_lifecycle, policy):
    claim = claim_policy(rig, real_lifecycle, policy)
    payload, event = finish_follow_up(rig, real_lifecycle, claim)
    assert NewSessionReceipt.record_provider_ack(
        rig.engine, request_id=event.run_id, bot_id=rig.bot, event=event,
    )
    with rig.engine.begin() as conn:
        conn.execute(text("""
            UPDATE sessions SET session_metadata=jsonb_set(
                session_metadata, '{agent_session_keys,claude_code}', '"other-provider-session"'::jsonb)
            WHERE id=:id
        """), {"id": claim.new_session_id})
    real_lifecycle.store.mark_delivered(
        claim.id, claim.claim_token, response_model="fixture", response_chars=len(TASK_REPLY),
    )
    settled = BotPolicyReceipt.reconcile(rig.engine)
    assert len(settled) == 1
    assert settled[0].status == "FAILED"
    assert settled[0].metadata["new_session_ambiguous"] is True
    assert "new_session_receipt" not in settled[0].metadata
    with rig.engine.connect() as conn:
        assert conn.execute(text("""
            SELECT count(*) FROM new_session_event_outbox WHERE delivery_id=:id
        """), {"id": claim.id}).scalar_one() == 0
        assert conn.execute(text("""
            SELECT content FROM messages WHERE bot_id=:bot AND id=:id
        """), {"bot": rig.bot, "id": payload["assistant_message_id"]}).scalar_one() == TASK_REPLY


def test_public_bot_send_preserves_formatted_task_through_exact_receipt(rig, real_lifecycle, monkeypatch):
    """The public endpoint must feed the same policy claim/receipt transaction."""
    import asyncio
    import json
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from llm_bawt.service.inter_bot_dispatcher import InterBotDeliveryDispatcher
    from llm_bawt.service.routes import inter_bot_deliveries as route
    from llm_bawt.service.session_lifecycle.receipt import RESET_CONFIRMATION

    bot = SimpleNamespace(slug=rig.bot, agent_backend="claude-code", harness="claude-code")
    from llm_bawt import inter_bot_send as send

    monkeypatch.setattr(send, "BotManager", lambda _config: SimpleNamespace(get_bot=lambda _slug: bot))
    monkeypatch.setattr(route, "get_service", lambda: real_lifecycle.service)
    monkeypatch.setattr(send, "resolve_delivery_user", lambda _config, _user: rig.user)
    real_lifecycle.dispatcher._emit = AsyncMock()
    app = FastAPI()
    app.include_router(route.router)
    response = TestClient(app).post("/v1/inter-bot-deliveries", json={
        "sender_bot_id": "test-sender", "target_bot_id": rig.bot,
        "message": "Perform one task", "session_policy": "reset_without_history",
        "idempotency_key": f"public-reset:{rig.bot}",
    })
    assert response.status_code == 202, response.text
    queued = real_lifecycle.store.get(response.json()["delivery_id"])
    assert queued.metadata["new_session_intent"]["origin"] == Origin.BOT_DELIVERY
    assert queued.metadata["new_session_intent"]["follow_up"] == "Perform one task"
    claimed = real_lifecycle.store.claim_next(rig.bot, claim_owner="public", steer_capable=False)
    assert claimed.id == queued.id and claimed.reset_status == "ROTATED"
    assert claimed.old_session_id == rig.session_id
    assert real_lifecycle.store.mark_transport_accepted(claimed.id, claimed.claim_token)
    payload = real_lifecycle.store.payload(claimed.id)
    supplied = "Message from bot 'test-sender': Perform one task"
    assert payload["messages"][-1]["content"] == supplied
    with rig.engine.begin() as conn:
        conn.execute(text("""
            UPDATE sessions SET session_metadata=jsonb_set(
                session_metadata, '{agent_session_keys}', CAST(:keys AS jsonb)) WHERE id=:session
        """), {"session": claimed.new_session_id, "keys": json.dumps({"claude_code": "fresh-sdk"})})
        conn.execute(text("""
            INSERT INTO messages (bot_id,id,role,content,timestamp,session_id,
                                  author_entity_type,author_entity_id)
            VALUES (:bot,:id,'user',:content,2,:session,'bot','test-sender')
        """), {"bot": rig.bot, "id": claimed.user_message_id,
               "content": supplied, "session": claimed.new_session_id})
        conn.execute(text("""
            INSERT INTO messages (bot_id,id,role,content,timestamp,session_id)
            VALUES (:bot,:id,'assistant',:reply,3,:session)
        """), {"bot": rig.bot, "id": payload["assistant_message_id"],
               "reply": TASK_REPLY, "session": claimed.new_session_id})
        conn.execute(text("""
            UPDATE turn_logs SET status='ok',ended_at=now(),response_text=:reply,
                assistant_message_id=:assistant,trigger_message_id=:trigger WHERE id=:turn
        """), {"reply": TASK_REPLY, "assistant": payload["assistant_message_id"],
               "trigger": claimed.user_message_id, "turn": claimed.turn_id})
    event = SimpleNamespace(
        run_id=payload["inter_bot_bridge_request_id"], provider="claude-code",
        trigger_message_id=claimed.user_message_id,
        raw={"new_session_ack": {"thread_session_id": claimed.new_session_id,
                                 "provider_session_id": "fresh-sdk", "clean_start": False}},
    )
    assert NewSessionReceipt.record_provider_ack(
        rig.engine, request_id=event.run_id, bot_id=rig.bot, event=event,
    )
    delivered = real_lifecycle.store.mark_delivered(
        claimed.id, claimed.claim_token, response_model="fixture", response_chars=len(TASK_REPLY),
    )
    dispatcher = InterBotDeliveryDispatcher.__new__(InterBotDeliveryDispatcher)
    dispatcher.store = real_lifecycle.store
    dispatcher._emit = AsyncMock()
    asyncio.run(dispatcher._finish_delivered(delivered))
    settled = real_lifecycle.store.get(claimed.id)
    assert settled.status == "DELIVERED" and settled.reset_status == "COMPLETED"
    assert settled.metadata["new_session_receipt"]["message_id"] != payload["assistant_message_id"]
    with rig.engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT content FROM messages WHERE bot_id=:bot AND session_id=:session
        """), {"bot": rig.bot, "session": claimed.new_session_id}).scalars().all()
        assert sorted(rows) == sorted([supplied, TASK_REPLY, RESET_CONFIRMATION])
        assert conn.execute(text("""
            SELECT count(*) FROM new_session_event_outbox WHERE delivery_id=:id
        """), {"id": claimed.id}).scalar_one() == 1
