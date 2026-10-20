"""Receipt cannot claim success from transport or model prose alone."""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from llm_bawt.memory.session_reset_lineage import SessionResetLineage
from llm_bawt.service.session_lifecycle.intent import Outcome
from llm_bawt.service.session_lifecycle.receipt import NewSessionReceipt, RESET_CONFIRMATION
from tests.session_lifecycle_pg_fixtures import scheduled_intent

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


def test_provider_ack_and_exact_persisted_receipt(rig, real_lifecycle):
    intent = scheduled_intent(rig)
    asyncio.run(real_lifecycle.submit(intent, timeout_seconds=0))
    claim = real_lifecycle.store.claim_next(rig.bot, claim_owner="fixture", steer_capable=False)
    assert claim is not None
    payload = real_lifecycle.store.payload(claim.id)
    new_id = str(uuid.uuid4())
    with rig.engine.begin() as conn:
        SessionResetLineage.rotate(
            None, connection=conn, bot_id=rig.bot, user_id=rig.user,
            new_id=new_id, idempotency_key=claim.id, accept_pointer=True,
        )
        conn.execute(text("""
            INSERT INTO messages (bot_id, id, role, content, timestamp, session_id,
                                  author_entity_type, author_entity_id)
            VALUES (:bot, :id, 'user', '/new', 2, :session, 'bot', 'system')
        """), {"bot": rig.bot, "id": claim.user_message_id, "session": new_id})
        conn.execute(text("""
            INSERT INTO messages (bot_id, id, role, content, timestamp, session_id)
            VALUES (:bot, :id, 'assistant', 'bridge prose is not proof', 3, :session)
        """), {"bot": rig.bot, "id": payload["assistant_message_id"], "session": new_id})
        conn.execute(text("""
            UPDATE turn_logs SET status='ok', response_text='bridge prose is not proof',
                assistant_message_id=:assistant, ended_at=now(),
                trigger_message_id=:trigger
            WHERE id=:turn
        """), {"assistant": payload["assistant_message_id"], "trigger": claim.user_message_id,
               "turn": claim.turn_id})
    assert NewSessionReceipt.record_provider_ack(
        rig.engine, request_id="wrong-run", bot_id=rig.bot,
        event=SimpleNamespace(run_id="wrong-run", provider="claude-code",
                              session_key=rig.bot, trigger_message_id=claim.user_message_id,
                              raw={"new_session_ack": {"thread_session_id": new_id, "clean_start": True}}),
    ) is False
    real_lifecycle.store.mark_delivered(claim.id, claim.claim_token,
                                        response_model="fixture", response_chars=20)
    missing = asyncio.run(real_lifecycle.submit(intent, timeout_seconds=0))
    assert missing.outcome is Outcome.AMBIGUOUS
    # A reset ack must arrive while the exact bridge run still holds the claim.
    with rig.engine.begin() as conn:
        conn.execute(text("""
            UPDATE inter_bot_deliveries SET status='DISPATCHING', claim_token=:token
            WHERE id=:id
        """), {"id": claim.id, "token": claim.claim_token})
    event = SimpleNamespace(
        run_id=payload["inter_bot_bridge_request_id"], provider="claude-code",
        session_key=rig.bot, trigger_message_id=claim.user_message_id,
        raw={"new_session_ack": {"thread_session_id": new_id, "clean_start": True,
                                   "provider_session_id": None}},
    )
    assert NewSessionReceipt.record_provider_ack(
        rig.engine, request_id=event.run_id, bot_id=rig.bot, event=event,
    ) is True
    real_lifecycle.store.mark_delivered(claim.id, claim.claim_token,
                                        response_model="fixture", response_chars=20)
    done = asyncio.run(real_lifecycle.submit(intent, timeout_seconds=0))
    replay = asyncio.run(real_lifecycle.submit(intent, timeout_seconds=0))
    assert done.outcome is Outcome.COMPLETED and done == replay
    assert done.old_session_id == rig.session_id and done.new_session_id == new_id
    assert done.receipt_message_id == payload["assistant_message_id"]
    with rig.engine.connect() as conn:
        row = conn.execute(text("""
            SELECT content, author_entity_id FROM messages WHERE bot_id=:bot AND id=:id
        """), {"bot": rig.bot, "id": done.receipt_message_id}).one()
        assert row == (RESET_CONFIRMATION, rig.bot)
        assert conn.execute(text("""
            SELECT count(*) FROM new_session_event_outbox WHERE delivery_id=:id
        """), {"id": claim.id}).scalar_one() == 1
