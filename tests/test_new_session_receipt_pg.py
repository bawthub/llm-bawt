"""Receipt cannot claim success from transport or model prose alone."""
from __future__ import annotations

import asyncio
import json
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from llm_bawt.memory.session_reset_lineage import SessionResetLineage
from llm_bawt.service.session_lifecycle.intent import Outcome
from llm_bawt.service.session_lifecycle.receipt import NewSessionReceipt, RESET_CONFIRMATION
from tests.session_lifecycle_pg_fixtures import scheduled_intent

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


def test_scheduler_missing_receipt_is_settled_without_replaying(rig, real_lifecycle):
    intent = scheduled_intent(rig)
    asyncio.run(real_lifecycle.submit(intent, timeout_seconds=0))
    claim = real_lifecycle.store.claim_next(rig.bot, claim_owner="fixture", steer_capable=False)
    assert claim is not None
    real_lifecycle.store.mark_transport_accepted(claim.id, claim.claim_token)
    real_lifecycle.store.mark_delivered(claim.id, claim.claim_token, response_model="fixture", response_chars=0)
    settled = NewSessionReceipt.reconcile_typed(rig.engine)
    assert any(row.id == claim.id and row.status == "FAILED" for row in settled)
    assert real_lifecycle.store.get(claim.id).metadata["new_session_ambiguous"] is True
    assert real_lifecycle.store.claim_next(rig.bot, claim_owner="retry", steer_capable=False) is None


@pytest.mark.parametrize("lifecycle_backend", ["claude-code", "codex"], indirect=True)
@pytest.mark.parametrize("finisher", ["observer", "dispatcher", "reconciler"])
def test_provider_ack_and_exact_persisted_receipt(rig, real_lifecycle, lifecycle_backend, finisher):
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
            working_memory_mode="pinned",
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
        event=SimpleNamespace(run_id="wrong-run", provider=lifecycle_backend,
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
        run_id=payload["inter_bot_bridge_request_id"], provider=lifecycle_backend,
        session_key=rig.bot, trigger_message_id=claim.user_message_id,
        raw={"new_session_ack": {"thread_session_id": new_id, "clean_start": True,
                                   "provider_session_id": None}},
    )
    assert NewSessionReceipt.record_provider_ack(
        rig.engine, request_id=event.run_id, bot_id=rig.bot, event=event,
    ) is True
    delivered = real_lifecycle.store.mark_delivered(claim.id, claim.claim_token,
                                                    response_model="fixture", response_chars=20)
    if finisher == "dispatcher":
        from unittest.mock import AsyncMock
        from llm_bawt.service.inter_bot_dispatcher import InterBotDeliveryDispatcher
        dispatcher = InterBotDeliveryDispatcher.__new__(InterBotDeliveryDispatcher)
        dispatcher.store = real_lifecycle.store
        dispatcher._emit = AsyncMock()
        asyncio.run(dispatcher._finish_delivered(delivered))
    elif finisher == "reconciler":
        assert any(row.id == claim.id for row in NewSessionReceipt.reconcile_typed(rig.engine))
    if finisher != "observer":
        assert real_lifecycle.store.get(claim.id).metadata.get("new_session_receipt")
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


@pytest.mark.parametrize("lifecycle_backend", ["claude-code", "codex"], indirect=True)
@pytest.mark.parametrize("finisher", ["observer", "dispatcher", "reconciler"])
def test_seeded_scheduler_receipt_reports_the_acknowledged_continuity(
    rig, real_lifecycle, lifecycle_backend, finisher,
):
    """A seeded reset's bubble cites the restored manifest, never bridge prose."""
    intent = scheduled_intent(rig)
    asyncio.run(real_lifecycle.submit(intent, timeout_seconds=0))
    claim = real_lifecycle.store.claim_next(rig.bot, claim_owner="fixture", steer_capable=False)
    assert claim is not None
    payload = real_lifecycle.store.payload(claim.id)
    new_id = str(uuid.uuid4())
    continuity = {"status": "restored", "summary_count": 0, "message_count": 40,
                  "approx_tokens": 11034, "oldest_timestamp": 1792553315.54,
                  "newest_timestamp": 1792617268.65}
    with rig.engine.begin() as conn:
        SessionResetLineage.rotate(
            None, connection=conn, bot_id=rig.bot, user_id=rig.user,
            new_id=new_id, idempotency_key=claim.id, accept_pointer=True,
            working_memory_mode="pinned",
        )
        conn.execute(text("""
            UPDATE sessions SET session_metadata=COALESCE(session_metadata, '{}'::jsonb)
                || CAST(:meta AS jsonb) WHERE id=:id
        """), {"id": new_id, "meta": json.dumps({
            "agent_session_keys": {
                "codex" if lifecycle_backend == "codex" else "claude_code": "sdk-seeded",
            }, "continuity": continuity})})
        conn.execute(text("""
            UPDATE inter_bot_deliveries SET metadata_json=metadata_json || CAST(:meta AS jsonb)
            WHERE id=:id
        """), {"id": claim.id, "meta": json.dumps({
            "new_session_seed": {"payload_digest": "exact-bound-seed", "count": 40},
        })})
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
                assistant_message_id=:assistant, ended_at=now(), trigger_message_id=:trigger
            WHERE id=:turn
        """), {"assistant": payload["assistant_message_id"], "trigger": claim.user_message_id,
               "turn": claim.turn_id})
    event = SimpleNamespace(
        run_id=payload["inter_bot_bridge_request_id"], provider=lifecycle_backend,
        session_key=rig.bot, trigger_message_id=claim.user_message_id,
        raw={"new_session_ack": {"thread_session_id": new_id, "provider_session_id": "sdk-seeded",
                                 "seed_digest": "exact-bound-seed", "seeded_count": 40}},
    )
    assert NewSessionReceipt.record_provider_ack(
        rig.engine, request_id=event.run_id, bot_id=rig.bot, event=event,
    ) is True
    delivered = real_lifecycle.store.mark_delivered(claim.id, claim.claim_token,
                                                    response_model="fixture", response_chars=20)
    if finisher == "dispatcher":
        from unittest.mock import AsyncMock
        from llm_bawt.service.inter_bot_dispatcher import InterBotDeliveryDispatcher
        dispatcher = InterBotDeliveryDispatcher.__new__(InterBotDeliveryDispatcher)
        dispatcher.store = real_lifecycle.store
        dispatcher._emit = AsyncMock()
        asyncio.run(dispatcher._finish_delivered(delivered))
    elif finisher == "reconciler":
        assert any(row.id == claim.id for row in NewSessionReceipt.reconcile_typed(rig.engine))
    if finisher != "observer":
        assert real_lifecycle.store.get(claim.id).metadata.get("new_session_receipt")
    done = asyncio.run(real_lifecycle.submit(intent, timeout_seconds=0))
    replay = asyncio.run(real_lifecycle.submit(intent, timeout_seconds=0))
    assert done.outcome is Outcome.COMPLETED and done == replay
    expected = ("Session reset. Seeded 40 recent messages (~11.0k tokens), from 2026-10-21. "
                "Continuity restored.")
    with rig.engine.connect() as conn:
        assert conn.execute(text("""
            SELECT content, author_entity_id FROM messages WHERE bot_id=:bot AND id=:id
        """), {"bot": rig.bot, "id": done.receipt_message_id}).one() == (expected, rig.bot)
        assert conn.execute(text("SELECT response_text FROM turn_logs WHERE id=:turn"),
                            {"turn": claim.turn_id}).scalar_one() == expected
        assert conn.execute(text("""
            SELECT metadata_json->'new_session_receipt'->>'content'
            FROM inter_bot_deliveries WHERE id=:id
        """), {"id": claim.id}).scalar_one() == expected
