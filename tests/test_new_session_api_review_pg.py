"""Standalone APIs must survive actual admission, claim and persisted receipt."""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import text

from llm_bawt.message_authorship import AuthorReference
from llm_bawt.service.inter_bot_dispatcher import InterBotDeliveryDispatcher
from llm_bawt.service.session_lifecycle.intent import (
    Admission, Continuity, NewSessionIntent, Origin, Outcome,
)
from llm_bawt.service.session_lifecycle.receipt import NewSessionReceipt, RESET_CONFIRMATION

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


def intent_for(rig, origin=Origin.CONTEXT_API, continuity=Continuity.CLEAN):
    return NewSessionIntent(
        bot_id=rig.bot, user_id=rig.user, author=AuthorReference.user(rig.user),
        continuity=continuity, admission=Admission.SKIP, origin=origin,
        idempotency_key=f"api-review:{rig.bot}",
    )


def finish(rig, lifecycle, record, *, backend, continuity):
    claim = lifecycle.store.claim_next(rig.bot, claim_owner="api-review", steer_capable=False)
    assert claim is not None and claim.id == record.id
    assert claim.old_session_id == rig.session_id and claim.new_session_id != rig.session_id
    assert claim.reset_status == "ROTATED"
    assert lifecycle.store.mark_transport_accepted(claim.id, claim.claim_token)
    payload = lifecycle.store.payload(claim.id)
    assert payload["inter_bot_reset_only"] is True
    provider = "retained-provider" if continuity is Continuity.RETAIN else None
    with rig.engine.begin() as conn:
        if provider:
            conn.execute(text("""
                UPDATE sessions SET session_metadata=jsonb_set(session_metadata,
                    '{agent_session_keys}', CAST(:keys AS jsonb)) WHERE id=:id
            """), {"id": claim.new_session_id,
                    "keys": json.dumps({backend.replace("-", "_"): provider})})
        conn.execute(text("""
            INSERT INTO messages (bot_id,id,role,content,timestamp,session_id,
                                  author_entity_type,author_entity_id)
            VALUES (:bot,:id,'user','/new',2,:session,'user',:owner)
        """), {"bot": rig.bot, "id": claim.user_message_id,
                "session": claim.new_session_id, "owner": rig.user})
        conn.execute(text("""
            INSERT INTO messages (bot_id,id,role,content,timestamp,session_id)
            VALUES (:bot,:id,'assistant','bridge reset prose',3,:session)
        """), {"bot": rig.bot, "id": payload["assistant_message_id"], "session": claim.new_session_id})
        conn.execute(text("""
            UPDATE turn_logs SET status='ok',ended_at=now(),response_text='bridge reset prose',
                assistant_message_id=:assistant,trigger_message_id=:trigger WHERE id=:turn
        """), {"assistant": payload["assistant_message_id"],
                "trigger": claim.user_message_id, "turn": claim.turn_id})
    event = SimpleNamespace(
        run_id=payload["inter_bot_bridge_request_id"], provider=backend,
        trigger_message_id=claim.user_message_id,
        raw={"new_session_ack": {"thread_session_id": claim.new_session_id,
                                 "provider_session_id": provider, "clean_start": not bool(provider)}},
    )
    return claim, payload, event


@pytest.mark.parametrize("origin", [Origin.CONTEXT_API, Origin.SESSION_API])
@pytest.mark.parametrize("backend,continuity", [
    ("claude-code", Continuity.CLEAN), ("claude-code", Continuity.RETAIN),
    ("codex", Continuity.CLEAN),
])
def test_api_admission_claim_receipt_and_retry_are_one_operation(rig, real_lifecycle, monkeypatch, origin, backend, continuity):
    bot = SimpleNamespace(slug=rig.bot, agent_backend=backend, harness=backend)
    monkeypatch.setattr("llm_bawt.service.session_lifecycle.lifecycle.BotManager",
                        lambda _config: SimpleNamespace(get_bot=lambda _slug: bot))
    intent = intent_for(rig, origin, continuity)
    record, reason = real_lifecycle._admit(intent)
    assert record is not None and reason is None
    assert record.message == "/new" and record.author == AuthorReference.user(rig.user)
    assert record.metadata["new_session_intent"]["expected_session_id"] == rig.session_id
    assert real_lifecycle._admit(intent)[0].id == record.id
    with rig.engine.connect() as conn:
        assert conn.execute(text("SELECT last_session_id FROM bot_profiles WHERE slug=:bot"),
                            {"bot": rig.bot}).scalar_one() == rig.session_id
    claim, payload, event = finish(rig, real_lifecycle, record, backend=backend, continuity=continuity)
    assert NewSessionReceipt.record_provider_ack(
        rig.engine, request_id=event.run_id, bot_id=rig.bot, event=event,
    )
    delivered = real_lifecycle.store.mark_delivered(claim.id, claim.claim_token,
                                                   response_model="fixture", response_chars=18)
    dispatcher = InterBotDeliveryDispatcher.__new__(InterBotDeliveryDispatcher)
    dispatcher.store = real_lifecycle.store
    dispatcher._emit = AsyncMock()
    asyncio.run(dispatcher._finish_delivered(delivered))
    result = real_lifecycle._observe(intent, record.id)
    assert result.outcome is Outcome.COMPLETED
    assert result.old_session_id == rig.session_id and result.new_session_id == claim.new_session_id
    assert result.receipt_message_id == payload["assistant_message_id"]
    assert real_lifecycle._admit(intent)[0].id == record.id
    assert real_lifecycle._observe(intent, record.id) == result
    with rig.engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT role,content FROM messages WHERE bot_id=:bot AND session_id=:session
        """), {"bot": rig.bot, "session": claim.new_session_id}).all()
        assert sorted(rows) == sorted([("user", "/new"), ("assistant", RESET_CONFIRMATION)])
        assert conn.execute(text("SELECT count(*) FROM new_session_event_outbox WHERE delivery_id=:id"),
                            {"id": record.id}).scalar_one() == 1
    assert real_lifecycle.store.claim_next(rig.bot, claim_owner="retry", steer_capable=False) is None


@pytest.mark.parametrize("changed", ["pointer", "active"])
def test_api_claim_rechecks_idle_pointer_before_any_rotation(rig, real_lifecycle, changed):
    from uuid import uuid4

    from llm_bawt.memory.session_reset_lineage import SessionResetLineage

    intent = intent_for(rig)
    record, reason = real_lifecycle._admit(intent)
    assert record is not None and reason is None
    expected_pointer = rig.session_id
    with rig.engine.begin() as conn:
        if changed == "pointer":
            expected_pointer = str(uuid4())
            SessionResetLineage.rotate(None, connection=conn, bot_id=rig.bot,
                                       user_id=rig.user, new_id=expected_pointer,
                                       accept_pointer=True)
        else:
            conn.execute(text("UPDATE turn_logs SET ended_at=NULL,status='running' WHERE id=:id"),
                         {"id": rig.turn_id})
    assert real_lifecycle.store.claim_next(rig.bot, claim_owner="race", steer_capable=False) is None
    skipped = real_lifecycle.store.get(record.id)
    assert skipped.status == "FAILED" and skipped.metadata["new_session_skip_reason"]
    assert real_lifecycle._observe(intent, record.id).outcome is Outcome.SKIPPED
    with rig.engine.connect() as conn:
        assert conn.execute(text("SELECT last_session_id FROM bot_profiles WHERE slug=:bot"),
                            {"bot": rig.bot}).scalar_one() == expected_pointer


@pytest.mark.parametrize("blocked", ["wrong-owner", "stale-session", "active", "queued"])
def test_api_admission_skips_without_rotation_or_delayed_work(rig, real_lifecycle, blocked):
    intent = intent_for(rig)
    if blocked == "wrong-owner":
        intent = replace(intent, user_id="other-owner", author=AuthorReference.user("other-owner"))
    elif blocked == "stale-session":
        intent = replace(intent, expected_session_id="stale-session")
    elif blocked == "active":
        with rig.engine.begin() as conn:
            conn.execute(text("UPDATE turn_logs SET ended_at=NULL,status='running' WHERE id=:id"),
                         {"id": rig.turn_id})
    else:
        real_lifecycle.store.enqueue(sender_bot_id="sender", target_bot_id=rig.bot,
                                     message="queued work", payload={"user": rig.user})
    record, reason = real_lifecycle._admit(intent)
    assert record is None and reason
    with rig.engine.connect() as conn:
        assert conn.execute(text("SELECT last_session_id FROM bot_profiles WHERE slug=:bot"),
                            {"bot": rig.bot}).scalar_one() == rig.session_id
        assert conn.execute(text("""
            SELECT count(*) FROM inter_bot_deliveries
            WHERE target_bot_id=:bot AND idempotency_key=:key
        """), {"bot": rig.bot, "key": intent.idempotency_key}).scalar_one() == 0
