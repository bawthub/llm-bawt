"""Independent adversarial checks of reset completion and event publication."""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import event, text

from llm_bawt.memory.session_reset_lineage import SessionResetLineage
from llm_bawt.service.session_lifecycle.outbox import NewSessionOutbox
from llm_bawt.service.session_lifecycle.receipt import NewSessionReceipt
from tests.session_lifecycle_pg_fixtures import scheduled_intent

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.fixture
def pending_receipt(rig, real_lifecycle):
    intent = scheduled_intent(rig)
    result = asyncio.run(real_lifecycle.submit(intent, timeout_seconds=0))
    claim = real_lifecycle.store.claim_next(rig.bot, claim_owner="review", steer_capable=False)
    assert claim is not None and claim.id == result.delivery_id
    payload = real_lifecycle.store.payload(claim.id)
    successor = str(uuid.uuid4())
    with rig.engine.begin() as conn:
        SessionResetLineage.rotate(
            None, connection=conn, bot_id=rig.bot, user_id=rig.user,
            new_id=successor, idempotency_key=claim.id, accept_pointer=True,
        )
        conn.execute(text("""
            INSERT INTO messages (bot_id, id, role, content, timestamp,
                session_id, author_entity_type, author_entity_id)
            VALUES (:bot, :id, 'user', '/new', 2, :session, 'bot', 'system')
        """), {"bot": rig.bot, "id": claim.user_message_id, "session": successor})
        conn.execute(text("""
            INSERT INTO messages (bot_id, id, role, content, timestamp, session_id)
            VALUES (:bot, :id, 'assistant', 'untrusted bridge prose', 3, :session)
        """), {"bot": rig.bot, "id": payload["assistant_message_id"], "session": successor})
        conn.execute(text("""
            UPDATE turn_logs SET status='ok', ended_at=now(),
                response_text='untrusted bridge prose', assistant_message_id=:assistant,
                trigger_message_id=:trigger WHERE id=:turn
        """), {"assistant": payload["assistant_message_id"], "trigger": claim.user_message_id,
               "turn": claim.turn_id})
    outbox = NewSessionOutbox(rig.engine)
    event = SimpleNamespace(
        run_id=payload["inter_bot_bridge_request_id"], provider="claude-code",
        session_key=rig.bot, trigger_message_id=claim.user_message_id,
        raw={"new_session_ack": {"thread_session_id": successor, "clean_start": True}},
    )
    state = SimpleNamespace(rig=rig, lifecycle=real_lifecycle, intent=intent,
                            claim=claim, payload=payload, successor=successor,
                            event=event, outbox=outbox)
    try:
        yield state
    finally:
        with rig.engine.begin() as conn:
            conn.execute(text("DELETE FROM new_session_event_outbox WHERE bot_id=:bot"), {"bot": rig.bot})


def acknowledge(state):
    return NewSessionReceipt.record_provider_ack(
        state.rig.engine, request_id=state.payload["inter_bot_bridge_request_id"],
        bot_id=state.rig.bot, event=state.event,
    )


def finish(state):
    state.lifecycle.store.mark_delivered(state.claim.id, state.claim.claim_token,
                                         response_model="fixture", response_chars=10)
    return NewSessionReceipt.ensure(state.rig.engine,
                                   state.lifecycle.store.get(state.claim.id), state.intent)


@pytest.mark.parametrize("field,value", [
    ("run_id", "another-run"), ("provider", "another-provider"),
    ("trigger_message_id", "another-trigger"),
])
def test_wrong_provider_run_or_trigger_cannot_ack(pending_receipt, field, value):
    state = pending_receipt
    setattr(state.event, field, value)
    assert acknowledge(state) is False
    assert finish(state) is None


def test_empty_provider_evidence_cannot_ack(pending_receipt):
    state = pending_receipt
    state.event.raw["new_session_ack"]["clean_start"] = False
    assert acknowledge(state) is False
    assert finish(state) is None


@pytest.mark.parametrize("damage", [
    "user_author", "user_author_type", "user_session", "assistant_missing",
    "assistant_session", "assistant_role", "failed_turn", "unfinished_turn",
    "turn_owner", "turn_trigger", "provider_session", "ack_session",
])
def test_receipt_requires_exact_completed_turn_and_message_identity(pending_receipt, damage):
    state = pending_receipt
    rig = state.rig
    if damage == "provider_session":
        state.event.raw["new_session_ack"]["provider_session_id"] = "wrong-provider-key"
    if damage == "ack_session":
        state.event.raw["new_session_ack"]["thread_session_id"] = rig.session_id
    assert acknowledge(state) is True
    with rig.engine.begin() as conn:
        params = {"bot": rig.bot, "user_msg": state.claim.user_message_id,
                  "assistant": state.payload["assistant_message_id"],
                  "turn": state.claim.turn_id, "old": rig.session_id}
        statements = {
            "user_author": "UPDATE messages SET author_entity_id='someone-else' WHERE bot_id=:bot AND id=:user_msg",
            "user_author_type": "UPDATE messages SET author_entity_type='user' WHERE bot_id=:bot AND id=:user_msg",
            "user_session": "UPDATE messages SET session_id=:old WHERE bot_id=:bot AND id=:user_msg",
            "assistant_missing": "DELETE FROM messages WHERE bot_id=:bot AND id=:assistant",
            "assistant_session": "UPDATE messages SET session_id=:old WHERE bot_id=:bot AND id=:assistant",
            "assistant_role": "UPDATE messages SET role='user' WHERE bot_id=:bot AND id=:assistant",
            "failed_turn": "UPDATE turn_logs SET status='error', error_text='seed failed' WHERE id=:turn",
            "unfinished_turn": "UPDATE turn_logs SET ended_at=NULL WHERE id=:turn",
            "turn_owner": "UPDATE turn_logs SET user_id='wrong-owner' WHERE id=:turn",
            "turn_trigger": "UPDATE turn_logs SET trigger_message_id='wrong-trigger' WHERE id=:turn",
        }
        if damage in statements:
            conn.execute(text(statements[damage]), params)
    assert finish(state) is None
    with rig.engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM new_session_event_outbox WHERE delivery_id=:id"),
                            {"id": state.claim.id}).scalar_one() == 0


def test_repeated_receipt_and_drain_publish_once_sequentially(pending_receipt):
    state = pending_receipt
    assert acknowledge(state) is True
    first = finish(state)
    assert first is not None
    replay = NewSessionReceipt.ensure(state.rig.engine,
                                     state.lifecycle.store.get(state.claim.id), state.intent)
    assert replay == first
    subscriber = SimpleNamespace(publish_tool_event=AsyncMock(return_value="event-id"))
    assert asyncio.run(state.outbox.drain(subscriber)) == 1
    assert asyncio.run(state.outbox.drain(subscriber)) == 0
    subscriber.publish_tool_event.assert_awaited_once()
    payload = subscriber.publish_tool_event.call_args.args[2]
    assert payload["event_id"] == f"new-session:{state.claim.id}"
    assert payload["receipt_message_id"] == first.receipt_message_id


def test_delayed_old_event_never_rewinds_a_newer_pointer(pending_receipt):
    state = pending_receipt
    assert acknowledge(state) is True
    assert finish(state) is not None
    newer = str(uuid.uuid4())
    with state.rig.engine.begin() as conn:
        SessionResetLineage.rotate(None, connection=conn, bot_id=state.rig.bot,
                                   user_id=state.rig.user, new_id=newer, accept_pointer=True)
    subscriber = SimpleNamespace(publish_tool_event=AsyncMock(return_value="event-id"))
    assert asyncio.run(state.outbox.drain(subscriber)) == 1
    subscriber.publish_tool_event.assert_not_awaited()
    with state.rig.engine.connect() as conn:
        assert conn.execute(text("SELECT last_session_id FROM bot_profiles WHERE slug=:bot"),
                            {"bot": state.rig.bot}).scalar_one() == newer


def test_publish_exception_preserves_pending_event(pending_receipt):
    state = pending_receipt
    assert acknowledge(state) is True and finish(state) is not None
    subscriber = SimpleNamespace(publish_tool_event=AsyncMock(side_effect=RuntimeError("offline")))
    assert asyncio.run(state.outbox.drain(subscriber)) == 0
    with state.rig.engine.connect() as conn:
        assert conn.execute(text("SELECT published_at FROM new_session_event_outbox WHERE delivery_id=:id"),
                            {"id": state.claim.id}).scalar_one() is None
    subscriber.publish_tool_event.side_effect = None
    subscriber.publish_tool_event.return_value = "event-id"
    assert asyncio.run(state.outbox.drain(subscriber)) == 1


def test_receipt_and_event_insert_roll_back_together(pending_receipt):
    state = pending_receipt
    assert acknowledge(state) is True
    def reject_event(conn, cursor, statement, parameters, context, executemany):
        if "INSERT INTO new_session_event_outbox" in statement:
            raise RuntimeError("outbox write unavailable")
    event.listen(state.rig.engine, "before_cursor_execute", reject_event)
    try:
        with pytest.raises(RuntimeError, match="outbox write unavailable"):
            finish(state)
    finally:
        event.remove(state.rig.engine, "before_cursor_execute", reject_event)
    with state.rig.engine.connect() as conn:
        assert conn.execute(text("SELECT content FROM messages WHERE bot_id=:bot AND id=:id"),
                            {"bot": state.rig.bot, "id": state.payload["assistant_message_id"]}).scalar_one() == "untrusted bridge prose"
        metadata = conn.execute(text("SELECT metadata_json FROM inter_bot_deliveries WHERE id=:id"),
                                {"id": state.claim.id}).scalar_one()
        assert "new_session_receipt" not in metadata
    assert finish(state) is not None


def test_concurrent_drain_keeps_one_stable_invalidation_identity(pending_receipt):
    state = pending_receipt
    assert acknowledge(state) is True and finish(state) is not None
    events = []
    async def concurrent():
        started, release = asyncio.Event(), asyncio.Event()
        async def publish(bot, user, payload):
            events.append(payload)
            started.set()
            await release.wait()
            return "redis-id"
        subscriber = SimpleNamespace(publish_tool_event=publish)
        drains = [asyncio.create_task(state.outbox.drain(subscriber)) for _ in range(2)]
        await asyncio.wait_for(started.wait(), timeout=3)
        await asyncio.sleep(0)
        release.set()
        await asyncio.wait_for(asyncio.gather(*drains), timeout=3)
    asyncio.run(concurrent())
    # At-least-once transport is deliberate: both copies invalidate from DB,
    # never navigate to the event's session. The logical event key is stable.
    assert events and all(payload == events[0] for payload in events)
    assert events[0]["event_id"] == f"new-session:{state.claim.id}"


def accepted_reset(rig, lifecycle, mode):
    if mode == "structured":
        result = asyncio.run(lifecycle.submit(scheduled_intent(rig), timeout_seconds=0))
        assert result.delivery_id
    else:
        lifecycle.store.enqueue(
            sender_bot_id="test-sender", target_bot_id=rig.bot,
            message="follow-up task", session_policy="reset_retain_history",
            payload={"user": rig.user, "messages": [{"role": "user", "content": "follow-up task"}]},
        )
    claim = lifecycle.store.claim_next(rig.bot, claim_owner="review", steer_capable=False)
    assert claim is not None
    assert lifecycle.store.mark_transport_accepted(claim.id, claim.claim_token)
    return claim


@pytest.mark.parametrize("mode", ["structured", "policy"])
def test_accepted_reset_expiry_never_replays_unknown_provider_outcome(rig, real_lifecycle, mode):
    claim = accepted_reset(rig, real_lifecycle, mode)
    with rig.engine.begin() as conn:
        conn.execute(text("UPDATE inter_bot_deliveries SET lease_expires_at=now()-interval '1 minute' WHERE id=:id"),
                     {"id": claim.id})
    recovered = real_lifecycle.store.recover_expired()
    row = next(row for row in recovered if row.id == claim.id)
    assert row.status == "FAILED" and "ambiguous" in row.last_error.lower()
    assert row.next_retry_at is None
    assert real_lifecycle.store.claim_next(rig.bot, claim_owner="retry", steer_capable=False) is None
    if mode == "structured":
        from llm_bawt.service.session_lifecycle.intent import Outcome
        observed = asyncio.run(real_lifecycle.submit(scheduled_intent(rig), timeout_seconds=0))
        assert observed.outcome is Outcome.AMBIGUOUS


@pytest.mark.parametrize("mode", ["structured", "policy"])
def test_accepted_reset_dispatch_timeout_never_replays(rig, real_lifecycle, mode):
    from llm_bawt.service.inter_bot_dispatcher import InterBotDeliveryDispatcher
    from llm_bawt.service.turn_logs import TurnLogStore
    claim = accepted_reset(rig, real_lifecycle, mode)
    async def failing_stream(request):
        raise TimeoutError("transport disconnected after provider acceptance")
        yield  # Make this the same async-generator contract as chat_completion_stream.
    service = SimpleNamespace(config=rig.config, chat_completion_stream=failing_stream,
                              _turn_log_store=TurnLogStore(rig.config))
    dispatcher = InterBotDeliveryDispatcher(service)
    asyncio.run(dispatcher._dispatch(claim))
    row = real_lifecycle.store.get(claim.id)
    assert row.status == "FAILED" and "ambiguous" in row.last_error.lower()
    assert row.next_retry_at is None
    assert real_lifecycle.store.claim_next(rig.bot, claim_owner="retry", steer_capable=False) is None


def test_unacknowledged_publish_preserves_pending_event(pending_receipt):
    state = pending_receipt
    assert acknowledge(state) is True and finish(state) is not None
    subscriber = SimpleNamespace(publish_tool_event=AsyncMock(return_value=None))
    assert asyncio.run(state.outbox.drain(subscriber)) == 0
    with state.rig.engine.connect() as conn:
        assert conn.execute(text("SELECT published_at FROM new_session_event_outbox WHERE delivery_id=:id"),
                            {"id": state.claim.id}).scalar_one() is None
