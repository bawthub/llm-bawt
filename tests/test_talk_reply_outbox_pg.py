"""Private PostgreSQL proof for durable Talk reply subscription and send fencing."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import uuid

import pytest
from sqlalchemy import text

from llm_bawt.service.talk_reply_outbox import TalkReplyOutbox, TalkReplySubscription
from llm_bawt.utils.config import Config
from llm_bawt.utils.db import get_shared_engine


@pytest.mark.integration
def test_subscription_survives_reopen_and_only_one_worker_claims_send():
    config = Config()
    if not config.POSTGRES_DATABASE.endswith("_test") or config.POSTGRES_HOST != "127.0.0.1":
        pytest.skip("requires localhost private *_test PostgreSQL")
    engine = get_shared_engine(config)
    if engine is None:
        pytest.skip("private PostgreSQL unavailable")
    outbox = TalkReplyOutbox(engine)
    message_id = str(uuid.uuid4())
    subscription = TalkReplySubscription(
        message_id=message_id, bot_id="test-nova", user_id="test-nick",
        source_type="delivery", source_id=f"delivery-{message_id}",
        backend="https://talk.example", conversation_token="room", reply_to=42,
    )
    try:
        assert outbox.register(subscription) == ("PENDING", True)
        assert TalkReplyOutbox(engine).register(subscription) == ("PENDING", False)
        wrong = TalkReplySubscription(**{**vars(subscription), "user_id": "other"})
        with pytest.raises(ValueError, match="different reply destination"):
            outbox.register(wrong)
        assert subscription in TalkReplyOutbox(engine).pending()
        with ThreadPoolExecutor(max_workers=2) as workers:
            claims = list(workers.map(outbox.claim, [message_id, message_id]))
        assert sorted(claims) == [False, True]
        assert TalkReplyOutbox(engine).pending() == [] or subscription not in TalkReplyOutbox(engine).pending()
        with engine.connect() as conn:
            state = conn.execute(text(
                "SELECT state FROM talk_reply_subscriptions WHERE message_id=:id"
            ), {"id": message_id}).scalar_one()
        assert state == "SENDING"  # interrupted send is not replayed after restart
        outbox.settle(message_id, "UNCERTAIN", "outbound result unknown")
        assert not outbox.claim(message_id)
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM talk_reply_subscriptions WHERE message_id=:id"),
                         {"id": message_id})


@pytest.mark.integration
def test_pending_subscription_checks_cannot_starve_newer_replies():
    config = Config()
    if not config.POSTGRES_DATABASE.endswith("_test") or config.POSTGRES_HOST != "127.0.0.1":
        pytest.skip("requires localhost private *_test PostgreSQL")
    engine = get_shared_engine(config)
    if engine is None:
        pytest.skip("private PostgreSQL unavailable")
    outbox = TalkReplyOutbox(engine)
    ids = [str(uuid.uuid4()) for _ in range(21)]
    try:
        for message_id in ids:
            outbox.register(TalkReplySubscription(
                message_id=message_id, bot_id="test-nova", user_id="test-nick",
                source_type="turn", source_id=f"turn-{message_id}",
                backend="https://talk.example", conversation_token="room", reply_to=None,
            ))
        first = outbox.pending(limit=20)
        assert len(first) == 20
        for sub in first:
            outbox.defer(sub.message_id)
        assert {sub.message_id for sub in outbox.pending(limit=20)} == set(ids) - {
            sub.message_id for sub in first
        }
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM talk_reply_subscriptions WHERE message_id=ANY(:ids)"),
                         {"ids": ids})


@pytest.mark.integration
def test_finished_turn_late_reply_uses_real_turn_log_once(monkeypatch):
    from llm_bawt.integrations.nextcloud import manager, webhook
    from llm_bawt.service.turn_logs import TurnLogStore

    config = Config()
    if not config.POSTGRES_DATABASE.endswith("_test") or config.POSTGRES_HOST != "127.0.0.1":
        pytest.skip("requires localhost private *_test PostgreSQL")
    turns = TurnLogStore(config)
    if turns.engine is None:
        pytest.skip("private PostgreSQL unavailable")
    outbox = TalkReplyOutbox(turns.engine)
    message_id = str(uuid.uuid4())
    turn_id = f"turn-{uuid.UUID(message_id).hex}"
    bot_id = f"test-talk-{uuid.uuid4().hex}"
    user_id = f"test-user-{uuid.uuid4().hex}"
    sent = []
    monkeypatch.setattr(manager, "get_nextcloud_manager", lambda: SimpleNamespace(
        get_bot_by_conversation=lambda token: SimpleNamespace(llm_bawt_bot=bot_id, secret="secret")
    ))

    async def send(**kwargs):
        sent.append(kwargs)
        return True

    monkeypatch.setattr(webhook, "send_nextcloud_message", send)
    try:
        turns.save_turn(
            turn_id=turn_id, request_id="test", path="/test", stream=False,
            model="test", bot_id=bot_id, user_id=user_id, status="ok",
            latency_ms=0, user_prompt="help", request_payload={"messages": []},
            response_text="Finished after webhook exited", tool_calls=None,
            trigger_message_id=message_id,
        )
        outbox.register(TalkReplySubscription(
            message_id=message_id, bot_id=bot_id, user_id=user_id,
            source_type="turn", source_id=turn_id,
            backend="https://talk.example", conversation_token="room", reply_to=42,
        ))
        asyncio.run(outbox.drain(None, turns))
        asyncio.run(TalkReplyOutbox(turns.engine).drain(None, turns))
        assert len(sent) == 1
        assert sent[0]["message"] == "Finished after webhook exited"
        assert sent[0]["reply_to"] == 42
        with turns.engine.connect() as conn:
            assert conn.execute(text(
                "SELECT state FROM talk_reply_subscriptions WHERE message_id=:id"
            ), {"id": message_id}).scalar_one() == "SENT"
    finally:
        with turns.engine.begin() as conn:
            conn.execute(text("DELETE FROM talk_reply_subscriptions WHERE message_id=:id"),
                         {"id": message_id})
            conn.execute(text("DELETE FROM turn_logs WHERE id=:id"), {"id": turn_id})
