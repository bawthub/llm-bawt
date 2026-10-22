"""The existing delivery sweep sends at most one verified Talk answer."""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from llm_bawt.integrations.nextcloud import manager, webhook
from llm_bawt.service.talk_reply_outbox import TalkReplySubscription, TalkReplyOutbox


class Subscriptions:
    drain = TalkReplyOutbox.drain

    def __init__(self, subscription):
        self.subscription = subscription
        self.state = "PENDING"
        self.claims = 0
        self.last_error = None

    def pending(self):
        return [self.subscription] if self.state == "PENDING" else []

    def defer(self, message_id):
        assert message_id == self.subscription.message_id

    def claim(self, message_id):
        if self.state != "PENDING":
            return False
        self.claims += 1
        self.state = "SENDING"
        return True

    def settle(self, message_id, state, error=None):
        assert self.state == "SENDING"
        self.state = state
        self.last_error = error


def source(status="DELIVERED", *, message_id="original", text="Exact answer"):
    return SimpleNamespace(
        id="delivery-original", status=status, sender_bot_id="nova", target_bot_id="nova",
        author_entity_type="user", author_entity_id="nick",
        user_message_id=message_id, turn_id="turn-original",
        metadata={"ordinary_chat": {"digest": "original"}}, last_error=None,
    ), SimpleNamespace(
        ended_at=datetime.now(timezone.utc), status="ok", error_text=None,
        bot_id="nova", user_id="nick", trigger_message_id=message_id,
        response_text=text,
    )


def setup(monkeypatch, *, kind="delivery"):
    calls = []
    sub = TalkReplySubscription(
        message_id="original", bot_id="nova", user_id="nick",
        source_type=kind, source_id="delivery-original" if kind == "delivery" else "turn-original",
        backend="https://talk.example", conversation_token="room", reply_to=42,
    )
    outbox = Subscriptions(sub)
    monkeypatch.setattr(manager, "get_nextcloud_manager", lambda: SimpleNamespace(
        get_bot_by_conversation=lambda token: SimpleNamespace(llm_bawt_bot="nova", secret="secret"),
    ))

    async def send(**kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr(webhook, "send_nextcloud_message", send)
    return outbox, calls


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["delivery", "turn"])
async def test_late_verified_reply_sent_once_after_pending_sweeps(monkeypatch, kind):
    outbox, calls = setup(monkeypatch, kind=kind)
    record, turn = source(status="QUEUED")
    deliveries = SimpleNamespace(get=lambda _id: record)
    turns = SimpleNamespace(get_turn=lambda _id: turn)
    if kind == "turn":
        turn.ended_at = None
    await outbox.drain(deliveries, turns)
    assert outbox.state == "PENDING" and calls == []
    record.status = "DELIVERED"
    turn.ended_at = datetime.now(timezone.utc)
    await outbox.drain(deliveries, turns)
    await outbox.drain(deliveries, turns)
    assert outbox.state == "SENT" and outbox.claims == 1
    assert len(calls) == 1
    assert calls[0]["message"] == "Exact answer"
    assert calls[0]["reply_to"] == 42


@pytest.mark.asyncio
async def test_mismatched_turn_identity_never_posts(monkeypatch):
    outbox, calls = setup(monkeypatch)
    record, turn = source(message_id="someone-else")
    await outbox.drain(SimpleNamespace(get=lambda _id: record), SimpleNamespace(get_turn=lambda _id: turn))
    assert outbox.state == "PENDING" and calls == []


@pytest.mark.asyncio
async def test_ambiguous_outbound_send_is_not_retried(monkeypatch):
    outbox, calls = setup(monkeypatch)
    record, turn = source()

    async def uncertain(**kwargs):
        calls.append(kwargs)
        raise ConnectionError("sent but connection dropped")

    monkeypatch.setattr(webhook, "send_nextcloud_message", uncertain)
    deliveries = SimpleNamespace(get=lambda _id: record)
    turns = SimpleNamespace(get_turn=lambda _id: turn)
    await outbox.drain(deliveries, turns)
    await outbox.drain(deliveries, turns)
    assert outbox.state == "UNCERTAIN" and outbox.claims == 1
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["FAILED", "CANCELLED"])
async def test_terminal_delivery_sends_truthful_status_once_not_fabricated_answer(monkeypatch, status):
    outbox, calls = setup(monkeypatch)
    record, turn = source(status=status)
    # ``list`` backs the FAILED-original retry-child lookup (TASK-1231B); none here.
    deliveries = SimpleNamespace(get=lambda _id: record, list=lambda **_kw: [])
    turns = SimpleNamespace(get_turn=lambda _id: turn)
    await outbox.drain(deliveries, turns)
    await outbox.drain(deliveries, turns)
    assert outbox.state == "SENT" and len(calls) == 1
    assert "without a verified answer" in calls[0]["message"]
    assert status in calls[0]["message"]
    assert "Exact answer" not in calls[0]["message"]
