"""Queued ordinary replies cannot be inferred from outbox completion alone."""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from llm_bawt.service.routes import inter_bot_deliveries
from llm_bawt.service.delivery_reply import (
    DeliveryReplyMismatch, resolve_delivery_reply, resolve_direct_turn_reply,
    wait_for_delivery_reply, wait_for_direct_turn_reply,
)


class Store:
    def __init__(self, record, children=()):
        self.record = record
        self.children = list(children)

    def get(self, delivery_id):
        return self.record if delivery_id == "delivery-test" else None

    def list(self, *, sender_bot_id, target_bot_id, idempotency_key, limit):
        return [child for child in self.children
                if child.sender_bot_id == sender_bot_id and child.target_bot_id == target_bot_id
                and child.idempotency_key == idempotency_key][:limit]


class Turns:
    def __init__(self, turn):
        self.turn = turn

    def get_turn(self, turn_id):
        assert turn_id == "turn-test"
        return self.turn


def lookup(record, turn=None, **overrides):
    return resolve_delivery_reply(
        Store(record), Turns(turn), "delivery-test",
        bot_id=overrides.get("bot_id", "bot"),
        user_id=overrides.get("user_id", "owner"),
        user_message_id=overrides.get("user_message_id", "message"),
    )


def delivery(status="QUEUED", **overrides):
    data = dict(id="delivery-test", status=status, target_bot_id="bot", sender_bot_id="owner",
                author_entity_type="user", author_entity_id="owner",
                user_message_id="message", turn_id="turn-test",
                metadata={"ordinary_chat": {"digest": "test"}}, last_error=None)
    data.update(overrides)
    return SimpleNamespace(**data)


def turn(**overrides):
    data = dict(ended_at=datetime.now(timezone.utc), status="ok", error_text=None,
                bot_id="bot", user_id="owner", trigger_message_id="message",
                response_text="Actual reply")
    data.update(overrides)
    return SimpleNamespace(**data)


def test_unknown_delivery_is_not_a_reply():
    assert resolve_delivery_reply(Store(None), Turns(None), "delivery-test",
                                  bot_id="bot", user_id="owner", user_message_id="message") is None


@pytest.mark.parametrize("field,value", [
    ("target_bot_id", "other"), ("author_entity_type", "bot"),
    ("author_entity_id", "other"), ("user_message_id", "other"),
    ("metadata", {}),
])
def test_receipt_requires_exact_original_identity(field, value):
    with pytest.raises(DeliveryReplyMismatch):
        lookup(delivery(**{field: value}), turn())


@pytest.mark.parametrize("status", ["QUEUED", "DISPATCHING", "FAILED", "CANCELLED"])
def test_pending_and_failed_deliveries_never_expose_reply(status):
    result = lookup(delivery(status), turn())
    assert "response_text" not in result
    assert result["status"] == status


@pytest.mark.parametrize("field,value", [
    ("ended_at", None), ("status", "aborted"), ("error_text", "failure"),
    ("bot_id", "other"), ("user_id", "other"),
    ("trigger_message_id", "other"),
])
def test_delivered_without_matching_finished_turn_is_unresolved(field, value):
    result = lookup(delivery("DELIVERED"), turn(**{field: value}))
    assert result["status"] == "UNRESOLVED"
    assert "response_text" not in result


def test_exact_finished_turn_returns_its_reply():
    assert lookup(delivery("DELIVERED"), turn())["response_text"] == "Actual reply"


@pytest.mark.parametrize("changes,status", [
    ({}, "DELIVERED"), ({"ended_at": None}, "IN_FLIGHT"),
    ({"status": "error"}, "FAILED"),
])
def test_direct_retry_reads_only_original_turn(changes, status):
    result = resolve_direct_turn_reply(
        Turns(turn(**changes)), "turn-test",
        bot_id="bot", user_id="owner", user_message_id="message",
    )
    assert result["status"] == status
    assert ("response_text" in result) == (status == "DELIVERED")


def test_direct_retry_rejects_wrong_original_identity():
    with pytest.raises(DeliveryReplyMismatch):
        resolve_direct_turn_reply(Turns(turn(user_id="other")), "turn-test",
                                  bot_id="bot", user_id="owner", user_message_id="message")


def test_reply_route_only_exposes_matching_finished_turn(monkeypatch):
    record = delivery("DELIVERED")
    monkeypatch.setattr(inter_bot_deliveries, "_dispatcher", lambda: SimpleNamespace(store=Store(record)))
    monkeypatch.setattr(inter_bot_deliveries, "get_service", lambda: SimpleNamespace(
        _turn_log_store=Turns(turn())
    ))
    reply = inter_bot_deliveries.get_delivery_reply(
        "delivery-test", bot_id="bot", user_id="owner", user_message_id="message"
    )
    assert reply["response_text"] == "Actual reply"
    with pytest.raises(HTTPException) as error:
        inter_bot_deliveries.get_delivery_reply(
            "delivery-test", bot_id="bot", user_id="other", user_message_id="message"
        )
    assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_queued_waiter_returns_verified_reply_when_already_delivered():
    result = await wait_for_delivery_reply(
        Store(delivery("DELIVERED")), Turns(turn()), "delivery-test",
        bot_id="bot", user_id="owner", user_message_id="message", timeout_seconds=0,
    )
    assert result["response_text"] == "Actual reply"


@pytest.mark.asyncio
async def test_queued_waiter_timeout_stays_pending_not_fake_answer():
    result = await wait_for_delivery_reply(
        Store(delivery()), Turns(turn()), "delivery-test",
        bot_id="bot", user_id="owner", user_message_id="message", timeout_seconds=0,
    )
    assert result["status"] == "QUEUED"
    assert "response_text" not in result


@pytest.mark.asyncio
async def test_direct_waiter_does_not_replay_in_flight_turn():
    result = await wait_for_direct_turn_reply(
        Turns(turn(ended_at=None)), "turn-test",
        bot_id="bot", user_id="owner", user_message_id="message", timeout_seconds=0,
    )
    assert result["status"] == "IN_FLIGHT"
    assert "response_text" not in result


# --- TASK-1231B: a FAILED original reports its verified restart-retry child ---

class ChildTurns:
    def __init__(self, turns):
        self.turns = turns

    def get_turn(self, turn_id):
        return self.turns.get(turn_id)


def restart_child(status="DELIVERED", **overrides):
    data = dict(id="delivery-child", status=status, target_bot_id="bot", sender_bot_id="owner",
                author_entity_type="bot", author_entity_id="system",
                user_message_id="child-message", turn_id="turn-child",
                idempotency_key="restart-retry:delivery-test",
                metadata={"ordinary_chat": {}, "restart_retry_of": "delivery-test",
                          "_request_identity_producer": "retry"},
                last_error=None)
    data.update(overrides)
    return SimpleNamespace(**data)


def follow(child, child_turn=None):
    original = delivery("FAILED", last_error="streamed durable turn ended with status=aborted")
    return resolve_delivery_reply(
        Store(original, [child] if child else []),
        ChildTurns({"turn-child": child_turn}), "delivery-test",
        bot_id="bot", user_id="owner", user_message_id="message",
    )


def test_failed_original_reports_delivered_restart_child_reply():
    result = follow(restart_child(), turn(trigger_message_id="child-message",
                                          response_text="Continued reply"))
    assert result == {"delivery_id": "delivery-test", "retry_delivery_id": "delivery-child",
                      "status": "DELIVERED", "turn_id": "turn-child",
                      "user_message_id": "child-message", "response_text": "Continued reply"}


def test_failed_original_with_queued_child_stays_pending():
    result = follow(restart_child("QUEUED"))
    assert result["status"] == "QUEUED" and "response_text" not in result


@pytest.mark.parametrize("override", [
    {"metadata": {"ordinary_chat": {}, "restart_retry_of": "other",
                  "_request_identity_producer": "retry"}},
    {"metadata": {"ordinary_chat": {}, "restart_retry_of": "delivery-test",
                  "_request_identity_producer": "bot"}},   # forged by a sender
    {"sender_bot_id": "someone-else"},
    {"idempotency_key": "restart-retry:other"},
], ids=["wrong-marker", "forged-producer", "other-sender", "other-key"])
def test_unverified_child_is_ignored(override):
    result = follow(restart_child(**override), turn(trigger_message_id="child-message"))
    assert result["status"] == "FAILED" and result["delivery_id"] == "delivery-test"
    assert "retry_delivery_id" not in result


def test_failed_original_without_child_keeps_its_error():
    result = follow(None)
    assert result["status"] == "FAILED" and "aborted" in result["error"]
