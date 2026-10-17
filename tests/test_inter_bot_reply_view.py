"""Sender reply projections only contain verified canonical target messages."""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from llm_bawt.service.inter_bot_reply_view import ReplyViewMismatch, inter_bot_reply_view
from llm_bawt.service.routes import inter_bot_deliveries


class Memory:
    def __init__(self, message=None, incoming=None):
        self.message = message
        self.incoming = incoming
        self.lookups = []

    def get_message_by_id(self, message_id):
        self.lookups.append(message_id)
        if self.message and self.message.get("id") == message_id:
            return self.message
        if message_id == "user-msg":
            return self.incoming or dict(id="user-msg", role="user", author_entity_type="bot",
                                         author_entity_id="sender", session_id="target-session")
        return self.message


class Turns:
    def __init__(self, turn=None):
        self.turn = turn
        self.lookups = []

    def get_turn(self, turn_id):
        self.lookups.append(turn_id)
        return self.turn


def delivery(**changes):
    data = dict(id="delivery-test", sender_bot_id="sender", author_entity_type="bot",
                author_entity_id="sender", target_bot_id="target", user_message_id="user-msg",
                turn_id="target-turn", target_turn_id=None, delivery_mode="turn", reply_mode="async",
                status="DELIVERED",
                last_error=None, metadata={}, message_kind=None)
    data.update(changes)
    return SimpleNamespace(**data)


def turn(**changes):
    data = dict(ended_at=datetime.now(timezone.utc), status="ok", error_text=None,
                bot_id="target", user_id="owner", trigger_message_id="user-msg",
                assistant_message_id="assistant-msg", response_text="stale snapshot")
    data.update(changes)
    return SimpleNamespace(**data)


def message(**changes):
    data = dict(id="assistant-msg", role="assistant", content="Canonical reply",
                session_id="target-session", author_entity_type="bot", author_entity_id="target")
    data.update(changes)
    return data


def project(record=None, target_turn=None, target_message=None, incoming_message=None, **kwargs):
    turns = Turns(turn() if target_turn is None else target_turn)
    memory = Memory(message() if target_message is None else target_message, incoming_message)
    view = inter_bot_reply_view(record or delivery(), turns=turns, memory_client=memory,
                                sender_bot_id=kwargs.get("sender_bot_id", "sender"),
                                user_id=kwargs.get("user_id", "owner"))
    return view, turns, memory


def test_verified_answer_is_canonical_not_a_turn_snapshot():
    view, turns, memory = project()
    assert view["reply_status"] == "answered"
    assert view["reply_text"] == "Canonical reply"
    assert view["reply_mode"] == "async"
    assert project(delivery(reply_mode="waited"))[0]["reply_mode"] == "waited"
    assert view["target_session_id"] == "target-session"
    assert view["target_message_id"] == "assistant-msg"
    assert turns.lookups == ["target-turn"]
    assert memory.lookups == ["assistant-msg", "user-msg"]


@pytest.mark.parametrize("status,reply_status", [
    ("QUEUED", "pending"), ("STEERING", "pending"),
    ("DISPATCHING", "pending"), ("FAILED", "unavailable"),
    ("CANCELLED", "unavailable"),
])
def test_nonanswer_statuses_never_read_target_text(status, reply_status):
    view, turns, memory = project(delivery(status=status))
    assert view["reply_status"] == reply_status
    assert view["reply_text"] is None
    assert turns.lookups == memory.lookups == []


def test_accepted_steer_never_claims_shared_turn_output():
    incoming = message(id="user-msg", role="user", content="From sender",
                       author_entity_type="bot", author_entity_id="sender")
    view, turns, memory = project(delivery(delivery_mode="steer", target_turn_id="shared-turn"),
                                  target_message=incoming)
    assert view["reply_status"] == "unattributed"
    assert view["reply_text"] is None
    assert view["target_turn_id"] == "shared-turn"
    assert view["target_message_id"] == "user-msg"
    assert view["target_session_id"] == "target-session"
    assert turns.lookups == []
    assert memory.lookups == ["user-msg"]


def test_steer_cannot_link_an_unrelated_message():
    view, _, _ = project(delivery(delivery_mode="steer"),
                         target_message=message(id="user-msg", role="user",
                                                author_entity_type="bot", author_entity_id="other"))
    assert view["target_message_id"] is None
    assert view["reply_status"] == "unattributed"


@pytest.mark.parametrize("field,value", [
    ("sender_bot_id", "another"), ("author_entity_type", "user"),
    ("author_entity_id", "another"), ("metadata", {"ordinary_chat": {}}),
    ("metadata", {"prompt_schedule": {}}), ("message_kind", "CONTEXT_MAINTENANCE"),
])
def test_wrong_sender_or_nonconversation_receipt_is_rejected(field, value):
    with pytest.raises(ReplyViewMismatch):
        project(delivery(**{field: value}))
    with pytest.raises(ReplyViewMismatch):
        project(sender_bot_id="another")


@pytest.mark.parametrize("field,value", [
    ("ended_at", None), ("status", "aborted"), ("error_text", "bad"),
    ("bot_id", "other"), ("user_id", "other"),
    ("trigger_message_id", "unrelated"), ("assistant_message_id", None),
])
def test_missing_or_wrong_target_turn_stays_unresolved(field, value):
    view, _, _ = project(target_turn=turn(**{field: value}))
    assert view["reply_status"] == "unresolved"
    assert view["reply_text"] is None


@pytest.mark.parametrize("field,value", [
    ("id", "other"), ("role", "user"), ("content", None), ("session_id", None),
    ("author_entity_type", "user"), ("author_entity_id", "other"),
])
def test_only_matching_canonical_assistant_message_is_exposed(field, value):
    view, _, _ = project(target_message=message(**{field: value}))
    assert view["reply_status"] == "unresolved"
    assert view["reply_text"] is None


def test_mismatched_origin_session_or_author_never_exposes_answer():
    for change in ({"session_id": "another-session"}, {"author_entity_id": "another-bot"},
                   {"author_entity_type": "user"}):
        incoming = dict(id="user-msg", role="user", session_id="target-session",
                        author_entity_type="bot", author_entity_id="sender")
        incoming.update(change)
        view, _, _ = project(incoming_message=incoming)
        assert view["reply_status"] == "unresolved"
        assert view["reply_text"] is None


def test_empty_canonical_reply_is_not_invented():
    view, _, _ = project(target_message=message(content=""))
    assert view["reply_status"] == "answered"
    assert view["reply_text"] == ""


def test_answer_route_requires_matching_stored_user_and_sender(monkeypatch):
    class Store:
        def get(self, _id):
            return delivery()
        def payload(self, _id):
            return {"user": "owner"}

    monkeypatch.setattr(inter_bot_deliveries, "_dispatcher", lambda: SimpleNamespace(store=Store()))
    monkeypatch.setattr(inter_bot_deliveries, "get_service", lambda: SimpleNamespace(
        _turn_log_store=Turns(turn()), get_memory_client=lambda *_: Memory(message())))
    assert inter_bot_deliveries.get_inter_bot_answer("delivery-test", sender_bot_id="sender", user_id="owner")["reply_text"] == "Canonical reply"
    for sender, user in [("other", "owner"), ("sender", "other")]:
        with pytest.raises(HTTPException) as error:
            inter_bot_deliveries.get_inter_bot_answer("delivery-test", sender_bot_id=sender, user_id=user)
        assert error.value.status_code == 404

    app = FastAPI()
    app.include_router(inter_bot_deliveries.router)
    client = TestClient(app)
    url = "/v1/inter-bot-deliveries/delivery-test/answer"
    assert client.get(url, params={"sender_bot_id": "sender", "user_id": "owner"}).json()["reply_text"] == "Canonical reply"
    assert client.get(url, params={"sender_bot_id": "sender", "user_id": "other"}).status_code == 404


def test_manual_retry_uses_the_original_send_card_as_its_single_anchor(monkeypatch):
    original = delivery(status="FAILED", last_error="temporary failure")
    child = delivery(id="delivery-retry", metadata={"manual_retry_of": original.id})

    class Store:
        def get(self, _id):
            return original
        def payload(self, _id):
            return {"user": "owner"}
        def list(self, **filters):
            assert filters["idempotency_key"] == "manual-retry:delivery-test"
            return [child]

    monkeypatch.setattr(inter_bot_deliveries, "_dispatcher", lambda: SimpleNamespace(store=Store()))
    monkeypatch.setattr(inter_bot_deliveries, "get_service", lambda: SimpleNamespace(
        _turn_log_store=Turns(turn()), get_memory_client=lambda *_: Memory(message())))
    view = inter_bot_deliveries.get_inter_bot_answer(
        "delivery-test", sender_bot_id="sender", user_id="owner")
    assert view["delivery_id"] == original.id
    assert view["retry_delivery_id"] == child.id
    assert view["reply_status"] == "answered"
    assert view["reply_text"] == "Canonical reply"


def test_unrelated_manual_retry_cannot_replace_original_failure(monkeypatch):
    original = delivery(status="FAILED")
    child = delivery(id="delivery-retry", metadata={"manual_retry_of": "unrelated"})

    class Store:
        def get(self, _id):
            return original
        def payload(self, _id):
            return {"user": "owner"}
        def list(self, **_filters):
            return [child]

    monkeypatch.setattr(inter_bot_deliveries, "_dispatcher", lambda: SimpleNamespace(store=Store()))
    monkeypatch.setattr(inter_bot_deliveries, "get_service", lambda: SimpleNamespace(
        _turn_log_store=Turns(turn()), get_memory_client=lambda *_: Memory(message())))
    view = inter_bot_deliveries.get_inter_bot_answer(
        "delivery-test", sender_bot_id="sender", user_id="owner")
    assert view["reply_status"] == "unavailable"
    assert "retry_delivery_id" not in view
