"""TASK-792 UI-originated delivery authorship regressions."""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from llm_bawt.inter_bot_reply_channel import (
    DEFAULT_REPLY_CHANNEL_ASYNC, DEFAULT_REPLY_CHANNEL_WAITED, append_request_reply_channel_note,
    delivery_reply_channel_note,
    request_reply_channel_note, resolve_reply_channel_note,
)
from llm_bawt.message_authorship import AuthorReference
from llm_bawt.service.routes.inter_bot_deliveries import (
    DeliveryCreate, _format_delivery_message, create_delivery,
)


def test_user_authored_delivery_is_not_wrapped_as_a_bot_message() -> None:
    message = "Repository workspace decision request: TASK-792"

    assert _format_delivery_message(message, AuthorReference.user("nick")) == message


def test_bot_authored_delivery_keeps_the_bot_wrapper() -> None:
    assert _format_delivery_message(
        "Review this change",
        AuthorReference.bot("loopy"),
    ) == "Message from bot 'loopy': Review this change"


@pytest.mark.parametrize("reply_mode", ["async", "waited"])
def test_stored_delivery_text_never_carries_the_reply_channel_note(reply_mode) -> None:
    """The note is instruction text: it rides the outbound copy only (TASK-1210 follow-up)."""
    assert _format_delivery_message(
        "Choose SPLIT or HOLD", AuthorReference.bot("nova"),
    ) == "Message from bot 'nova': Choose SPLIT or HOLD"


def _record(reply_mode="async", *, author=None, message_kind=None, target="loopy"):
    return SimpleNamespace(
        reply_mode=reply_mode, author=author or AuthorReference.bot("vex"),
        message_kind=message_kind, target_bot_id=target,
    )


def test_waited_delivery_note_tells_target_final_answer_is_the_reply() -> None:
    note = delivery_reply_channel_note(_record("waited"))
    assert note == DEFAULT_REPLY_CHANNEL_WAITED
    assert "Your final response on this turn is the answer" in note
    assert "Do not send a separate message back merely to repeat that answer" in note
    assert "intentional, distinct follow-up" in note


def test_async_delivery_note_names_the_sender_to_call_back() -> None:
    note = delivery_reply_channel_note(_record("async"))
    assert note == DEFAULT_REPLY_CHANNEL_ASYNC.replace("{sender_bot_id}", "vex")


@pytest.mark.parametrize("record", [
    None,
    _record(author=AuthorReference.user("nick")),
    _record(author=AuthorReference.bot("unknown")),
    _record(message_kind="CONTEXT_MAINTENANCE"),
])
def test_no_note_without_a_bot_reply_channel_or_for_maintenance(record) -> None:
    assert delivery_reply_channel_note(record) is None


class _Store:
    def __init__(self, record=None, error=None):
        self.record, self.error, self.calls = record, error, []

    def get(self, delivery_id):
        self.calls.append(delivery_id)
        if self.error:
            raise self.error
        return self.record


def _service(store):
    return SimpleNamespace(_inter_bot_dispatcher=SimpleNamespace(store=store))


def test_request_note_comes_from_the_claimed_delivery_record() -> None:
    store = _Store(_record("async"))
    request = SimpleNamespace(inter_bot_delivery_id="delivery-1")
    note = request_reply_channel_note(_service(store), request)
    assert 'bots_send_message(target_bot_id="vex")' in note
    assert store.calls == ["delivery-1"]


@pytest.mark.parametrize("service,request_", [
    (_service(_Store(_record())), SimpleNamespace(inter_bot_delivery_id=None)),
    (SimpleNamespace(_inter_bot_dispatcher=None), SimpleNamespace(inter_bot_delivery_id="d")),
    (_service(_Store(error=RuntimeError("db down"))), SimpleNamespace(inter_bot_delivery_id="d")),
])
def test_request_note_is_absent_for_ordinary_turns_and_lookup_failures(service, request_) -> None:
    assert request_reply_channel_note(service, request_) is None


def test_async_note_tells_target_to_reply_manually_to_the_sender() -> None:
    """The Loopy miss: an async review verdict stayed in the reviewer's chat."""
    note = resolve_reply_channel_note("async", sender_bot_id="vex", target_bot_id="loopy")
    assert note.startswith("[Reply channel: async delivery.")
    assert "is NOT returned to it" in note
    assert 'bots_send_message(target_bot_id="vex")' in note
    assert "{sender_bot_id}" not in note
    assert "If no response is needed, do not send one." in note


class _Resolved:
    def __init__(self, body: str) -> None:
        self.body = body


class _Resolver:
    def __init__(self, body: str | None = None, error: Exception | None = None) -> None:
        self.body, self.error, self.calls = body, error, []

    def resolve(self, key, scope_type="global", scope_id=None):
        self.calls.append((key, scope_type, scope_id))
        if self.error:
            raise self.error
        return _Resolved(self.body) if self.body is not None else None


def test_reply_channel_note_is_resolved_from_the_registry_per_target(monkeypatch) -> None:
    resolver = _Resolver("[Custom: reply to {sender_bot_id}; keep {\"json\": 1}]")
    monkeypatch.setattr("llm_bawt.prompt_registry.get_prompt_resolver", lambda: resolver)

    note = delivery_reply_channel_note(_record("waited"))

    assert note == '[Custom: reply to vex; keep {"json": 1}]'
    assert resolver.calls == [("inter_bot.reply_channel.waited", "bot", "loopy")]


@pytest.mark.parametrize("resolver", [_Resolver(body="   "), _Resolver(error=RuntimeError("db down"))])
def test_blank_or_failed_registry_falls_back_to_seed_note(monkeypatch, resolver) -> None:
    monkeypatch.setattr("llm_bawt.prompt_registry.get_prompt_resolver", lambda: resolver)
    note = resolve_reply_channel_note("async", sender_bot_id="vex", target_bot_id="loopy")
    assert note == DEFAULT_REPLY_CHANNEL_ASYNC.replace("{sender_bot_id}", "vex")


def test_reply_channel_keys_are_registered_with_seed_bodies() -> None:
    from llm_bawt.prompt_registry import DEFAULT_PROMPT_DEFINITIONS

    waited = DEFAULT_PROMPT_DEFINITIONS["inter_bot.reply_channel.waited"]
    async_ = DEFAULT_PROMPT_DEFINITIONS["inter_bot.reply_channel.async"]
    assert waited.default_body() == DEFAULT_REPLY_CHANNEL_WAITED
    assert async_.default_body() == DEFAULT_REPLY_CHANNEL_ASYNC
    assert async_.required_vars == ("sender_bot_id",)


@pytest.mark.parametrize("options", [
    {"prefer_steer": True},
    {"prefer_steer": False, "author_entity_type": "user", "author_entity_id": "nick"},
    {"prefer_steer": False, "sender_bot_id": "unknown"},
])
def test_waited_contract_rejects_steers_and_non_bot_senders(options) -> None:
    body = DeliveryCreate(target_bot_id="target", message="choose", reply_mode="waited",
                          **{"sender_bot_id": "sender", **options})
    with pytest.raises(HTTPException, match="Waited replies require"):
        asyncio.run(create_delivery(body))


def test_unknown_reply_mode_is_rejected_at_boundary() -> None:
    with pytest.raises(ValidationError):
        DeliveryCreate(target_bot_id="target", message="choose", reply_mode="steer")


def test_agent_suffix_keeps_the_manifest_and_appends_the_note() -> None:
    service = _service(_Store(_record("async")))
    request = SimpleNamespace(inter_bot_delivery_id="delivery-1")
    note = delivery_reply_channel_note(_record("async"))
    assert append_request_reply_channel_note(service, request, None) == note
    assert append_request_reply_channel_note(service, request, "MANIFEST") == f"MANIFEST\n\n{note}"
    plain = SimpleNamespace(inter_bot_delivery_id=None)
    assert append_request_reply_channel_note(service, plain, "MANIFEST") == "MANIFEST"
    assert append_request_reply_channel_note(service, plain, None) is None


def test_reset_follow_up_swap_keeps_the_outbound_note() -> None:
    """A typed reset follow-up replaces only the command span, not the note."""
    from llm_bawt.models.message import Message
    from llm_bawt.service.session_lifecycle.execution import NewSessionExecution

    command = "Message from bot 'vex': /new Review TASK-1"
    outbound = Message(role="user", content=f"{command}\n\n[Reply channel: async delivery.]")
    swapped = NewSessionExecution.bridge_messages([outbound], "Review TASK-1", command)
    assert swapped[-1].content == "Review TASK-1\n\n[Reply channel: async delivery.]"
    assert outbound.content.startswith(command)
