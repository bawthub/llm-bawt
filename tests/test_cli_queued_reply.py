"""Queued CLI replies reattach to one accepted delivery, never resubmit."""
from __future__ import annotations

import io
from types import SimpleNamespace

import pytest
from rich.console import Console

from llm_bawt.service.client import ServiceClient


ACK = {
    "object": "chat.completion.queued", "delivery_id": "delivery-1",
    "user_message_id": "user-message-1", "turn_id": "turn-delivery-1",
}


def test_ordinary_receipt_exposes_backend_resolved_scope():
    from llm_bawt.service.ordinary_turn_queue import OrdinaryTurnDecision

    receipt = OrdinaryTurnDecision(
        "queued", "turn-delivery-1", "user-message-1", "assistant-message-1",
        "delivery-1", "QUEUED", bot_id="resolved-bot", user_id="resolved-user",
    ).receipt()
    assert receipt["bot_id"] == "resolved-bot"
    assert receipt["user_id"] == "resolved-user"


def test_client_resolves_missing_scope_from_exact_delivery(monkeypatch):
    client = ServiceClient(http_url="http://example.invalid")
    paths = []

    def request(method, path, **kwargs):
        paths.append((method, path, kwargs.get("params")))
        if path.endswith("/reply"):
            assert kwargs["params"] == {
                "bot_id": "resolved-bot", "user_id": "resolved-user",
                "user_message_id": "user-message-1",
            }
            return {"delivery_id": "delivery-1", "status": "DELIVERED", "response_text": "done"}
        return {"delivery_id": "delivery-1", "user_message_id": "user-message-1",
                "target_bot_id": "resolved-bot",
                "author": {"entity_type": "user", "entity_id": "resolved-user"}}

    monkeypatch.setattr(client, "_request", request)
    assert client.wait_for_queued_reply(ACK, bot_id=None, user_id=None) == {
        "delivery_id": "delivery-1", "status": "DELIVERED", "response_text": "done",
    }
    assert len(paths) == 2


def test_client_rejects_delivery_scope_mismatch(monkeypatch):
    client = ServiceClient(http_url="http://example.invalid")
    monkeypatch.setattr(client, "_request", lambda *_args, **_kwargs: {
        "delivery_id": "delivery-1", "user_message_id": "user-message-1",
        "target_bot_id": "another-bot", "author": {"entity_type": "user", "entity_id": "resolved-user"},
    })
    with pytest.raises(ValueError, match="changed scope"):
        client.wait_for_queued_reply(ACK, bot_id="selected-bot", user_id=None)


def test_queued_cli_pending_is_handled_without_second_query(monkeypatch):
    from llm_bawt.cli import display_cmd

    client = ServiceClient(http_url="http://example.invalid")
    monkeypatch.setattr(client, "is_available", lambda **_kwargs: True)
    monkeypatch.setattr(client, "chat_completion", lambda **_kwargs: dict(ACK))
    monkeypatch.setattr(client, "wait_for_queued_reply", lambda *_args, **_kwargs: None)
    config = SimpleNamespace(VERBOSE=False)
    monkeypatch.setattr(display_cmd, "Config", lambda: config)
    monkeypatch.setattr(display_cmd, "BotManager", lambda _config: SimpleNamespace(
        get_bot=lambda _id: SimpleNamespace(name="Selected bot", color="green")))
    monkeypatch.setattr(display_cmd, "get_service_client", lambda _config: client)
    output = io.StringIO()
    monkeypatch.setattr(display_cmd, "console", Console(file=output))
    assert display_cmd.query_via_service("hello", model=None, bot_id="selected-bot",
                                         user_id="selected-user", plaintext_output=True,
                                         stream=False, config=config) is True
    assert "accepted" in output.getvalue()
