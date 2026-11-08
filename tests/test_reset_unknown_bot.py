"""TASK-2156: /new to an unknown bot is refused before anything is written."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from llm_bawt import bots
from llm_bawt.service.routes import chat as chat_routes

SERVICE = SimpleNamespace(config=None)


def _request(bot_id, text):
    return SimpleNamespace(bot_id=bot_id, messages=[{"role": "user", "content": text}])


@pytest.fixture
def registry(monkeypatch):
    """A warm registry with one known bot and no pending reload."""
    monkeypatch.setattr(bots, "BUILTIN_BOTS", {"nova": object()})
    monkeypatch.setattr(bots, "_LOADED", True)
    monkeypatch.setattr(bots, "_FORCE_RELOAD", False)
    monkeypatch.setattr(bots, "_RETRY_AFTER", 0.0)
    return bots


def test_new_to_unknown_bot_is_404(registry):
    with pytest.raises(HTTPException) as exc:
        chat_routes._reject_reset_for_unknown_bot(SERVICE, _request("nosuchbot", "/new"))
    assert exc.value.status_code == 404
    assert "nosuchbot" in exc.value.detail


def test_new_to_known_bot_passes(registry):
    chat_routes._reject_reset_for_unknown_bot(SERVICE, _request("Nova ", "/new"))


def test_pending_registry_reload_finds_newly_committed_bot(registry, monkeypatch):
    """A failed earlier reload must be retried before declaring the bot unknown."""
    beta = object()
    monkeypatch.setattr(bots, "_FORCE_RELOAD", True)

    def recovered_load():
        bots.BUILTIN_BOTS = {"nova": object(), "beta": beta}
        bots._LOADED = True

    monkeypatch.setattr(bots, "_load_bots_config", recovered_load)
    chat_routes._reject_reset_for_unknown_bot(SERVICE, _request("beta", "/new"))
    assert bots._FORCE_RELOAD is False


def test_ordinary_message_and_default_bot_skip_lookup():
    with patch.object(bots.BotManager, "get_bot") as get_bot:
        chat_routes._reject_reset_for_unknown_bot(SERVICE, _request("nosuchbot", "hello"))
        chat_routes._reject_reset_for_unknown_bot(SERVICE, _request(None, "/new"))
    get_bot.assert_not_called()


@pytest.mark.asyncio
async def test_route_refuses_before_admission(registry):
    request = chat_routes.ChatCompletionRequest(
        bot_id="nosuchbot", messages=[{"role": "user", "content": "/new"}],
    )
    http_request = SimpleNamespace(headers={})
    with patch.object(chat_routes, "get_service", return_value=SERVICE), \
         patch.object(chat_routes, "_internal_inter_bot_sender", return_value=None), \
         patch("llm_bawt.service.chat_response_adapter.completion_response") as respond:
        with pytest.raises(HTTPException) as exc:
            await chat_routes.chat_completions(request, http_request)
    assert exc.value.status_code == 404
    respond.assert_not_called()
