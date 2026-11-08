"""TASK-2156: /new to an unknown bot is refused before anything is written."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from llm_bawt.service.routes import chat as chat_routes


def _request(bot_id, text):
    return SimpleNamespace(bot_id=bot_id, messages=[{"role": "user", "content": text}])


def test_new_to_unknown_bot_is_404():
    with patch("llm_bawt.bots.get_bot", return_value=None) as get_bot:
        with pytest.raises(HTTPException) as exc:
            chat_routes._reject_reset_for_unknown_bot(_request("nosuchbot", "/new"))
    assert exc.value.status_code == 404
    assert "nosuchbot" in exc.value.detail
    get_bot.assert_called_once_with("nosuchbot")


def test_new_to_known_bot_passes():
    with patch("llm_bawt.bots.get_bot", return_value=object()):
        chat_routes._reject_reset_for_unknown_bot(_request("nova", "/new"))


def test_ordinary_message_and_default_bot_skip_lookup():
    with patch("llm_bawt.bots.get_bot") as get_bot:
        chat_routes._reject_reset_for_unknown_bot(_request("nosuchbot", "hello"))
        chat_routes._reject_reset_for_unknown_bot(_request(None, "/new"))
    get_bot.assert_not_called()


@pytest.mark.asyncio
async def test_route_refuses_before_admission():
    request = chat_routes.ChatCompletionRequest(
        bot_id="nosuchbot", messages=[{"role": "user", "content": "/new"}],
    )
    http_request = SimpleNamespace(headers={})
    with patch("llm_bawt.bots.get_bot", return_value=None), \
         patch.object(chat_routes, "get_service", return_value=SimpleNamespace()), \
         patch.object(chat_routes, "_internal_inter_bot_sender", return_value=None), \
         patch("llm_bawt.service.chat_response_adapter.completion_response") as respond:
        with pytest.raises(HTTPException) as exc:
            await chat_routes.chat_completions(request, http_request)
    assert exc.value.status_code == 404
    respond.assert_not_called()
