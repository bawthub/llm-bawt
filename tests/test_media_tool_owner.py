"""TASK-1383: media tools resolve their owner before any provider work."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_bawt.mcp_server import home_audio_tools, media_generation_tools


def _no_user(_user_id):
    raise ValueError("user_id is required")


def test_generate_image_without_user_does_not_generate(monkeypatch):
    service = MagicMock(generate_image=AsyncMock())
    monkeypatch.setattr(media_generation_tools, "_generation_service", service)
    monkeypatch.setattr(media_generation_tools, "resolve_user_id", _no_user)
    monkeypatch.setattr(media_generation_tools, "get_media_store", MagicMock())
    result = asyncio.run(media_generation_tools.generate_image(prompt="a dog"))
    assert result == [{"error": "user_id is required"}]
    service.generate_image.assert_not_called()


def test_speech_generate_without_user_does_not_render(monkeypatch):
    load = MagicMock()
    monkeypatch.setattr(home_audio_tools, "resolve_user_id", _no_user)
    monkeypatch.setattr(home_audio_tools.HomeAudioSettings, "load", load)
    with pytest.raises(ValueError, match="user_id is required"):
        asyncio.run(home_audio_tools.speech_generate(text="hi", bot_id="bot"))
    load.assert_not_called()
