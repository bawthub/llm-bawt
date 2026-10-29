"""TASK-1702: one effective-voice resolver (bot ?? global ?? none)."""

from types import SimpleNamespace

import pytest

from llm_bawt import voice_resolution
from llm_bawt.mcp_server import home_audio_tools
from llm_bawt.setting_definitions import SETTING_DEFINITIONS
from llm_bawt.voice_resolution import EffectiveVoice, resolve_bot_voice


def test_bot_voice_wins_over_global():
    assert resolve_bot_voice("bot-v", "global-v") == EffectiveVoice("bot-v", "bot")


def test_global_voice_used_when_bot_has_none():
    assert resolve_bot_voice(None, "global-v") == EffectiveVoice("global-v", "global")
    assert resolve_bot_voice("  ", "global-v") == EffectiveVoice("global-v", "global")


def test_no_voice_when_neither_is_set():
    assert resolve_bot_voice(None, None) == EffectiveVoice(None, None)
    assert resolve_bot_voice("", " ") == EffectiveVoice(None, None)


def test_global_default_voice_is_global_only_with_no_code_default():
    definition = SETTING_DEFINITIONS["default_voice"]
    assert definition.applies_to == ()
    assert definition.default is None


def _patch_bot(monkeypatch, bot_voice, global_voice):
    import llm_bawt.bots as bots

    monkeypatch.setattr(bots, "get_bot", lambda slug: SimpleNamespace(default_voice=bot_voice))
    monkeypatch.setattr(voice_resolution, "global_default_voice", lambda config: global_voice)


def test_home_audio_uses_global_voice_for_bot_without_one(monkeypatch):
    _patch_bot(monkeypatch, None, "global-v")
    assert home_audio_tools._voice("somebot", None) == "global-v"


def test_home_audio_refuses_when_no_voice_resolves(monkeypatch):
    _patch_bot(monkeypatch, None, None)
    with pytest.raises(ValueError, match="no global default voice"):
        home_audio_tools._voice("somebot", None)


def test_home_audio_explicit_voice_still_wins(monkeypatch):
    _patch_bot(monkeypatch, "bot-v", "global-v")
    assert home_audio_tools._voice("somebot", " given ") == "given"
