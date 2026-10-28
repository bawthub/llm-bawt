"""Regression tests for bot profile loading and prompt cache invalidation."""

from __future__ import annotations

from typing import Any

from llm_bawt import bots as bots_module
from llm_bawt.service.background_service import BackgroundService


def test_db_bot_profiles_load_correctly(monkeypatch) -> None:
    """Bot profiles should load from DB and build Bot instances correctly."""
    db_profiles = {
        "nova": {
            "name": "Nova DB",
            "description": "from db",
            "system_prompt": "db prompt",
            "requires_memory": False,
            "voice_optimized": True,
            "tts_mode": False,
            "include_summaries": True,
            "include_in_global_search": False,
            "uses_tools": True,
            "uses_search": True,
            "uses_home_assistant": True,
            "default_model": "db-model",
            "nextcloud": {"bot_id": "nc-nova"},
        },
        "ember": {
            "name": "Ember",
            "description": "db only bot",
            "system_prompt": "ember prompt",
            "requires_memory": True,
            "voice_optimized": False,
            "tts_mode": False,
            "include_summaries": True,
            "include_in_global_search": True,
            "uses_tools": False,
            "uses_search": False,
            "uses_home_assistant": False,
            "default_model": None,
            "nextcloud": None,
        },
        "claw": {
            "name": "Claw",
            "description": "codex agent bot",
            "system_prompt": "claw prompt",
            "requires_memory": True,
            "voice_optimized": False,
            "tts_mode": False,
            "include_summaries": True,
            "include_in_global_search": True,
            "uses_tools": False,
            "uses_search": False,
            "uses_home_assistant": False,
            "default_model": None,
            "nextcloud": None,
            "agent_backend": "codex",
            "agent_backend_config": {"session_key": "agent:claw:main"},
        },
    }

    monkeypatch.setattr(bots_module, "_load_db_bot_profiles", lambda: db_profiles)

    bots_module._load_bots_config()

    nova = bots_module.get_bot("nova")
    assert nova is not None
    assert nova.system_prompt == "db prompt"
    assert nova.description == "from db"
    assert nova.requires_memory is False
    assert nova.voice_optimized is True
    assert nova.include_in_global_search is False
    assert nova.default_model == "db-model"
    assert nova.bot_type == "chat"
    assert bots_module.get_raw_bot_data("nova")["nextcloud"]["bot_id"] == "nc-nova"

    ember = bots_module.get_bot("ember")
    assert ember is not None
    assert ember.system_prompt == "ember prompt"
    assert ember.bot_type == "chat"
    assert ember.include_in_global_search is True

    claw = bots_module.get_bot("claw")
    assert claw is not None
    assert claw.bot_type == "agent"
    assert claw.agent_backend == "codex"
    assert bots_module.get_raw_bot_data("claw")["bot_type"] == "agent"


def test_background_service_invalidates_bot_instances() -> None:
    """Bot-level cache invalidation should only evict matching bot keys."""
    service = BackgroundService.__new__(BackgroundService)
    service._llm_bawt_cache = {
        ("model-a", "nova", "u1"): object(),
        ("model-b", "nova", "u2"): object(),
        ("model-a", "spark", "u1"): object(),
    }

    removed = service.invalidate_bot_instances("nova")

    assert removed == 2
    assert ("model-a", "spark", "u1") in service._llm_bawt_cache
    assert ("model-a", "nova", "u1") not in service._llm_bawt_cache
    assert ("model-b", "nova", "u2") not in service._llm_bawt_cache


def test_background_service_invalidates_all_instances() -> None:
    """Global cache invalidation should clear all cached instances."""
    service = BackgroundService.__new__(BackgroundService)
    service._llm_bawt_cache = {
        ("model-a", "nova", "u1"): object(),
        ("model-b", "spark", "u2"): object(),
    }

    removed = service.invalidate_all_instances()

    assert removed == 2
    assert service._llm_bawt_cache == {}


def test_background_service_clears_session_model_overrides_for_bot() -> None:
    """Clearing model overrides by bot should only remove matching bot sessions."""
    service = BackgroundService.__new__(BackgroundService)
    service._session_model_overrides = {
        ("nova", "u1"): "grok-4-fast",
        ("nova", "u2"): "grok-4-fast",
        ("spark", "u1"): "gpt-4.1",
    }

    removed = service.clear_session_model_overrides(bot_id="nova")

    assert removed == 2
    assert ("spark", "u1") in service._session_model_overrides
    assert ("nova", "u1") not in service._session_model_overrides
    assert ("nova", "u2") not in service._session_model_overrides


def test_background_service_clears_all_session_model_overrides() -> None:
    """Clearing with no scope should remove every session override."""
    service = BackgroundService.__new__(BackgroundService)
    service._session_model_overrides = {
        ("nova", "u1"): "grok-4-fast",
        ("spark", "u1"): "gpt-4.1",
    }

    removed = service.clear_session_model_overrides()

    assert removed == 2
    assert service._session_model_overrides == {}
