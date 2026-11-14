"""TASK-2284: settings read only at global scope are not bot-editable, and a
stale bot row for one never resolves (or reports) as the effective value."""

from types import SimpleNamespace

import pytest

from llm_bawt.config_resolver import ConfigResolver
from llm_bawt.runtime_settings import RuntimeSettingsResolver
from llm_bawt.setting_definitions import (
    SETTING_DEFINITIONS,
    STORAGE_RUNTIME_SETTING,
    applies_to_bot_type,
    is_global_only_runtime_setting,
)

# Every reader of these resolves the global scope only (resolve_job_model,
# resolve_global_runtime_setting, or RuntimeSettingsResolver(bot=None)).
GLOBAL_READ_KEYS = (
    "skills_skill_token_cap",
    "skills_total_token_cap",
    "claude_code_disallowed_tools",
    "maintenance_model",
    "extraction_model",
    "profile_maintenance_model",
    "maintenance_model_fallback",
    "maintenance_reasoning_effort",
    "maintenance_fallback_cooldown_seconds",
    "model_context_window_default",
    "max_output_tokens",
)


@pytest.mark.parametrize("key", GLOBAL_READ_KEYS)
def test_global_read_settings_are_declared_global_only(key):
    d = SETTING_DEFINITIONS[key]
    assert d.storage == STORAGE_RUNTIME_SETTING
    # Empty applies_to is the bot-create refusal and the "not consumed" flag.
    assert d.applies_to == ()
    assert applies_to_bot_type(key, "chat") is False
    assert applies_to_bot_type(key, "agent") is False
    assert is_global_only_runtime_setting(key) is True


def test_bot_editable_and_unknown_keys_are_not_global_only():
    assert is_global_only_runtime_setting("temperature") is False
    assert is_global_only_runtime_setting("no_such_setting_key") is False


def _resolver(bot_rows, global_rows):
    r = RuntimeSettingsResolver.__new__(RuntimeSettingsResolver)
    r.config = SimpleNamespace()
    r.bot = None
    r.bot_id = "somebot"
    r.cache_ttl_seconds = 5.0
    r.store = SimpleNamespace(engine=None)  # no refresh: use the seeded caches
    r._bot_cache = dict(bot_rows)
    r._global_cache = dict(global_rows)
    r._cache_loaded_at = 0.0
    return r


def test_stale_bot_row_for_global_only_key_resolves_global():
    r = _resolver({"max_output_tokens": 999}, {"max_output_tokens": 8192})
    assert r.resolve_with_source("max_output_tokens", fallback=4096) == (8192, "global_override")


def test_stale_bot_row_for_global_only_key_falls_to_default():
    r = _resolver({"maintenance_model": "bot-only-model"}, {})
    assert r.resolve_with_source("maintenance_model", fallback="d") == ("d", "code_default")


def test_bot_editable_key_still_prefers_bot_row():
    r = _resolver({"temperature": 0.2}, {"temperature": 0.7})
    assert r.resolve_with_source("temperature") == (0.2, "bot_override")


def test_effective_config_scalar_reports_global_scope_for_global_only_key():
    cr = ConfigResolver.__new__(ConfigResolver)
    cr._settings = _resolver({"max_output_tokens": 999}, {"max_output_tokens": 8192})
    cr.bot_id = "somebot"
    resolved = cr.resolve_scalar("max_output_tokens", fallback=4096)
    assert (resolved.value, resolved.scope_type, resolved.scope_id) == (8192, "global", "*")
    assert resolved.source != "bot_override"
