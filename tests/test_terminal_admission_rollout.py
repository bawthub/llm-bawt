"""Rollout is explicit, reset-safe, and never upgrades reserved/supplied IDs."""
import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llm_bawt.service import turn_terminal_admission as mod
from llm_bawt.setting_definitions import SETTING_DEFINITIONS


def test_defaults_leave_admission_off_and_pump_report_only():
    assert SETTING_DEFINITIONS["turn_terminal_protocol_enabled"].default is False
    assert SETTING_DEFINITIONS["terminal_recovery_mode"].default == "report"
    assert SETTING_DEFINITIONS["terminal_recovery_mode"].applies_to == ()


@pytest.mark.parametrize("enabled, mode, expected", [
    (False, "recover", False), (True, "report", False), (True, "off", False),
    (True, "invalid", False), (True, "recover", True), ("true", "recover", False),
])
def test_admission_requires_explicit_enable_and_mutating_recovery(monkeypatch, enabled, mode, expected):
    monkeypatch.setattr(mod, "resolve_bot_runtime_setting", Mock(return_value=enabled))
    monkeypatch.setattr(mod, "resolve_global_runtime_setting", Mock(return_value=mode))
    assert mod.protocol_admission_enabled(SimpleNamespace(REDIS_URL="redis://fixture"), "fixture",
                                          user_prompt="ordinary prompt") is expected


@pytest.mark.parametrize("config", [SimpleNamespace(), SimpleNamespace(REDIS_URL=None), SimpleNamespace(REDIS_URL="")])
def test_admission_without_recovery_transport_stays_legacy(monkeypatch, config):
    resolve = Mock(side_effect=AssertionError("no transport must not consult rollout"))
    monkeypatch.setattr(mod, "resolve_bot_runtime_setting", resolve)
    assert not mod.protocol_admission_enabled(config, "fixture", user_prompt="prompt")
    resolve.assert_not_called()


@pytest.mark.parametrize("options", [
    {"user_prompt": "/new"}, {"user_prompt": "/new follow up"},
    {"session_policy": "reset_retain_history"}, {"session_policy": "reset_without_history"},
    {"reset_intent": True}, {"reset_only": True},
])
def test_reset_receipt_paths_stay_legacy_without_setting_queries(monkeypatch, options):
    resolve = Mock(side_effect=AssertionError("ineligible reset must not consult rollout"))
    monkeypatch.setattr(mod, "resolve_bot_runtime_setting", resolve)
    assert not mod.protocol_admission_enabled(SimpleNamespace(REDIS_URL="redis://fixture"), "fixture",
                                               **{"user_prompt": "prompt", **options})
    resolve.assert_not_called()


@pytest.mark.parametrize("supplied, claimed", [(True, False), (False, True), (True, True)])
def test_existing_identity_uses_initial_marker_not_current_rollout(monkeypatch, supplied, claimed):
    resolve = Mock(side_effect=AssertionError("never adopt existing admission"))
    monkeypatch.setattr(mod, "protocol_admission_enabled", resolve)
    request = SimpleNamespace(inter_bot_turn_id="existing" if supplied else None)
    assert asyncio.run(mod.direct_protocol_options(object(), "fixture", request, "prompt",
                                                   object() if claimed else None)) == {}
    resolve.assert_not_called()


def test_fresh_direct_admission_resolves_settings_off_loop(monkeypatch):
    import threading

    loop_thread = threading.get_ident()

    def resolve(*args, **kwargs):
        assert threading.get_ident() != loop_thread
        return True

    monkeypatch.setattr(mod, "protocol_admission_enabled", resolve)
    result = asyncio.run(mod.direct_protocol_options(object(), "fixture", SimpleNamespace(), "prompt", None))
    assert result == {"terminal_protocol": True}
