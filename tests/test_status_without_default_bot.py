"""Status must answer when no default bot is configured (TASK-2404)."""

from types import SimpleNamespace

import pytest

import llm_bawt.bots as bots_mod
import llm_bawt.core.status as status_mod
import llm_bawt.model_manager as mm
import llm_bawt.utils.bot_identity as ident


class _Manager:
    def __init__(self, config, local_only=False):
        self._bots = {"alpha": SimpleNamespace(slug="alpha", name="Alpha")}

    def get_default_bot(self):  # pragma: no cover - must not be called
        raise AssertionError("status must not require a default bot")

    def get_bot(self, slug):
        return self._bots.get(slug)

    def list_bots(self):
        return list(self._bots.values())

    def select_model(self, alias, bot_slug=None):
        return SimpleNamespace(alias=alias or f"{bot_slug}-model", source="bot")


@pytest.fixture
def status(monkeypatch):
    monkeypatch.setattr(bots_mod, "BotManager", _Manager)
    monkeypatch.setattr(mm, "is_service_mode_enabled", lambda config: False)
    monkeypatch.setattr(status_mod, "_collect_service_info", lambda config, local_only=False: (None, None))
    seen = {}
    monkeypatch.setattr(status_mod, "_collect_mcp_info", lambda config, slug: seen.setdefault("mcp", slug))
    monkeypatch.setattr(status_mod, "_collect_memory_info", lambda config, slug: seen.setdefault("mem", slug))
    monkeypatch.setattr(status_mod, "_collect_model_info", lambda config, alias: None)
    monkeypatch.setattr(status_mod, "_collect_dependencies", lambda config: None)
    config = SimpleNamespace(
        DEFAULT_USER="nick", SERVICE_HOST="127.0.0.1", SERVICE_PORT=8642, defined_models={"models": {}},
        SCHEDULER_ENABLED=False, SCHEDULER_CHECK_INTERVAL_SECONDS=60,
    )
    return config, seen


def test_status_without_default_bot(status, monkeypatch):
    config, seen = status
    monkeypatch.setattr(ident, "optional_default_bot_id", lambda config: None)
    result = status_mod.collect_system_status(config)
    assert result.config.bot_slug == ""
    assert result.config.model_alias is None
    assert [b.is_default for b in result.config.all_bots] == [False]
    assert seen == {"mcp": None, "mem": None}


def test_status_without_default_bot_uses_explicit_bot(status, monkeypatch):
    config, _ = status
    monkeypatch.setattr(ident, "optional_default_bot_id", lambda config: None)
    result = status_mod.collect_system_status(config, bot_slug="alpha")
    assert result.config.bot_slug == "alpha"
    assert result.config.model_alias == "alpha-model"


def test_status_with_default_bot(status, monkeypatch):
    config, seen = status
    monkeypatch.setattr(ident, "optional_default_bot_id", lambda config: "alpha")
    result = status_mod.collect_system_status(config)
    assert result.config.bot_slug == "alpha"
    assert [b.is_default for b in result.config.all_bots] == [True]
    assert seen == {"mcp": "alpha", "mem": "alpha"}
