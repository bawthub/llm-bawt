"""A fresh install with no default bot starts; bot-scoped use still fails loudly (TASK-2404)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from llm_bawt.service.background_service import BackgroundService
from llm_bawt.service.routes import setup as setup_routes
from llm_bawt.utils import bot_identity
from llm_bawt.utils.bot_identity import MissingBotIdentity, optional_default_bot_id


def _botless(**extra):
    return SimpleNamespace(DEFAULT_BOT="", **extra)


def test_env_default_wins_without_db_lookup():
    with patch("llm_bawt.runtime_setting_resolution.resolve_global_runtime_setting") as resolve:
        assert optional_default_bot_id(SimpleNamespace(DEFAULT_BOT=" Alpha ")) == "alpha"
    resolve.assert_not_called()


def test_botless_install_has_no_default():
    with patch("llm_bawt.utils.config.has_database_credentials", return_value=False):
        assert optional_default_bot_id(_botless()) is None


def test_db_default_used_when_env_unset():
    with (
        patch("llm_bawt.utils.config.has_database_credentials", return_value=True),
        patch("llm_bawt.runtime_setting_resolution.resolve_global_runtime_setting", return_value="Helper"),
    ):
        assert optional_default_bot_id(_botless()) == "helper"
        assert bot_identity.configured_default_bot_id(_botless()) == "helper"


def test_unreadable_db_default_counts_as_unset():
    with (
        patch("llm_bawt.utils.config.has_database_credentials", return_value=True),
        patch("llm_bawt.runtime_setting_resolution.resolve_global_runtime_setting", side_effect=RuntimeError("db down")),
    ):
        assert optional_default_bot_id(_botless()) is None


def test_service_default_is_resolved_on_use_and_still_loud():
    service = BackgroundService.__new__(BackgroundService)
    service.config = _botless()
    service._default_bot_override = None
    with patch("llm_bawt.utils.config.has_database_credentials", return_value=False):
        with pytest.raises(MissingBotIdentity):
            _ = service._default_bot
    service._default_bot = "beta"
    assert service._default_bot == "beta"


def test_first_bot_becomes_default_only_when_none_configured():
    store = MagicMock()
    store.engine = MagicMock()
    service = SimpleNamespace(config=_botless())
    profiles = MagicMock()
    with (
        patch.object(setup_routes, "get_bot_profile_store", return_value=profiles),
        patch.object(setup_routes, "get_runtime_settings_store", return_value=store),
        patch.object(setup_routes, "optional_default_bot_id", return_value=None),
    ):
        setup_routes._record_first_default_bot(service, "Helper")
    store.set_value.assert_called_once_with("global", "*", "default_bot", "helper")

    store.reset_mock()
    with (
        patch.object(setup_routes, "get_runtime_settings_store", return_value=store),
        patch.object(setup_routes, "optional_default_bot_id", return_value="existing"),
    ):
        setup_routes._record_first_default_bot(service, "Helper")
    store.set_value.assert_not_called()
