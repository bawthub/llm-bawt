"""Bounded regressions for TASK-2280; no live state or credentials."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine, text

from llm_bawt import bots, runtime_settings


def configured_store(monkeypatch, store):
    monkeypatch.setattr("llm_bawt.utils.config.has_database_credentials", lambda _: True)
    monkeypatch.setattr(runtime_settings, "BotProfileStore", lambda _: store)


def test_registry_reload_does_not_dispose_shared_engine(monkeypatch):
    engine = SimpleNamespace(dispose=Mock())
    store = SimpleNamespace(engine=engine, list_all=lambda: [], init_error=None)
    configured_store(monkeypatch, store)
    monkeypatch.setattr(bots, "_load_bot_tags", lambda _: {})
    assert bots._load_db_bot_profiles() == {}
    engine.dispose.assert_not_called()


def test_nextcloud_config_cannot_create_missing_bot(monkeypatch):
    store = SimpleNamespace(engine=object(), get=lambda _: None, upsert=Mock())
    configured_store(monkeypatch, store)
    with pytest.raises(ValueError, match="not found"):
        bots.save_user_bot_config("missing-fixture", "nextcloud", {"bot_id": "fixture"})
    store.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_nextcloud_route_rejects_missing_profile_before_remote_provision(monkeypatch):
    from fastapi import HTTPException
    from llm_bawt.service.routes import nextcloud as routes
    from llm_bawt.integrations.nextcloud import provisioner
    from llm_bawt.service.schemas import NextcloudProvisionRequest
    monkeypatch.setattr(routes, "get_service", lambda: SimpleNamespace(config=None))
    monkeypatch.setattr(routes, "get_bot_profile_store",
                        lambda config: SimpleNamespace(get=lambda slug: None))
    factory = Mock()
    monkeypatch.setattr(provisioner, "get_provisioner_client", factory)
    with pytest.raises(HTTPException) as raised:
        await routes.provision_nextcloud_talk(NextcloudProvisionRequest(bot_id="missing-fixture"))
    assert raised.value.status_code == 404
    factory.assert_not_called()


def test_nextcloud_config_update_requires_existing_profile(monkeypatch):
    store = SimpleNamespace(engine=object(), get=lambda _: SimpleNamespace(), upsert=Mock())
    configured_store(monkeypatch, store)
    monkeypatch.setattr(bots, "_load_bots_config", lambda: None)
    bots.save_user_bot_config("fixture", "nextcloud", {"bot_id": "fixture"})
    store.upsert.assert_called_once_with(
        {"slug": "fixture", "nextcloud_config": {"bot_id": "fixture"}}, require_existing=True)


def test_nextcloud_removal_clears_only_the_requested_section(monkeypatch):
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE bot_profiles (slug TEXT, name TEXT, nextcloud_config TEXT, updated_at TEXT)"))
        conn.execute(text("INSERT INTO bot_profiles VALUES ('fixture', 'Kept', '{}', NULL)"))
    store = SimpleNamespace(engine=engine, get=lambda _: SimpleNamespace(nextcloud_config={}))
    configured_store(monkeypatch, store)
    invalidate = Mock()
    monkeypatch.setattr(bots, "invalidate_bots_cache", invalidate)
    try:
        assert bots.remove_user_bot_section("fixture", "nextcloud")
        with engine.connect() as conn:
            row = conn.execute(text("SELECT name, nextcloud_config FROM bot_profiles")).one()
        assert tuple(row) == ("Kept", None)
        assert not bots.remove_user_bot_section("fixture", "nextcloud")
    finally:
        engine.dispose()


@pytest.mark.parametrize("raw,expected", [
    ("Compute 2 * 3 = 6, then 6 * 4 = 24.", "Compute 2 * 3 = 6, then 6 * 4 = 24."),
    ("std::vector and std::map", "std::vector and std::map"),
    ("*smiles warmly* Hello ::waves::", "Hello"),
    ("*line one\nline two*", "*line one\nline two*"),
])
def test_emote_removal_preserves_ordinary_text(raw, expected):
    assert bots.strip_emotes(raw) == expected


def test_purge_evicts_actual_partition_caches(monkeypatch):
    from llm_bawt.bot_purge_execute import evict_partition_caches
    from llm_bawt.memory import message_store, postgresql
    memory = {"memories_p_fixtureagent": object(), "memories_p_other": object()}
    messages = {"messages_p_fixtureagent": object(), "messages_p_other": object()}
    forgotten = {"forgotten_messages_p_fixtureagent": object()}
    monkeypatch.setattr(postgresql, "_memory_table_cache", memory)
    monkeypatch.setattr(message_store, "_message_table_cache", messages)
    monkeypatch.setattr(message_store, "_forgotten_table_cache", forgotten)
    evict_partition_caches("fixture-agent")
    assert set(memory) == {"memories_p_other"}
    assert set(messages) == {"messages_p_other"}
    assert forgotten == {}


@pytest.mark.parametrize("bot,global_value,expected,source", [
    (None, None, 10, "code_default"),
    (None, 20, 20, "global_override"),
    (False, True, False, "bot_override"),
    (0, 20, 0, "bot_override"),
])
def test_null_runtime_override_is_absent_not_a_value(bot, global_value, expected, source):
    resolver = runtime_settings.RuntimeSettingsResolver.__new__(runtime_settings.RuntimeSettingsResolver)
    resolver._bot_cache = {"inbox_batch": bot}
    resolver._global_cache = {"inbox_batch": global_value}
    resolver._refresh_cache_if_needed = lambda: None
    assert resolver.resolve_with_source("inbox_batch", fallback=10) == (expected, source)
