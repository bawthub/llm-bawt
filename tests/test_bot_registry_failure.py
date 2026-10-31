"""A failed bot-profile load is an error state, never an empty registry (TASK-1819).

Incident: a startup lock timeout on bot_profiles DDL made the loader return {}
and mark the registry loaded. /v1/bots answered 200 [] for good, and BawtHub
sent an installed system to first-run setup.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from llm_bawt import bots as bots_module
from llm_bawt.bots import BotRegistryUnavailable

PROFILE = {"name": "Alpha", "description": "", "system_prompt": "p", "requires_memory": False}


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    """Isolate the module registry and its clock."""
    bots = dict(bots_module.BUILTIN_BOTS)
    raw = dict(bots_module._RAW_BOT_DATA)
    for name in ("_LOADED", "_FORCE_RELOAD", "_LOAD_ERROR", "_RETRY_AFTER"):
        monkeypatch.setattr(bots_module, name, getattr(bots_module, name))
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(bots_module, "time", SimpleNamespace(monotonic=lambda: clock.now))
    bots_module.BUILTIN_BOTS.clear()
    bots_module._RAW_BOT_DATA.clear()
    bots_module._LOADED = False
    bots_module._FORCE_RELOAD = False
    bots_module._LOAD_ERROR = None
    bots_module._RETRY_AFTER = 0.0
    yield clock
    bots_module.BUILTIN_BOTS.clear()
    bots_module.BUILTIN_BOTS.update(bots)
    bots_module._RAW_BOT_DATA.clear()
    bots_module._RAW_BOT_DATA.update(raw)


def _loader(monkeypatch, results: list):
    calls = []

    def load():
        calls.append(1)
        result = results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(bots_module, "_load_db_bot_profiles", load)
    return calls


def test_cold_failure_is_an_error_not_zero_bots_and_retries_after_backoff(monkeypatch, registry) -> None:
    calls = _loader(monkeypatch, [BotRegistryUnavailable("lock timeout"), {"alpha": PROFILE}])

    assert bots_module.BotManager().list_bots() == []
    assert bots_module.bot_registry_error() == "lock timeout"
    assert bots_module._LOADED is False
    assert len(calls) == 1  # inside the backoff: no reload storm

    registry.now += bots_module._LOAD_RETRY_SECONDS
    assert [b.slug for b in bots_module.BotManager().list_bots()] == ["alpha"]
    assert bots_module.bot_registry_error() is None
    assert len(calls) == 2


def test_failed_reload_keeps_the_last_good_registry(monkeypatch, registry) -> None:
    calls = _loader(monkeypatch, [{"alpha": PROFILE}, BotRegistryUnavailable("db gone"), {"alpha": PROFILE, "beta": PROFILE}])
    bots_module._load_bots_config()

    bots_module.invalidate_bots_cache()
    assert [b.slug for b in bots_module.BotManager().list_bots()] == ["alpha"]
    assert bots_module.bot_registry_error() == "db gone"

    # A failed reload stays pending; an explicit invalidate skips the backoff.
    bots_module.invalidate_bots_cache()
    assert [b.slug for b in bots_module.BotManager().list_bots()] == ["alpha", "beta"]
    assert bots_module.bot_registry_error() is None
    assert len(calls) == 3


def test_named_prompts_do_not_depend_on_the_db_read(monkeypatch) -> None:
    _loader(monkeypatch, [BotRegistryUnavailable("down")])
    bots_module._load_bots_config()
    assert bots_module._SYSTEM_PROMPTS.keys() == bots_module._BUILTIN_SYSTEM_PROMPTS.keys()


def _configured_db(monkeypatch, store):
    from llm_bawt import runtime_settings
    from llm_bawt.utils import config as config_module

    monkeypatch.setattr(config_module, "has_database_credentials", lambda config: True)
    monkeypatch.setattr(runtime_settings, "BotProfileStore", lambda config: store)


def test_unusable_store_raises_instead_of_returning_no_profiles(monkeypatch) -> None:
    _configured_db(monkeypatch, SimpleNamespace(engine=None, init_error="bot profiles DB unavailable: lock timeout"))
    with pytest.raises(BotRegistryUnavailable, match="lock timeout"):
        bots_module._load_db_bot_profiles()


def test_failed_profile_read_raises_instead_of_returning_no_profiles(monkeypatch) -> None:
    def list_all():
        raise RuntimeError("connection reset")

    engine = SimpleNamespace(dispose=lambda: None)
    _configured_db(monkeypatch, SimpleNamespace(engine=engine, init_error=None, list_all=list_all))
    with pytest.raises(BotRegistryUnavailable, match="connection reset"):
        bots_module._load_db_bot_profiles()


def test_no_local_database_is_still_an_empty_registry(monkeypatch) -> None:
    from llm_bawt.utils import config as config_module

    monkeypatch.setattr(config_module, "has_database_credentials", lambda config: False)
    assert bots_module._load_db_bot_profiles() == {}


def _list_bots_route(monkeypatch):
    from llm_bawt.service.routes import models as models_routes

    monkeypatch.setattr(models_routes, "get_service", lambda: SimpleNamespace(config=None))
    monkeypatch.setattr(models_routes, "global_default_voice", lambda config: None)
    return models_routes.list_bots


def test_v1_bots_is_503_not_an_empty_list_when_the_registry_is_unreadable(monkeypatch) -> None:
    _loader(monkeypatch, [BotRegistryUnavailable("lock timeout")])
    with pytest.raises(HTTPException) as raised:
        _list_bots_route(monkeypatch)()
    assert raised.value.status_code == 503
    assert "lock timeout" not in str(raised.value.detail)  # DB error text stays in the log


def test_v1_bots_lists_a_genuinely_empty_registry(monkeypatch) -> None:
    _loader(monkeypatch, [{}])
    assert _list_bots_route(monkeypatch)().data == []
