"""Bot imports do not bootstrap the database before service logging (TASK-1361)."""
from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from llm_bawt import bots, runtime_settings


@pytest.fixture
def fresh_bots(monkeypatch):
    """Execute a separate registry without resetting other tests' module state."""
    constructions = []

    def store(config):
        constructions.append(True)
        raise AssertionError("import must not construct a bot profile store")

    monkeypatch.setattr(runtime_settings, "BotProfileStore", store)
    monkeypatch.setattr("llm_bawt.utils.config.has_database_credentials", lambda _: True)
    name = "llm_bawt._startup_test_bots"
    spec = importlib.util.spec_from_file_location(name, Path(bots.__file__))
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module, constructions


def test_import_does_not_load_profiles_or_construct_the_store(fresh_bots):
    module, constructions = fresh_bots
    assert constructions == []
    assert module._LOADED is False
    assert module.BUILTIN_BOTS == {}


@pytest.mark.parametrize("first_access", [
    lambda module: module.get_bot("alpha"),
    lambda module: module.BotManager().get_bot("alpha"),
    lambda module: module.get_raw_bot_data("alpha"),
    lambda module: module.get_all_raw_bot_data(),
    lambda module: module.get_system_prompt("refine"),
    lambda module: module.bot_registry_error(),
])
def test_first_access_loads_the_same_bot_set_once(fresh_bots, monkeypatch, first_access):
    module, constructions = fresh_bots
    loads = []
    profiles = {
        "alpha": {"name": "Alpha", "system_prompt": "alpha prompt"},
        "beta": {"name": "Beta", "system_prompt": "beta prompt"},
    }

    def load():
        loads.append(True)
        return profiles

    monkeypatch.setattr(module, "_load_db_bot_profiles", load)
    first_access(module)
    assert [bot.slug for bot in module.BotManager().list_bots()] == ["alpha", "beta"]
    assert module.get_bot(" ALPHA ").system_prompt == "alpha prompt"
    assert module.get_raw_bot_data("beta")["name"] == "Beta"
    assert len(loads) == 1
    assert constructions == []


def test_service_logging_precedes_first_profile_bootstrap(fresh_bots, monkeypatch):
    from llm_bawt.service import api
    import uvicorn

    module, constructions = fresh_bots
    records = []
    logger = logging.getLogger(module.__name__)

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = Capture()
    old_level = logger.level
    old_propagate = logger.propagate

    def setup_logging(**kwargs):
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    def store(config):
        constructions.append(True)
        assert handler in logger.handlers
        logger.info("Runtime schema ready: bot-profile-store")
        return SimpleNamespace(engine=object(), list_all=lambda: [])

    def run_server(*args, **kwargs):
        assert module.BotManager().list_bots() == []

    monkeypatch.setattr(runtime_settings, "BotProfileStore", store)
    monkeypatch.setattr(module, "_load_bot_tags", lambda _: {})
    monkeypatch.setattr(api, "setup_service_logging", setup_logging)
    monkeypatch.setattr(api, "_is_service_running", lambda host, port: False)
    monkeypatch.setattr(uvicorn, "run", run_server)
    monkeypatch.setattr(sys, "argv", ["llm-service", "--host", "127.0.0.1", "--port", "65000"])
    try:
        api.main()
        assert constructions and len(constructions) == 1
        assert records == ["Runtime schema ready: bot-profile-store"]
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
        logger.propagate = old_propagate
