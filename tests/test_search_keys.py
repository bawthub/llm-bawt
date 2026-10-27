"""Search-tool keys are record-first and fail-closed (TASK-1476, TASK-826).

Brave, Tavily and NewsAPI keys live in ``provider_connection:<id>``. Their
legacy env vars and ``Config`` fields apply only while no record exists, and a
record that can't be used never falls back to them.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import httpx
import pytest

from llm_bawt.integrations.newsapi.client import NewsAPIClient
from llm_bawt.search import credentials
from llm_bawt.search.base import SearchProvider
from llm_bawt.search.credentials import resolve_key, search_key
from llm_bawt.service.connection_secrets import ConnectionUnavailable
from llm_bawt.service.providers import search_keys
from llm_bawt.service.providers.base import ConnectionRecord
from llm_bawt.service.providers.registry import _ADAPTER_CLASSES
from llm_bawt.service.providers.search_keys import (
    SEARCH_KEY_ADAPTERS,
    BraveSearchKeyAdapter,
    NewsApiKeyAdapter,
    TavilySearchKeyAdapter,
)
from tests.stored_provider_keys import real_credential_store, stub_stored_keys

_ENV_NAMES = [name for cls in SEARCH_KEY_ADAPTERS for name in cls.env_vars]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(credentials, "_legacy_warned", set())


def _config(**fields):
    """A config with database credentials, so the record is consulted."""
    base = {"POSTGRES_PASSWORD": "synthetic", "BRAVE_API_KEY": "", "TAVILY_API_KEY": "",
            "NEWSAPI_API_KEY": "", "SEARCH_PROVIDER": None, "SEARCH_MAX_RESULTS": 5,
            "SEARCH_TIMEOUT": 10, "SEARCH_INCLUDE_ANSWER": False, "SEARCH_DEPTH": "basic",
            "BRAVE_SAFESEARCH": "moderate", "REDDIT_CLIENT_ID": "", "REDDIT_CLIENT_SECRET": ""}
    return SimpleNamespace(**{**base, **fields})


# --- registry ----------------------------------------------------------------


@pytest.mark.parametrize("cls", SEARCH_KEY_ADAPTERS)
def test_each_search_key_is_a_registered_tool_credential(real_store, cls):
    assert _ADAPTER_CLASSES[cls.id] is cls
    assert cls.keeps_disconnected_record is True
    assert cls.access_path_keys == ()  # a tool key backs no model route

    descriptor = cls(SimpleNamespace()).descriptor()

    assert descriptor["category"] == "search"
    assert descriptor["setup_url"].startswith("https://")
    assert descriptor["health"]["state"] == "unconfigured"


# --- resolution order ----------------------------------------------------------


@pytest.mark.parametrize("cls", SEARCH_KEY_ADAPTERS)
def test_the_record_wins_over_env_and_config(monkeypatch, cls):
    monkeypatch.setenv(cls.env_vars[0], "env-key")
    stub_stored_keys(monkeypatch, {cls.id: "stored-key"})

    key = resolve_key(_config(**{cls.config_attr: "config-key"}), cls.id)

    assert key.value == "stored-key" and key.problem is None


@pytest.mark.parametrize("cls", SEARCH_KEY_ADAPTERS)
@pytest.mark.parametrize("source", ["env", "config"])
def test_legacy_sources_apply_only_without_a_record_and_warn_once(monkeypatch, caplog, cls, source):
    secret = "legacy-secret-value"
    fields = {}
    if source == "env":
        monkeypatch.setenv(cls.env_vars[-1], f"  {secret}  ")
    else:
        fields[cls.config_attr] = secret
    reads = stub_stored_keys(monkeypatch, {})

    with caplog.at_level(logging.WARNING, logger=credentials.logger.name):
        first = resolve_key(_config(**fields), cls.id)
        second = resolve_key(_config(**fields), cls.id)

    assert first.value == second.value == secret
    assert reads == [cls.id, cls.id]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert secret not in warnings[0]
    expected = cls.env_vars[-1] if source == "env" else f"Config.{cls.config_attr}"
    assert expected in warnings[0]


@pytest.mark.parametrize("cls", SEARCH_KEY_ADAPTERS)
@pytest.mark.parametrize("code", ["not_connected", "unreadable", "store_unavailable"])
def test_an_unusable_record_never_falls_back_to_legacy(monkeypatch, caplog, cls, code):
    monkeypatch.setenv(cls.env_vars[0], "env-key")
    stub_stored_keys(monkeypatch, {cls.id: ConnectionUnavailable(code)})
    config = _config(**{cls.config_attr: "config-key"})

    with caplog.at_level(logging.WARNING, logger=credentials.logger.name):
        first = resolve_key(config, cls.id)
        second = resolve_key(config, cls.id)

    assert first == second == credentials.SearchKey(problem=code)
    assert not first
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_without_database_credentials_only_legacy_sources_are_read(monkeypatch):
    reads = stub_stored_keys(monkeypatch, {"brave": "stored-key"})

    no_db = _config(POSTGRES_PASSWORD="", BRAVE_API_KEY="config-key")
    duck_typed = SimpleNamespace(BRAVE_API_KEY="config-key")  # no POSTGRES_PASSWORD at all

    assert resolve_key(no_db, "brave").value == "config-key"
    assert resolve_key(duck_typed, "brave").value == "config-key"
    assert resolve_key(SimpleNamespace(), "tavily") == credentials.SearchKey()
    assert reads == []


def test_a_real_store_disconnect_or_root_rotation_turns_the_key_off(monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "env-key")
    config = _config(BRAVE_API_KEY="config-key")
    with real_credential_store(monkeypatch) as (store, rotate_root):
        store.save(ConnectionRecord(provider="brave", status="connected", auth_method="api_key",
                                    secret={"api_key": "stored-key"}))
        assert search_key(config, SearchProvider.BRAVE).value == "stored-key"

        rotate_root()
        assert search_key(config, SearchProvider.BRAVE).problem == "unreadable"

        assert BraveSearchKeyAdapter(SimpleNamespace()).disconnect() is True
        assert search_key(config, SearchProvider.BRAVE).problem == "not_connected"


# --- readers -------------------------------------------------------------------


@pytest.mark.parametrize("code", ["not_connected", "unreadable", "store_unavailable"])
def test_web_search_skips_a_provider_whose_record_is_unusable(monkeypatch, code):
    from llm_bawt.search import factory, multi
    from llm_bawt.search.brave_client import BraveSearchClient

    monkeypatch.setattr(multi, "is_search_available", lambda provider=None: True)
    monkeypatch.setattr(factory, "is_search_available", lambda provider=None: True)
    stub_stored_keys(monkeypatch, {"brave": ConnectionUnavailable(code), "tavily": None})
    config = _config(BRAVE_API_KEY="config-key")

    providers = multi.available_providers(config)
    client = factory.get_search_client(config, provider=SearchProvider.BRAVE)
    reason = factory.get_search_unavailable_reason(config, provider=SearchProvider.BRAVE)

    assert SearchProvider.BRAVE not in providers and SearchProvider.TAVILY not in providers
    assert SearchProvider.DUCKDUCKGO in providers
    assert not isinstance(client, BraveSearchClient)
    assert "Settings → Providers" in reason or "credential store" in reason
    assert "config-key" not in reason


def test_web_search_uses_a_stored_key(monkeypatch):
    from llm_bawt.search import multi

    monkeypatch.setattr(multi, "is_search_available", lambda provider=None: True)
    stub_stored_keys(monkeypatch, {"tavily": "stored-key"})

    assert multi.available_providers(_config()) == [SearchProvider.TAVILY]


def test_news_tool_follows_the_record_on_every_call(monkeypatch):
    from llm_bawt.tools.executor import ToolExecutor

    executor = object.__new__(ToolExecutor)
    executor.config = _config()
    executor.news_client = None
    outcomes = {"newsapi": "stored-key"}
    stub_stored_keys(monkeypatch, outcomes)

    executor._ensure_news_client()
    assert executor.news_client is not None and executor.news_client.is_available()

    outcomes["newsapi"] = ConnectionUnavailable("not_connected")
    executor._ensure_news_client()
    assert executor.news_client is None


def test_news_client_reads_no_env(monkeypatch):
    monkeypatch.setenv("NEWSAPI_API_KEY", "env-key")
    monkeypatch.setenv("LLM_BAWT_NEWSAPI_API_KEY", "env-key")

    assert NewsAPIClient().is_available() is False


# --- key probes ----------------------------------------------------------------


@pytest.fixture
def real_store(monkeypatch):
    with real_credential_store(monkeypatch) as handles:
        yield handles


def _fake_get(monkeypatch, status: int | Exception):
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append((url, headers, timeout))
        if isinstance(status, Exception):
            raise status
        return httpx.Response(status, request=httpx.Request("GET", url))

    monkeypatch.setattr(search_keys.httpx, "get", fake_get)
    return calls


@pytest.mark.parametrize(("cls", "rejected"), [
    (BraveSearchKeyAdapter, 422),
    (TavilySearchKeyAdapter, 401),
    (NewsApiKeyAdapter, 401),
])
def test_probe_rejects_a_bad_key_and_never_puts_it_in_the_url(monkeypatch, real_store, cls, rejected):
    calls = _fake_get(monkeypatch, rejected)
    adapter = cls(SimpleNamespace())

    result = adapter.set_api_key("synthetic-key")

    assert result.ok is False and "rejected" in result.detail
    url, headers, _timeout = calls[0]
    assert "synthetic-key" not in url
    assert "synthetic-key" in " ".join(headers.values())
    assert adapter.store.load(cls.id) is None


@pytest.mark.parametrize("cls", SEARCH_KEY_ADAPTERS)
def test_probe_accepts_a_good_key_and_reports_outages(monkeypatch, real_store, cls):
    _fake_get(monkeypatch, 200)
    assert cls(SimpleNamespace()).set_api_key("synthetic-key").ok is True
    assert cls(SimpleNamespace()).health()["state"] == "ok"

    _fake_get(monkeypatch, 503)
    assert "HTTP 503" in cls(SimpleNamespace())._probe("synthetic-key")[1]

    _fake_get(monkeypatch, httpx.ConnectError("down"))
    assert "unreachable" in cls(SimpleNamespace())._probe("synthetic-key")[1]
