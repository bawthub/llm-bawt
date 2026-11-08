"""TASK-1612: the former env-only credential readers resolve the stored
connection first and keep env only as a fallback."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from llm_bawt import model_manager
from llm_bawt.media.clients.grok_media import GrokMediaClient
from llm_bawt.media.clients.openai_images import OpenAIImageClient
from llm_bawt.service.providers import registry as provider_registry
from llm_bawt.service.providers.api_key import ApiKeyAdapter

_ENV_NAMES = (
    "XAI_API_KEY", "LLM_BAWT_XAI_API_KEY", "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY", "LLM_BAWT_ANTHROPIC_API_KEY",
)


class _StoredKey(ApiKeyAdapter):
    def __init__(self, key):  # no store/config needed for the resolver
        self._key = key

    def api_key(self):
        return self._key


@pytest.fixture
def stored(monkeypatch):
    """Patch the provider registry: ``stored({"xai": "rec"})`` sets records."""
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    calls: list[str] = []

    def install(records, *, broken=False):
        def get_adapter(_config, provider_id):
            calls.append(provider_id)
            if broken:
                raise RuntimeError("credential store down")
            return _StoredKey(records.get(provider_id))

        monkeypatch.setattr(provider_registry, "get_adapter", get_adapter)
        return calls

    return install


def _close(client):
    asyncio.run(client.close())


CONFIG = SimpleNamespace()


@pytest.mark.parametrize(
    ("client_cls", "provider_id", "env_name"),
    [
        (GrokMediaClient, "xai", "XAI_API_KEY"),
        (GrokMediaClient, "xai", "LLM_BAWT_XAI_API_KEY"),
        (OpenAIImageClient, "openai-api", "OPENAI_API_KEY"),
    ],
)
def test_media_client_prefers_stored_key_over_env(stored, monkeypatch, client_cls, provider_id, env_name):
    calls = stored({provider_id: "stored-key"})
    monkeypatch.setenv(env_name, "env-key")
    client = client_cls(config=CONFIG)
    try:
        assert client.api_key == "stored-key"
        assert calls == [provider_id]
    finally:
        _close(client)


@pytest.mark.parametrize(
    ("client_cls", "env_name"),
    [(GrokMediaClient, "LLM_BAWT_XAI_API_KEY"), (OpenAIImageClient, "OPENAI_API_KEY")],
)
@pytest.mark.parametrize("broken", [False, True])
def test_media_client_falls_back_to_env_without_usable_record(stored, monkeypatch, client_cls, env_name, broken):
    stored({}, broken=broken)
    monkeypatch.setenv(env_name, "env-key")
    client = client_cls(config=CONFIG)
    try:
        assert client.api_key == "env-key"
    finally:
        _close(client)


@pytest.mark.parametrize("client_cls", [GrokMediaClient, OpenAIImageClient])
def test_media_client_explicit_key_skips_lookup(stored, client_cls):
    calls = stored({"xai": "stored-key", "openai-api": "stored-key"})
    client = client_cls(api_key="explicit", config=CONFIG)
    try:
        assert client.api_key == "explicit"
        assert calls == []
    finally:
        _close(client)


@pytest.mark.parametrize("client_cls", [GrokMediaClient, OpenAIImageClient])
def test_media_client_without_any_key_is_empty(stored, client_cls):
    stored({})
    client = client_cls(config=CONFIG)
    try:
        assert client.api_key == ""
    finally:
        _close(client)


def test_media_registry_factory_resolves_stored_key(stored, monkeypatch):
    """The registry builds clients with no arguments; that path must still find the record."""
    from llm_bawt.media.clients import media_provider_registry

    stored({"xai": "stored-xai", "openai-api": "stored-openai"})
    monkeypatch.setattr(model_manager, "Config", lambda: CONFIG)
    monkeypatch.setattr("llm_bawt.utils.config.Config", lambda: CONFIG)
    for provider, expected in (("grok", "stored-xai"), ("openai", "stored-openai")):
        client = media_provider_registry.create_client(provider, "image")
        try:
            assert client.api_key == expected
        finally:
            _close(client)


def _manager():
    manager = object.__new__(model_manager.ModelManager)
    manager.config = CONFIG
    return manager


def test_cli_grok_discovery_uses_stored_key(stored, monkeypatch):
    stored({"xai": "stored-xai"})
    monkeypatch.setenv("XAI_API_KEY", "env-xai")
    seen = []
    monkeypatch.setattr(model_manager, "fetch_grok_api_models", lambda key: (seen.append(key), (True, []))[1])
    assert _manager()._fetch_api_models(model_manager.PROVIDER_GROK) == (True, [])
    assert seen == ["stored-xai"]


def test_cli_grok_discovery_without_key_reports_missing(stored, monkeypatch):
    stored({})
    monkeypatch.setattr(model_manager, "fetch_grok_api_models", lambda key: pytest.fail("must not fetch"))
    assert _manager()._fetch_api_models(model_manager.PROVIDER_GROK) == (False, [])


def test_cli_openai_discovery_uses_stored_key(stored, monkeypatch):
    stored({"openai-api": "stored-openai"})
    seen = []
    monkeypatch.setattr(model_manager, "fetch_openai_api_models", lambda key=None: (seen.append(key), (True, []))[1])
    assert _manager()._fetch_api_models(model_manager.PROVIDER_OPENAI) == (True, [])
    assert seen == ["stored-openai"]


def test_anthropic_discovery_without_explicit_key_uses_stored_record(stored, monkeypatch):
    calls = stored({"anthropic-api": "stored-anthropic"})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-anthropic")
    monkeypatch.setattr(model_manager, "Config", lambda: CONFIG)
    used = []

    class _Models:
        def list(self, **_kwargs):
            return SimpleNamespace(data=[], has_more=False)

    class _Anthropic:
        def __init__(self, api_key):
            used.append(api_key)
            self.models = _Models()

    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", _Anthropic)
    model_manager.fetch_anthropic_api_models()
    assert calls == ["anthropic-api"]
    assert used == ["stored-anthropic"]


def test_anthropic_discovery_falls_back_to_env(stored, monkeypatch):
    stored({})
    monkeypatch.setenv("LLM_BAWT_ANTHROPIC_API_KEY", "env-anthropic")
    monkeypatch.setattr(model_manager, "Config", lambda: CONFIG)
    used = []

    class _Anthropic:
        def __init__(self, api_key):
            used.append(api_key)
            self.models = SimpleNamespace(list=lambda **_k: SimpleNamespace(data=[], has_more=False))

    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", _Anthropic)
    model_manager.fetch_anthropic_api_models()
    assert used == ["env-anthropic"]


def test_anthropic_discovery_without_any_key_skips(stored, monkeypatch):
    stored({})
    monkeypatch.setattr(model_manager, "Config", lambda: CONFIG)
    assert model_manager.fetch_anthropic_api_models() == (False, [])
