"""TASK-1476: Home Assistant credentials come from the encrypted store.

The ``integration_connection:home-assistant`` record wins. The deprecated
``HA_NATIVE_MCP_URL``/``HA_NATIVE_MCP_TOKEN`` settings are read only when no
record exists; a record that exists but can't be used never falls back to them.
Every reader resolves per operation, so a rotated token needs no restart.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import httpx
import pytest
from cryptography.fernet import Fernet
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel

from llm_bawt.core.initialization import OrchestratorInitializationMixin
from llm_bawt.integrations import ha_credentials
from llm_bawt.integrations.connections import IntegrationStore
from llm_bawt.integrations.credential_integrations import CredentialIntegrations
from llm_bawt.integrations.ha_credentials import HACredentials, HomeAssistantUnavailable, resolve_ha_credentials
from llm_bawt.integrations.ha_mcp.client import HomeAssistantMCPClient, HomeAssistantNativeClient
from llm_bawt.integrations.home_audio import HomeAudioClient, HomeAudioSettings
from llm_bawt.runtime_settings import RuntimeSetting, RuntimeSettingsStore
from llm_bawt.service import connection_secrets
from llm_bawt.service.providers import crypto
from llm_bawt.service.providers.base import STATUS_DISCONNECTED
from llm_bawt.service.routes import ha_weather
from llm_bawt.tools.executor import ToolExecutor

STORED_TOKEN = "stored-ha-token-do-not-echo"
ROTATED_TOKEN = "rotated-ha-token-do-not-echo"
ENV_TOKEN = "env-ha-token-do-not-echo"


def _config(url: str = "http://ha-env.example/api/mcp", token: str = ENV_TOKEN, **extra):
    return SimpleNamespace(HA_NATIVE_MCP_URL=url, HA_NATIVE_MCP_TOKEN=token, HA_MCP_TIMEOUT=10,
                           HA_MCP_TOOL_EXCLUDE="", **extra)


@pytest.fixture
def store(monkeypatch):
    fernet = Fernet(Fernet.generate_key())
    monkeypatch.setattr(crypto, "_get_fernet", lambda: fernet)
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine, tables=[RuntimeSetting.__table__])
    with engine.begin() as conn:
        conn.execute(text("CREATE UNIQUE INDEX idx_runtime_settings_scope_key "
                          "ON runtime_settings (scope_type, scope_id, key)"))
    runtime = object.__new__(RuntimeSettingsStore)
    runtime.config = None
    runtime.engine = engine
    made = object.__new__(IntegrationStore)
    made._store = runtime
    monkeypatch.setattr(connection_secrets, "_store", lambda config, kind: made)
    monkeypatch.setattr(ha_credentials, "_env_fallback_warned", False)
    return made


@pytest.fixture
def forms(store):
    return CredentialIntegrations(None, store=store)


def _save(forms, url="https://ha.example", token=STORED_TOKEN):
    forms.configure("home-assistant", {"url": url, "token": token})


# --- the resolver -----------------------------------------------------------

def test_record_wins_over_env_and_strips_the_mcp_path(forms):
    _save(forms, url="https://ha.example/api/mcp/")
    found = resolve_ha_credentials(_config())
    assert found == HACredentials("https://ha.example", STORED_TOKEN)
    assert found.mcp_url == "https://ha.example/api/mcp"


def test_no_record_falls_back_to_env_with_one_warning_naming_vars_only(store, caplog):
    with caplog.at_level(logging.WARNING, logger=ha_credentials.logger.name):
        assert resolve_ha_credentials(_config()) == HACredentials("http://ha-env.example", ENV_TOKEN)
        resolve_ha_credentials(_config())
    warnings = [r.getMessage() for r in caplog.records]
    assert len(warnings) == 1
    assert "LLM_BAWT_HA_NATIVE_MCP_URL" in warnings[0] and "LLM_BAWT_HA_NATIVE_MCP_TOKEN" in warnings[0]
    assert ENV_TOKEN not in caplog.text


@pytest.mark.parametrize("url, token", [("", ENV_TOKEN), ("http://ha-env.example", ""), ("", "")])
def test_no_record_and_incomplete_env_is_not_configured(store, url, token):
    assert resolve_ha_credentials(_config(url=url, token=token)) is None


def test_disconnect_through_the_form_never_falls_back_to_env(store, forms):
    _save(forms)
    assert forms.disconnect("home-assistant") is True
    with pytest.raises(HomeAssistantUnavailable) as caught:
        resolve_ha_credentials(_config())
    assert caught.value.code == "not_connected"
    record = store.load("home-assistant")
    assert record.status == STATUS_DISCONNECTED and not record.secret and not record.meta
    assert forms.descriptor("home-assistant")["configured"] is False


def test_undecryptable_record_never_falls_back_to_env(forms, monkeypatch):
    _save(forms)
    monkeypatch.setattr(crypto, "_get_fernet", lambda: Fernet(Fernet.generate_key()))  # rotated root
    with pytest.raises(HomeAssistantUnavailable) as caught:
        resolve_ha_credentials(_config())
    assert caught.value.code == "unreadable"


def test_store_down_never_falls_back_to_env(monkeypatch):
    def down(config, kind):
        raise RuntimeError("db down")

    monkeypatch.setattr(connection_secrets, "_store", down)
    with pytest.raises(HomeAssistantUnavailable) as caught:
        resolve_ha_credentials(_config())
    assert caught.value.code == "store_unavailable"


def test_repr_never_shows_the_token():
    shown = repr(HACredentials("https://ha.example", STORED_TOKEN))
    assert STORED_TOKEN not in shown and "https://ha.example" in shown


# --- readers ----------------------------------------------------------------

def _discovered(config) -> tuple[HomeAssistantNativeClient, list]:
    client = HomeAssistantNativeClient(config)
    client._initialized = True
    client._tool_names = {"HassTurnOn"}
    seen: list[HACredentials] = []

    async def fake_call(credentials, name, arguments):
        seen.append(credentials)
        return SimpleNamespace(content=[SimpleNamespace(text="ok")])

    client._call_tool_async = fake_call
    return client, seen


def test_native_client_uses_a_rotated_token_without_a_restart(forms):
    _save(forms)
    client, seen = _discovered(_config())
    assert client.available is True
    client.call_tool("HassTurnOn", {})
    _save(forms, token=ROTATED_TOKEN)
    client.call_tool("HassTurnOn", {})
    assert [c.token for c in seen] == [STORED_TOKEN, ROTATED_TOKEN]
    assert seen[0].mcp_url == "https://ha.example/api/mcp"


def test_native_client_stops_after_a_disconnect_and_never_uses_env(forms, caplog):
    _save(forms)
    client, seen = _discovered(_config())
    forms.disconnect("home-assistant")
    with caplog.at_level(logging.WARNING):
        result = client.call_tool("HassTurnOn", {})
    assert result.startswith("Error") and "not configured" in result
    assert seen == [] and client.available is False
    assert "not_connected" in caplog.text and ENV_TOKEN not in caplog.text


def _audio_client(config, requests):
    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=[])

    return HomeAudioClient(config, HomeAudioSettings(), httpx.AsyncClient(transport=httpx.MockTransport(respond)))


def test_home_audio_uses_the_current_record(forms):
    _save(forms)
    requests: list[httpx.Request] = []
    client = _audio_client(_config(), requests)
    asyncio.run(client._ha("GET", "states"))
    _save(forms, token=ROTATED_TOKEN)
    asyncio.run(client._ha("GET", "states"))
    assert [str(r.url) for r in requests] == ["https://ha.example/api/states"] * 2
    assert [r.headers["authorization"] for r in requests] == [f"Bearer {STORED_TOKEN}", f"Bearer {ROTATED_TOKEN}"]


def test_home_audio_surfaces_an_unusable_record(forms):
    _save(forms)
    forms.disconnect("home-assistant")
    requests: list[httpx.Request] = []
    client = _audio_client(_config(), requests)
    with pytest.raises(ValueError, match="unavailable: not_connected"):
        asyncio.run(client._ha("GET", "states"))
    assert requests == []


def _weather(monkeypatch, config, requests):
    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"state": "sunny", "attributes": {"temperature": 70}})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(ha_weather.httpx, "AsyncClient",
                        lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs))
    monkeypatch.setattr(ha_weather, "get_service", lambda: SimpleNamespace(config=config))
    monkeypatch.setattr(ha_weather, "_cache", {})
    return asyncio.run(ha_weather.get_ha_weather(entity_id="weather.home"))


def test_weather_route_uses_the_record(forms, monkeypatch):
    _save(forms)
    requests: list[httpx.Request] = []
    result = _weather(monkeypatch, _config(), requests)
    assert result["state"] == "sunny"
    assert str(requests[0].url) == "https://ha.example/api/states/weather.home"
    assert requests[0].headers["authorization"] == f"Bearer {STORED_TOKEN}"


def test_weather_route_surfaces_an_unusable_record_without_legacy_fallback(forms, monkeypatch):
    _save(forms)
    forms.disconnect("home-assistant")
    requests: list[httpx.Request] = []
    config = _config(HA_MCP_URL="http://ha-legacy.example/api/mcp", HA_MCP_AUTH_TOKEN=ENV_TOKEN)
    result = _weather(monkeypatch, config, requests)
    assert result == {"error": "Home Assistant connection unavailable: not_connected"}
    assert requests == []


# --- the legacy env-only MCP client (Loopy r1 B1) ---------------------------

def _make_unusable(forms, monkeypatch, state):
    _save(forms)
    if state == "disconnected":
        forms.disconnect("home-assistant")
    elif state == "unreadable":
        monkeypatch.setattr(crypto, "_get_fernet", lambda: Fernet(Fernet.generate_key()))  # rotated root
    else:
        def down(config, kind):
            raise RuntimeError("db down")

        monkeypatch.setattr(connection_secrets, "_store", down)


def _legacy_config():
    # No native settings, so only the legacy server could serve Home Assistant.
    return _config(url="", token="", HA_MCP_ENABLED=True, HA_MCP_URL="http://ha-legacy.example",
                   HA_MCP_AUTH_TOKEN=ENV_TOKEN)


def _initialized(config):
    owner = SimpleNamespace(bot=SimpleNamespace(uses_tools=True, uses_home_assistant=True),
                            home_client=None, ha_native_client=None)
    OrchestratorInitializationMixin._init_home_assistant(owner, config)
    return owner


UNUSABLE = ["disconnected", "unreadable", "store_down"]


@pytest.mark.parametrize("state", UNUSABLE)
def test_initialization_never_enables_legacy_on_an_unusable_record(forms, monkeypatch, state):
    _make_unusable(forms, monkeypatch, state)
    owner = _initialized(_legacy_config())
    assert owner.home_client is None and owner.ha_native_client is None


@pytest.mark.parametrize("state", UNUSABLE)
def test_executor_lazy_init_never_enables_legacy_on_an_unusable_record(forms, monkeypatch, state):
    _make_unusable(forms, monkeypatch, state)
    executor = ToolExecutor(user_id="test-user", config=_legacy_config())
    executor._ensure_home_client()
    assert executor.home_client is None


def test_without_a_record_legacy_still_initializes(store):
    owner = _initialized(_legacy_config())
    assert isinstance(owner.home_client, HomeAssistantMCPClient)
    executor = ToolExecutor(user_id="test-user", config=_legacy_config())
    executor._ensure_home_client()
    assert isinstance(executor.home_client, HomeAssistantMCPClient)


def test_legacy_client_stops_on_the_next_call_after_a_disconnect(forms, monkeypatch):
    client = HomeAssistantMCPClient(_legacy_config())
    assert client.available is True
    calls = []

    async def fake_call(sse_url, name, arguments):
        calls.append(name)
        return SimpleNamespace(content=[SimpleNamespace(text="ok")])

    monkeypatch.setattr(client, "_call_tool_async", fake_call)
    assert client.status() == "ok"
    _save(forms)
    forms.disconnect("home-assistant")
    with pytest.raises(RuntimeError, match="unavailable: not_connected"):
        client.status()
    assert calls == ["status"] and client.available is False
