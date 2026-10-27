"""TASK-1476: Sonarr/Radarr/SABnzbd credentials come from the encrypted store.

The ``integration_connection:<service>`` record wins. The legacy env vars are
read only when no record exists; a record that exists but can't be used never
falls back to them.
"""

from __future__ import annotations

import asyncio
import importlib
import logging

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel

from llm_bawt.integrations.connections import IntegrationStore
from llm_bawt.integrations.credential_integrations import CredentialIntegrations
from llm_bawt.mcp_server import media_tools
from llm_bawt.media.arr import (
    ArrNotConfigured,
    ArrUnavailable,
    get_radarr_client,
    get_sabnzbd_client,
    get_sonarr_client,
)
from llm_bawt.media.arr import base as arr_base
from llm_bawt.runtime_settings import RuntimeSetting, RuntimeSettingsStore
from llm_bawt.service import connection_secrets
from llm_bawt.service.connection_secrets import BAWTHUB_READABLE
from llm_bawt.service.providers import crypto
from llm_bawt.service.providers.base import STATUS_DISCONNECTED

STORED_KEY = "stored-arr-key-do-not-echo"
ENV_KEY = "env-arr-key-do-not-echo"
SERVICES = {
    "sonarr": ("LLM_BAWT_SONARR_URL", "LLM_BAWT_SONARR_API_KEY", get_sonarr_client),
    "radarr": ("LLM_BAWT_RADARR_URL", "LLM_BAWT_RADARR_API_KEY", get_radarr_client),
    "sabnzbd": ("LLM_BAWT_SABNZBD_URL", "LLM_BAWT_SABNZBD_API_KEY", get_sabnzbd_client),
}


def _store(engine) -> IntegrationStore:
    runtime = object.__new__(RuntimeSettingsStore)
    runtime.config = None
    runtime.engine = engine
    made = object.__new__(IntegrationStore)
    made._store = runtime
    return made


@pytest.fixture
def store(monkeypatch):
    fernet = Fernet(Fernet.generate_key())
    monkeypatch.setattr(crypto, "_get_fernet", lambda: fernet)
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine, tables=[RuntimeSetting.__table__])
    with engine.begin() as conn:
        conn.execute(text("CREATE UNIQUE INDEX idx_runtime_settings_scope_key "
                          "ON runtime_settings (scope_type, scope_id, key)"))
    made = _store(engine)
    monkeypatch.setattr(connection_secrets, "_store", lambda config, kind: made)
    monkeypatch.setattr(arr_base, "_ENV_FALLBACK_WARNED", set())
    for url_var, key_var, _getter in SERVICES.values():
        monkeypatch.delenv(url_var, raising=False)
        monkeypatch.delenv(key_var, raising=False)
    return made


def _env(monkeypatch, service: str) -> None:
    url_var, key_var, _getter = SERVICES[service]
    monkeypatch.setenv(url_var, f"http://{service}-env.example")
    monkeypatch.setenv(key_var, ENV_KEY)


def _save(store, service: str) -> None:
    CredentialIntegrations(None, store=store).configure(
        service, {"url": f"https://{service}.example", "api_key": STORED_KEY})


def _resolve(service: str):
    url_var, key_var, _getter = SERVICES[service]
    return arr_base.resolve_credentials(service, url_var, key_var, config=object())


@pytest.mark.parametrize("service", list(SERVICES))
def test_stored_connection_wins_over_env(store, monkeypatch, service):
    _env(monkeypatch, service)
    _save(store, service)
    assert _resolve(service) == (f"https://{service}.example", STORED_KEY)


@pytest.mark.parametrize("service", list(SERVICES))
def test_no_record_falls_back_to_env_with_one_warning_naming_vars_only(store, monkeypatch, caplog, service):
    _env(monkeypatch, service)
    url_var, key_var, _getter = SERVICES[service]
    with caplog.at_level(logging.WARNING, logger=arr_base.logger.name):
        assert _resolve(service) == (f"http://{service}-env.example", ENV_KEY)
        _resolve(service)
    warnings = [r.getMessage() for r in caplog.records]
    assert len(warnings) == 1
    assert url_var in warnings[0] and key_var in warnings[0]
    assert ENV_KEY not in caplog.text


def test_no_record_and_no_env_is_not_configured(store):
    with pytest.raises(ArrNotConfigured) as caught:
        _resolve("sonarr")
    assert caught.value.missing == ["LLM_BAWT_SONARR_URL", "LLM_BAWT_SONARR_API_KEY"]


def test_disconnected_record_never_falls_back_to_env(store, monkeypatch):
    _env(monkeypatch, "radarr")
    _save(store, "radarr")
    record = store.load("radarr")
    record.status = STATUS_DISCONNECTED
    store.save(record)
    with pytest.raises(ArrUnavailable) as caught:
        _resolve("radarr")
    assert caught.value.code == "not_connected"


def test_undecryptable_record_never_falls_back_to_env(store, monkeypatch):
    _env(monkeypatch, "sabnzbd")
    _save(store, "sabnzbd")
    monkeypatch.setattr(crypto, "_get_fernet", lambda: Fernet(Fernet.generate_key()))  # rotated root
    with pytest.raises(ArrUnavailable) as caught:
        _resolve("sabnzbd")
    assert caught.value.code == "unreadable"
    assert STORED_KEY not in str(caught.value) and ENV_KEY not in str(caught.value)


def test_store_down_never_falls_back_to_env(monkeypatch):
    monkeypatch.setattr(connection_secrets, "_store", lambda config, kind: _store(None))
    _env(monkeypatch, "sonarr")
    with pytest.raises(ArrUnavailable) as caught:
        _resolve("sonarr")
    assert caught.value.code == "store_unavailable"


def test_unavailable_surfaces_as_a_structured_tool_error():
    assert media_tools._err(ArrUnavailable("sonarr", "unreadable")) == {
        "error": "[sonarr] stored connection unavailable: unreadable",
        "service": "sonarr", "configured": True, "code": "unreadable",
    }


def test_media_keys_are_not_readable_by_bawthub():
    assert not {k for k in BAWTHUB_READABLE if k[1] in SERVICES}


# --- client reuse ---------------------------------------------------------------

class _FakeClient:
    def __init__(self, url, key):
        self.pair = (url, key)
        self.closed = False

    async def aclose(self):
        self.closed = True


def test_client_slot_reuses_until_credentials_change_and_closes_the_old_one():
    async def scenario():
        slot = arr_base.ClientSlot(_FakeClient)
        first = slot.get(("https://a.example", "k1"))
        assert slot.get(("https://a.example", "k1")) is first
        second = slot.get(("https://a.example", "k2"))
        assert second is not first and second.pair == ("https://a.example", "k2")
        await asyncio.sleep(0)
        assert first.closed and not second.closed

    asyncio.run(scenario())


@pytest.mark.parametrize("service", list(SERVICES))
def test_getter_picks_up_a_rotated_key_without_restart(store, monkeypatch, service):
    _getter = SERVICES[service][2]
    _save(store, service)

    async def scenario():
        first = await _getter()
        assert first.api_key == STORED_KEY
        assert await _getter() is first
        CredentialIntegrations(None, store=store).configure(service, {"api_key": "rotated-key"})
        second = await _getter()
        assert second is not first and second.api_key == "rotated-key"
        await second.aclose()

    module = importlib.import_module(f"llm_bawt.media.arr.{service}")
    original = arr_base.resolve_credentials
    monkeypatch.setattr(module, "resolve_credentials", lambda *a, **kw: original(*a, config=object(), **kw))
    try:
        asyncio.run(scenario())
    finally:
        module._slot.reset()
