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

from llm_bawt.integrations.connections import IntegrationError, IntegrationStore
from llm_bawt.integrations.credential_integrations import CredentialIntegrations
from llm_bawt.mcp_server import media_tools
from llm_bawt.media.arr import (
    ArrNotConfigured,
    ArrUnavailable,
    RadarrClient,
    SabnzbdClient,
    SonarrClient,
    radarr_client,
    sabnzbd_client,
    sonarr_client,
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
    "sonarr": ("LLM_BAWT_SONARR_URL", "LLM_BAWT_SONARR_API_KEY", sonarr_client),
    "radarr": ("LLM_BAWT_RADARR_URL", "LLM_BAWT_RADARR_API_KEY", radarr_client),
    "sabnzbd": ("LLM_BAWT_SABNZBD_URL", "LLM_BAWT_SABNZBD_API_KEY", sabnzbd_client),
}
CLIENTS = {"sonarr": SonarrClient, "radarr": RadarrClient, "sabnzbd": SabnzbdClient}


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


@pytest.mark.parametrize("service", list(SERVICES))
def test_disconnect_through_the_form_never_falls_back_to_env(store, monkeypatch, service):
    # Loopy r1 B1: the real writer (DELETE /v1/integrations/<id>), not a hand-made row.
    _env(monkeypatch, service)
    _save(store, service)
    forms = CredentialIntegrations(None, store=store)
    assert forms.disconnect(service) is True
    with pytest.raises(ArrUnavailable) as caught:
        _resolve(service)
    assert caught.value.code == "not_connected"
    record = store.load(service)
    assert record.status == STATUS_DISCONNECTED and not record.secret and not record.meta
    assert forms.descriptor(service)["configured"] is False
    assert forms.disconnect(service) is False  # nothing left to disconnect


def test_reconnecting_after_a_disconnect_needs_every_field_again(store):
    _save(store, "sonarr")
    forms = CredentialIntegrations(None, store=store)
    forms.disconnect("sonarr")
    with pytest.raises(IntegrationError) as caught:
        forms.configure("sonarr", {"api_key": "new-key-only"})
    assert caught.value.args[0] == "missing_required_field"
    _save(store, "sonarr")
    assert _resolve("sonarr") == ("https://sonarr.example", STORED_KEY)
    assert forms.descriptor("sonarr")["configured"] is True


def test_integrations_without_an_env_fallback_still_delete_on_disconnect(store):
    forms = CredentialIntegrations(None, store=store)
    forms.configure("azure-speech", {"region": "eastus", "key": STORED_KEY})
    assert forms.disconnect("azure-speech") is True
    assert not store.snapshot("azure-speech").exists


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


# --- client leases --------------------------------------------------------------

class _FakeClient:
    def __init__(self, url, key, *, fail_close=False):
        self.pair = (url, key)
        self.closed = False
        self.fail_close = fail_close

    async def aclose(self):
        self.closed = True
        if self.fail_close:
            raise RuntimeError("close failed")


def _const(pair):
    return lambda: pair


OLD, NEW = ("https://a.example", "k1"), ("https://a.example", "k2")


def test_lease_reuses_the_client_until_credentials_change():
    async def scenario():
        slot = arr_base.ClientSlot(_FakeClient)
        async with slot.lease(_const(OLD)) as first:
            pass
        async with slot.lease(_const(OLD)) as again:
            assert again is first and not first.closed
        async with slot.lease(_const(NEW)) as second:
            assert second is not first and second.pair == NEW
            assert first.closed  # idle when replaced: closed before the new lease is used
        assert not second.closed

    asyncio.run(scenario())


def test_rotation_never_closes_a_client_another_operation_still_holds():
    # Loopy r1 B2: media_add / media_library_stats make several awaited calls
    # on one client while another tool resolves a rotated key.
    async def scenario():
        slot = arr_base.ClientSlot(_FakeClient)
        holding, rotated, done = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def long_tool():
            async with slot.lease(_const(OLD)) as client:
                holding.set()  # OLD is acquired before the rotation starts
                await rotated.wait()
                assert not client.closed  # still usable mid-operation
                return client

        async def rotating_tool():
            async with slot.lease(_const(NEW)) as client:
                rotated.set()
                await done.wait()
                return client

        holder = asyncio.create_task(long_tool())
        await holding.wait()
        rotator = asyncio.create_task(rotating_tool())
        old = await holder
        assert old.closed  # closed by the holder's own exit, awaited to completion
        done.set()
        new = await rotator
        assert not new.closed

    asyncio.run(scenario())


def test_a_failing_operation_still_releases_and_closes_its_replaced_client():
    async def scenario():
        slot = arr_base.ClientSlot(lambda url, key: _FakeClient(url, key, fail_close=True))
        with pytest.raises(ValueError):
            async with slot.lease(_const(OLD)) as old:
                async with slot.lease(_const(NEW)):
                    pass
                raise ValueError("tool error")
        assert old.closed  # the close error was logged, not raised over the tool error

    asyncio.run(scenario())


def test_a_cancelled_operation_still_closes_its_replaced_client():
    async def scenario():
        slot = arr_base.ClientSlot(_FakeClient)
        holding = asyncio.Event()
        captured = []

        async def tool():
            async with slot.lease(_const(OLD)) as client:
                captured.append(client)
                holding.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(tool())
        await holding.wait()
        async with slot.lease(_const(NEW)):
            pass
        assert not captured[0].closed
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.gather(*tuple(arr_base._CLOSING))
        assert captured[0].closed

    asyncio.run(scenario())


def test_credential_errors_build_nothing():
    def unavailable():
        raise ArrUnavailable("sonarr", "unreadable")

    async def scenario():
        built = []
        slot = arr_base.ClientSlot(lambda *pair: built.append(pair))
        with pytest.raises(ArrUnavailable):
            async with slot.lease(unavailable):
                pass
        assert built == []

    asyncio.run(scenario())


@pytest.mark.parametrize("service", list(SERVICES))
def test_a_real_client_survives_a_rotation_mid_operation(service):
    import httpx

    async def scenario():
        rotations = []

        def build(url, key):
            client = CLIENTS[service](url, key)
            client._client = httpx.AsyncClient(base_url=url, transport=httpx.MockTransport(handle))
            return client

        async def handle(request):
            if not rotations:
                # Another tool saves and uses a rotated key while this one waits.
                async with slot.lease(_const(("https://arr.example", "new-key"))) as other:
                    rotations.append(other)
            return httpx.Response(200, json={"queue": {"slots": []}, "records": []})

        slot = arr_base.ClientSlot(build)
        async with slot.lease(_const(("https://arr.example", "old-key"))) as borrowed:
            await borrowed.queue()
            await borrowed.queue()  # the second call of a multi-call tool
        assert borrowed._client.is_closed
        assert not rotations[0]._client.is_closed
        await rotations[0].aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("service", list(SERVICES))
def test_lease_picks_up_a_rotated_key_without_restart(store, monkeypatch, service):
    lease = SERVICES[service][2]
    _save(store, service)

    async def scenario():
        async with lease() as first:
            assert first.api_key == STORED_KEY
        async with lease() as again:
            assert again is first
        CredentialIntegrations(None, store=store).configure(service, {"api_key": "rotated-key"})
        async with lease() as second:
            assert second is not first and second.api_key == "rotated-key"
        await second.aclose()

    module = importlib.import_module(f"llm_bawt.media.arr.{service}")
    original = arr_base.resolve_credentials
    monkeypatch.setattr(module, "resolve_credentials", lambda *a, **kw: original(*a, config=object(), **kw))
    try:
        asyncio.run(scenario())
    finally:
        module._slot.reset()
