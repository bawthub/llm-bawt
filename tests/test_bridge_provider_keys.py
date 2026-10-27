"""Z.AI, Kimi For Coding and Moonshot keys come from the encrypted store (TASK-1476).

The ``provider_connection:<id>`` record wins everywhere a key is read: the
app's usage cards and Kimi discovery (via ``resolve_stored_api_key``) and the
bridge proxy (via the app broker). Legacy env keys apply only while no record
exists, and a disconnect leaves a marker record so they can't come back.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from claude_code_bridge.provider_key_broker import BrokerKeyError, BrokerKeyNotInstalled
from claude_code_bridge.proxy.adapters import anthropic_passthrough
from claude_code_bridge.proxy.adapters import kimi_coding as kimi_proxy
from claude_code_bridge.proxy.adapters.kimi_coding import KimiCodingAdapter
from claude_code_bridge.proxy.adapters.moonshot import MoonshotAdapter
from claude_code_bridge.proxy.adapters.zai import ZaiAdapter
from llm_bawt.service.connection_secrets import ConnectionUnavailable
from llm_bawt.service.providers import api_key as api_key_module
from llm_bawt.service.providers.api_key import resolve_stored_api_key
from llm_bawt.service.providers.base import ConnectionRecord
from llm_bawt.service.providers.bridge_keys import (
    KimiCodingKeyAdapter,
    MoonshotKeyAdapter,
    ZaiKeyAdapter,
)
from llm_bawt.service.providers.registry import _ACCESS_PATH_OWNERS, _ADAPTER_CLASSES
from llm_bawt.service.routes import providers as provider_routes
from llm_bawt.service.usage.adapters import zai as zai_usage
from llm_bawt.service.usage.adapters.kimi_coding import KimiCodingUsageAdapter
from llm_bawt.service.usage.adapters.moonshot import MoonshotUsageAdapter
from llm_bawt.service.usage.adapters.zai import ZaiUsageAdapter
from tests.stored_provider_keys import real_credential_store, stub_stored_keys

ADAPTERS = (ZaiKeyAdapter, KimiCodingKeyAdapter, MoonshotKeyAdapter)


class FakeStore:
    def __init__(self, record: ConnectionRecord | None = None) -> None:
        self.record = record
        self.deleted = False

    def load(self, provider: str) -> ConnectionRecord | None:
        return self.record if self.record and self.record.provider == provider else None

    def save(self, record: ConnectionRecord) -> None:
        self.record = record

    def delete(self, provider: str) -> bool:
        self.deleted = True
        self.record = None
        return True


def _adapter(cls, record: ConnectionRecord | None = None):
    adapter = object.__new__(cls)
    adapter.config = SimpleNamespace()
    adapter.store = FakeStore(record)
    return adapter


# --- app: registry, disconnect marker, strict resolver -----------------------


def test_each_provider_is_registered_and_owns_its_access_path():
    owned = {"zai": "zai-anthropic", "kimi-coding": "kimi-coding-chat", "moonshot": "moonshot-anthropic"}
    for provider_id, path in owned.items():
        assert _ADAPTER_CLASSES[provider_id].keeps_disconnected_record is True
        assert _ACCESS_PATH_OWNERS[path].id == provider_id


@pytest.mark.parametrize("cls", ADAPTERS)
def test_disconnect_leaves_a_key_free_marker_record(cls):
    adapter = _adapter(cls, ConnectionRecord(
        provider=cls.id, status="connected", secret={"api_key": "stored-key"},
    ))

    assert adapter.disconnect() is True
    assert adapter.store.deleted is False
    assert adapter.store.record.status == "disconnected"
    assert not adapter.store.record.secret
    assert adapter.disconnect() is False  # already disconnected
    assert _adapter(cls).disconnect() is False  # nothing stored


def test_resolver_prefers_the_record_over_env(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "env-key")
    stub_stored_keys(monkeypatch, {"zai": "stored-key"})

    assert resolve_stored_api_key(SimpleNamespace(), "zai", ZaiKeyAdapter.env_vars) == "stored-key"


def test_resolver_uses_env_only_without_a_record_and_warns_once(monkeypatch, caplog):
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    monkeypatch.setenv("Z_AI_API_KEY", "env-secret-value")
    monkeypatch.setattr(api_key_module, "_env_fallback_warned", set())
    stub_stored_keys(monkeypatch, {})

    with caplog.at_level(logging.WARNING, logger=api_key_module.logger.name):
        first = resolve_stored_api_key(SimpleNamespace(), "zai", ZaiKeyAdapter.env_vars)
        second = resolve_stored_api_key(SimpleNamespace(), "zai", ZaiKeyAdapter.env_vars)

    assert first == second == "env-secret-value"
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "Z_AI_API_KEY" in warnings[0] and "env-secret-value" not in warnings[0]


@pytest.mark.parametrize("code", ["not_connected", "unreadable", "store_unavailable"])
def test_resolver_never_falls_back_to_env_past_an_unusable_record(monkeypatch, code):
    monkeypatch.setenv("ZAI_API_KEY", "env-key")
    stub_stored_keys(monkeypatch, {"zai": ConnectionUnavailable(code)})

    with pytest.raises(ConnectionUnavailable) as exc:
        resolve_stored_api_key(SimpleNamespace(), "zai", ZaiKeyAdapter.env_vars)
    assert exc.value.code == code


# --- app: health decides route eligibility -----------------------------------


@pytest.fixture
def real_store(monkeypatch):
    """A real CredentialStore over SQLite with a swappable Fernet key."""
    with real_credential_store(monkeypatch) as handles:
        yield handles


def _route(cls):
    from llm_bawt.service.providers.registry import access_path_connections, route_eligibility

    path = cls.access_path_keys[0]
    connection = access_path_connections(SimpleNamespace(), [path])[path]
    return connection, route_eligibility(path, "api-key", connection)


@pytest.mark.parametrize("cls", ADAPTERS)
def test_a_readable_stored_key_makes_the_route_assignable(real_store, cls):
    store, _rotate = real_store
    store.save(ConnectionRecord(provider=cls.id, status="connected", auth_method="api_key",
                                secret={"api_key": "synthetic-key"}))

    connection, eligibility = _route(cls)

    assert connection["state"] == "ok"
    assert eligibility["eligible"] is True


@pytest.mark.parametrize("cls", ADAPTERS)
def test_an_undecryptable_key_blocks_new_assignment(real_store, cls):
    store, rotate_root = real_store
    store.save(ConnectionRecord(provider=cls.id, status="connected", auth_method="api_key",
                                secret={"api_key": "synthetic-key"}))
    rotate_root()

    connection, eligibility = _route(cls)

    assert connection["state"] == "broken"
    assert eligibility["eligible"] is False
    assert "synthetic-key" not in str(connection)


@pytest.mark.parametrize("cls", ADAPTERS)
def test_a_disconnect_marker_or_no_record_is_not_assignable(real_store, cls):
    store, _rotate = real_store
    assert _route(cls)[0]["state"] == "unconfigured"
    store.save(ConnectionRecord(provider=cls.id, status="connected", auth_method="api_key",
                                secret={"api_key": "synthetic-key"}))
    assert cls(SimpleNamespace()).disconnect() is True

    connection, eligibility = _route(cls)

    assert connection["state"] == "unconfigured"
    assert eligibility["eligible"] is False


def test_an_unavailable_store_is_unknown_not_connected(monkeypatch):
    stub_stored_keys(monkeypatch, {"zai": ConnectionUnavailable("store_unavailable")})

    assert _adapter(ZaiKeyAdapter).health()["state"] == "unknown"


# --- app: the broker route --------------------------------------------------


def _broker_get(monkeypatch, provider_id: str):
    adapter = _adapter(_ADAPTER_CLASSES[provider_id])
    monkeypatch.delenv("BRIDGE_CLAUDE_TOKEN_SECRET", raising=False)
    monkeypatch.setattr(provider_routes, "_adapter_or_404", lambda _id: adapter)
    app = FastAPI()
    app.include_router(provider_routes.router)
    return TestClient(app).get(f"/v1/providers/{provider_id}/token")


@pytest.mark.parametrize(
    ("outcome", "status"),
    [
        (None, 503),  # no record: the one answer that allows the bridge's env key
        (ConnectionUnavailable("not_connected"), 409),
        (ConnectionUnavailable("unreadable"), 409),
        (ConnectionUnavailable("store_unavailable"), 500),
    ],
)
def test_broker_answers_503_only_when_no_record_exists(monkeypatch, outcome, status):
    stub_stored_keys(monkeypatch, {"moonshot": outcome})

    response = _broker_get(monkeypatch, "moonshot")

    assert response.status_code == status
    assert "env-key" not in response.text


def test_broker_serves_the_stored_key(monkeypatch):
    stub_stored_keys(monkeypatch, {"kimi-coding": "stored-key"})

    response = _broker_get(monkeypatch, "kimi-coding")

    assert response.status_code == 200
    assert response.json() == {"access_token": "stored-key", "expires_at": None, "state": "ok"}


# --- app: usage cards --------------------------------------------------------


def test_zai_usage_authenticates_with_the_stored_key(monkeypatch):
    sent: list[str] = []

    class _Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def get(self, _url, *, headers):
            sent.append(headers["x-api-key"])
            return SimpleNamespace(status_code=401, json=lambda: {}, text="")

    monkeypatch.setenv("ZAI_API_KEY", "env-key")
    stub_stored_keys(monkeypatch, {"zai": "stored-key"})
    monkeypatch.setattr(zai_usage.httpx, "AsyncClient", _Client)

    snap = asyncio.run(ZaiUsageAdapter().fetch())

    assert sent == ["stored-key"]
    assert snap.status == "unauthorized"  # what the fake upstream answered


@pytest.mark.parametrize(
    ("adapter_cls", "provider_id", "env"),
    [
        (ZaiUsageAdapter, "zai", "ZAI_API_KEY"),
        (KimiCodingUsageAdapter, "kimi-coding", "KIMI_CODING_API_KEY"),
        (MoonshotUsageAdapter, "moonshot", "MOONSHOT_API_KEY"),
    ],
)
@pytest.mark.parametrize(
    ("code", "status"),
    [("not_connected", "unauthorized"), ("unreadable", "unauthorized"), ("store_unavailable", "error")],
)
def test_usage_cards_fail_closed_on_an_unusable_record(monkeypatch, adapter_cls, provider_id, env, code, status):
    def no_network(*_args, **_kwargs):
        raise AssertionError("no request may be sent")

    monkeypatch.setenv(env, "env-key")
    reads = stub_stored_keys(monkeypatch, {provider_id: ConnectionUnavailable(code)})
    monkeypatch.setattr("httpx.AsyncClient", no_network)

    snap = asyncio.run(adapter_cls().fetch())

    assert reads == [provider_id]
    assert snap.available is False and snap.status == status
    assert "env-key" not in (snap.error or "")


# --- bridge: broker-first proxy adapters -------------------------------------


def _broker(monkeypatch, module, outcome):
    asked: list[str] = []

    def fake_fetch(provider_id):
        asked.append(provider_id)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(module, "fetch_provider_api_key", fake_fetch)
    return asked


BRIDGE_CASES = [
    (ZaiAdapter, anthropic_passthrough, "zai", "ZAI_API_KEY"),
    (MoonshotAdapter, anthropic_passthrough, "moonshot", "MOONSHOT_API_KEY"),
    (KimiCodingAdapter, kimi_proxy, "kimi-coding", "KIMI_CODING_API_KEY"),
]


@pytest.mark.parametrize(("cls", "module", "provider_id", "env"), BRIDGE_CASES)
def test_bridge_prefers_the_broker_key_over_env(monkeypatch, cls, module, provider_id, env):
    monkeypatch.setenv(env, "env-key")
    asked = _broker(monkeypatch, module, "stored-key")

    assert cls()._api_key() == "stored-key"
    assert asked == [provider_id]


@pytest.mark.parametrize(("cls", "module", "provider_id", "env"), BRIDGE_CASES)
def test_bridge_uses_env_only_when_the_broker_has_no_key(monkeypatch, cls, module, provider_id, env):
    monkeypatch.setenv(env, "env-key")
    _broker(monkeypatch, module, BrokerKeyNotInstalled("503"))

    assert cls()._api_key() == "env-key"


@pytest.mark.parametrize(("cls", "module", "provider_id", "env"), BRIDGE_CASES)
def test_bridge_never_masks_a_broker_failure_with_env(monkeypatch, cls, module, provider_id, env):
    monkeypatch.setenv(env, "env-key")
    _broker(monkeypatch, module, BrokerKeyError(f"{provider_id} API-key broker returned 409"))

    with pytest.raises(RuntimeError, match="409"):
        cls()._api_key()


@pytest.mark.parametrize(("cls", "module", "provider_id", "env"), BRIDGE_CASES)
def test_bridge_reads_the_key_off_the_event_loop(monkeypatch, cls, module, provider_id, env):
    threads: list[int] = []

    def fake_fetch(_provider_id):
        threads.append(threading.get_ident())
        return "stored-key"

    monkeypatch.setattr(module, "fetch_provider_api_key", fake_fetch)

    async def run():
        key, _base_url = await cls().authorize()
        return key, threading.get_ident()

    key, loop_thread = asyncio.run(run())

    assert key == "stored-key"
    assert threads and threads[0] != loop_thread
