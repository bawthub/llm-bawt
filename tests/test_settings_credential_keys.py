"""TASK-1520: the generic /v1/settings routes don't touch credential rows.

``provider_connection:*`` and ``integration_connection:*`` rows in
``runtime_settings`` belong to a CredentialStore. The generic settings routes
have no service-token auth, so they must neither write, delete nor list those
rows; the provider and integration APIs are the only way in.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel

from llm_bawt.integrations.connections import IntegrationStore
from llm_bawt.integrations.nextcloud.setup import NextcloudSetupStore
from llm_bawt.runtime_settings import RuntimeSetting, RuntimeSettingsStore
from llm_bawt.service.providers import base as providers_base
from llm_bawt.service.providers.base import CredentialStore
from llm_bawt.service.routes import settings as settings_routes
from llm_bawt.service.schemas import (
    RuntimeSettingBatchItem,
    RuntimeSettingBatchUpsertRequest,
    RuntimeSettingUpsertRequest,
)

PROVIDER_KEY = "provider_connection:example"
INTEGRATION_KEY = "integration_connection:example"
RESERVED_KEYS = [
    PROVIDER_KEY,
    INTEGRATION_KEY,
    "integration_connection:pending:example",
    # Case and padding variants are refused too: nothing that looks like a
    # credential key is writable through the generic route.
    " Provider_Connection:example",
    "INTEGRATION_CONNECTION:example",
]


@pytest.fixture
def store(monkeypatch):
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine, tables=[RuntimeSetting.__table__])
    settings_store = object.__new__(RuntimeSettingsStore)
    settings_store.config = None
    settings_store.engine = engine
    service = SimpleNamespace(config=None, _default_bot="examplebot")
    monkeypatch.setattr(settings_routes, "get_service", lambda: service)
    monkeypatch.setattr(settings_routes, "get_runtime_settings_store", lambda _config: settings_store)
    monkeypatch.setattr(settings_routes, "_invalidate_bot_instance_cache", lambda *_a: None)
    monkeypatch.setattr(settings_routes, "_invalidate_all_instance_cache", lambda *_a: None)
    # Rows a CredentialStore wrote (ciphertext stand-ins) next to a normal setting.
    settings_store.set_value("global", "*", PROVIDER_KEY, {"secret_enc": "enc:v1:original"})
    settings_store.set_value("global", "*", INTEGRATION_KEY, {"secret_enc": "enc:v1:original"})
    settings_store.set_value("global", "*", "ordinary_setting", 1)
    return settings_store


def _stored(store, key):
    return store.get_scope_settings("global", "*").get(key)


@pytest.mark.parametrize("key", RESERVED_KEYS)
def test_put_refuses_credential_keys(store, key):
    with pytest.raises(HTTPException) as raised:
        settings_routes.upsert_runtime_setting(
            RuntimeSettingUpsertRequest(scope_type="global", key=key, value={"secret_enc": "forged"})
        )
    assert raised.value.status_code == 403
    assert _stored(store, PROVIDER_KEY) == {"secret_enc": "enc:v1:original"}
    assert _stored(store, INTEGRATION_KEY) == {"secret_enc": "enc:v1:original"}


@pytest.mark.parametrize("scope", [("global", None), ("bot", "examplebot")])
def test_put_refuses_credential_keys_in_every_scope(store, scope):
    with pytest.raises(HTTPException) as raised:
        settings_routes.upsert_runtime_setting(
            RuntimeSettingUpsertRequest(scope_type=scope[0], scope_id=scope[1], key=PROVIDER_KEY, value=1)
        )
    assert raised.value.status_code == 403


@pytest.mark.parametrize("key", [PROVIDER_KEY, INTEGRATION_KEY])
def test_delete_refuses_credential_keys(store, key):
    with pytest.raises(HTTPException) as raised:
        settings_routes.delete_runtime_setting(scope_type="global", key=key, scope_id=None)
    assert raised.value.status_code == 403
    assert _stored(store, key) == {"secret_enc": "enc:v1:original"}


def test_batch_refuses_whole_batch_with_a_credential_key(store):
    with pytest.raises(HTTPException) as raised:
        settings_routes.batch_upsert_runtime_settings(RuntimeSettingBatchUpsertRequest(items=[
            RuntimeSettingBatchItem(scope_type="global", key="ordinary_setting", value=2),
            RuntimeSettingBatchItem(scope_type="global", key=PROVIDER_KEY, value={"secret_enc": "forged"}),
        ]))
    assert raised.value.status_code == 403
    assert _stored(store, "ordinary_setting") == 1  # nothing in the batch was applied
    assert _stored(store, PROVIDER_KEY) == {"secret_enc": "enc:v1:original"}


def test_scope_listing_hides_credential_rows(store):
    response = settings_routes.list_runtime_settings(scope_type="global", scope_id=None)
    assert [item.key for item in response.settings] == ["ordinary_setting"]


@pytest.mark.parametrize("filters", [
    {},
    {"key_prefix": "provider_connection:"},
    {"key_prefix": "integration_connection:"},
    {"key": PROVIDER_KEY},
])
def test_all_listing_hides_credential_rows(store, filters):
    params = {"scope_type": None, "scope_id": None, "key": None, "key_prefix": None,
              "limit": 200, "offset": 0, **filters}
    response = settings_routes.list_all_runtime_settings(**params)
    keys = [item.key for item in response.settings]
    assert not any(key.startswith(("provider_connection:", "integration_connection:")) for key in keys)
    assert response.total_count == len(keys)


def test_ordinary_settings_still_write_and_delete(store):
    settings_routes.upsert_runtime_setting(
        RuntimeSettingUpsertRequest(scope_type="global", key="ordinary_setting", value=5)
    )
    assert _stored(store, "ordinary_setting") == 5
    settings_routes.batch_upsert_runtime_settings(RuntimeSettingBatchUpsertRequest(items=[
        RuntimeSettingBatchItem(scope_type="global", key="ordinary_setting", value=6),
    ]))
    assert _stored(store, "ordinary_setting") == 6
    assert settings_routes.delete_runtime_setting(scope_type="global", key="ordinary_setting", scope_id=None)["deleted"]


def _credential_store_classes() -> list[type]:
    found, pending = [], [CredentialStore]
    while pending:
        cls = pending.pop()
        found.append(cls)
        pending.extend(cls.__subclasses__())
    return found


def test_every_credential_store_key_is_reserved():
    """Ratchet: every CredentialStore (and subclass) writes under a reserved prefix."""
    classes = _credential_store_classes()
    assert {CredentialStore, IntegrationStore, NextcloudSetupStore} <= set(classes)
    for store_cls in classes:
        key = object.__new__(store_cls)._key("example")
        assert providers_base.is_credential_setting_key(key), store_cls.__name__
        assert key.startswith(providers_base.CREDENTIAL_SETTING_PREFIXES), store_cls.__name__
