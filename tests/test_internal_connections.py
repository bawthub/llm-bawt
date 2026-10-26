"""TASK-1476: BawtHub reads stored connection secrets through llm-bawt.

``GET /v1/internal/connections/{kind}/{id}`` is service-token gated, returns
only the allowlisted fields, and distinguishes "absent" (404, BawtHub may use
its legacy env) from "present but unusable" (409/503, it must not).
"""

from __future__ import annotations

import logging

import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel

from llm_bawt.bawthub_auth import TOKEN_ENV, TOKEN_FILE_ENV
from llm_bawt.integrations.connections import IntegrationStore
from llm_bawt.runtime_settings import RuntimeSetting, RuntimeSettingsStore
from llm_bawt.service import connection_secrets
from llm_bawt.service.connection_secrets import BAWTHUB_READABLE
from llm_bawt.service.providers import crypto
from llm_bawt.service.providers.base import (
    STATUS_CONNECTED,
    STATUS_ERROR,
    ConnectionRecord,
    CredentialStore,
)
from llm_bawt.service.routes import internal_connections as route

TOKEN = "fixture-service-token-0123456789"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
URL = "/v1/internal/connections"
SECRET_VALUE = "fixture-secret-value-do-not-leak"

FIXTURES = {
    ("provider", "xai"): ({}, {"api_key": SECRET_VALUE}),
    ("integration", "unraid"): ({"url": "https://unraid.example"}, {"api_key": SECRET_VALUE}),
    ("integration", "object-store"): (
        {"endpoint": "https://s3.example", "region": "garage", "bucket": "blobs"},
        {"access_key_id": "fixture-access-id", "secret_access_key": SECRET_VALUE},
    ),
    ("integration", "azure-speech"): ({"region": "eastus"}, {"key": SECRET_VALUE}),
    ("integration", "home-assistant"): ({"url": "https://ha.example"}, {"token": SECRET_VALUE}),
}


@pytest.fixture
def stores(monkeypatch):
    fernet = Fernet(Fernet.generate_key())
    monkeypatch.setattr(crypto, "_get_fernet", lambda: fernet)
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    SQLModel.metadata.create_all(engine, tables=[RuntimeSetting.__table__])
    settings_store = object.__new__(RuntimeSettingsStore)
    settings_store.config = None
    settings_store.engine = engine
    made = {}
    for kind, cls in (("provider", CredentialStore), ("integration", IntegrationStore)):
        store = object.__new__(cls)
        store._store = settings_store
        made[kind] = store
    monkeypatch.setattr(connection_secrets, "_store", lambda _config, kind: made[kind])
    return made


@pytest.fixture
def client(stores, monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    monkeypatch.delenv(TOKEN_FILE_ENV, raising=False)
    app = FastAPI()
    app.include_router(route.router)
    app.dependency_overrides[route.get_config] = lambda: None
    return TestClient(app)


def _save(stores, kind, cid, *, meta, secret, status=STATUS_CONNECTED):
    stores[kind].save(ConnectionRecord(provider=cid, status=status, meta=meta, secret=secret))


# --- auth ---------------------------------------------------------------------

@pytest.mark.parametrize("headers", [
    {},
    {"Authorization": "Bearer wrong-token"},
    {"Authorization": f"Basic {TOKEN}"},
])
def test_bearer_is_required_before_anything_else(client, stores, headers):
    _save(stores, "provider", "xai", meta={}, secret={"api_key": SECRET_VALUE})
    for path in ("provider/xai", "integration/nope", "bogus/kind"):
        response = client.get(f"{URL}/{path}", headers=headers)
        assert response.status_code == 401
        assert SECRET_VALUE not in response.text


def test_unconfigured_service_token_is_503(client, monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    response = client.get(f"{URL}/provider/xai", headers=AUTH)
    assert response.status_code == 503


# --- contract -----------------------------------------------------------------

def test_allowlist_is_the_frozen_contract():
    assert set(BAWTHUB_READABLE) == set(FIXTURES)


@pytest.mark.parametrize("key", list(FIXTURES))
def test_connected_record_returns_only_spec_fields(client, stores, key):
    kind, cid = key
    meta, secret = FIXTURES[key]
    # Extra fields in the record never leave llm-bawt.
    _save(stores, kind, cid, meta={**meta, "internal_note": "x"}, secret={**secret, "refresh_token": "nope"})
    response = client.get(f"{URL}/{kind}/{cid}", headers=AUTH)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {"kind": kind, "id": cid, "status": "connected", "meta": meta, "secret": secret}


@pytest.mark.parametrize("key", list(FIXTURES))
def test_absent_record_is_404_not_configured(client, key):
    kind, cid = key
    response = client.get(f"{URL}/{kind}/{cid}", headers=AUTH)
    assert response.status_code == 404
    assert response.json() == {"detail": "not_configured"}


@pytest.mark.parametrize("path", ["provider/anthropic-api", "integration/google-keep", "bogus/xai",
                                  "integration/xai", "provider/unraid"])
def test_not_allowlisted_is_404_even_when_stored(client, stores, path):
    kind, cid = path.split("/")
    if kind in stores:
        _save(stores, kind, cid, meta={}, secret={"api_key": SECRET_VALUE})
    response = client.get(f"{URL}/{path}", headers=AUTH)
    assert response.status_code == 404
    assert SECRET_VALUE not in response.text


def test_not_connected_is_409(client, stores):
    _save(stores, "integration", "unraid", meta={"url": "https://u"}, secret={"api_key": SECRET_VALUE},
          status=STATUS_ERROR)
    response = client.get(f"{URL}/integration/unraid", headers=AUTH)
    assert response.status_code == 409
    assert response.json() == {"detail": "not_connected"}
    assert SECRET_VALUE not in response.text


@pytest.mark.parametrize("meta,secret", [
    ({"url": "https://u"}, {}),                       # secret field missing
    ({"url": "https://u"}, {"api_key": "  "}),        # blank
    ({}, {"api_key": SECRET_VALUE}),                  # required meta missing
])
def test_malformed_record_is_503_unreadable(client, stores, meta, secret):
    _save(stores, "integration", "unraid", meta=meta, secret=secret)
    response = client.get(f"{URL}/integration/unraid", headers=AUTH)
    assert response.status_code == 503
    assert response.json() == {"detail": "unreadable"}
    assert SECRET_VALUE not in response.text


def test_azure_needs_region_or_endpoint(client, stores):
    _save(stores, "integration", "azure-speech", meta={}, secret={"key": SECRET_VALUE})
    assert client.get(f"{URL}/integration/azure-speech", headers=AUTH).status_code == 503
    _save(stores, "integration", "azure-speech", meta={"endpoint": "https://a"}, secret={"key": SECRET_VALUE})
    response = client.get(f"{URL}/integration/azure-speech", headers=AUTH)
    assert response.status_code == 200
    assert response.json()["meta"] == {"endpoint": "https://a"}


def test_undecryptable_record_is_503_and_logs_no_value(client, stores, monkeypatch, caplog):
    _save(stores, "provider", "xai", meta={}, secret={"api_key": SECRET_VALUE})
    other = Fernet(Fernet.generate_key())
    monkeypatch.setattr(crypto, "_get_fernet", lambda: other)  # root rotated: ciphertext is unreadable
    with caplog.at_level(logging.DEBUG):
        response = client.get(f"{URL}/provider/xai", headers=AUTH)
    assert response.status_code == 503
    assert response.json() == {"detail": "unreadable"}
    assert SECRET_VALUE not in caplog.text


def test_store_unavailable_is_503(client, stores):
    stores["provider"]._store.engine = None
    response = client.get(f"{URL}/provider/xai", headers=AUTH)
    assert response.status_code == 503
    assert response.json() == {"detail": "store_unavailable"}


def test_route_is_registered_with_the_app_routers():
    from llm_bawt.service.routes import all_routers
    assert route.router in all_routers
