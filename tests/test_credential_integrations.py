"""TASK-1476: integrations configured by entering credentials.

The credentials form writes the same ``integration_connection:<id>`` row that
``GET /v1/internal/connections/integration/<id>`` reads. Secrets are
write-only: descriptors and save responses never echo them.
"""

from __future__ import annotations

import json

import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel

from llm_bawt.integrations.connections import IntegrationError, IntegrationStore
from llm_bawt.integrations.credential_integrations import (
    AUTH_METHOD,
    CREDENTIAL_INTEGRATIONS,
    CredentialIntegrations,
)
from llm_bawt.runtime_settings import RuntimeSetting, RuntimeSettingsStore
from llm_bawt.service.connection_secrets import BAWTHUB_READABLE, FieldSpec, read_connection
from llm_bawt.service.providers import crypto
from llm_bawt.service.providers.base import STATUS_CONNECTED, ConnectionRecord
from llm_bawt.service.routes import integrations as routes

SECRET = "fixture-secret-do-not-echo"
VALID = {
    "unraid": {"url": "https://unraid.example", "api_key": SECRET},
    "object-store": {
        "endpoint": "https://s3.example", "region": "garage", "bucket": "blobs",
        "access_key_id": "fixture-access-id", "secret_access_key": SECRET,
    },
    "azure-speech": {"region": "eastus", "key": SECRET},
    "home-assistant": {"url": "https://ha.example", "token": SECRET},
}


@pytest.fixture
def store(monkeypatch):
    fernet = Fernet(Fernet.generate_key())
    monkeypatch.setattr(crypto, "_get_fernet", lambda: fernet)
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine, tables=[RuntimeSetting.__table__])
    runtime = object.__new__(RuntimeSettingsStore)
    runtime.config = None
    runtime.engine = engine
    made = object.__new__(IntegrationStore)
    made._store = runtime
    return made


@pytest.fixture
def manager(store):
    return CredentialIntegrations(None, store=store)


def _raw_row(store, integration_id):
    return store._store.get_scope_settings("global", "*")[store._key(integration_id)]


# --- contract -----------------------------------------------------------------

def test_read_contract_field_names_are_frozen():
    # BawtHub's consumer depends on these exact names; the reader derives them
    # from the writer's definitions, so a rename must fail here first.
    assert {k: v for k, v in BAWTHUB_READABLE.items() if k[0] == "integration"} == {
        ("integration", "unraid"): FieldSpec(secret=("api_key",), meta=("url",)),
        ("integration", "object-store"): FieldSpec(
            secret=("access_key_id", "secret_access_key"), meta=("endpoint", "region", "bucket"),
        ),
        ("integration", "azure-speech"): FieldSpec(
            secret=("key",), meta_any=("region", "endpoint"), meta_optional=("region", "endpoint"),
        ),
        ("integration", "home-assistant"): FieldSpec(secret=("token",), meta=("url",)),
    }


def test_every_credential_integration_is_readable_by_bawthub():
    assert {("integration", i) for i in CREDENTIAL_INTEGRATIONS} == {
        k for k in BAWTHUB_READABLE if k[0] == "integration"
    }


# --- save and read back ---------------------------------------------------------

@pytest.mark.parametrize("integration_id", list(VALID))
def test_saved_credentials_read_back_through_the_internal_contract(manager, store, integration_id):
    descriptor = manager.configure(integration_id, VALID[integration_id])
    assert descriptor["connection"]["connected"] is True
    result = read_connection(None, "integration", integration_id,
                             BAWTHUB_READABLE[("integration", integration_id)], store=store)
    assert result is not None
    assert {**result.meta, **result.secret} == VALID[integration_id]


@pytest.mark.parametrize("integration_id", list(VALID))
def test_secrets_are_encrypted_at_rest_and_never_echoed(manager, store, integration_id):
    descriptor = manager.configure(integration_id, VALID[integration_id])
    assert SECRET not in json.dumps(descriptor)
    assert SECRET not in json.dumps(manager.descriptor(integration_id))
    raw = _raw_row(store, integration_id)
    assert SECRET not in json.dumps({k: v for k, v in raw.items() if k != "secret_enc"})
    assert raw["status"] == STATUS_CONNECTED and raw["auth_method"] == AUTH_METHOD
    secret_fields = [f for f in descriptor["fields"] if f["kind"] == "secret"]
    assert secret_fields and all(f["set"] is True and "value" not in f for f in secret_fields)


def test_descriptor_before_save_reports_nothing_set(manager):
    descriptor = manager.descriptor("unraid")
    assert descriptor["configured"] is False
    assert descriptor["connection"]["connected"] is False
    assert {f["name"]: f.get("set", f.get("value")) for f in descriptor["fields"]} == {"url": "", "api_key": False}


def test_blank_secret_keeps_the_stored_one(manager, store):
    manager.configure("unraid", VALID["unraid"])
    manager.configure("unraid", {"url": "https://unraid2.example", "api_key": "  "})
    result = read_connection(None, "integration", "unraid", BAWTHUB_READABLE[("integration", "unraid")], store=store)
    assert result.meta == {"url": "https://unraid2.example"}
    assert result.secret == {"api_key": SECRET}


def test_new_secret_replaces_the_stored_one(manager, store):
    manager.configure("unraid", VALID["unraid"])
    manager.configure("unraid", {"api_key": "rotated-key"})
    result = read_connection(None, "integration", "unraid", BAWTHUB_READABLE[("integration", "unraid")], store=store)
    assert result.secret == {"api_key": "rotated-key"}
    assert result.meta == {"url": "https://unraid.example"}


def test_reconfigure_keeps_original_connected_at(manager, store):
    manager.configure("unraid", VALID["unraid"])
    first = _raw_row(store, "unraid")["connected_at"]
    manager.configure("unraid", {"url": "https://other.example"})
    assert _raw_row(store, "unraid")["connected_at"] == first


# --- validation -------------------------------------------------------------------

@pytest.mark.parametrize("values", [
    {"url": "https://unraid.example"},                    # no key yet
    {"api_key": SECRET},                                  # no url
    {"url": "https://unraid.example", "api_key": ""},
])
def test_missing_required_field_saves_nothing(manager, store, values):
    with pytest.raises(IntegrationError, match="missing_required_field"):
        manager.configure("unraid", values)
    assert store.load("unraid") is None


def test_azure_needs_region_or_endpoint(manager, store):
    with pytest.raises(IntegrationError, match="missing_required_field"):
        manager.configure("azure-speech", {"key": SECRET})
    manager.configure("azure-speech", {"endpoint": "https://speech.example", "key": SECRET})
    with pytest.raises(IntegrationError, match="missing_required_field"):
        manager.configure("azure-speech", {"endpoint": ""})
    assert store.load("azure-speech").meta == {"endpoint": "https://speech.example"}


@pytest.mark.parametrize("url", ["ftp://unraid.example", "unraid.example", "https://", "https://user:pw@unraid.example"])
def test_url_fields_must_be_http_urls_without_userinfo(manager, url):
    with pytest.raises(IntegrationError, match="invalid_url"):
        manager.configure("unraid", {"url": url, "api_key": SECRET})


def test_unknown_field_is_refused(manager, store):
    with pytest.raises(IntegrationError, match="invalid_field"):
        manager.configure("unraid", {**VALID["unraid"], "extra": "x"})
    assert store.load("unraid") is None


def test_overlong_value_is_refused(manager):
    with pytest.raises(IntegrationError, match="field_too_long"):
        manager.configure("unraid", {"url": "https://unraid.example", "api_key": "x" * 5000})


def test_unknown_integration_is_refused(manager):
    with pytest.raises(IntegrationError, match="unknown_integration"):
        manager.configure("google-keep", {})


def test_store_unavailable_is_refused(manager, store):
    store._store.engine = None
    with pytest.raises(IntegrationError, match="storage_unavailable"):
        manager.configure("unraid", VALID["unraid"])


def test_record_from_another_auth_method_is_not_merged(manager, store):
    store.save(ConnectionRecord(provider="unraid", status=STATUS_CONNECTED, auth_method="imported",
                                meta={"url": "https://old.example", "stale": "x"}, secret={"api_key": "old"}))
    assert manager.descriptor("unraid")["configured"] is False
    with pytest.raises(IntegrationError, match="missing_required_field"):
        manager.configure("unraid", {"url": "https://unraid.example"})


def test_disconnect_removes_the_record(manager, store):
    manager.configure("unraid", VALID["unraid"])
    assert manager.disconnect("unraid") is True
    assert store.load("unraid") is None
    assert manager.disconnect("unraid") is False


# --- routes -----------------------------------------------------------------------

@pytest.fixture
def client(manager, monkeypatch):
    monkeypatch.setattr(routes, "get_service", lambda: type("Service", (), {"config": None})())
    monkeypatch.setattr(routes, "CredentialIntegrations", lambda _config: manager)
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def test_put_credentials_saves_and_never_echoes_secret(client, store):
    response = client.put("/v1/integrations/unraid/credentials", json={"values": VALID["unraid"]})
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert SECRET not in response.text
    assert response.json()["connection"]["connected"] is True
    assert store.load("unraid").secret == {"api_key": SECRET}


def test_put_credentials_validation_error_is_400(client, store):
    response = client.put("/v1/integrations/unraid/credentials", json={"values": {"url": "nope", "api_key": SECRET}})
    assert response.status_code == 400
    assert response.json()["detail"] == "invalid_url"
    assert SECRET not in response.text


def test_put_credentials_unknown_integration_is_404():
    app = FastAPI()
    app.include_router(routes.router)
    response = TestClient(app).put("/v1/integrations/google-keep/credentials", json={"values": {}})
    assert response.status_code == 404


def test_delete_routes_credential_integrations_to_their_manager(client, store):
    client.put("/v1/integrations/unraid/credentials", json={"values": VALID["unraid"]})
    response = client.delete("/v1/integrations/unraid")
    assert response.status_code == 200 and response.json() == {"disconnected": True}
    assert store.load("unraid") is None
