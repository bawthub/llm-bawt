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
from sqlalchemy import create_engine, text
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
    with engine.begin() as conn:  # mirrors the production unique index
        conn.execute(text("CREATE UNIQUE INDEX idx_runtime_settings_scope_key "
                          "ON runtime_settings (scope_type, scope_id, key)"))
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


def test_only_bawthub_readable_integrations_are_on_the_bawthub_allowlist():
    assert {("integration", i) for i, d in CREDENTIAL_INTEGRATIONS.items() if d.bawthub_readable} == {
        k for k in BAWTHUB_READABLE if k[0] == "integration"
    }
    # The media stack's, Reddit's and the Talk provisioner's credentials are
    # read in-process by llm-bawt only; the OpenClaw gateway token goes to the
    # openclaw bridge through the bridge-gated route, never to BawtHub.
    assert {i for i, d in CREDENTIAL_INTEGRATIONS.items() if not d.bawthub_readable} == {
        "sonarr", "radarr", "sabnzbd", "reddit", "talk-provisioner", "openclaw",
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


def _plant_foreign(store, integration_id="unraid", value=None):
    """Write a row this form didn't create; returns its exact stored bytes."""
    if value is None:
        store.save(ConnectionRecord(provider=integration_id, status=STATUS_CONNECTED, auth_method="imported",
                                    meta={"url": "https://old.example"}, secret={"api_key": "old"}))
    else:
        store._store.set_value("global", "*", store._key(integration_id), value)
    return store.snapshot(integration_id).raw


@pytest.mark.parametrize("value", [None, "not-an-object", ["list"]])
def test_foreign_or_unreadable_row_is_reported_and_never_overwritten(manager, store, value):
    before = _plant_foreign(store, value=value)
    descriptor = manager.descriptor("unraid")
    assert descriptor["configured"] is False and descriptor["managed_elsewhere"] is True
    with pytest.raises(IntegrationError, match="managed_elsewhere"):
        manager.configure("unraid", VALID["unraid"])
    assert store.snapshot("unraid").raw == before


@pytest.mark.parametrize("value", [None, "not-an-object"])
def test_foreign_or_unreadable_row_is_never_disconnected(manager, store, value):
    before = _plant_foreign(store, value=value)
    with pytest.raises(IntegrationError, match="managed_elsewhere"):
        manager.disconnect("unraid")
    assert store.snapshot("unraid").raw == before


def test_own_row_reports_not_managed_elsewhere(manager):
    assert manager.descriptor("unraid")["managed_elsewhere"] is False
    assert manager.configure("unraid", VALID["unraid"])["managed_elsewhere"] is False


def test_disconnect_removes_the_record(manager, store):
    manager.configure("unraid", VALID["unraid"])
    assert manager.disconnect("unraid") is True
    assert store.load("unraid") is None
    assert manager.disconnect("unraid") is False


# --- concurrent writers -----------------------------------------------------------

def _interleave(monkeypatch, store, write):
    """Run ``write`` once, right after the next snapshot read (a racing writer)."""
    real = store.snapshot
    pending = [write]

    def snapshot(provider):
        result = real(provider)
        if pending:
            pending.pop()()
        return result

    monkeypatch.setattr(store, "snapshot", snapshot)


def test_blank_secret_save_racing_a_rotation_keeps_the_new_key(manager, store, monkeypatch):
    manager.configure("unraid", {"url": "https://unraid.example", "api_key": "K0"})
    other = CredentialIntegrations(None, store=store)
    _interleave(monkeypatch, store, lambda: other.configure("unraid", {"api_key": "K1"}))
    manager.configure("unraid", {"url": "https://unraid2.example", "api_key": ""})
    result = store.load("unraid")
    assert result.secret == {"api_key": "K1"}
    assert result.meta == {"url": "https://unraid2.example"}


def test_first_save_racing_another_first_save_retries_onto_its_row(manager, store, monkeypatch):
    # Both see no row; the loser's insert hits the unique index and re-merges.
    other = CredentialIntegrations(None, store=store)
    _interleave(monkeypatch, store, lambda: other.configure(
        "object-store", {**VALID["object-store"], "region": "a", "access_key_id": "first-id"}))
    manager.configure("object-store", {**VALID["object-store"], "access_key_id": "second-id"})
    result = store.load("object-store")
    assert result.meta["region"] == "garage" and result.secret["access_key_id"] == "second-id"
    with store._store.engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM runtime_settings")).scalar() == 1


def test_save_racing_a_foreign_writer_does_not_overwrite_it(manager, store, monkeypatch):
    manager.configure("unraid", VALID["unraid"])
    _interleave(monkeypatch, store, lambda: _plant_foreign(store))
    with pytest.raises(IntegrationError, match="managed_elsewhere"):
        manager.configure("unraid", {"url": "https://unraid2.example"})
    assert store.load("unraid").auth_method == "imported"


def test_disconnect_racing_a_foreign_writer_does_not_delete_it(manager, store, monkeypatch):
    manager.configure("unraid", VALID["unraid"])
    _interleave(monkeypatch, store, lambda: _plant_foreign(store))
    with pytest.raises(IntegrationError, match="managed_elsewhere"):
        manager.disconnect("unraid")
    assert store.load("unraid").auth_method == "imported"


def test_save_that_keeps_losing_the_race_reports_conflict(manager, store, monkeypatch):
    manager.configure("unraid", VALID["unraid"])
    monkeypatch.setattr(store, "replace", lambda *_args: False)
    with pytest.raises(IntegrationError, match="conflict"):
        manager.configure("unraid", {"url": "https://unraid2.example"})
    assert store.load("unraid").meta == {"url": "https://unraid.example"}


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


@pytest.mark.parametrize("body", [
    {"values": {f"f{i}": SECRET for i in range(17)}},          # too many values
    {"values": {"url": "https://unraid.example", "api_key": [SECRET]}},  # nested type error
    {"values": {"api_key": {"nested": SECRET}}},
    {"values": {"api_key": 7}, "extra": SECRET},
    {"values": [SECRET]},
    [SECRET],
    SECRET,
])
def test_put_credentials_malformed_body_never_echoes_input(client, store, body):
    response = client.put("/v1/integrations/unraid/credentials", json=body)
    assert response.status_code == 400
    assert response.json() == {"detail": "invalid_field"}
    assert SECRET not in response.text
    assert store.load("unraid") is None


def test_put_credentials_invalid_json_never_echoes_input(client):
    response = client.put("/v1/integrations/unraid/credentials", content=f'{{"values": "{SECRET}',
                          headers={"Content-Type": "application/json"})
    assert response.status_code == 400
    assert SECRET not in response.text


def test_put_and_delete_refuse_a_foreign_row(client, store):
    before = _plant_foreign(store)
    put = client.put("/v1/integrations/unraid/credentials", json={"values": VALID["unraid"]})
    assert put.status_code == 400 and put.json()["detail"] == "managed_elsewhere"
    assert SECRET not in put.text
    delete = client.delete("/v1/integrations/unraid")
    assert delete.status_code == 400 and delete.json()["detail"] == "managed_elsewhere"
    assert store.snapshot("unraid").raw == before


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
