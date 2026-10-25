"""TASK-1472: the service-token-gated named-secret API for BawtHub auth."""

from __future__ import annotations

import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from llm_bawt.bawthub_auth import TOKEN_ENV, TOKEN_FILE_ENV
from llm_bawt.bawthub_auth.secret_store import (
    BawtHubAuthSecretStore,
    SecretUnreadable,
    valid_secret_name,
)
from llm_bawt.service.providers import crypto
from llm_bawt.service.routes import bawthub_auth_secrets as route

TOKEN = "fixture-service-token-0123456789"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
URL = "/v1/internal/secrets/bawthub-auth"


@pytest.fixture
def store(monkeypatch):
    fernet = Fernet(Fernet.generate_key())
    monkeypatch.setattr(crypto, "_get_fernet", lambda: fernet)
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    return BawtHubAuthSecretStore(engine)


@pytest.fixture
def client(store, monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    monkeypatch.delenv(TOKEN_FILE_ENV, raising=False)
    app = FastAPI()
    app.include_router(route.router)
    app.dependency_overrides[route.get_secret_store] = lambda: store
    return TestClient(app)


# --- auth --------------------------------------------------------------------

@pytest.mark.parametrize("headers", [
    {},
    {"Authorization": "Bearer wrong-token"},
    {"Authorization": f"Basic {TOKEN}"},
    {"Authorization": TOKEN},
])
@pytest.mark.parametrize("method,path,body", [
    ("GET", "session-secret", None),
    ("PUT", "session-secret", {"value": "s"}),
    ("GET", "oidc/authelia", None),
    ("PUT", "oidc/authelia", {"value": "s"}),
    ("GET", "not-a-name", None),  # unknown names don't leak before auth
    ("PUT", "session-secret", {"wrong": "shape"}),  # auth runs before body validation
])
def test_no_or_wrong_token_is_401(client, store, headers, method, path, body):
    store.put("oidc/authelia", "existing")
    response = client.request(method, f"{URL}/{path}", headers=headers, json=body)
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert response.headers["Cache-Control"] == "no-store"
    assert "existing" not in response.text
    assert store.get("session-secret") is None
    assert store.get("oidc/authelia") == "existing"


def test_unconfigured_token_fails_closed(client, monkeypatch):
    monkeypatch.delenv(TOKEN_ENV)
    for headers in ({}, AUTH, {"Authorization": "Bearer "}):
        response = client.get(f"{URL}/session-secret", headers=headers)
        assert response.status_code == 503
        assert response.json() == {"detail": "service_token_unconfigured"}


def test_token_from_secret_file(client, monkeypatch, tmp_path):
    secret = tmp_path / "token"
    secret.write_text("file-token\n")
    monkeypatch.setenv(TOKEN_FILE_ENV, str(secret))
    assert client.get(f"{URL}/session-secret", headers=AUTH).status_code == 401
    ok = client.get(f"{URL}/session-secret", headers={"Authorization": "Bearer file-token"})
    assert ok.status_code == 404


# --- behaviour ---------------------------------------------------------------

def test_session_secret_round_trip_and_create_if_absent(client):
    assert client.get(f"{URL}/session-secret", headers=AUTH).status_code == 404

    created = client.put(f"{URL}/session-secret", headers=AUTH, json={"value": "first"})
    assert created.status_code == 201
    assert created.json() == {"name": "session-secret", "created": True}
    assert "first" not in created.text

    again = client.put(f"{URL}/session-secret", headers=AUTH, json={"value": "second"})
    assert again.status_code == 409
    assert again.json() == {"detail": "exists"}

    read = client.get(f"{URL}/session-secret", headers=AUTH)
    assert read.status_code == 200
    assert read.json() == {"name": "session-secret", "value": "first"}
    assert read.headers["Cache-Control"] == "no-store"


def test_oidc_secret_upserts(client):
    first = client.put(f"{URL}/oidc/authelia", headers=AUTH, json={"value": "one"})
    assert (first.status_code, first.json()) == (200, {"name": "oidc/authelia", "created": True})
    second = client.put(f"{URL}/oidc/authelia", headers=AUTH, json={"value": "two"})
    assert (second.status_code, second.json()) == (200, {"name": "oidc/authelia", "created": False})
    assert client.get(f"{URL}/oidc/authelia", headers=AUTH).json()["value"] == "two"
    assert client.get(f"{URL}/oidc/other", headers=AUTH).status_code == 404


@pytest.mark.parametrize("name", [
    "", "sessions-secret", "session-secret/x", "oidc", "oidc/", "oidc/Authelia",
    "oidc/a/b", "oidc/-x", "oidc/" + "a" * 64, "OIDC/x",
])
def test_unknown_names_are_404_with_a_valid_token(client, name):
    assert not valid_secret_name(name)
    for response in (client.get(f"{URL}/{name}", headers=AUTH),
                     client.put(f"{URL}/{name}", headers=AUTH, json={"value": "v"})):
        assert response.status_code in {404, 405}


@pytest.mark.parametrize("name", ["oidc/../session-secret", "oidc/a.b", "oidc/a b",
                                  "session-secret\n", " session-secret"])
def test_name_grammar_rejects_traversal_and_padding(name):
    # Not sent over HTTP: clients normalise dot segments before the server sees them.
    assert not valid_secret_name(name)


def test_no_list_route(client):
    for path in (URL, f"{URL}/", "/v1/internal/secrets"):
        assert client.get(path, headers=AUTH).status_code in {404, 405}


@pytest.mark.parametrize("body", [{}, {"value": ""}, {"value": "x" * 8193},
                                  {"value": "v", "extra": 1}, {"value": 5}])
def test_bad_bodies_are_422(client, body):
    assert client.put(f"{URL}/session-secret", headers=AUTH, json=body).status_code == 422
    assert client.get(f"{URL}/session-secret", headers=AUTH).status_code == 404


# --- storage -----------------------------------------------------------------

def test_values_are_stored_encrypted(store):
    store.create_if_absent("session-secret", "plain-value")
    with store.engine.connect() as conn:
        stored = conn.execute(text("SELECT value_enc FROM bawthub_auth_secrets")).scalar_one()
    assert crypto.is_encrypted(stored)
    assert "plain-value" not in stored


def test_plaintext_row_is_refused_not_trusted(client, store):
    with store.engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO bawthub_auth_secrets (name, value_enc, created_at, updated_at) "
            "VALUES ('session-secret', 'attacker-known', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"))
    with pytest.raises(SecretUnreadable):
        store.get("session-secret")
    response = client.get(f"{URL}/session-secret", headers=AUTH)
    assert response.status_code == 503
    assert "attacker-known" not in response.text


def test_wrong_key_is_unreadable(store, monkeypatch):
    store.create_if_absent("session-secret", "v")
    other = Fernet(Fernet.generate_key())
    monkeypatch.setattr(crypto, "_get_fernet", lambda: other)
    with pytest.raises(SecretUnreadable):
        store.get("session-secret")


def test_store_rejects_unknown_names_directly(store):
    for call in (lambda: store.get("x"), lambda: store.put("x", "v"),
                 lambda: store.create_if_absent("oidc/", "v")):
        with pytest.raises(ValueError):
            call()


def test_secrets_are_not_in_runtime_settings(store):
    store.create_if_absent("session-secret", "v")
    with store.engine.connect() as conn:
        tables = set(conn.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table'")).scalars())
    assert "runtime_settings" not in tables
