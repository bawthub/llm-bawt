from __future__ import annotations

import json
import time
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine
from sqlalchemy.pool import StaticPool

from llm_bawt.integrations.connections import (
    CALLBACK_PATH, KEEP_SCOPE, IntegrationConnections, IntegrationError, IntegrationStore, keep_failure_code,
)
from llm_bawt.runtime_settings import RuntimeSetting, RuntimeSettingsStore
from llm_bawt.service.providers import crypto
from llm_bawt.service.routes import integrations as routes

ORIGIN = "https://dev.bawthub.com"
CLIENT_ID = "test.apps.googleusercontent.com"


@pytest.fixture
def manager(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine, tables=[RuntimeSetting.__table__])
    runtime = object.__new__(RuntimeSettingsStore)
    runtime.engine = engine
    store = object.__new__(IntegrationStore)
    store._store = runtime
    cipher = Fernet(Fernet.generate_key())
    monkeypatch.setattr(crypto, "encrypt", lambda text: cipher.encrypt(text.encode()).decode())
    monkeypatch.setattr(crypto, "decrypt", lambda text: cipher.decrypt(text.encode()).decode())
    manager = IntegrationConnections(None, store=store)
    manager.configure("google-keep", CLIENT_ID, "client-secret", ORIGIN + CALLBACK_PATH)
    return manager


def start(manager, return_to="/tools/settings/integrations"):
    flow = manager.start("google-keep", return_to, ORIGIN)
    params = parse_qs(urlsplit(flow["authorization_url"]).query)
    return flow, params


def upstream(manager, *, token=None, user=None, keep_status=200, token_status=200, keep_body=None):
    calls = []
    def handle(request):
        calls.append(request)
        if request.url.path == "/token":
            return httpx.Response(token_status, json=token if token is not None else {
                "scope": f"openid email {KEEP_SCOPE}", "access_token": "access-secret",
                "refresh_token": "refresh-secret", "expires_in": 3600,
            })
        if request.url.path == "/v1/userinfo":
            return httpx.Response(200, json=user if user is not None else {
                "sub": "google-123", "email": "operator@example.com", "email_verified": True,
            })
        return httpx.Response(keep_status, json=keep_body if keep_body is not None else {"notes": [{"title": "PRIVATE NOTE"}]})
    transport = httpx.MockTransport(handle)
    manager.client_factory = lambda **kwargs: httpx.Client(transport=transport, **kwargs)
    return calls


def finish(manager, flow, params):
    return manager.finish("google-keep", params["state"][0], flow["browser"], "code", "")


@pytest.mark.parametrize("status,body,expected", [
    (403, {"error": {"details": [{"reason": "SERVICE_DISABLED"}]}}, "keep_api_disabled"),
    (403, {"error": {"errors": [{"reason": "accessNotConfigured"}]}}, "keep_api_disabled"),
    (403, {"error": {"details": [{"reason": "ACCESS_TOKEN_SCOPE_INSUFFICIENT"}]}}, "keep_scope_insufficient"),
    (403, {"error": {"details": [{"reason": "ORG_RESTRICTION_VIOLATION"}]}}, "keep_organization_policy_denied"),
    (403, {"error": {"message": "API is not available for your account"}}, "keep_account_unavailable"),
    (403, {"error": {"message": "Google Keep is not enabled for this user"}}, "keep_account_unavailable"),
    (401, {}, "keep_token_rejected"),
    (429, {}, "keep_rate_limited"),
    (500, {}, "keep_verification_failed"),
    (403, {"error": {"details": "malformed", "message": "SECRET"}}, "keep_access_denied"),
    (403, {"error": "SECRET"}, "keep_access_denied"),
    (403, [], "keep_access_denied"),
])
def test_keep_failure_classification_is_bounded(status, body, expected, caplog):
    assert keep_failure_code(httpx.Response(status, json=body)) == expected
    assert "SECRET" not in caplog.text
    assert expected in caplog.text


def test_non_json_keep_denial_is_not_misreported_as_network_failure():
    assert keep_failure_code(httpx.Response(403, text="SECRET upstream HTML")) == "keep_access_denied"


def test_descriptor_never_exposes_secrets(manager):
    descriptor = manager.descriptor("google-keep")
    assert descriptor["configured"] is True
    assert descriptor["connection"]["connected"] is False
    assert "client-secret" not in json.dumps(descriptor)
    raw = manager.store._store.get_scope_settings("global", "*")
    assert "client-secret" not in json.dumps(raw)
    assert "integration_connection:client:google-keep" in raw
    assert not any(key.startswith("provider_connection:") for key in raw)


def test_start_uses_state_pkce_offline_and_fixed_callback(manager):
    flow, params = start(manager)
    assert params["access_type"] == ["offline"]
    # No saved refresh token must NOT force Keep scopes onto a consent screen.
    assert params["prompt"] == ["select_account"]
    assert set(params["scope"][0].split()) == {
        "https://www.googleapis.com/auth/keep", "openid", "https://www.googleapis.com/auth/userinfo.email",
    }
    assert set(params["scope"][0].split()) == set(manager.descriptor("google-keep")["admin_scopes"])
    assert params["code_challenge_method"] == ["S256"]
    assert params["redirect_uri"] == [ORIGIN + CALLBACK_PATH]
    pending = manager.store.load("pending:google-keep")
    assert pending.secret["state_hash"] != params["state"][0]
    assert pending.secret["browser_hash"] != flow["browser"]


@pytest.mark.parametrize("return_to,origin", [
    ("https://evil.example", ORIGIN), ("//evil.example", ORIGIN),
    ("/tools/settings/integrations", "https://evil.example"),
])
def test_start_rejects_redirect_substitution(manager, return_to, origin):
    with pytest.raises(IntegrationError):
        manager.start("google-keep", return_to, origin)


@pytest.mark.parametrize("uri", [
    "http://dev.bawthub.com" + CALLBACK_PATH,
    ORIGIN + "/wrong", ORIGIN + CALLBACK_PATH + "?x=1",
    "https://user@dev.bawthub.com" + CALLBACK_PATH,
])
def test_config_requires_exact_https_callback(manager, uri):
    with pytest.raises(IntegrationError, match="invalid_redirect_uri"):
        manager.configure("google-keep", CLIENT_ID, None, uri)


def test_success_persists_verified_identity_encrypted_tokens_without_notes(manager):
    calls = upstream(manager)
    flow, params = start(manager, "/setup")
    assert finish(manager, flow, params) == "/setup"
    record = manager.store.load("google-keep")
    assert record.account == "operator@example.com"
    assert record.secret["refresh_token"] == "refresh-secret"
    assert record.meta["read_only"] is False
    assert not manager.descriptor("google-keep")["reauthorization_required"]
    assert calls[-1].method == "GET"  # Setup never writes to the user's notes.
    public = json.dumps(manager.descriptor("google-keep"))
    assert "access-secret" not in public and "refresh-secret" not in public and "PRIVATE NOTE" not in public
    assert len(calls) == 3
    assert calls[-1].url.params["pageSize"] == "1"
    assert "code_verifier=" in calls[0].content.decode()
    raw = json.dumps(manager.store._store.get_scope_settings("global", "*"))
    assert "refresh-secret" not in raw and "PRIVATE NOTE" not in raw
    # New manager after process recreation sees the durable connection.
    assert IntegrationConnections(None, store=manager.store).descriptor("google-keep")["connection"]["connected"]


def test_registry_does_not_claim_checklist_editing(manager):
    descriptor = manager.descriptor("google-keep")
    assert descriptor["capabilities"] == {
        "read_notes": True, "create_notes": True, "delete_notes": True, "edit_existing_notes": False,
    }
    assert "domain-wide delegation" in descriptor["requirements"]


def test_old_readonly_connection_needs_reauthorization_without_losing_tokens(manager):
    upstream(manager)
    flow, params = start(manager)
    finish(manager, flow, params)
    record = manager.store.load("google-keep")
    record.meta["scopes"] = ["https://www.googleapis.com/auth/keep.readonly"]
    record.meta["read_only"] = True
    manager.store.save(record)
    assert manager.descriptor("google-keep")["reauthorization_required"]
    assert manager.store.load("google-keep").secret["refresh_token"] == "refresh-secret"


def test_reconnect_can_reuse_same_account_client_refresh_token(manager):
    upstream(manager)
    flow, params = start(manager)
    finish(manager, flow, params)
    upstream(manager, token={"scope": KEEP_SCOPE, "access_token": "new-access"})
    flow, params = start(manager)
    finish(manager, flow, params)
    assert manager.store.load("google-keep").secret == {
        "access_token": "new-access", "refresh_token": "refresh-secret",
    }


@pytest.mark.parametrize("mismatch", ["subject", "client", "scope"])
def test_reconnect_never_reuses_refresh_token_for_mismatched_grant(manager, mismatch):
    upstream(manager)
    flow, params = start(manager)
    finish(manager, flow, params)
    previous = manager.store.load("google-keep")
    if mismatch == "client":
        previous.meta["client_id"] = "another.apps.googleusercontent.com"
    elif mismatch == "scope":
        previous.meta["scopes"] = ["https://www.googleapis.com/auth/keep.readonly"]
    else:
        previous.meta["google_subject"] = "another-user"
    manager.store.save(previous)
    upstream(manager, token={"scope": KEEP_SCOPE, "access_token": "new-access"})
    flow, params = start(manager)
    with pytest.raises(IntegrationError, match="offline_access_required"):
        finish(manager, flow, params)
    assert manager.store.load("google-keep").secret["access_token"] == "access-secret"


def test_failed_keep_probe_retains_encrypted_refresh_without_connecting(manager):
    upstream(manager, keep_status=403)
    flow, params = start(manager)
    with pytest.raises(IntegrationError, match="keep_access_denied"):
        finish(manager, flow, params)
    assert not manager.descriptor("google-keep")["connection"]["connected"]
    staged = manager.store.load("unverified:google-keep")
    assert staged.secret["refresh_token"] == "refresh-secret"
    assert not staged.public()["connected"]
    assert staged.connected_at is None
    raw = json.dumps(manager.store._store.get_scope_settings("global", "*"))
    assert "refresh-secret" not in raw and "access-secret" not in raw
    upstream(manager, token={"scope": KEEP_SCOPE, "access_token": "new-access"})
    flow, params = start(manager)
    assert params["prompt"] == ["select_account"]
    finish(manager, flow, params)
    assert manager.store.load("google-keep").secret["refresh_token"] == "refresh-secret"
    assert manager.store.load("unverified:google-keep") is None


def test_missing_refresh_does_not_hide_keep_denial(manager):
    calls = upstream(manager, token={"scope": KEEP_SCOPE, "access_token": "access"},
                     keep_status=403, keep_body={"error": {"details": [{"reason": "SERVICE_DISABLED"}]}})
    flow, params = start(manager)
    with pytest.raises(IntegrationError, match="keep_api_disabled"):
        finish(manager, flow, params)
    assert len(calls) == 3
    assert manager.store.load("google-keep") is None


@pytest.mark.parametrize("action", ["disconnect", "configure"])
def test_unverified_credentials_removed_on_disconnect_or_configure(manager, action):
    upstream(manager, keep_status=403)
    flow, params = start(manager)
    with pytest.raises(IntegrationError):
        finish(manager, flow, params)
    if action == "disconnect":
        assert manager.disconnect("google-keep")
    else:
        manager.configure("google-keep", CLIENT_ID, None, ORIGIN + CALLBACK_PATH)
    assert manager.store.load("unverified:google-keep") is None


def test_disconnect_race_cannot_persist_unverified_tokens(manager):
    upstream(manager, keep_status=403)
    factory = manager.client_factory
    def disconnecting_factory(**kwargs):
        manager.disconnect("google-keep")
        return factory(**kwargs)
    manager.client_factory = disconnecting_factory
    flow, params = start(manager)
    with pytest.raises(IntegrationError, match="invalid_state"):
        finish(manager, flow, params)
    assert manager.store.load("unverified:google-keep") is None


def test_callback_is_single_use(manager):
    calls = upstream(manager)
    flow, params = start(manager)
    finish(manager, flow, params)
    with pytest.raises(IntegrationError, match="invalid_state"):
        finish(manager, flow, params)
    assert len(calls) == 3


def test_wrong_browser_does_not_consume_state(manager):
    upstream(manager)
    flow, params = start(manager)
    with pytest.raises(IntegrationError, match="invalid_state"):
        manager.finish("google-keep", params["state"][0], "wrong-browser", "code", "")
    finish(manager, flow, params)


def test_expired_state_never_reaches_google(manager):
    calls = upstream(manager)
    flow, params = start(manager)
    pending = manager.store.load("pending:google-keep")
    pending.secret["expires_at"] = time.time() - 1
    manager.store.save(pending)
    with pytest.raises(IntegrationError, match="invalid_state"):
        finish(manager, flow, params)
    assert not calls


@pytest.mark.parametrize("options,error", [
    ({"keep_status": 403}, "keep_access_denied"),
    ({"keep_status": 403, "keep_body": {"error": {"details": [{"reason": "SERVICE_DISABLED"}]}}}, "keep_api_disabled"),
    ({"keep_status": 500}, "keep_verification_failed"),
    ({"token_status": 400}, "token_exchange_failed"),
    ({"token": {"scope": "email"}}, "missing_keep_scope"),
    ({"token": {"scope": "https://www.googleapis.com/auth/keep.readonly", "access_token": "access", "refresh_token": "refresh"}}, "missing_keep_scope"),
    ({"token": {"scope": KEEP_SCOPE, "access_token": "access"}}, "offline_access_required"),
    ({"user": {"email": "operator@example.com", "sub": "123", "email_verified": False}}, "account_verification_failed"),
])
def test_failure_never_creates_connection(manager, options, error):
    upstream(manager, **options)
    flow, params = start(manager)
    with pytest.raises(IntegrationError, match=error):
        finish(manager, flow, params)
    assert manager.store.load("google-keep") is None


def test_reconnect_failure_preserves_prior_connection(manager):
    upstream(manager)
    flow, params = start(manager)
    finish(manager, flow, params)
    upstream(manager, keep_status=403)
    flow, params = start(manager)
    with pytest.raises(IntegrationError):
        finish(manager, flow, params)
    assert manager.store.load("google-keep").secret["refresh_token"] == "refresh-secret"


def test_disconnect_removes_connection_and_pending_not_client(manager):
    upstream(manager)
    flow, params = start(manager)
    finish(manager, flow, params)
    flow, params = start(manager)
    assert manager.disconnect("google-keep")
    assert manager.store.load("google-keep") is None
    assert manager.descriptor("google-keep")["configured"]
    with pytest.raises(IntegrationError):
        finish(manager, flow, params)


def test_denial_consumes_state_without_google_request(manager):
    calls = upstream(manager)
    flow, params = start(manager)
    with pytest.raises(IntegrationError, match="authorization_denied"):
        manager.finish("google-keep", params["state"][0], flow["browser"], "", "secret-provider-error")
    assert not calls
    assert manager.store.load("pending:google-keep").status == "exchanging"
    with pytest.raises(IntegrationError, match="invalid_state"):
        finish(manager, flow, params)


def test_invalid_scope_callback_explains_admin_preapproval(manager):
    calls = upstream(manager)
    flow, params = start(manager)
    with pytest.raises(IntegrationError, match="google_scope_not_authorized"):
        manager.finish("google-keep", params["state"][0], flow["browser"], "", "invalid_scope")
    assert not calls
    assert manager.store.load("google-keep") is None


def test_disconnect_during_exchange_cannot_restore_connection(manager):
    upstream(manager)
    original_factory = manager.client_factory
    def factory(**kwargs):
        # Simulate disconnect after claiming callback, before Google replies.
        manager.disconnect("google-keep")
        return original_factory(**kwargs)
    manager.client_factory = factory
    flow, params = start(manager)
    with pytest.raises(IntegrationError, match="invalid_state"):
        finish(manager, flow, params)
    assert manager.store.load("google-keep") is None


def test_changed_client_requires_new_secret(manager):
    with pytest.raises(IntegrationError, match="missing_client_secret"):
        manager.configure("google-keep", "other.apps.googleusercontent.com", None, ORIGIN + CALLBACK_PATH)


def test_route_cookie_and_callback_redirect(manager, monkeypatch):
    upstream(manager)
    monkeypatch.setattr(routes, "manager", lambda _: manager)
    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app, base_url=ORIGIN)
    response = client.post("/v1/integrations/google-keep/connect", json={}, headers={"origin": ORIGIN})
    assert response.status_code == 200
    assert "HttpOnly" in response.headers["set-cookie"]
    assert "Secure" in response.headers["set-cookie"]
    assert "SameSite=lax" in response.headers["set-cookie"]
    state = parse_qs(urlsplit(response.json()["authorization_url"]).query)["state"][0]
    browser = client.cookies.get(routes.COOKIE)
    response = client.get(f"/v1/integrations/google-keep/callback?state={state}&code=code",
                          headers={"cookie": f"{routes.COOKIE}={browser}"}, follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/tools/settings/integrations?integration_result=connected"
    assert response.headers["cache-control"] == "no-store"
