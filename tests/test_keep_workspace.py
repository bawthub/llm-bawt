"""Delegated Keep authorization contracts; no live Google credentials required."""
import json
import time
from urllib.parse import parse_qs

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, NoEncryption
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_bawt.integrations.connections import IntegrationError, KEEP_SCOPE
from llm_bawt.integrations.keep_workspace import TOKEN_URL, WorkspaceAuthError, validate_service_account
from llm_bawt.service.routes import integrations as routes
from tests.test_integration_connections import manager as integration_manager
from llm_bawt.service.routes.service_auth import require_service_token

manager = pytest.fixture(name="manager")(integration_manager.__wrapped__)  # noqa: F401 -- shared isolated encrypted store fixture


@pytest.fixture
def key():
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    data = {"type": "service_account", "client_id": "123456789012345678901",
            "client_email": "bawthub-keep@test-project.iam.gserviceaccount.com", "project_id": "test-project",
            "private_key_id": "key-id", "token_uri": TOKEN_URL,
            "private_key": private.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()).decode()}
    return data, private


def configure(manager, key):
    return manager.configure_workspace("google-keep", json.dumps(key[0]), "Operator@Example.com")


def google(manager, key, *, keep_status=200, token_error=None, on_probe=None):
    calls = []
    def handle(request):
        calls.append(request)
        if request.url.path == "/token":
            form = parse_qs(request.content.decode())
            assert form["grant_type"] == ["urn:ietf:params:oauth:grant-type:jwt-bearer"]
            assertion = form["assertion"][0]
            claims = jwt.decode(assertion, key[1].public_key(), algorithms=["RS256"], audience=TOKEN_URL)
            assert claims["iss"] == key[0]["client_email"]
            assert claims["sub"] == "operator@example.com"
            assert claims["scope"] == KEEP_SCOPE
            assert claims["exp"] - claims["iat"] == 3600
            assert jwt.get_unverified_header(assertion)["kid"] == "key-id"
            if token_error:
                return httpx.Response(400, json={"error": token_error, "error_description": "DO NOT LEAK"})
            return httpx.Response(200, json={"access_token": "private-access", "expires_in": 3600})
        assert request.url.host == "keep.googleapis.com"
        assert request.method == "GET"
        assert request.headers["authorization"] == "Bearer private-access"
        assert request.url.params["pageSize"] in {"1", "100"}
        if on_probe:
            on_probe()
        return httpx.Response(keep_status, json={"notes": [{"text": "PRIVATE NOTE"}]})
    manager.client_factory = lambda **kwargs: httpx.Client(transport=httpx.MockTransport(handle), **kwargs)
    return calls


def test_configuration_is_encrypted_and_redacted(manager, key):
    result = configure(manager, key)
    assert result["workspace"]["subject"] == "operator@example.com"
    assert result["workspace"]["client_id"] == key[0]["client_id"]
    assert result["auth_method"] == "service_account_delegation"
    assert result["workspace_admin_scopes"] == [KEEP_SCOPE]
    assert not result["connection"]["connected"]
    raw = json.dumps(manager.store._store.get_scope_settings("global", "*"))
    assert "PRIVATE KEY" not in raw + json.dumps(result)


@pytest.mark.parametrize("field,value", [
    ("type", "authorized_user"), ("client_id", "web.apps.googleusercontent.com"),
    ("token_uri", "https://evil.example/token"), ("client_email", "evil@example.com"),
    ("private_key", "not a key"), ("project_id", ""),
])
def test_invalid_key_rejected(manager, key, field, value):
    key[0][field] = value
    with pytest.raises(IntegrationError, match="invalid_service_account_key"):
        configure(manager, key)
    assert manager.store.load("workspace:google-keep") is None


@pytest.mark.parametrize("subject,error", [
    ("user@gmail.com", "workspace_required"), ("user@googlemail.com", "workspace_required"),
    ("not-email", "invalid_workspace_user"), ("user@example.com\nextra", "invalid_workspace_user"),
])
def test_subject_validation(key, subject, error):
    with pytest.raises(WorkspaceAuthError, match=error):
        validate_service_account(json.dumps(key[0]), subject)


def test_verified_connection_contains_no_private_key_or_refresh_token(manager, key):
    configure(manager, key)
    calls = google(manager, key)
    result = manager.verify_workspace("google-keep")
    assert len(calls) == 2
    assert result["connection"]["connected"]
    assert result["connection"]["account"] == "operator@example.com"
    assert result["connection"]["auth_method"] == "service_account_delegation"
    assert result["connection"]["meta"]["expires_at"] > time.time()
    assert manager.store.load("google-keep").secret == {"access_token": "private-access"}
    assert "PRIVATE NOTE" not in json.dumps(result)
    assert "private-access" not in json.dumps(result)


@pytest.mark.parametrize("reason,code", [
    ("unauthorized_client", "workspace_delegation_denied"), ("invalid_grant", "workspace_assertion_rejected"),
    ("invalid_scope", "workspace_scope_denied"), ("DO NOT LEAK", "workspace_token_failed"),
])
def test_token_errors_bounded(manager, key, reason, code):
    configure(manager, key)
    google(manager, key, token_error=reason)
    with pytest.raises(IntegrationError, match=code):
        manager.verify_workspace("google-keep")
    assert manager.store.load("google-keep") is None
    assert manager.descriptor("google-keep")["workspace"]["configured"]


def test_keep_denial_retains_config_but_does_not_connect(manager, key):
    configure(manager, key)
    google(manager, key, keep_status=403)
    with pytest.raises(IntegrationError, match="keep_access_denied"):
        manager.verify_workspace("google-keep")
    assert manager.store.load("google-keep") is None
    assert manager.store.load("workspace:google-keep").secret["private_key"]


def test_failed_reverify_preserves_previous_connection(manager, key):
    configure(manager, key)
    google(manager, key)
    manager.verify_workspace("google-keep")
    google(manager, key, keep_status=403)
    with pytest.raises(IntegrationError):
        manager.verify_workspace("google-keep")
    assert manager.store.load("google-keep").status == "connected"


@pytest.mark.parametrize("action", ["disconnect", "reconfigure"])
def test_stale_verification_cannot_restore_connection(manager, key, action):
    configure(manager, key)
    def invalidate():
        if action == "disconnect":
            manager.disconnect("google-keep")
        else:
            configure(manager, key)
    google(manager, key, on_probe=invalidate)
    with pytest.raises(IntegrationError, match="invalid_state"):
        manager.verify_workspace("google-keep")
    assert manager.store.load("google-keep") is None


def test_changing_target_user_invalidates_old_verified_connection(manager, key):
    configure(manager, key)
    google(manager, key)
    manager.verify_workspace("google-keep")
    manager.configure_workspace("google-keep", json.dumps(key[0]), "other@example.com")
    assert not manager.descriptor("google-keep")["connection"]["connected"]
    assert manager.descriptor("google-keep")["workspace"]["subject"] == "other@example.com"


def test_disconnect_removes_delegated_signing_key(manager, key):
    configure(manager, key)
    assert manager.disconnect("google-keep")
    assert manager.store.load("workspace:google-keep") is None
    # Existing network-independent Web OAuth config is preserved, not repurposed.
    assert manager.store.load("client:google-keep") is not None


def test_note_preview_paginates_without_exposing_contents(manager, key):
    configure(manager, key)
    google(manager, key)
    manager.verify_workspace("google-keep")
    pages = [
        {"notes": [{"name": "notes/a", "title": "Inbox", "body": {"list": {"listItems": [
            {"text": {"text": "PRIVATE CONTENT"}, "checked": False}, {"text": {"text": "another"}, "checked": True}
        ]}}}], "nextPageToken": "next"},
        {"notes": [{"name": "notes/b", "title": "Archive", "body": {"text": {"text": "PRIVATE TEXT"}}},
                   {"name": "notes/c", "title": "Trashed", "trashed": True, "body": {"list": {}}}]},
    ]
    def handle(request):
        if request.url.path == "/token":
            return httpx.Response(200, json={"access_token": "private-access", "expires_in": 3600})
        return httpx.Response(200, json=pages[1 if request.url.params.get("pageToken") == "next" else 0])
    manager.client_factory = lambda **kw: httpx.Client(transport=httpx.MockTransport(handle), **kw)
    preview = manager.list_keep_notes("google-keep")
    assert preview["total"] == 2
    assert preview["notes"][0]["unchecked_count"] == 1
    assert preview["notes"][0]["item_count"] == 2
    assert preview["notes"][1]["type"] == "text"
    assert "PRIVATE CONTENT" not in json.dumps(preview)
    assert "PRIVATE TEXT" not in json.dumps(preview)


def test_selected_list_reads_only_unchecked_flat_items(manager, key):
    configure(manager, key)
    google(manager, key)
    manager.verify_workspace("google-keep")
    calls = []
    def handle(request):
        calls.append(request)
        if request.url.path == "/token":
            return httpx.Response(200, json={"access_token": "private-access", "expires_in": 3600})
        return httpx.Response(200, json={"name": "notes/a", "title": "Selected list",
            "body": {"list": {"listItems": [
                {"checked": False, "text": {"text": " first task "}},
                {"checked": True, "text": {"text": "completed"}},
                {"checked": False, "text": {"text": "parent"},
                 "childListItems": [{"text": {"text": "nested"}}]},
                {"checked": False, "text": {"text": ""}},
            ]}}})
    manager.client_factory = lambda **kw: httpx.Client(transport=httpx.MockTransport(handle), **kw)
    result = manager.read_keep_list("google-keep", "notes/a")
    assert result["unchecked_items"] == ["first task"]
    assert result["account"] == "operator@example.com"
    assert calls[-1].url.path == "/v1/notes/a"
    assert calls[-1].method == "GET"


def test_selected_list_rejects_bad_note_and_disconnected(manager, key):
    with pytest.raises(IntegrationError, match="invalid_keep_note"):
        manager.read_keep_list("google-keep", "notes/a/extra")
    with pytest.raises(IntegrationError, match="keep_not_connected"):
        manager.read_keep_list("google-keep", "notes/a")
    configure(manager, key)
    google(manager, key)
    manager.verify_workspace("google-keep")
    manager.disconnect("google-keep")
    with pytest.raises(IntegrationError, match="keep_not_connected"):
        manager.read_keep_list("google-keep", "notes/a")


def test_preview_requires_verified_workspace_connection(manager, key):
    configure(manager, key)
    with pytest.raises(IntegrationError, match="keep_not_connected"):
        manager.list_keep_notes("google-keep")


def test_workspace_routes(manager, key, monkeypatch):
    monkeypatch.setattr(routes, "manager", lambda _: manager)
    app = FastAPI()
    app.dependency_overrides[require_service_token] = lambda: None  # BawtHub's bearer (TASK-2370)
    app.include_router(routes.router)
    client = TestClient(app)
    response = client.put("/v1/integrations/google-keep/workspace", json={
        "service_account_json": json.dumps(key[0]), "subject": "operator@example.com",
    })
    assert response.status_code == 200
    assert "PRIVATE KEY" not in response.text
    google(manager, key)
    response = client.post("/v1/integrations/google-keep/workspace/verify")
    assert response.status_code == 200
    assert response.json()["connection"]["connected"]
    response = client.get("/v1/integrations/google-keep/notes")
    assert response.status_code == 200
    assert response.json()["total"] == 0
    assert "PRIVATE NOTE" not in response.text
