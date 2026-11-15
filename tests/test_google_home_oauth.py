import re
import time
from dataclasses import replace
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select, update
from sqlalchemy.pool import StaticPool

from llm_bawt.integrations.google_home_oauth import (
    OAuthError, ProbeAuthenticator, ProbeOAuthStore, ProbeSettings,
    ProbeUnauthorized, ProbeUnavailable, _tokens,
)
from llm_bawt.integrations.google_home_probe import GoogleHomeProbe
from llm_bawt.service.routes import google_home_probe as route

SETTINGS = ProbeSettings(
    enabled=True, project_id="bawthub-00a0bf", client_id="loopy-google-home-probe",
    allowed_subject="operator", allowed_email="owner@example.com", public_origin="https://app.example.com",
    sso_auth_url="http://sso.internal:4180/oauth2/auth", sso_signin_url="https://auth.example.com/oauth2/start",
)
SECRET = "test-only-client-secret-" + "x" * 32


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def store():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    return ProbeOAuthStore(engine, SETTINGS)


def params(**changes):
    return dict(client_id=SETTINGS.client_id, response_type="code", redirect_uri=SETTINGS.redirects[0],
                scope=SETTINGS.required_scope, state="google-state + unicode-é") | changes


def grant(store):
    consent = store.consent(params())
    code = store.approve(consent, True)["code"]
    return dict(grant_type="authorization_code", code=code, redirect_uri=SETTINGS.redirects[0])


def tokens(store):
    return store.exchange(grant(store))


@pytest.mark.parametrize("changes", [
    {"enabled": False}, {"enabled": "true"}, {"project_id": "bad/id"}, {"allowed_subject": ""},
    {"allowed_email": ""}, {"public_origin": "http://app.example.com"},
    {"public_origin": "https://app.example.com/path"}, {"public_origin": "https://u:p@app.example.com"},
    {"sso_signin_url": "https://auth.example.com/start?rd=bad"},
    {"sso_auth_url": "file:///etc/passwd"}, {"required_scope": "a b"},
])
def test_configuration_fails_closed(changes):
    with pytest.raises(ProbeUnavailable):
        replace(SETTINGS, **changes).validate()


@pytest.mark.parametrize("changes", [
    {"client_id": "other"}, {"redirect_uri": SETTINGS.redirects[0] + "/"},
    {"redirect_uri": SETTINGS.redirects[0] + "?next=bad"}, {"redirect_uri": "https://attacker.example"},
    {"response_type": "token"}, {"scope": "admin"}, {"state": ""},
    {"code_challenge": "challenge"}, {"code_challenge_method": "S256"},
])
def test_authorization_boundaries(store, changes):
    with pytest.raises(OAuthError):
        store.consent(params(**changes))


def test_one_use_consent_code_refresh_and_hash_only_storage(store):
    consent = store.consent(params())
    result = store.approve(consent, True)
    assert result["state"] == params()["state"]
    with pytest.raises(OAuthError):
        store.approve(consent, True)
    exchange = dict(grant_type="authorization_code", code=result["code"], redirect_uri=result["redirect_uri"])
    with pytest.raises(OAuthError):
        store.exchange(exchange | {"redirect_uri": "https://attacker.example"})
    issued = store.exchange(exchange)
    with pytest.raises(OAuthError):
        store.exchange(exchange)
    assert store.authenticate("Bearer " + issued["access_token"]) == "operator"
    refreshed = store.exchange(dict(grant_type="refresh_token", refresh_token=issued["refresh_token"]))
    assert "refresh_token" not in refreshed
    assert refreshed["access_token"] != issued["access_token"]
    assert store.authenticate("Bearer " + refreshed["access_token"]) == "operator"
    with store.engine.connect() as conn:
        rows = str(conn.execute(select(_tokens)).all())
    for secret in (consent, result["code"], issued["access_token"], issued["refresh_token"], refreshed["access_token"]):
        assert secret not in rows


def test_cancel_does_not_issue_code(store):
    result = store.approve(store.consent(params()), False)
    assert result["error"] == "access_denied"
    assert "code" not in result


@pytest.mark.parametrize("kind", ["consent", "code", "access"])
def test_expired_credentials_rejected(store, kind):
    if kind == "consent":
        value = store.consent(params())
    elif kind == "code":
        value = grant(store)
    else:
        value = tokens(store)["access_token"]
    with store.engine.begin() as conn:
        conn.execute(update(_tokens).where(_tokens.c.kind == kind).values(expires_at=time.time() - 1))
    with pytest.raises((OAuthError, ProbeUnauthorized)):
        if kind == "consent":
            store.approve(value, True)
        elif kind == "code":
            store.exchange(value)
        else:
            store.authenticate("Bearer " + value)


def test_disconnect_revokes_pending_and_issued_credentials(store):
    issued = tokens(store)
    pending = grant(store)
    consent = store.consent(params())
    store.revoke("Bearer " + issued["access_token"])
    with pytest.raises(ProbeUnauthorized):
        store.authenticate("Bearer " + issued["access_token"])
    with pytest.raises(OAuthError):
        store.exchange(dict(grant_type="refresh_token", refresh_token=issued["refresh_token"]))
    with pytest.raises(OAuthError):
        store.exchange(pending)
    with pytest.raises(OAuthError):
        store.approve(consent, True)
    fresh = tokens(store)
    with pytest.raises(ProbeUnauthorized):
        store.revoke("Bearer " + issued["access_token"])
    assert store.authenticate("Bearer " + fresh["access_token"]) == "operator"


@pytest.mark.parametrize("changes", [{"client_id": "other"}, {"allowed_subject": "other"}, {"required_scope": "other"}])
def test_access_and_refresh_bound_to_client_subject_and_scope(store, changes):
    issued = tokens(store)
    other = ProbeOAuthStore(store.engine, replace(SETTINGS, **changes))
    with pytest.raises(ProbeUnauthorized):
        other.authenticate("Bearer " + issued["access_token"])
    with pytest.raises(OAuthError):
        other.exchange(dict(grant_type="refresh_token", refresh_token=issued["refresh_token"]))


@pytest.mark.parametrize("header", ["", "Basic a", "Bearer ", "Bearer a b", "Bearer " + "x" * 8193])
def test_bad_bearer_rejected(store, header):
    with pytest.raises(ProbeUnauthorized):
        store.authenticate(header)


@pytest.mark.anyio
@pytest.mark.parametrize("status,email,expected", [
    (202, "owner@example.com", None), (200, "OWNER@example.com", None),
    (202, "other@example.com", ProbeUnauthorized), (202, "", ProbeUnauthorized),
    (401, "owner@example.com", ProbeUnauthorized), (403, "owner@example.com", ProbeUnauthorized),
    (500, "owner@example.com", ProbeUnavailable), (302, "owner@example.com", ProbeUnavailable),
])
async def test_sso_response_not_browser_headers_proves_identity(store, status, email, expected):
    def sso(request):
        assert str(request.url) == SETTINGS.sso_auth_url
        assert request.headers["cookie"] == "sso=session"
        assert "authorization" not in request.headers
        assert "x-auth-request-email" not in request.headers
        return httpx.Response(status, headers={"X-Auth-Request-Email": email})
    auth = ProbeAuthenticator(SETTINGS, SECRET, store)
    async with httpx.AsyncClient(transport=httpx.MockTransport(sso)) as http:
        if expected:
            with pytest.raises(expected):
                await auth.browser_identity("sso=session", http)
        else:
            assert await auth.browser_identity("sso=session", http) == "operator"


@pytest.fixture
def client(store, monkeypatch):
    auth = ProbeAuthenticator(SETTINGS, SECRET, store)
    async def browser(cookie, http):
        if "sso=session" not in cookie:
            raise ProbeUnauthorized()
        return "operator"
    monkeypatch.setattr(auth, "browser_identity", browser)
    class Captures:
        def record(self, key, value):
            pass
    monkeypatch.setattr(route, "get_probe", lambda: GoogleHomeProbe(Captures()))
    app = FastAPI()
    app.include_router(route.router)
    app.dependency_overrides[route.get_probe_auth] = lambda: auth
    http = TestClient(app, base_url=SETTINGS.public_origin, follow_redirects=False)
    return http, store


def begin(http):
    http.cookies.set("sso", "session")
    response = http.get(route.BASE + "/authorize", params=params())
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["referrer-policy"] == "same-origin"
    policy = response.headers["content-security-policy"]
    assert "form-action 'self' " + " ".join(SETTINGS.redirects) + ";" in policy
    assert "*" not in policy
    assert "HttpOnly" in response.headers["set-cookie"]
    assert "Secure" in response.headers["set-cookie"]
    return re.search(r'name="consent" value="([^"]+)"', response.text).group(1)


def approve(http):
    consent = begin(http)
    response = http.post(route.BASE + "/authorize", data={"consent": consent, "decision": "allow"},
                         headers={"Origin": SETTINGS.public_origin})
    assert response.status_code == 303
    target = urlsplit(response.headers["location"])
    result = parse_qs(target.query)
    assert result["state"] == [params()["state"]]
    return result["code"][0]


def test_full_http_link_refresh_capture_unlink(client):
    http, store = client
    code = approve(http)
    credentials = dict(client_id=SETTINGS.client_id, client_secret=SECRET)
    response = http.post(route.BASE + "/token", data=credentials | dict(
        grant_type="authorization_code", code=code, redirect_uri=SETTINGS.redirects[0]))
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    issued = response.json()
    assert http.post(route.BASE + "/token", data=credentials | dict(
        grant_type="refresh_token", refresh_token=issued["refresh_token"])).status_code == 200
    bearer = {"Authorization": "Bearer " + issued["access_token"]}
    body = {"requestId": "sync", "inputs": [{"intent": "action.devices.SYNC"}]}
    assert http.post(route.BASE + "/fulfillment", json=body, headers=bearer).json()["payload"]["agentUserId"] == "operator"
    body["inputs"][0]["intent"] = "action.devices.DISCONNECT"
    result = http.post(route.BASE + "/fulfillment", json=body, headers=bearer)
    assert result.status_code == 200 and result.content == b""
    assert http.post(route.BASE + "/fulfillment", json=body, headers=bearer).status_code == 401
    assert http.post(route.BASE + "/token", data=credentials | dict(
        grant_type="refresh_token", refresh_token=issued["refresh_token"])).json() == {"error": "invalid_grant"}


def test_spoofed_browser_email_does_not_skip_login(client):
    http, _ = client
    response = http.get(route.BASE + "/authorize", params=params(), headers={"X-Auth-Request-Email": SETTINGS.allowed_email})
    assert "Sign in to BawtHub" in response.text
    assert "Agree and link" not in response.text
    assert "set-cookie" not in response.headers


@pytest.mark.parametrize("mutation", ["origin", "null_origin", "cookie", "consent", "identity", "decision"])
def test_consent_csrf_and_identity_boundaries(client, mutation):
    http, _ = client
    consent = begin(http)
    origin = SETTINGS.public_origin
    decision = "allow"
    if mutation == "origin":
        origin = "https://attacker.example"
    if mutation == "null_origin":
        origin = "null"
    if mutation == "cookie":
        http.cookies.clear()
        http.cookies.set("sso", "session")
    if mutation == "consent":
        consent = "wrong"
    if mutation == "identity":
        http.cookies.delete("sso")
    if mutation == "decision":
        decision = "yes"
    response = http.post(route.BASE + "/authorize", data={"consent": consent, "decision": decision}, headers={"Origin": origin})
    assert response.status_code == 403


def test_no_open_redirect_and_duplicate_params_rejected(client):
    http, _ = client
    response = http.get(route.BASE + "/authorize", params=params(redirect_uri="https://attacker.example"))
    assert response.status_code == 400 and "location" not in response.headers
    response = http.get(route.BASE + "/authorize?" + urlencode(params()) + "&client_id=evil")
    assert response.status_code == 400


def test_client_auth_before_code_consumption(client):
    http, _ = client
    code = approve(http)
    data = dict(client_id=SETTINGS.client_id, client_secret="wrong", grant_type="authorization_code",
                code=code, redirect_uri=SETTINGS.redirects[0])
    assert http.post(route.BASE + "/token", data=data).status_code == 401
    data["client_secret"] = SECRET
    assert http.post(route.BASE + "/token", data=data, headers={"Authorization": "Basic junk"}).status_code == 401
    assert http.post(route.BASE + "/token", data=data).status_code == 200
    assert http.post(route.BASE + "/token", data=data).json() == {"error": "invalid_grant"}


def test_form_limits_and_bad_encoding(client):
    http, _ = client
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    assert http.post(route.BASE + "/token", json={}).status_code == 400
    assert http.post(route.BASE + "/token", content=b"x" * 8193, headers=headers).status_code == 413
    assert http.post(route.BASE + "/token", content=b"x=\xff", headers=headers).status_code == 400
    assert http.post(route.BASE + "/token", content="client_id=a&client_id=b", headers=headers).status_code == 400


def test_store_survives_reconstruction(store):
    issued = tokens(store)
    restored = ProbeOAuthStore(store.engine, SETTINGS)
    assert restored.authenticate("Bearer " + issued["access_token"]) == "operator"
    assert restored.exchange(dict(grant_type="refresh_token", refresh_token=issued["refresh_token"]))["access_token"]


def test_token_types_cannot_be_substituted(store):
    issued = tokens(store)
    with pytest.raises(ProbeUnauthorized):
        store.authenticate("Bearer " + issued["refresh_token"])
    with pytest.raises(OAuthError):
        store.exchange(dict(grant_type="refresh_token", refresh_token=issued["access_token"]))


def test_sandbox_redirect_and_cancel_preserve_state(client):
    http, _ = client
    http.cookies.set("sso", "session")
    response = http.get(route.BASE + "/authorize", params=params(redirect_uri=SETTINGS.redirects[1]))
    consent = re.search(r'name="consent" value="([^\"]+)"', response.text).group(1)
    response = http.post(route.BASE + "/authorize", data={"consent": consent, "decision": "deny"},
                         headers={"Origin": SETTINGS.public_origin})
    assert response.status_code == 303
    assert response.headers["location"].startswith(SETTINGS.redirects[1] + "?")
    result = parse_qs(urlsplit(response.headers["location"]).query)
    assert result == {"error": ["access_denied"], "state": [params()["state"]]}


def test_unlink_does_not_depend_on_capture_database(client, monkeypatch):
    http, store = client
    issued = tokens(store)
    monkeypatch.setattr(route, "get_probe", lambda: pytest.fail("Unlink touched capture store"))
    result = http.post(route.BASE + "/fulfillment", headers={"Authorization": "Bearer " + issued["access_token"]},
                       json={"requestId": "unlink", "inputs": [{"intent": "action.devices.DISCONNECT"}]})
    assert result.status_code == 200
    with pytest.raises(ProbeUnauthorized):
        store.authenticate("Bearer " + issued["access_token"])


@pytest.mark.anyio
async def test_sso_outage_and_missing_cookie_fail_closed(store):
    auth = ProbeAuthenticator(SETTINGS, SECRET, store)
    def unavailable(request):
        raise httpx.ConnectError("unavailable", request=request)
    async with httpx.AsyncClient(transport=httpx.MockTransport(unavailable)) as http:
        with pytest.raises(ProbeUnauthorized):
            await auth.browser_identity("", http)
        with pytest.raises(ProbeUnavailable):
            await auth.browser_identity("sso=session", http)


def test_disabled_dependency_creates_no_store(monkeypatch):
    class Service:
        config = object()
    class Settings:
        def get_scope_settings(self, *args):
            return {}
    monkeypatch.setattr(route, "get_service", lambda: Service())
    monkeypatch.setattr(route, "get_runtime_settings_store", lambda _: Settings())
    monkeypatch.setattr(route, "ProbeOAuthStore", lambda *args: pytest.fail("Disabled handler initialized store"))
    app = FastAPI()
    app.include_router(route.router)
    http = TestClient(app)
    for path in ("token", "fulfillment", "authorize"):
        assert http.post(route.BASE + "/" + path).status_code == 503
    assert http.get(route.BASE + "/authorize").status_code == 503
