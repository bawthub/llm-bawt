"""TASK-2610: MCP-standard OAuth client for owner-added remote MCP servers, against a fake server."""
from __future__ import annotations

import json
import time
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from cryptography.fernet import Fernet
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, create_engine

from llm_bawt.integrations.connections import IntegrationError, IntegrationStore
from llm_bawt.integrations.mcp_servers import oauth as oauth_module
from llm_bawt.integrations.mcp_servers.oauth import McpOAuth, callback_path, check_destination, store_id
from llm_bawt.integrations.mcp_servers.store import McpServer, McpServerError, validate_slug, validate_url
from llm_bawt.runtime_settings import RuntimeSetting, RuntimeSettingsStore
from llm_bawt.service.providers import crypto

ORIGIN = "https://hub.example.com"
MCP_URL = "https://tools.example.net/mcp"
AS = "https://auth.example.net"
SERVER = McpServer("tools", "Tools", MCP_URL, "streamable_http", "oauth", True, "gen-1")
SETTINGS = "/settings/mcp-servers"
PUBLIC = {"auth.example.net": ["93.184.216.34"], "public.example.org": ["2606:2800:220:1::1"],
          "private.example.org": ["10.0.0.5"], "mixed.example.org": ["93.184.216.34", "127.0.0.1"]}


@pytest.fixture(autouse=True)
def resolver(monkeypatch):
    def addresses(host):
        if host not in PUBLIC:
            raise OSError("unknown host")
        return PUBLIC[host]
    monkeypatch.setattr(oauth_module, "_addresses", addresses)


@pytest.fixture
def store(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine, tables=[RuntimeSetting.__table__])
    runtime = object.__new__(RuntimeSettingsStore)
    runtime.engine = engine
    store = object.__new__(IntegrationStore)
    store._store = runtime
    cipher = Fernet(Fernet.generate_key())
    monkeypatch.setattr(crypto, "encrypt", lambda text: cipher.encrypt(text.encode()).decode())
    monkeypatch.setattr(crypto, "decrypt", lambda text: cipher.decrypt(text.encode()).decode())
    return store


class FakeRemote:
    """A protected MCP resource plus its authorization server."""

    def __init__(self, *, pkce=("S256",), registration=True, www_auth=True, iss_param=False,
                 resource=MCP_URL, issuer=AS, endpoint=f"{AS}/authorize", prm_url=f"{AS}/prm"):
        self.pkce, self.registration, self.www_auth, self.iss_param = pkce, registration, www_auth, iss_param
        self.resource, self.issuer, self.endpoint, self.prm_url = resource, issuer, endpoint, prm_url
        self.calls: list[httpx.Request] = []
        self.registrations = 0
        self.refresh_status = 200
        self.as_scopes: list[str] | None = None
        self.resource_scopes = True  # the challenge and resource metadata name "mcp"
        self.registered_scopes: list[str | None] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        url = str(request.url).split("?")[0]
        if url == MCP_URL and request.method == "POST":
            scope = ', scope="mcp"' if self.resource_scopes else ""
            headers = ({"WWW-Authenticate": f'Bearer resource_metadata="{self.prm_url}"{scope}'}
                       if self.www_auth else {})
            return httpx.Response(401, headers=headers)
        if url in (f"{AS}/prm", "https://tools.example.net/.well-known/oauth-protected-resource/mcp"):
            return httpx.Response(200, json={
                "resource": self.resource, "authorization_servers": [AS],
                **({"scopes_supported": ["mcp", "offline_access"]} if self.resource_scopes else {})})
        if url == f"{AS}/.well-known/oauth-authorization-server":
            return httpx.Response(200, json={
                **({"issuer": self.issuer} if self.issuer else {}),
                "authorization_endpoint": self.endpoint, "token_endpoint": f"{AS}/token",
                **({"registration_endpoint": f"{AS}/register"} if self.registration else {}),
                "code_challenge_methods_supported": self.pkce if isinstance(self.pkce, str) else list(self.pkce),
                "authorization_response_iss_parameter_supported": self.iss_param,
                **({"scopes_supported": self.as_scopes} if self.as_scopes is not None else {})})
        if url == f"{AS}/register":
            self.registrations += 1
            body = httpx.Request("POST", url, content=request.content).read()
            assert b'"token_endpoint_auth_method":"none"' in body.replace(b" ", b"")
            self.registered_scopes.append(json.loads(body).get("scope"))
            return httpx.Response(201, json={"client_id": f"client-{self.registrations}"})
        if url == f"{AS}/token":
            form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
            assert form["resource"] == MCP_URL
            if form["grant_type"] == "refresh_token":
                if self.refresh_status != 200:
                    return httpx.Response(self.refresh_status, json={"error": "invalid_grant"})
                return httpx.Response(200, json={"access_token": "access-2", "refresh_token": "refresh-2",
                                                 "token_type": "Bearer", "expires_in": 3600})
            assert form["code"] == "the-code" and form["code_verifier"]
            return httpx.Response(200, json={"access_token": "access-1", "refresh_token": "refresh-1",
                                             "token_type": "Bearer", "expires_in": 3600, "scope": "mcp"})
        return httpx.Response(404)


def oauth_for(store, remote):
    return McpOAuth(None, store=store,
                    client_factory=lambda **kw: httpx.Client(transport=httpx.MockTransport(remote.handle), **kw))


def sign_in(oauth, **finish):
    flow = oauth.start(SERVER, ORIGIN, SETTINGS)
    params = {k: v[0] for k, v in parse_qs(urlsplit(flow["authorization_url"]).query).items()}
    args = {"state": params["state"], "browser": flow["browser"], "code": "the-code", "error": "", "iss": "", **finish}
    return flow, params, oauth.finish(SERVER, args["state"], args["browser"], args["code"], args["error"], args["iss"])


def test_discovers_registers_and_signs_in_with_pkce_and_resource(store):
    remote = FakeRemote()
    oauth = oauth_for(store, remote)
    flow, params, return_to = sign_in(oauth)
    assert flow["authorization_url"].startswith(f"{AS}/authorize?")
    assert params["code_challenge_method"] == "S256" and params["resource"] == MCP_URL
    assert params["redirect_uri"] == ORIGIN + callback_path("tools") and params["scope"] == "mcp"
    assert params["client_id"] == "client-1" and return_to == SETTINGS
    assert oauth.access_token(SERVER) == "access-1"
    assert oauth.status(SERVER)["connected"] is True
    # The browser-facing result never carries a token.
    for token in ("access-1", "refresh-1"):
        assert token not in str(flow) and token not in str(oauth.status(SERVER))


def test_reuses_registration_for_same_issuer_and_callback(store):
    remote = FakeRemote()
    oauth = oauth_for(store, remote)
    sign_in(oauth)
    sign_in(oauth)
    assert remote.registrations == 1


def test_requests_offline_access_when_the_authorization_server_offers_it(store):
    # Without offline_access some servers issue no refresh token and the sign-in dies in an hour.
    remote = FakeRemote()
    remote.as_scopes = ["openid", "offline_access", "mcp"]
    _, params, _ = sign_in(oauth_for(store, remote))
    assert params["scope"] == "mcp offline_access"
    assert remote.registered_scopes == ["mcp offline_access"]


def test_re_registers_when_the_requested_scope_changes(store):
    remote = FakeRemote()
    oauth = oauth_for(store, remote)
    sign_in(oauth)
    remote.as_scopes = ["mcp", "offline_access"]
    _, params, _ = sign_in(oauth)
    assert remote.registrations == 2 and params["client_id"] == "client-2"
    assert remote.registered_scopes == ["mcp", "mcp offline_access"]
    assert store.load(f"client:{store_id('tools')}").meta["client_id"] == "client-2"


def test_omitted_resource_scope_stays_omitted(store):
    # No scope from the resource means the server's default; offline_access alone must not replace it.
    remote = FakeRemote()
    remote.resource_scopes = False
    remote.as_scopes = ["mcp", "offline_access"]
    _, params, _ = sign_in(oauth_for(store, remote))
    assert "scope" not in params and remote.registered_scopes == [None]


def _make_legacy(store):
    """A registration and token saved before the client recorded its scope."""
    client = store.load(f"client:{store_id('tools')}")
    client.meta = {k: v for k, v in client.meta.items() if k != "scope"}
    store.save(client)


def test_legacy_registration_is_reused_when_the_scope_did_not_change(store):
    remote = FakeRemote()
    oauth = oauth_for(store, remote)
    sign_in(oauth)
    _make_legacy(store)
    sign_in(oauth)
    assert remote.registrations == 1


def test_abandoned_reconnect_keeps_the_working_refresh(store):
    remote = FakeRemote()
    oauth = oauth_for(store, remote)
    sign_in(oauth)
    _make_legacy(store)
    remote.as_scopes = ["mcp", "offline_access"]
    oauth.start(SERVER, ORIGIN, SETTINGS)  # registers client-2, then the browser never returns
    assert remote.registrations == 2
    record = store.load(store_id("tools"))
    record.meta = {**record.meta, "expires_at": time.time() - 1}
    store.save(record)
    assert oauth.access_token(SERVER) == "access-2"
    assert store.load(f"client:{store_id('tools')}").meta["client_id"] == "client-1"


def test_falls_back_to_well_known_without_www_authenticate(store):
    oauth = oauth_for(store, FakeRemote(www_auth=False))
    _, params, _ = sign_in(oauth)
    assert params["resource"] == MCP_URL and "mcp" in params["scope"]


@pytest.mark.parametrize("pkce", [("plain",), "S256", ("s256",)])
def test_refuses_server_without_s256(store, pkce):
    with pytest.raises(IntegrationError, match="pkce_unsupported"):
        oauth_for(store, FakeRemote(pkce=pkce)).start(SERVER, ORIGIN, SETTINGS)


@pytest.mark.parametrize("resource", ["https://other.example.net/mcp", "https://tools.example.net/other", None])
def test_protected_resource_metadata_must_name_this_server(store, resource):
    with pytest.raises(IntegrationError, match="oauth_metadata_invalid"):
        oauth_for(store, FakeRemote(resource=resource)).start(SERVER, ORIGIN, SETTINGS)


@pytest.mark.parametrize("issuer", [None, "https://evil.example.org", f"{AS}/tenant"])
def test_authorization_server_metadata_must_declare_its_issuer(store, issuer):
    with pytest.raises(IntegrationError, match="issuer_mismatch"):
        oauth_for(store, FakeRemote(issuer=issuer)).start(SERVER, ORIGIN, SETTINGS)


def test_advertised_endpoints_must_be_public_https(store):
    with pytest.raises(IntegrationError, match="oauth_metadata_invalid"):
        oauth_for(store, FakeRemote(endpoint="http://auth.example.net/authorize")).start(SERVER, ORIGIN, SETTINGS)
    with pytest.raises(IntegrationError, match="destination_not_allowed"):
        oauth_for(store, FakeRemote(endpoint="https://private.example.org/authorize")).start(SERVER, ORIGIN, SETTINGS)
    with pytest.raises(IntegrationError, match="destination_not_allowed"):
        oauth_for(store, FakeRemote(prm_url="https://mixed.example.org/prm")).start(SERVER, ORIGIN, SETTINGS)


@pytest.mark.parametrize("url,error", [
    ("https://public.example.org/token", None),
    ("https://tools.example.net/token", None),  # the origin the operator chose, even if private
    ("https://tools.example.net:443/token", None),
    ("https://tools.example.net:8443/token", "destination_not_allowed"),  # same host, another service
    ("https://private.example.org/token", "destination_not_allowed"),
    ("https://unknown.example.org/token", "destination_not_allowed"),
    ("https://10.0.0.5/token", "destination_not_allowed"),
    ("https://169.254.169.254/latest", "destination_not_allowed"),
    ("http://public.example.org/token", "oauth_metadata_invalid"),
    ("https://user@public.example.org/token", "oauth_metadata_invalid"),
    ("https://public.example.org/to ken", "oauth_metadata_invalid"),
])
def test_destination_policy(url, error):
    if error is None:
        assert check_destination(url, MCP_URL) == url
    else:
        with pytest.raises(IntegrationError, match=error):
            check_destination(url, MCP_URL)


def test_refuses_without_registration_endpoint(store):
    with pytest.raises(IntegrationError, match="registration_unsupported"):
        oauth_for(store, FakeRemote(registration=False)).start(SERVER, ORIGIN, SETTINGS)


def test_requires_https_origin_and_known_return_path(store):
    oauth = oauth_for(store, FakeRemote())
    with pytest.raises(IntegrationError, match="https_required"):
        oauth.start(SERVER, "http://hub.example.com", SETTINGS)
    with pytest.raises(IntegrationError, match="invalid_return_path"):
        oauth.start(SERVER, ORIGIN, "https://evil.example/")


def test_callback_needs_matching_browser_and_is_single_use(store):
    oauth = oauth_for(store, FakeRemote())
    flow = oauth.start(SERVER, ORIGIN, SETTINGS)
    state = parse_qs(urlsplit(flow["authorization_url"]).query)["state"][0]
    with pytest.raises(IntegrationError, match="invalid_state"):
        oauth.finish(SERVER, state, "another-browser", "the-code", "", "")
    oauth.finish(SERVER, state, flow["browser"], "the-code", "", "")
    with pytest.raises(IntegrationError, match="invalid_state"):
        oauth.finish(SERVER, state, flow["browser"], "the-code", "", "")


def test_issuer_mismatch_is_refused(store):
    oauth = oauth_for(store, FakeRemote(iss_param=True))
    with pytest.raises(IntegrationError, match="issuer_mismatch"):
        sign_in(oauth, iss="https://other.example")
    assert oauth.status(SERVER)["connected"] is False


def test_refresh_is_central_and_rotates(store):
    remote = FakeRemote()
    oauth = oauth_for(store, remote)
    sign_in(oauth)
    record = store.load(store_id("tools"))
    record.meta = {**record.meta, "expires_at": time.time() - 1}
    store.save(record)
    assert oauth.access_token(SERVER) == "access-2"
    assert store.load(store_id("tools")).secret["refresh_token"] == "refresh-2"


def test_revoked_refresh_requires_reauthorization(store):
    remote = FakeRemote()
    oauth = oauth_for(store, remote)
    sign_in(oauth)
    record = store.load(store_id("tools"))
    record.meta = {**record.meta, "expires_at": time.time() - 1}
    store.save(record)
    remote.refresh_status = 400
    with pytest.raises(IntegrationError, match="reauthorization_required"):
        oauth.access_token(SERVER)


def test_token_is_bound_to_the_server_url(store):
    oauth = oauth_for(store, FakeRemote())
    sign_in(oauth)
    moved = McpServer("tools", "Tools", "https://elsewhere.example/mcp", "streamable_http", "oauth", True, "gen-1")
    with pytest.raises(IntegrationError, match="not_connected"):
        oauth.access_token(moved)


def test_a_re_added_server_never_inherits_the_old_sign_in(store):
    oauth = oauth_for(store, FakeRemote())
    flow = oauth.start(SERVER, ORIGIN, SETTINGS)
    state = parse_qs(urlsplit(flow["authorization_url"]).query)["state"][0]
    readded = McpServer("tools", "Tools", MCP_URL, "streamable_http", "oauth", True, "gen-2")
    with pytest.raises(IntegrationError, match="invalid_state"):
        oauth.finish(readded, state, flow["browser"], "the-code", "", "")
    sign_in(oauth)
    assert oauth.access_token(SERVER) == "access-1"
    assert oauth.status(readded)["connected"] is False
    with pytest.raises(IntegrationError, match="not_connected"):
        oauth.access_token(readded)


def test_disconnect_forgets_tokens_and_registration(store):
    oauth = oauth_for(store, FakeRemote())
    sign_in(oauth)
    assert oauth.disconnect("tools") is True
    assert store.load(store_id("tools")) is None and store.load(f"client:{store_id('tools')}") is None


@pytest.mark.parametrize("slug", ["bawthub", "playwright", "crawl4ai"])
def test_reserved_slugs_refused(slug):
    with pytest.raises(McpServerError, match="reserved_slug"):
        validate_slug(slug)


@pytest.mark.parametrize("slug", ["A", "1x", "a__b", "x_", "", "a" * 40, "a b", "abc\n"])
def test_invalid_slugs_refused(slug):
    with pytest.raises(McpServerError, match="invalid_slug"):
        validate_slug(slug)


@pytest.mark.parametrize("url", ["ftp://x/y", "https://user:pw@x/mcp", "https://x/mcp#f", "not a url",
                                 "https://x/mc\np", "https://x/m cp", "https://x\\y/mcp", "https://x:0/mcp",
                                 "https://x/mcp#"])
def test_invalid_urls_refused(url):
    with pytest.raises(McpServerError, match="invalid_url"):
        validate_url(url)


def test_writes_are_fenced_to_the_row_they_began_for(store):
    """A sign-in for a removed or re-added server never writes its pending flow or tokens."""
    rows = {"tools": SERVER}
    oauth = oauth_for(store, FakeRemote())
    oauth.current = rows.get
    flow = oauth.start(SERVER, ORIGIN, SETTINGS)
    state = parse_qs(urlsplit(flow["authorization_url"]).query)["state"][0]
    rows["tools"] = McpServer("tools", "Tools", MCP_URL, "streamable_http", "oauth", True, "gen-2")
    with pytest.raises(IntegrationError, match="server_changed"):
        oauth.finish(SERVER, state, flow["browser"], "the-code", "", "")
    assert store.load(store_id("tools")) is None
    with pytest.raises(IntegrationError, match="server_changed"):
        oauth.start(SERVER, ORIGIN, SETTINGS)
    del rows["tools"]
    with pytest.raises(IntegrationError, match="server_changed"):
        oauth.start(SERVER, ORIGIN, SETTINGS)


class _Stream:
    def __init__(self, host):
        self.host = host


def _backend(monkeypatch):
    connected = []

    def connect(self, host, port, timeout=None, local_address=None, socket_options=None):
        connected.append((host, port))
        return _Stream(host)

    monkeypatch.setattr(oauth_module.httpcore.SyncBackend, "connect_tcp", connect)
    return oauth_module._PolicyBackend(("tools.example.net", 443)), connected


def test_connections_go_only_to_addresses_that_passed(monkeypatch):
    backend, connected = _backend(monkeypatch)
    backend.connect_tcp("tools.example.net", 443)  # the configured origin, as chosen
    backend.connect_tcp("auth.example.net", 443)   # resolved here, connected by address
    assert connected == [("tools.example.net", 443), ("93.184.216.34", 443)]
    for host, port in [("private.example.org", 443), ("mixed.example.org", 443),
                       ("tools.example.net", 8443), ("10.0.0.5", 443), ("unknown.example.org", 443)]:
        with pytest.raises(IntegrationError, match="destination_not_allowed"):
            backend.connect_tcp(host, port)
    assert len(connected) == 2


def test_the_default_client_uses_the_policy_transport():
    transport = oauth_module.policy_transport(MCP_URL)
    assert isinstance(transport._pool._network_backend, oauth_module._PolicyBackend)


@pytest.mark.parametrize("url,shown", [
    ("https://tools.example.net/mcp", "https://tools.example.net/…"),
    ("https://tools.example.net/k/secret-path", "https://tools.example.net/…"),
    ("https://tools.example.net:8443/?key=secret", "https://tools.example.net:8443/…"),
    ("https://tools.example.net/", "https://tools.example.net"),
])
def test_members_see_only_the_origin(url, shown):
    public = McpServer("tools", "Tools", url, "streamable_http", "none", True, "g").public()
    assert public["url"] == shown and "secret" not in str(public)
