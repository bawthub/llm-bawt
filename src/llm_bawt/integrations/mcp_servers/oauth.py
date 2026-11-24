"""MCP-standard OAuth client for remote MCP servers (TASK-2610).

Discovery follows the MCP authorization spec: protected-resource metadata
(RFC 9728, from ``WWW-Authenticate`` or the well-known paths), then
authorization-server metadata (RFC 8414 / OIDC), dynamic client registration
(RFC 7591), and an authorization-code grant with PKCE S256 and the resource
indicator (RFC 8707). The app owns refresh: callers ask :meth:`access_token`
and never see a refresh token.

Every state record lives in the encrypted integration store under
``integration_connection:{pending:,client:,}mcp:<slug>``; nothing here is
returned to a browser except the authorization URL.
"""
from __future__ import annotations

import base64
import hashlib
import ipaddress
import secrets
import socket
import time
from urllib.parse import urlencode, urlsplit

import httpcore
import httpx

from mcp.client.auth.utils import (
    build_oauth_authorization_server_metadata_discovery_urls,
    build_protected_resource_metadata_discovery_urls,
    extract_resource_metadata_from_www_auth,
    extract_scope_from_www_auth,
)
from mcp.shared.auth_utils import check_resource_allowed, resource_url_from_server_url

from ..connections import IntegrationError, IntegrationStore, digest
from ...service.providers.base import ConnectionRecord
from .store import McpServer

CALLBACK_PREFIX = "/api/chat/proxy/v1/integrations/mcp-servers"
RETURN_PATHS = {"/settings/mcp-servers"}
PENDING_SECONDS = 600
CLIENT_NAME = "BawtHub"
_INIT_BODY = {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {
    "protocolVersion": "2025-06-18", "capabilities": {},
    "clientInfo": {"name": CLIENT_NAME, "version": "1"}}}


def store_id(slug: str) -> str:
    return f"mcp:{slug}"


def callback_path(slug: str) -> str:
    return f"{CALLBACK_PREFIX}/{slug}/callback"


def owned_by(record: ConnectionRecord, server: McpServer) -> bool:
    """Whether a stored secret belongs to this exact server row (URL and generation)."""
    return (bool(server.generation) and record.meta.get("generation") == server.generation
            and record.meta.get("server_url") == server.url)


def _json(response: httpx.Response) -> dict | None:
    if response.status_code != 200:
        return None
    try:
        body = response.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


def _addresses(host: str) -> list[str]:
    """Every address ``host`` resolves to (patched in tests)."""
    return [info[4][0] for info in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)]


def _origin(url: str) -> tuple[str, int]:
    parts = urlsplit(url)
    return (parts.hostname or "").lower(), parts.port or (443 if parts.scheme == "https" else 80)


def public_addresses(host: str) -> list[str]:
    """``host``'s addresses, provided every one is public; else destination_not_allowed."""
    try:
        addresses = [ipaddress.ip_address(a.split("%")[0]) for a in _addresses(host)]
    except (OSError, ValueError) as exc:
        raise IntegrationError("destination_not_allowed") from exc
    if not addresses or not all(a.is_global for a in addresses):
        raise IntegrationError("destination_not_allowed")
    return [str(a) for a in addresses]


class _PolicyBackend(httpcore.SyncBackend):
    """Enforces the egress policy on the connection itself, not on a separate lookup.

    The configured server's exact origin (host and port) connects as the
    operator chose it, private or not. Any other destination is resolved
    here, once, and the socket goes only to an address that passed; TLS still
    verifies the certificate against the URL's hostname (httpcore sends the
    request host as SNI, not the address).
    """

    def __init__(self, allowed: tuple[str, int]):
        self._allowed = allowed

    def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        if (host.lower(), port) == self._allowed:
            return super().connect_tcp(host, port, timeout, local_address, socket_options)
        error: Exception | None = None
        for address in public_addresses(host):
            try:
                return super().connect_tcp(address, port, timeout, local_address, socket_options)
            except httpcore.ConnectError as exc:
                error = exc
        raise error or httpcore.ConnectError("no address")


def policy_transport(server_url: str) -> httpx.HTTPTransport:
    """An HTTPS transport whose every connection obeys :class:`_PolicyBackend`."""
    transport = httpx.HTTPTransport()
    transport._pool._network_backend = _PolicyBackend(_origin(server_url))
    return transport


def check_destination(url: str, server_url: str) -> str:
    """Early refusal for a URL the remote server advertised (metadata, endpoints).

    HTTPS only, since these endpoints receive OAuth codes and secrets. The
    configured server's exact origin is trusted as the operator chose it;
    anything else must resolve only to public addresses. This check gives a
    clear error before any request; :class:`_PolicyBackend` enforces the same
    policy on the actual connection. Redirects are never followed.
    """
    try:
        parts = urlsplit(url)
        parts.port  # noqa: B018 - raises on a malformed port
    except ValueError as exc:
        raise IntegrationError("oauth_metadata_invalid") from exc
    if (parts.scheme != "https" or not parts.hostname or "@" in parts.netloc
            or any(ord(c) < 0x21 or ord(c) == 0x7F for c in url)):
        raise IntegrationError("oauth_metadata_invalid")
    if _origin(url) != _origin(server_url):
        public_addresses(parts.hostname)
    return url


def _same_issuer(a: str, b: str) -> bool:
    return bool(a) and a.rstrip("/") == b.rstrip("/")


def discover(http: httpx.Client, server_url: str) -> dict:
    """Authorization metadata for ``server_url``: endpoints, resource and scope.

    The protected-resource metadata must name a resource covering this server
    (RFC 9728) and the authorization-server metadata must declare the issuer it
    was discovered for (RFC 8414), before anything is registered or sent.
    """
    try:
        probe = http.post(server_url, json=_INIT_BODY,
                          headers={"Accept": "application/json, text/event-stream"})
    except httpx.HTTPError as exc:
        raise IntegrationError("server_unreachable") from exc
    www_auth_url = extract_resource_metadata_from_www_auth(probe) if probe.status_code == 401 else None
    scope = extract_scope_from_www_auth(probe) if probe.status_code == 401 else None
    if www_auth_url:
        check_destination(www_auth_url, server_url)
    resource, auth_server = resource_url_from_server_url(server_url), None
    for url in build_protected_resource_metadata_discovery_urls(www_auth_url, server_url):
        try:
            prm = _json(http.get(url))
        except httpx.HTTPError:
            prm = None
        servers = prm.get("authorization_servers") if prm else None
        if isinstance(servers, list) and servers and isinstance(servers[0], str):
            named = prm.get("resource")
            if not isinstance(named, str) or not check_resource_allowed(
                    resource_url_from_server_url(server_url), named):
                raise IntegrationError("oauth_metadata_invalid")
            auth_server, resource = check_destination(servers[0], server_url), named
            if not scope and isinstance(prm.get("scopes_supported"), list):
                scope = " ".join(str(s) for s in prm["scopes_supported"] if isinstance(s, str))
            break
    parts = urlsplit(server_url)
    expected_issuer = auth_server or f"{parts.scheme}://{parts.netloc}"
    metadata = None
    for url in build_oauth_authorization_server_metadata_discovery_urls(auth_server, server_url):
        try:
            metadata = _json(http.get(url))
        except httpx.HTTPError:
            metadata = None
        if metadata and metadata.get("authorization_endpoint") and metadata.get("token_endpoint"):
            break
        metadata = None
    if metadata is None:
        raise IntegrationError("oauth_metadata_not_found")
    issuer = metadata.get("issuer")
    if not isinstance(issuer, str) or not _same_issuer(issuer, expected_issuer):
        raise IntegrationError("issuer_mismatch")
    # The spec requires refusing an authorization server that does not advertise S256.
    methods = metadata.get("code_challenge_methods_supported")
    if not isinstance(methods, list) or "S256" not in [m for m in methods if isinstance(m, str)]:
        raise IntegrationError("pkce_unsupported")
    for key in ("authorization_endpoint", "token_endpoint", "registration_endpoint"):
        value = metadata.get(key)
        if value is not None and not isinstance(value, str):
            raise IntegrationError("oauth_metadata_invalid")
        if value:
            check_destination(value, server_url)
    # Some authorization servers issue a refresh token only for offline_access. Without
    # one, the sign-in dies when the first access token expires, so ask when offered.
    offered = metadata.get("scopes_supported")
    if isinstance(offered, list) and "offline_access" in offered:
        requested = (scope or "").split()
        if "offline_access" not in requested:
            scope = " ".join([*requested, "offline_access"])
    return {
        "issuer": issuer,
        "authorization_endpoint": str(metadata["authorization_endpoint"]),
        "token_endpoint": str(metadata["token_endpoint"]),
        "registration_endpoint": str(metadata.get("registration_endpoint") or ""),
        "iss_supported": bool(metadata.get("authorization_response_iss_parameter_supported")),
        "resource": resource,
        "scope": scope or "",
    }


def _register(http: httpx.Client, meta: dict, redirect_uri: str) -> dict:
    if not meta["registration_endpoint"]:
        raise IntegrationError("registration_unsupported")
    try:
        response = http.post(meta["registration_endpoint"], json={
            "client_name": CLIENT_NAME, "redirect_uris": [redirect_uri],
            "grant_types": ["authorization_code", "refresh_token"], "response_types": ["code"],
            "token_endpoint_auth_method": "none",
            **({"scope": meta["scope"]} if meta["scope"] else {}),
        })
    except httpx.HTTPError as exc:
        raise IntegrationError("registration_failed") from exc
    if response.status_code not in (200, 201):
        raise IntegrationError("registration_failed")
    try:
        body = response.json()
    except ValueError as exc:
        raise IntegrationError("registration_failed") from exc
    if not isinstance(body, dict) or not body.get("client_id"):
        raise IntegrationError("registration_failed")
    return body


def _client_auth(client: ConnectionRecord, data: dict) -> tuple[dict, tuple[str, str] | None]:
    """Token-endpoint client authentication for a public or confidential DCR client."""
    data = {**data, "client_id": client.meta["client_id"]}
    secret = client.secret.get("client_secret")
    if not secret:
        return data, None
    if client.meta.get("token_endpoint_auth_method") == "client_secret_post":
        return {**data, "client_secret": secret}, None
    return data, (client.meta["client_id"], secret)


def _token_request(http: httpx.Client, url: str, client: ConnectionRecord, data: dict) -> httpx.Response:
    body, auth = _client_auth(client, data)
    return http.post(url, data=body, auth=auth, headers={"Accept": "application/json"})


class McpOAuth:
    def __init__(self, config, *, store: IntegrationStore | None = None, client_factory=httpx.Client,
                 current=None):
        self.store = store if store is not None else IntegrationStore(config)
        self.client_factory = client_factory
        # slug -> the server row now, so writes can be fenced to the row they began for.
        self.current = current

    def fenced(self, server: McpServer):
        """Serialize this server's credential writes with create/remove (see McpServers)."""
        return self.store.lifecycle(store_id(server.slug))

    def require_current(self, server: McpServer) -> None:
        """Inside :meth:`fenced`: refuse a write for a row that was removed or replaced."""
        if self.current is None:
            return
        now = self.current(server.slug)
        if now is None or now.generation != server.generation or now.url != server.url:
            raise IntegrationError("server_changed")

    def _http(self, server: McpServer):
        if self.client_factory is httpx.Client:
            return httpx.Client(timeout=20, follow_redirects=False, transport=policy_transport(server.url))
        return self.client_factory(timeout=20, follow_redirects=False)  # tests inject a mock transport

    def start(self, server: McpServer, origin: str, return_to: str) -> dict:
        if server.auth_kind != "oauth":
            raise IntegrationError("not_oauth")
        if return_to not in RETURN_PATHS:
            raise IntegrationError("invalid_return_path")
        parts = urlsplit(origin or "")
        if parts.scheme != "https" or not parts.hostname or parts.path not in ("", "/"):
            raise IntegrationError("https_required")
        redirect_uri = f"https://{parts.netloc}{callback_path(server.slug)}"
        sid = store_id(server.slug)
        with self._http(server) as http:
            meta = discover(http, server.url)
            client = self.store.load(f"client:{sid}")
            # Reuse a registration only for the same authorization server, callback and scope.
            register = not (client and client.meta.get("issuer") == meta["issuer"]
                            and client.meta.get("redirect_uri") == redirect_uri
                            and client.meta.get("scope", "") == meta["scope"] and owned_by(client, server))
            if register:
                body = _register(http, meta, redirect_uri)
                client = ConnectionRecord(
                    provider=f"client:{sid}", status="configured",
                    meta={"client_id": str(body["client_id"]), "issuer": meta["issuer"],
                          "redirect_uri": redirect_uri, "server_url": server.url,
                          "generation": server.generation, "scope": meta["scope"],
                          "token_endpoint_auth_method": str(body.get("token_endpoint_auth_method") or "none")},
                    secret={"client_secret": str(body["client_secret"])} if body.get("client_secret") else {})
        state, browser, verifier = (secrets.token_urlsafe(32) for _ in range(3))
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
        with self.fenced(server):
            self.require_current(server)
            if register:
                self.store.save(client)
            self.store.save(ConnectionRecord(provider=f"pending:{sid}", status="pending", secret={
                "state_hash": digest(state), "browser_hash": digest(browser), "verifier": verifier,
                "expires_at": time.time() + PENDING_SECONDS, "redirect_uri": redirect_uri,
                "return_to": return_to, "client_id": client.meta["client_id"], "server_url": server.url,
                "generation": server.generation,
                **{k: meta[k] for k in ("issuer", "token_endpoint", "resource", "scope", "iss_supported")},
            }))
        params = {"response_type": "code", "client_id": client.meta["client_id"], "redirect_uri": redirect_uri,
                  "state": state, "code_challenge": challenge, "code_challenge_method": "S256",
                  "resource": meta["resource"], **({"scope": meta["scope"]} if meta["scope"] else {})}
        sep = "&" if urlsplit(meta["authorization_endpoint"]).query else "?"
        return {"authorization_url": meta["authorization_endpoint"] + sep + urlencode(params), "browser": browser}

    def finish(self, server: McpServer, state: str, browser: str, code: str, error: str, iss: str) -> str:
        sid = store_id(server.slug)
        pending = self.store.consume_state(sid, state, browser)
        if error:
            raise IntegrationError("authorization_denied")
        if not code:
            raise IntegrationError("missing_code")
        if pending["server_url"] != server.url or pending.get("generation") != server.generation:
            raise IntegrationError("invalid_state")
        if (iss or pending["iss_supported"]) and iss != pending["issuer"]:
            raise IntegrationError("issuer_mismatch")  # RFC 9207 mix-up defense
        client = self.store.load(f"client:{sid}")
        if not client or client.meta.get("client_id") != pending["client_id"] or not owned_by(client, server):
            raise IntegrationError("client_not_registered")
        check_destination(pending["token_endpoint"], server.url)
        try:
            with self._http(server) as http:
                response = _token_request(http, pending["token_endpoint"], client, {
                    "grant_type": "authorization_code", "code": code, "code_verifier": pending["verifier"],
                    "redirect_uri": pending["redirect_uri"], "resource": pending["resource"]})
        except httpx.HTTPError as exc:
            raise IntegrationError("server_unreachable") from exc
        token = _json(response)
        if not token or not token.get("access_token") or str(token.get("token_type", "bearer")).lower() != "bearer":
            raise IntegrationError("token_exchange_failed")
        record = ConnectionRecord(
            provider=sid, status="connected", auth_method="oauth_mcp",
            meta={"issuer": pending["issuer"], "token_endpoint": pending["token_endpoint"],
                  "client_id": pending["client_id"], "resource": pending["resource"],
                  "server_url": server.url, "generation": server.generation,
                  "scope": str(token.get("scope") or pending["scope"]),
                  "expires_at": _expires_at(token), "verified_at": time.time()},
            secret={"access_token": token["access_token"],
                    **({"refresh_token": token["refresh_token"]} if token.get("refresh_token") else {})})
        with self.fenced(server):
            self.require_current(server)
            self.store.commit_connection(sid, pending["claim"], record)
        return pending["return_to"]

    def access_token(self, server: McpServer) -> str:
        """A live access token, refreshed centrally near expiry (compare-and-set)."""
        sid = store_id(server.slug)
        snap = self.store.snapshot(sid)
        record = snap.record
        if (not record or record.status != "connected" or record.auth_method != "oauth_mcp"
                or not owned_by(record, server)):
            raise IntegrationError("not_connected")
        expires_at = record.meta.get("expires_at")
        if record.secret.get("access_token") and (expires_at is None or expires_at > time.time() + 60):
            return record.secret["access_token"]
        refresh = record.secret.get("refresh_token")
        client = self.store.load(f"client:{sid}")
        if (not refresh or not client or client.meta.get("client_id") != record.meta.get("client_id")
                or not owned_by(client, server)):
            raise IntegrationError("reauthorization_required")
        check_destination(record.meta["token_endpoint"], server.url)
        try:
            with self._http(server) as http:
                response = _token_request(http, record.meta["token_endpoint"], client, {
                    "grant_type": "refresh_token", "refresh_token": refresh,
                    "resource": record.meta["resource"]})
        except httpx.HTTPError as exc:
            raise IntegrationError("server_unreachable") from exc
        if response.status_code in (400, 401):
            raise IntegrationError("reauthorization_required")
        token = _json(response)
        if not token or not token.get("access_token"):
            raise IntegrationError("server_unreachable")
        record.secret = {**record.secret, "access_token": token["access_token"],
                         **({"refresh_token": token["refresh_token"]} if token.get("refresh_token") else {})}
        record.meta = {**record.meta, "expires_at": _expires_at(token)}
        if not self.store.replace(sid, snap, record):
            # A concurrent refresher won; with rotating refresh tokens only its result is valid.
            winner = self.store.load(sid)
            if (winner and winner.status == "connected" and winner.auth_method == "oauth_mcp"
                    and owned_by(winner, server) and winner.secret.get("access_token")):
                return winner.secret["access_token"]
            raise IntegrationError("reauthorization_required")
        return token["access_token"]

    def disconnect(self, slug: str) -> bool:
        sid = store_id(slug)
        removed = False
        for key in (f"pending:{sid}", f"client:{sid}", f"unverified:{sid}", sid):
            removed = self.store.delete(key) or removed
        return removed

    def status(self, server: McpServer) -> dict:
        record = self.store.load(store_id(server.slug))
        if server.auth_kind == "none":
            return {"connected": True}
        if not record or record.status != "connected":
            return {"connected": False}
        if server.auth_kind == "bearer":
            return {"connected": record.auth_method == "bearer" and owned_by(record, server)}
        return {"connected": record.auth_method == "oauth_mcp" and owned_by(record, server),
                "scope": record.meta.get("scope", ""), "connected_at": record.connected_at,
                "refreshable": "refresh_token" in record.secret}


def _expires_at(token: dict) -> float | None:
    try:
        expires_in = int(token.get("expires_in"))
    except (TypeError, ValueError):
        return None
    return time.time() + max(expires_in, 0)
