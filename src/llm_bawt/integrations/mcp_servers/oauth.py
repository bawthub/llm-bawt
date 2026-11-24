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
import secrets
import time
from urllib.parse import urlencode, urlsplit

import httpx

from mcp.client.auth.utils import (
    build_oauth_authorization_server_metadata_discovery_urls,
    build_protected_resource_metadata_discovery_urls,
    extract_resource_metadata_from_www_auth,
    extract_scope_from_www_auth,
)

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


def _json(response: httpx.Response) -> dict | None:
    if response.status_code != 200:
        return None
    try:
        body = response.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


def _same_origin_or_https(url: str) -> bool:
    parts = urlsplit(url)
    return parts.scheme in ("https", "http") and bool(parts.hostname) and not parts.username


def discover(http: httpx.Client, server_url: str) -> dict:
    """Authorization metadata for ``server_url``: endpoints, resource and scope."""
    try:
        probe = http.post(server_url, json=_INIT_BODY,
                          headers={"Accept": "application/json, text/event-stream"})
    except httpx.HTTPError as exc:
        raise IntegrationError("server_unreachable") from exc
    www_auth_url = extract_resource_metadata_from_www_auth(probe) if probe.status_code == 401 else None
    scope = extract_scope_from_www_auth(probe) if probe.status_code == 401 else None
    resource, auth_server = server_url, None
    for url in build_protected_resource_metadata_discovery_urls(www_auth_url, server_url):
        try:
            prm = _json(http.get(url))
        except httpx.HTTPError:
            prm = None
        if prm and isinstance(prm.get("authorization_servers"), list) and prm["authorization_servers"]:
            auth_server = str(prm["authorization_servers"][0])
            resource = str(prm.get("resource") or server_url)
            if not scope and isinstance(prm.get("scopes_supported"), list):
                scope = " ".join(str(s) for s in prm["scopes_supported"])
            break
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
    # The spec requires refusing an authorization server that does not advertise S256.
    if "S256" not in (metadata.get("code_challenge_methods_supported") or []):
        raise IntegrationError("pkce_unsupported")
    for key in ("authorization_endpoint", "token_endpoint", "registration_endpoint"):
        if metadata.get(key) and not _same_origin_or_https(str(metadata[key])):
            raise IntegrationError("oauth_metadata_invalid")
    return {
        "issuer": str(metadata.get("issuer") or auth_server or ""),
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
    def __init__(self, config, *, store: IntegrationStore | None = None, client_factory=httpx.Client):
        self.store = store if store is not None else IntegrationStore(config)
        self.client_factory = client_factory

    def _http(self):
        return self.client_factory(timeout=20, follow_redirects=False)

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
        with self._http() as http:
            meta = discover(http, server.url)
            client = self.store.load(f"client:{sid}")
            # Reuse a registration only for the same authorization server and callback.
            if not (client and client.meta.get("issuer") == meta["issuer"]
                    and client.meta.get("redirect_uri") == redirect_uri):
                body = _register(http, meta, redirect_uri)
                client = ConnectionRecord(
                    provider=f"client:{sid}", status="configured",
                    meta={"client_id": str(body["client_id"]), "issuer": meta["issuer"],
                          "redirect_uri": redirect_uri,
                          "token_endpoint_auth_method": str(body.get("token_endpoint_auth_method") or "none")},
                    secret={"client_secret": str(body["client_secret"])} if body.get("client_secret") else {})
                self.store.save(client)
        state, browser, verifier = (secrets.token_urlsafe(32) for _ in range(3))
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
        self.store.save(ConnectionRecord(provider=f"pending:{sid}", status="pending", secret={
            "state_hash": digest(state), "browser_hash": digest(browser), "verifier": verifier,
            "expires_at": time.time() + PENDING_SECONDS, "redirect_uri": redirect_uri,
            "return_to": return_to, "client_id": client.meta["client_id"], "server_url": server.url,
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
        if pending["server_url"] != server.url:
            raise IntegrationError("invalid_state")
        if (iss or pending["iss_supported"]) and iss != pending["issuer"]:
            raise IntegrationError("issuer_mismatch")  # RFC 9207 mix-up defense
        client = self.store.load(f"client:{sid}")
        if not client or client.meta.get("client_id") != pending["client_id"]:
            raise IntegrationError("client_not_registered")
        try:
            with self._http() as http:
                response = _token_request(http, pending["token_endpoint"], client, {
                    "grant_type": "authorization_code", "code": code, "code_verifier": pending["verifier"],
                    "redirect_uri": pending["redirect_uri"], "resource": pending["resource"]})
        except httpx.HTTPError as exc:
            raise IntegrationError("server_unreachable") from exc
        token = _json(response)
        if not token or not token.get("access_token") or str(token.get("token_type", "bearer")).lower() != "bearer":
            raise IntegrationError("token_exchange_failed")
        self.store.commit_connection(sid, pending["claim"], ConnectionRecord(
            provider=sid, status="connected", auth_method="oauth_mcp",
            meta={"issuer": pending["issuer"], "token_endpoint": pending["token_endpoint"],
                  "client_id": pending["client_id"], "resource": pending["resource"],
                  "server_url": server.url, "scope": str(token.get("scope") or pending["scope"]),
                  "expires_at": _expires_at(token), "verified_at": time.time()},
            secret={"access_token": token["access_token"],
                    **({"refresh_token": token["refresh_token"]} if token.get("refresh_token") else {})}))
        return pending["return_to"]

    def access_token(self, server: McpServer) -> str:
        """A live access token, refreshed centrally near expiry (compare-and-set)."""
        sid = store_id(server.slug)
        snap = self.store.snapshot(sid)
        record = snap.record
        if (not record or record.status != "connected" or record.auth_method != "oauth_mcp"
                or record.meta.get("server_url") != server.url):
            raise IntegrationError("not_connected")
        expires_at = record.meta.get("expires_at")
        if record.secret.get("access_token") and (expires_at is None or expires_at > time.time() + 60):
            return record.secret["access_token"]
        refresh = record.secret.get("refresh_token")
        client = self.store.load(f"client:{sid}")
        if not refresh or not client or client.meta.get("client_id") != record.meta.get("client_id"):
            raise IntegrationError("reauthorization_required")
        try:
            with self._http() as http:
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
            if winner and winner.status == "connected" and winner.secret.get("access_token"):
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
            return {"connected": record.auth_method == "bearer"}
        return {"connected": record.auth_method == "oauth_mcp" and record.meta.get("server_url") == server.url,
                "scope": record.meta.get("scope", ""), "connected_at": record.connected_at,
                "refreshable": "refresh_token" in record.secret}


def _expires_at(token: dict) -> float | None:
    try:
        expires_in = int(token.get("expires_in"))
    except (TypeError, ValueError):
        return None
    return time.time() + max(expires_in, 0)
