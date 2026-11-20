"""Calling a paired install's peer endpoint (TASK-2517).

Every HTTP request carries a fresh signed assertion (``identity.assertion``),
because the receiver accepts each ``jti`` once and a stateless MCP exchange is
several requests. Errors come back as ``PeerUnavailable`` with an honest
reason (offline, refused, revoked); nothing here retries on its own.
"""
from __future__ import annotations

import json
from collections.abc import Generator
from contextlib import asynccontextmanager

import httpx2

from . import identity, pairing, store, tls

TIMEOUT = 10.0


class PeerUnavailable(Exception):
    def __init__(self, reason: str, status: int | None = None):
        super().__init__(reason)
        self.reason = reason
        self.status = status


class AssertionAuth(httpx2.Auth):
    def __init__(self, own: identity.Identity, audience: str):
        self.own = own
        self.audience = audience
        self.refused: int | None = None

    def auth_flow(self, request) -> Generator:
        request.headers["Authorization"] = f"Bearer {self.own.assertion(self.audience)}"
        response = yield request
        if response.status_code in (401, 403):
            self.refused = response.status_code


def verify_for(base_url: str, tls_cert: str | None):
    """The TLS policy for one peer: HTTPS trusts only its pinned certificate."""
    if not base_url.startswith("https://"):
        return True  # plain HTTP, only reachable when explicitly allowed
    if not tls_cert:
        raise PeerUnavailable("no pinned certificate for this peer; pair it again")
    return tls.client_context(tls_cert)


async def fetch_certificate(base_url: str) -> str:
    """The certificate a peer URL presents; the caller must check it before use."""
    try:
        return await tls.fetch_certificate(base_url)
    except (OSError, ValueError) as exc:
        raise PeerUnavailable(f"cannot reach {base_url}: {type(exc).__name__}") from exc


async def fetch_identity(base_url: str, challenge: str, tls_cert: str | None = None) -> dict:
    try:
        async with httpx2.AsyncClient(timeout=TIMEOUT, verify=verify_for(base_url, tls_cert)) as http:
            resp = await http.get(f"{base_url}/peer/identity", params={"challenge": challenge})
    except httpx2.HTTPError as exc:
        raise PeerUnavailable(f"cannot reach {base_url}: {type(exc).__name__}") from exc
    if resp.status_code != 200:
        raise PeerUnavailable(f"peer identity returned HTTP {resp.status_code}", resp.status_code)
    return resp.json()


async def send_pair(base_url: str, body: dict, tls_cert: str | None = None) -> dict:
    try:
        async with httpx2.AsyncClient(timeout=TIMEOUT, verify=verify_for(base_url, tls_cert)) as http:
            resp = await http.post(f"{base_url}/peer/pair", json=body)
    except httpx2.HTTPError as exc:
        raise PeerUnavailable(f"cannot reach {base_url}: {type(exc).__name__}") from exc
    if resp.status_code != 200:
        reason = _error(resp)
        raise PeerUnavailable(f"peer refused pairing: {reason}", resp.status_code)
    return resp.json()


def _error(resp) -> str:
    try:
        return str(resp.json().get("error") or resp.status_code)
    except (ValueError, AttributeError):
        return str(resp.status_code)


@asynccontextmanager
async def session(own: identity.Identity, link: store.PeerLink, auth: AssertionAuth):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    verify = verify_for(link.base_url, link.tls_cert)
    async with httpx2.AsyncClient(auth=auth, timeout=httpx2.Timeout(TIMEOUT),
                                  verify=verify) as http:
        async with streamable_http_client(f"{link.base_url}/peer/mcp", http_client=http) as (r, w):
            async with ClientSession(r, w) as sess:
                await sess.initialize()
                yield sess


async def call(own: identity.Identity, link: store.PeerLink, tool: str,
               arguments: dict | None = None) -> dict:
    if link.state != "active":
        raise PeerUnavailable(f"peer is {link.state}")
    try:  # the current transport policy, not whatever was allowed at pairing
        pairing.normalize_base_url(link.base_url)
    except pairing.PairingError as exc:
        raise PeerUnavailable(f"peer URL not allowed now: {exc.reason}") from exc
    auth = AssertionAuth(own, link.install_id)
    try:
        async with session(own, link, auth) as sess:
            result = await sess.call_tool(tool, arguments or {})
    except PeerUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - transport/HTTP failures surface as offline
        status = auth.refused or _status_of(exc)
        reason = {401: "peer refused our credentials", 403: "peer refused access (revoked or not confirmed)"}.get(
            status or 0, f"peer unreachable: {type(_leaf(exc)).__name__}")
        raise PeerUnavailable(reason, status) from exc
    text = "".join(getattr(c, "text", "") for c in result.content)
    if result.is_error:
        raise PeerUnavailable(f"peer tool error: {text[:200]}")
    if result.structured_content:
        return dict(result.structured_content)
    return json.loads(text) if text else {}


def _leaf(exc: BaseException) -> BaseException:
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc


def _status_of(exc: BaseException) -> int | None:
    """HTTP status from an error, looking inside the transport's exception groups."""
    if isinstance(exc, BaseExceptionGroup):
        for inner in exc.exceptions:
            status = _status_of(inner)
            if status:
                return status
        return None
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None)
