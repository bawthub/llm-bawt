"""OAuth bearer sign-in for human MCP clients on :8001 (TASK-2426).

BawtHub is the authorization server; this module makes the MCP server an
OAuth protected resource in front of the TASK-2381 gate:

* ``GET /.well-known/oauth-protected-resource[/<path>]`` (RFC 9728) names the
  resource and BawtHub as its authorization server. Public.
* ``Authorization: Bearer <token>`` is checked at BawtHub's internal
  ``/internal/auth/mcp-introspect`` with the service token; active answers are
  cached briefly (keyed by a hash of the token, never past ``exp``), failures
  are not cached.
* Only entities in the principal set are admitted (403 otherwise).
* Every 401 from :8001 carries ``WWW-Authenticate: Bearer resource_metadata=...``
  so clients such as Claude Code discover sign-in on their own.

Dormant unless ``LLM_BAWT_MCP_OAUTH_RESOURCE`` (the public MCP URL) and
``LLM_BAWT_MCP_OAUTH_ISSUER`` are both set. The issuer must be the exact
``issuer`` BawtHub's authorization-server metadata declares (Better Auth's
base URL, e.g. ``https://hub.example.com/api/auth``), not the bare origin:
strict clients compare it byte for byte (RFC 8414 section 3.3). The admitted
principal is ``Principal(<entityId>, "user")``, and the verified user and
client are exposed to tool execution through ``current_oauth_caller()`` for
audit; a bearer alongside a turn capability or key is refused as ambiguous.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from urllib.parse import urlsplit

from .middleware import PRE_AUTH_KEY, TURN_HEADER, Principal

logger = logging.getLogger(__name__)

RESOURCE_ENV = "LLM_BAWT_MCP_OAUTH_RESOURCE"
ISSUER_ENV = "LLM_BAWT_MCP_OAUTH_ISSUER"
PRINCIPALS_ENV = "LLM_BAWT_MCP_OAUTH_PRINCIPALS"
INTROSPECT_URL_ENV = "LLM_BAWT_MCP_OAUTH_INTROSPECT_URL"
DEFAULT_INTERNAL_URL_ENV = "BAWTHUB_TASK_ASSOCIATION_INTERNAL_URL"
INTROSPECT_PATH = "/internal/auth/mcp-introspect"
METADATA_PREFIX = "/.well-known/oauth-protected-resource"
CACHE_TTL = 60.0
CACHE_CAP = 512
SCOPES = ("mcp",)


@dataclass(frozen=True)
class OAuthConfig:
    resource: str
    issuer: str
    principals: frozenset[str]
    introspect_url: str

    @property
    def metadata_url(self) -> str:
        parts = urlsplit(self.resource)
        return f"{parts.scheme}://{parts.netloc}{METADATA_PREFIX}{parts.path.rstrip('/')}"

    def metadata(self) -> dict:
        return {"resource": self.resource, "authorization_servers": [self.issuer],
                "scopes_supported": list(SCOPES), "bearer_methods_supported": ["header"],
                "resource_name": "BawtHub MCP"}


def _url(raw: str) -> str | None:
    raw = raw.strip().rstrip("/")
    parts = urlsplit(raw)
    if parts.scheme not in ("http", "https") or not parts.netloc or parts.query or parts.fragment:
        return None
    return raw


def load_config(environ=None) -> OAuthConfig | None:
    env = os.environ if environ is None else environ
    resource, issuer = _url(env.get(RESOURCE_ENV, "")), _url(env.get(ISSUER_ENV, ""))
    if not (env.get(RESOURCE_ENV) or env.get(ISSUER_ENV)):
        return None
    if not resource or not issuer:
        logger.error("MCP OAuth: %s and %s must both be plain http(s) URLs; OAuth stays off",
                     RESOURCE_ENV, ISSUER_ENV)
        return None
    base = (env.get(INTROSPECT_URL_ENV) or env.get(DEFAULT_INTERNAL_URL_ENV)
            or "http://frontend-prod:3002").strip().rstrip("/")
    url = base if base.endswith(INTROSPECT_PATH) else base + INTROSPECT_PATH
    principals = frozenset(p.strip() for p in env.get(PRINCIPALS_ENV, "").split(",") if p.strip())
    if not urlsplit(issuer).path:
        logger.warning("MCP OAuth: %s has no path; Better Auth's issuer is normally "
                       "<origin>/api/auth and strict clients reject a mismatch", ISSUER_ENV)
    if not principals:
        logger.warning("MCP OAuth on but %s is empty; every signed-in user gets 403",
                       PRINCIPALS_ENV)
    return OAuthConfig(resource, issuer, principals, url)


@dataclass(frozen=True)
class OAuthCaller:
    """A BawtHub user signed in through an OAuth client, verified by introspection."""

    entity_id: str
    client_id: str


_oauth_caller: ContextVar[OAuthCaller | None] = ContextVar("mcp_oauth_caller", default=None)


def current_oauth_caller() -> OAuthCaller | None:
    """The verified OAuth caller of the current :8001 request, if any."""
    return _oauth_caller.get()


class IntrospectionUnavailable(Exception):
    pass


Introspect = Callable[[str], Awaitable[dict]]


def http_introspect(config: OAuthConfig) -> Introspect:
    async def call(token: str) -> dict:
        import httpx

        from ..bawthub_auth.service_token import service_auth

        try:
            async with httpx.AsyncClient(timeout=5.0, auth=service_auth()) as client:
                response = await client.post(config.introspect_url, json={"token": token})
        except httpx.HTTPError as error:
            raise IntrospectionUnavailable(type(error).__name__) from None
        if response.status_code != 200:
            raise IntrospectionUnavailable(f"http_{response.status_code}")
        try:
            body = response.json()
        except ValueError:
            raise IntrospectionUnavailable("bad_json") from None
        if not isinstance(body, dict):
            raise IntrospectionUnavailable("bad_json")
        return body

    return call


class TokenCache:
    """Short cache of introspection answers, keyed by the token's hash."""

    def __init__(self, ttl: float = CACHE_TTL, cap: int = CACHE_CAP,
                 clock: Callable[[], float] = time.time):
        self._ttl, self._cap, self._clock = ttl, cap, clock
        self._rows: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def key(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    def get(self, token: str) -> dict | None:
        key, now = self.key(token), self._clock()
        with self._lock:
            row = self._rows.get(key)
            if row is None:
                return None
            if row[0] <= now:
                del self._rows[key]
                return None
            return row[1]

    def put(self, token: str, answer: dict) -> None:
        now = self._clock()
        until = now + self._ttl
        exp = answer.get("exp")
        if answer.get("active") and isinstance(exp, (int, float)) and exp > 0:
            until = min(until, float(exp))
        if until <= now:
            return
        with self._lock:
            self._rows[self.key(token)] = (until, answer)
            self._rows.move_to_end(self.key(token))
            while len(self._rows) > self._cap:
                self._rows.popitem(last=False)


def _header(scope, name: bytes) -> list[bytes]:
    return [v for k, v in scope.get("headers") or () if k.lower() == name]


class McpOAuthLayer:
    """Outer ASGI layer: metadata, bearer validation, principal set, challenges."""

    def __init__(self, gate, config: OAuthConfig, *, introspect: Introspect | None = None,
                 cache: TokenCache | None = None):
        self.gate = gate
        self.config = config
        self._introspect = introspect or http_introspect(config)
        self.cache = cache or TokenCache()
        self._challenge = f'Bearer resource_metadata="{config.metadata_url}", scope="{" ".join(SCOPES)}"'

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.gate(scope, receive, send)
        path = scope.get("path") or ""
        if path == METADATA_PREFIX or path.startswith(METADATA_PREFIX + "/"):
            if scope.get("method") in ("GET", "HEAD"):
                return await self._json(send, 200, self.config.metadata(),
                                        cache="public, max-age=300")
            return await self._json(send, 405, {"detail": "method_not_allowed"})
        authz = _header(scope, b"authorization")
        bearer = None
        if len(authz) == 1:
            value = authz[0].decode("latin-1")
            if value[:7].lower() == "bearer ":
                bearer = value[7:].strip()
        elif len(authz) > 1:
            return await self._json(send, 401, {"detail": "repeated_authorization"},
                                    challenge='error="invalid_request"')
        if bearer is not None:
            if _header(scope, TURN_HEADER) or _header(scope, b"x-llm-bawt-key"):
                return await self._json(send, 401, {"detail": "ambiguous_credentials"},
                                        challenge='error="invalid_request"')
            answer = self.cache.get(bearer)
            if answer is None:
                try:
                    answer = await self._introspect(bearer)
                except IntrospectionUnavailable as error:
                    logger.warning("MCP OAuth: introspection unavailable (%s)", error)
                    return await self._json(send, 503, {"detail": "introspection_unavailable"})
                self.cache.put(bearer, answer)
            if not answer.get("active") or answer.get("resource") != self.config.resource:
                return await self._json(send, 401, {"detail": "invalid_token"},
                                        challenge='error="invalid_token"')
            entity = answer.get("entityId")
            client = answer.get("clientId")
            if not (isinstance(client, str) and client):
                return await self._json(send, 401, {"detail": "invalid_token"},
                                        challenge='error="invalid_token"')
            if not isinstance(entity, str) or entity not in self.config.principals:
                return await self._json(send, 403, {"detail": "principal_not_allowed"})
            scope.setdefault("state", {})[PRE_AUTH_KEY] = Principal(entity, "user")
            token = _oauth_caller.set(OAuthCaller(entity, client))
            try:
                return await self.gate(scope, receive, self._with_challenge(send))
            finally:
                _oauth_caller.reset(token)
        return await self.gate(scope, receive, self._with_challenge(send))

    def _with_challenge(self, send):
        async def wrapped(message):
            if message.get("type") == "http.response.start" and message.get("status") == 401:
                headers = list(message.get("headers") or ())
                if not any(k.lower() == b"www-authenticate" for k, _ in headers):
                    headers.append((b"www-authenticate", self._challenge.encode()))
                message = {**message, "headers": headers}
            await send(message)
        return wrapped

    async def _json(self, send, status: int, payload: dict, *, challenge: str | None = None,
                    cache: str = "no-store") -> None:
        body = json.dumps(payload).encode()
        headers = [(b"content-type", b"application/json"), (b"cache-control", cache.encode()),
                   (b"content-length", str(len(body)).encode())]
        if status == 401:
            value = self._challenge + (f", {challenge}" if challenge else "")
            headers.append((b"www-authenticate", value.encode()))
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})
