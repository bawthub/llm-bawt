"""ASGI middleware that authenticates every llm-bawt HTTP request (TASK-2380).

It resolves a principal and stores it in ``scope["state"]["api_principal"]``;
per-route checks (service bearer, bridge gate, turn capability) stay in place.

Credential precedence: turn capability (``bot:<id>``, identity only), then
``X-LLM-Bawt-Key``, then the BawtHub service bearer, then ``X-Bridge-Token``.
A malformed, invalid or repeated ``X-LLM-Bawt-Key`` is refused in every mode
but ``off``, as is a route-limited key used outside its routes. Missing credentials are counted in ``log`` and refused (401, or
WebSocket close 4401) in ``enforce``; while the mode is unresolved protected
routes answer 503 ``auth_mode_unresolved``.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from . import keys
from .mode import UNRESOLVED, ModeResolver

logger = logging.getLogger(__name__)

TURN_HEADER = b"x-llm-bawt-task-turn-context"
PRE_AUTH_KEY = "mcp_oauth_principal"
ALLOWED_PATHS = frozenset({
    "/health",
    "/integrations/google-home/authorize",
    "/integrations/google-home/token",
    "/integrations/google-home/fulfillment",
    "/webhook/nextcloud",
})
# Key principals that may only call these exact (method, raw path) pairs. The
# list lives in code so a garbled setting can never widen it. A single-purpose
# caller outside the hub (a desktop's "local model is back" notifier) gets one
# of these instead of a general key.
ROUTE_LIMITED_PRINCIPALS: dict[str, frozenset[tuple[str, str]]] = {
    "primary-available-notifier": frozenset({("POST", "/v1/maintenance/primary-available")}),
}
OBSERVATION_CAP = 2000
OBSERVATION_MAX_AGE = 48 * 3600
_LOG_INTERVAL = 60.0


@dataclass(frozen=True)
class Principal:
    name: str
    kind: str  # "key", "service", "bridge", "bot", "user" (MCP OAuth)


Checker = Callable[[dict[bytes, list[bytes]]], "Principal | None"]


def _headers(scope) -> dict[bytes, list[bytes]]:
    out: dict[bytes, list[bytes]] = {}
    for name, value in scope.get("headers") or ():
        out.setdefault(name.lower(), []).append(value)
    return out


class Observations:
    """Bounded counts of who called what, for the log phase."""

    def __init__(self, cap: int = OBSERVATION_CAP, max_age: float = OBSERVATION_MAX_AGE,
                 clock: Callable[[], float] = time.time):
        self._cap, self._max_age, self._clock = cap, max_age, clock
        self._rows: OrderedDict[tuple, list] = OrderedDict()
        self._lock = threading.Lock()

    def record(self, key: tuple) -> None:
        now = self._clock()
        with self._lock:
            self._evict(now)
            if key not in self._rows and len(self._rows) >= self._cap:
                key = ("<overflow>",)
            row = self._rows.setdefault(key, [0, now, now])
            row[0] += 1
            row[2] = now
            self._rows.move_to_end(key)

    def _evict(self, now: float) -> None:
        while self._rows:
            key, row = next(iter(self._rows.items()))
            if now - row[2] <= self._max_age:
                break
            del self._rows[key]

    def snapshot(self) -> list[dict]:
        with self._lock:
            self._evict(self._clock())
            return [{"key": list(k), "count": r[0], "first": r[1], "last": r[2]}
                    for k, r in self._rows.items()]


class ApiAuthMiddleware:
    def __init__(self, app, *, resolver: ModeResolver, checkers: Iterable[Checker] = (),
                 root: Callable[[], bytes] | None = None,
                 observations: Observations | None = None,
                 allowed_paths: frozenset[str] = ALLOWED_PATHS,
                 route_template: Callable[[dict], str] | None = None,
                 key_principal_allowed: Callable[[str], bool] | None = None,
                 refuse_turn_with_key: bool = False):
        self.app = app
        self.resolver = resolver
        self.checkers = tuple(checkers)
        self._root = root
        self.observations = observations or Observations()
        self.allowed_paths = allowed_paths
        self._template = route_template or (lambda scope: "<unmatched>")
        self._last_log = 0.0
        # Per-instance policy (TASK-2381 :8001): which key principals count,
        # and whether a turn capability alongside a key is ambiguous.
        self._key_allowed = key_principal_allowed
        self._refuse_turn_with_key = refuse_turn_with_key

    def _verify_key(self, value: str):
        kwargs = {"denylist": self.resolver.denylist()}
        if self._root is not None:
            kwargs["root"] = self._root
        return keys.verify(value, **kwargs)

    def authenticate(self, scope) -> tuple[Principal | None, str | None]:
        """(principal, refusal reason). A refusal is a bad key, never a missing one."""
        # An outer layer that already verified the caller (TASK-2426 MCP OAuth)
        # leaves its principal in scope state, which a client cannot set.
        pre = (scope.get("state") or {}).get(PRE_AUTH_KEY)
        if isinstance(pre, Principal):
            return pre, None
        headers = _headers(scope)
        presented = headers.get(keys.HEADER.encode(), [])
        key_principal = None
        if len(presented) > 1:
            return None, "repeated_key"
        if presented:
            result = self._verify_key(presented[0].decode("latin-1"))
            if not result.ok:
                return None, f"key_{result.reason}"
            if self._key_allowed is not None and not self._key_allowed(result.principal):
                return None, "key_not_permitted"
            limited = ROUTE_LIMITED_PRINCIPALS.get(result.principal)
            if limited is not None and (scope.get("type") != "http" or scope.get("query_string")
                                        or (scope.get("method"), scope.get("path")) not in limited):
                return None, "key_route_not_permitted"
            key_principal = Principal(result.principal, "key")
        for check in self.checkers:  # turn capability first, then key, then legacy
            found = check(headers)
            if found is not None and found.kind == "bot":
                if key_principal and self._refuse_turn_with_key:
                    return None, "ambiguous_credentials"
                return found, None
        if key_principal:
            return key_principal, None
        for check in self.checkers:
            found = check(headers)
            if found is not None:
                return found, None
        return None, None

    async def __call__(self, scope, receive, send):
        kind = scope.get("type")
        if kind not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        path = scope.get("path") or ""
        if kind == "http" and (path in self.allowed_paths or scope.get("method") == "OPTIONS"):
            return await self.app(scope, receive, send)
        mode = self.resolver.mode()
        if mode == "off":
            return await self.app(scope, receive, send)
        if mode == UNRESOLVED:
            return await self._refuse(scope, send, 503, "auth_mode_unresolved")
        principal, refusal = self.authenticate(scope)
        if refusal:
            return await self._refuse(scope, send, 401, refusal)
        self._observe(scope, principal)
        if principal is None and mode == "enforce":
            return await self._refuse(scope, send, 401, "missing_credentials")
        scope.setdefault("state", {})["api_principal"] = principal
        return await self.app(scope, receive, send)

    def _observe(self, scope, principal: Principal | None) -> None:
        if principal is not None:
            who = (principal.kind, principal.name)
        else:
            client = scope.get("client") or ("?", 0)
            agent = b""
            for name, value in scope.get("headers") or ():
                if name.lower() == b"user-agent":
                    agent = value
            who = ("anon", client[0], agent.decode("latin-1")[:120])
        self.observations.record(who + (scope.get("method", "WS"), self._template(scope)))
        now = time.monotonic()
        if principal is None and now - self._last_log > _LOG_INTERVAL:
            self._last_log = now
            logger.warning("api auth: unauthenticated %s %s from %s",
                           scope.get("method", "WS"), self._template(scope), who[1])

    async def _refuse(self, scope, send, status: int, reason: str) -> None:
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 4401 if status == 401 else 1013})
            return
        body = json.dumps({"detail": reason}).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})
