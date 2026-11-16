"""Host allowlist and cross-site write guard for the HTTP API (TASK-2371).

Two browser attacks reach this API without it:

* DNS rebinding: a hostile domain resolves to this server, so the victim's
  browser treats the API as same-origin and can read and write anything. The
  request then carries the hostile name in ``Host``; an allowlist refuses it.
* Cross-site writes: any page can make a browser POST here (a ``text/plain``
  form needs no preflight, and body-less POSTs need no body at all). Browsers
  mark every such request with ``Origin`` and ``Sec-Fetch-Site``; server
  callers (BawtHub's proxy, bridges, houses, the CLI, curl) send neither, so
  refusing a foreign ``Origin`` closes the hole without breaking them. That is
  why this does not require ``Content-Type: application/json``: tools post
  form-encoded bodies with ``curl -d`` and must keep working.

Webhooks that authenticate their own callers are exempt from the Origin rule,
not from the Host rule.

Configuration (environment, read once at startup):

* ``LLM_BAWT_HOST_GUARD``: ``off``, ``report`` (log what enforcement would
  refuse, refuse nothing) or ``enforce``. Default ``report``.
* ``LLM_BAWT_ALLOWED_HOSTS``: extra comma-separated host names, without port.
  ``*.example.com`` matches subdomains; ``*`` disables the Host rule.
* ``LLM_BAWT_ALLOWED_ORIGINS``: extra comma-separated origins whose browser
  writes are accepted. ``LLM_BAWT_PUBLIC_ORIGIN`` is always accepted, as is an
  origin on an allowed host.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from urllib.parse import urlsplit

log = logging.getLogger(__name__)

MODES = ("off", "report", "enforce")
DEFAULT_MODE = "report"
#: Names every installation's own processes use: loopback and the compose service.
BUILTIN_HOSTS = ("localhost", "127.0.0.1", "::1", "app", "host.docker.internal")
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
#: Self-authenticating webhooks; browsers may legitimately post to these.
ORIGIN_EXEMPT_PREFIXES = ("/webhook/", "/integrations/google-home/")
#: Bound on distinct report-mode log lines, so a scanner cannot flood the log.
_REPORT_LIMIT = 200


def _split(value: str | None) -> list[str]:
    return [part.strip().lower() for part in (value or "").split(",") if part.strip()]


def _hostname(authority: str) -> str:
    """Host name of a ``Host`` header value: no port, no IPv6 brackets."""
    authority = authority.strip().lower()
    if authority.startswith("["):
        return authority[1:].split("]", 1)[0]
    if authority.count(":") == 1:
        return authority.rsplit(":", 1)[0]
    return authority  # bare name, or an unbracketed IPv6 literal


@dataclass(frozen=True)
class GuardPolicy:
    mode: str
    hosts: frozenset[str]
    wildcard_suffixes: tuple[str, ...]
    any_host: bool
    origins: frozenset[str]

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "GuardPolicy":
        env = os.environ if env is None else env
        mode = (env.get("LLM_BAWT_HOST_GUARD") or DEFAULT_MODE).strip().lower()
        if mode not in MODES:
            log.warning("LLM_BAWT_HOST_GUARD=%r is not one of %s; using %s", mode, MODES, DEFAULT_MODE)
            mode = DEFAULT_MODE
        names = [*BUILTIN_HOSTS, *_split(env.get("LLM_BAWT_ALLOWED_HOSTS"))]
        origins = {o.rstrip("/") for o in _split(env.get("LLM_BAWT_ALLOWED_ORIGINS"))}
        public = (env.get("LLM_BAWT_PUBLIC_ORIGIN") or "").strip().lower().rstrip("/")
        if public:
            origins.add(public)
        return cls(
            mode=mode,
            hosts=frozenset(n for n in names if not n.startswith("*")),
            wildcard_suffixes=tuple(n[1:] for n in names if n.startswith("*.")),
            any_host="*" in names,
            origins=frozenset(origins),
        )

    def host_allowed(self, name: str) -> bool:
        if self.any_host or name in self.hosts:
            return True
        return any(name.endswith(suffix) for suffix in self.wildcard_suffixes)

    def origin_allowed(self, origin: str) -> bool:
        origin = origin.strip().lower().rstrip("/")
        if origin in self.origins:
            return True
        parts = urlsplit(origin)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            return False  # includes the opaque "null" origin
        return not self.any_host and self.host_allowed(parts.hostname)

    def refusal(self, method: str, path: str, headers: dict[str, str]) -> str | None:
        """Why enforcement would refuse this request, or ``None``."""
        host = _hostname(headers.get("host", ""))
        if not self.host_allowed(host):
            return "host_not_allowed"
        if method in SAFE_METHODS or path.startswith(ORIGIN_EXEMPT_PREFIXES):
            return None
        origin = headers.get("origin")
        if origin is not None:
            return None if self.origin_allowed(origin) else "cross_site_origin"
        if headers.get("sec-fetch-site", "").lower() in ("cross-site", "same-site"):
            return "cross_site_origin"
        return None


class RequestGuardMiddleware:
    """Pure ASGI, so streaming responses pass through untouched."""

    def __init__(self, app, policy: GuardPolicy | None = None):
        self.app = app
        self.policy = policy or GuardPolicy.from_env()
        self._reported: set[tuple[str, str, str]] = set()
        log.info("HTTP request guard: mode=%s", self.policy.mode)

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket") or self.policy.mode == "off":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        # A WebSocket handshake is a GET, but a browser opens one cross-site
        # freely, so its Origin is checked like a write.
        method = "WEBSOCKET" if scope["type"] == "websocket" else scope.get("method", "GET")
        path = scope.get("path", "")
        reason = self.policy.refusal(method, path, headers)
        if reason is None:
            await self.app(scope, receive, send)
            return
        self._report(reason, method, path, headers)
        if self.policy.mode == "report":
            await self.app(scope, receive, send)
            return
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        status = 421 if reason == "host_not_allowed" else 403
        body = json.dumps({"detail": reason}).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})

    def _report(self, reason: str, method: str, path: str, headers: dict[str, str]) -> None:
        culprit = headers.get("host", "") if reason == "host_not_allowed" else (
            headers.get("origin") or f"sec-fetch-site={headers.get('sec-fetch-site', '')}")
        key = (reason, culprit, path.split("/")[1] if "/" in path else path)  # first segment
        if key in self._reported or len(self._reported) >= _REPORT_LIMIT:
            return
        self._reported.add(key)
        verb = "refused" if self.policy.mode == "enforce" else "would refuse"
        log.warning("request guard %s %s %s: %s %r", verb, method, path, reason, culprit[:200])
