"""Send this process's API key on every request to the llm-bawt app (TASK-2380).

Python callers reach the app from dozens of call sites through httpx and
urllib. Rather than threading a header through each one, a process calls
:func:`install` once at startup. It adds ``X-LLM-Bawt-Key`` (from
``LLM_BAWT_API_KEY_FILE``, read per request) to outgoing requests whose
``host:port`` is an app address, and to nothing else, so the key never leaves
for a third-party host. A header the caller already set is left alone.

App addresses: the built-ins below plus ``LLM_BAWT_API_KEY_HOSTS``
(comma-separated ``host:port``). Installing is idempotent; with no key file the
hook adds nothing.
"""
from __future__ import annotations

import os
import urllib.request
from urllib.parse import urlsplit

from . import client

HOSTS_ENV = "LLM_BAWT_API_KEY_HOSTS"
BUILTIN_HOSTS = frozenset({
    "app:8642", "localhost:8642", "127.0.0.1:8642", "[::1]:8642", "host.docker.internal:8642",
})

_installed = False


def app_hosts() -> frozenset[str]:
    extra = {h.strip().lower() for h in (os.environ.get(HOSTS_ENV) or "").split(",") if h.strip()}
    return BUILTIN_HOSTS | extra


def _hostport(scheme: str, host: str | None, port: int | None) -> str:
    host = (host or "").lower()
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if port is None:
        port = 443 if scheme == "https" else 80
    return f"{host}:{port}"


def targets_app(url: str) -> bool:
    parts = urlsplit(url)
    try:
        port = parts.port
    except ValueError:
        return False
    return _hostport(parts.scheme, parts.hostname, port) in app_hosts()


def _key_for(url: str, existing) -> str | None:
    if not targets_app(url):
        return None
    if any(k.lower() == client.HEADER_NAME.lower() for k in existing):
        return None
    return client.header_value()


class _UrllibHandler(urllib.request.BaseHandler):
    handler_order = 100

    def http_request(self, req):
        value = _key_for(req.full_url, list(req.headers) + list(req.unredirected_hdrs))
        if value:
            req.add_unredirected_header(client.HEADER_NAME, value)
        return req

    https_request = http_request


def _patch_httpx() -> None:
    try:
        import httpx
    except ImportError:
        return

    def _add(request) -> None:
        value = _key_for(str(request.url), request.headers.keys())
        if value:
            request.headers[client.HEADER_NAME] = value

    sync_send = httpx.Client.send
    async_send = httpx.AsyncClient.send

    def send(self, request, *args, **kwargs):
        _add(request)
        return sync_send(self, request, *args, **kwargs)

    async def asend(self, request, *args, **kwargs):
        _add(request)
        return await async_send(self, request, *args, **kwargs)

    httpx.Client.send = send
    httpx.AsyncClient.send = asend

    # httpx follows redirects inside send() and copies every header but
    # Authorization to the next hop, so a hop that leaves the app would carry
    # the key. Strip it from any redirect whose destination is not the app.
    base = httpx._client.BaseClient
    redirect_headers = base._redirect_headers

    def _redirect_headers(self, request, url, method):
        headers = redirect_headers(self, request, url, method)
        if not targets_app(str(url)):
            headers.pop(client.HEADER_NAME, None)
        return headers

    base._redirect_headers = _redirect_headers


def install() -> None:
    """Hook httpx and the global urllib opener once per process."""
    global _installed
    if _installed:
        return
    _installed = True
    urllib.request.install_opener(urllib.request.build_opener(_UrllibHandler()))
    _patch_httpx()
