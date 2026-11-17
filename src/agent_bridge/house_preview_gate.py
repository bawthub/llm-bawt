"""A sign-in gate in front of a house's preview port (TASK-2390).

Off unless ``BAWT_HOUSE_PREVIEW_GATE=on`` (a tenant install turns it on; an
install that reaches previews only through its own authenticating proxy keeps
the ungated preview). When on, the gate owns the published preview port for
the house's whole life, so no preview state is ever reachable without it:
Vite listens on loopback only (:data:`~agent_bridge.house_preview.INNER_PORT`)
and the fallback page is served by the gate itself.

Entry: the signed-in BawtHub frontend asks the gate for a one-use link
(``POST /__gate/mint`` with the BawtHub service token as a bearer). The link,
``/__gate/enter?n=<nonce>``, is good once, for :data:`NONCE_TTL_S`. Opening it
sets this house's cookie and redirects to ``/`` so the nonce leaves the
address bar. The cookie is an HMAC under a key drawn when the gate starts, over
the house id and expiry: another house's cookie, or one from before a restart,
doesn't open this gate. Everything else, WebSocket upgrades included, gets 401.

Proxying is plain HTTP/1.1: every request head on a connection is parsed and
checked (so a keep-alive connection can't carry an unchecked request), its
body is forwarded by length or chunk framing, the gate's own cookie is removed
and the client's address appended to ``X-Forwarded-For``. Responses stream back
byte for byte, which keeps SSE and HMR intact. An upgrade is checked like any
request, then piped both ways.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import html
import json
import logging
import os
import re
import secrets
import time
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs

from .house_preview import INNER_PORT, PREVIEW_PORT

logger = logging.getLogger(__name__)

NONCE_TTL_S = 60.0
COOKIE_TTL_S = 12 * 3600
MAX_NONCES = 256
HEAD_LIMIT = 64 * 1024
FIRST_HEAD_TIMEOUT_S = 10.0
IDLE_TIMEOUT_S = 120.0
CHUNK = 64 * 1024
TOKEN_FILE_ENV = "BAWTHUB_SERVICE_TOKEN_FILE"
TOKEN_ENV = "BAWTHUB_SERVICE_TOKEN"


def read_service_token() -> str:
    """The BawtHub service token (the file wins, re-read so a rotation applies)."""
    path = os.environ.get(TOKEN_FILE_ENV, "").strip()
    if path:
        try:
            return Path(path).read_text().strip()
        except OSError:
            return ""
    return os.environ.get(TOKEN_ENV, "").strip()


class BadRequest(Exception):
    pass


class Request:
    def __init__(self, method: str, target: str, version: str,
                 headers: list[tuple[str, str]]) -> None:
        self.method, self.target, self.version, self.headers = method, target, version, headers

    def get(self, name: str) -> str | None:
        name = name.lower()
        for key, value in self.headers:
            if key.lower() == name:
                return value
        return None

    def all(self, name: str) -> list[str]:
        name = name.lower()
        return [value for key, value in self.headers if key.lower() == name]

    @property
    def path(self) -> str:
        return self.target.split("?", 1)[0]

    @property
    def query(self) -> dict[str, list[str]]:
        return parse_qs(self.target.split("?", 1)[1]) if "?" in self.target else {}

    @property
    def is_upgrade(self) -> bool:
        conn = ",".join(self.all("connection")).lower()
        return self.get("upgrade") is not None and "upgrade" in [
            part.strip() for part in conn.split(",")]


_TOKEN_RE = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")


def parse_head(raw: bytes) -> Request:
    try:
        text = raw.decode("latin-1")
    except UnicodeDecodeError as exc:  # pragma: no cover - latin-1 decodes anything
        raise BadRequest("undecodable head") from exc
    lines = text[:-4].split("\r\n") if text.endswith("\r\n\r\n") else text.split("\r\n")
    parts = lines[0].split(" ")
    if len(parts) != 3 or not _TOKEN_RE.match(parts[0]) or not parts[2].startswith("HTTP/1."):
        raise BadRequest("bad request line")
    if not parts[1].startswith("/"):
        raise BadRequest("origin-form targets only")
    headers: list[tuple[str, str]] = []
    for line in lines[1:]:
        name, sep, value = line.partition(":")
        if not sep or not _TOKEN_RE.match(name):  # also rejects obsolete line folding
            raise BadRequest("bad header line")
        headers.append((name, value.strip(" \t")))
    return Request(parts[0], parts[1], parts[2], headers)


def body_framing(req: Request) -> tuple[str, int]:
    """("none"|"length"|"chunked", length). Ambiguous framing is refused."""
    te = [v.strip().lower() for v in ",".join(req.all("transfer-encoding")).split(",") if v.strip()]
    cl = {v.strip() for v in req.all("content-length")}
    if te:
        if cl or te != ["chunked"]:
            raise BadRequest("unsupported or ambiguous body framing")
        return "chunked", 0
    if len(cl) > 1:
        raise BadRequest("conflicting Content-Length")
    if cl:
        value = cl.pop()
        if not value.isdigit():
            raise BadRequest("bad Content-Length")
        return ("length", int(value)) if int(value) else ("none", 0)
    return "none", 0


def cookie_name(house: str) -> str:
    return "bawthub_preview_" + re.sub(r"[^a-z0-9_]", "_", house.lower())


def split_cookie(values: list[str], name: str) -> tuple[list[str], str | None]:
    """(other cookie pairs, this gate's cookie value) from the Cookie headers."""
    others: list[str] = []
    found: str | None = None
    for header in values:
        for pair in header.split(";"):
            pair = pair.strip()
            if not pair:
                continue
            key, _, value = pair.partition("=")
            if key.strip() == name:
                found = value.strip() if found is None else found
            else:
                others.append(pair)
    return others, found


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


class PreviewGate:
    """Owns the preview port; serves the page, or proxies to loopback Vite."""

    def __init__(self, house: str, *, port: int = PREVIEW_PORT,
                 upstream_port: int = INNER_PORT, host: str = "0.0.0.0",
                 token: Callable[[], str] = read_service_token,
                 clock: Callable[[], float] = time.time) -> None:
        self.house = house
        self.port = port
        self.upstream_port = upstream_port
        self.host = host
        self.page = ""
        #: Where the 401 page points people (BawtHub's Campus), if known.
        self.campus_url: str | None = None
        self._token = token
        self._clock = clock
        self._key = secrets.token_bytes(32)
        self._nonces: dict[str, float] = {}
        self._cookie = cookie_name(house)
        self._server: asyncio.base_events.Server | None = None

    # -- lifecycle (same surface as FallbackServer) ------------------------
    @property
    def serving(self) -> bool:
        return self._server is not None

    async def start(self) -> None:
        if self._server is None:
            self._server = await asyncio.start_server(
                self._handle, self.host, self.port, reuse_address=True, limit=HEAD_LIMIT)

    async def stop(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.close()
            await server.wait_closed()

    # -- links and cookies -------------------------------------------------
    def mint(self) -> str:
        now = self._clock()
        self._nonces = {n: exp for n, exp in self._nonces.items() if exp > now}
        if len(self._nonces) >= MAX_NONCES:
            oldest = min(self._nonces, key=self._nonces.__getitem__)
            del self._nonces[oldest]
        nonce = secrets.token_urlsafe(32)
        self._nonces[nonce] = now + NONCE_TTL_S
        return nonce

    def consume(self, nonce: str) -> bool:
        expires = self._nonces.pop(nonce, None)  # one event loop: pop is atomic
        return expires is not None and expires > self._clock()

    def _mac(self, expires: int) -> str:
        msg = f"{self.house}|{expires}".encode()
        return _b64(hmac.new(self._key, msg, hashlib.sha256).digest())

    def make_cookie(self) -> str:
        expires = int(self._clock()) + COOKIE_TTL_S
        return f"{expires}.{self._mac(expires)}"

    def cookie_ok(self, value: str | None) -> bool:
        if not value:
            return False
        expires, _, mac = value.partition(".")
        if not expires.isdigit() or int(expires) <= self._clock():
            return False
        return hmac.compare_digest(mac, self._mac(int(expires)))

    def _bearer_ok(self, req: Request) -> bool:
        token = self._token()
        auth = req.get("authorization") or ""
        scheme, _, given = auth.partition(" ")
        return bool(token) and scheme.lower() == "bearer" and hmac.compare_digest(
            given.strip().encode(), token.encode())

    # -- connection handling -----------------------------------------------
    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = (writer.get_extra_info("peername") or ("",))[0] or ""
        up_writer: asyncio.StreamWriter | None = None
        pump: asyncio.Task | None = None
        first = True
        try:
            while True:
                try:
                    raw = await asyncio.wait_for(
                        reader.readuntil(b"\r\n\r\n"),
                        FIRST_HEAD_TIMEOUT_S if first else IDLE_TIMEOUT_S)
                except asyncio.LimitOverrunError:
                    if up_writer is None:
                        await self._respond(writer, 431, "Request headers too large")
                    return await self._finish(up_writer, pump)
                try:
                    req = parse_head(raw)
                    framing = body_framing(req)
                except BadRequest as exc:
                    if up_writer is None:
                        await self._respond(writer, 400, str(exc))
                    return await self._finish(up_writer, pump)
                first = False
                if req.path.startswith("/__gate/"):
                    if up_writer is None:
                        await self._gate_endpoint(req, writer)
                    return await self._finish(up_writer, pump)
                others, value = split_cookie(req.all("cookie"), self._cookie)
                if not self.cookie_ok(value):
                    # Mid-connection (e.g. the cookie just expired) the request
                    # is never forwarded; earlier answers still arrive, then the
                    # connection closes (a 401 can't be interleaved with them).
                    if up_writer is None:
                        await self._unauthorized(writer)
                    return await self._finish(up_writer, pump)
                if up_writer is None:
                    try:
                        up_reader, up_writer = await asyncio.open_connection(
                            "127.0.0.1", self.upstream_port)
                    except OSError:
                        await self._respond(writer, 200, self.page, html_body=True)
                        return
                    pump = asyncio.create_task(self._pipe(up_reader, writer))
                up_writer.write(self._forward_head(req, others, peer))
                if req.is_upgrade:
                    await self._pipe(reader, up_writer)
                    await pump
                    return
                await self._forward_body(reader, up_writer, framing)
                await up_writer.drain()
        except (asyncio.TimeoutError, asyncio.IncompleteReadError, ConnectionError,
                BadRequest, ValueError):
            pass
        except Exception:
            logger.warning("house %s preview gate: connection failed", self.house, exc_info=True)
        finally:
            if pump is not None and not pump.done():
                pump.cancel()
            for w in (up_writer, writer):
                if w is not None:
                    w.close()

    @staticmethod
    async def _finish(up_writer: asyncio.StreamWriter | None, pump: asyncio.Task | None) -> None:
        """Stop forwarding requests, but let answers already asked for arrive."""
        if up_writer is None or pump is None:
            return
        try:
            if up_writer.can_write_eof():
                up_writer.write_eof()
            await asyncio.wait_for(asyncio.shield(pump), IDLE_TIMEOUT_S)
        except (asyncio.TimeoutError, OSError, RuntimeError):
            pass

    async def _pipe(self, src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
        try:
            while data := await src.read(CHUNK):
                dst.write(data)
                await dst.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            try:
                if dst.can_write_eof():
                    dst.write_eof()
            except (OSError, RuntimeError):
                pass

    def _forward_head(self, req: Request, cookies: list[str], peer: str) -> bytes:
        lines = [f"{req.method} {req.target} {req.version}"]
        forwarded = None
        for name, value in req.headers:
            low = name.lower()
            if low == "cookie":
                continue
            if low == "x-forwarded-for":
                forwarded = value if forwarded is None else f"{forwarded}, {value}"
                continue
            lines.append(f"{name}: {value}")
        if cookies:
            lines.append("Cookie: " + "; ".join(cookies))
        if peer:
            lines.append("X-Forwarded-For: " + (f"{forwarded}, {peer}" if forwarded else peer))
        elif forwarded:
            lines.append("X-Forwarded-For: " + forwarded)
        return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")

    async def _forward_body(self, reader: asyncio.StreamReader, dst: asyncio.StreamWriter,
                            framing: tuple[str, int]) -> None:
        kind, left = framing
        if kind == "length":
            while left:
                data = await reader.readexactly(min(left, CHUNK))
                dst.write(data)
                left -= len(data)
                await dst.drain()
        elif kind == "chunked":
            while True:
                line = await reader.readuntil(b"\r\n")
                size = int(line.split(b";", 1)[0].strip(), 16)
                dst.write(line)
                if size == 0:
                    while True:  # trailers, then the blank line
                        trailer = await reader.readuntil(b"\r\n")
                        dst.write(trailer)
                        if trailer == b"\r\n":
                            return
                while size + 2:
                    data = await reader.readexactly(min(size + 2, CHUNK))
                    dst.write(data)
                    size -= len(data)
                await dst.drain()

    async def _gate_endpoint(self, req: Request, writer: asyncio.StreamWriter) -> None:
        if req.path == "/__gate/mint" and req.method == "POST":
            if not self._bearer_ok(req):
                return await self._respond(writer, 401, "unauthorized")
            body = json.dumps({"nonce": self.mint(), "expires_in": int(NONCE_TTL_S)})
            return await self._respond(writer, 200, body, content_type="application/json")
        if req.path == "/__gate/enter" and req.method in ("GET", "HEAD"):
            nonce = (req.query.get("n") or [""])[0]
            if not nonce or not self.consume(nonce):
                return await self._unauthorized(writer, expired=bool(nonce))
            secure = (req.get("x-forwarded-proto") or "").lower() == "https"
            cookie = (f"{self._cookie}={self.make_cookie()}; Path=/; Max-Age={COOKIE_TTL_S}; "
                      "HttpOnly; SameSite=Lax" + ("; Secure" if secure else ""))
            return await self._respond(writer, 303, "", headers=[
                ("Location", "/"), ("Set-Cookie", cookie)])
        await self._respond(writer, 404, "not found")

    async def _unauthorized(self, writer: asyncio.StreamWriter, expired: bool = False) -> None:
        esc = html.escape
        why = ("This preview link was already used or has expired."
               if expired else "This house preview needs sign-in.")
        back = (f'<p><a href="{esc(self.campus_url)}/tools/campus">Open it from Campus</a></p>'
                if self.campus_url else "<p>Open it from Campus in BawtHub.</p>")
        body = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
                f'<meta name="viewport" content="width=device-width,initial-scale=1">'
                f"<title>House {esc(self.house)}: sign-in needed</title></head>"
                f'<body style="font:16px/1.5 system-ui,sans-serif;background:#0b0f17;'
                f'color:#e5e7eb;display:grid;place-items:center;min-height:100vh;margin:0">'
                f'<main style="max-width:30rem;padding:1.5rem"><h1 style="font-size:1.3rem">'
                f"House {esc(self.house)}</h1><p>{why}</p>{back}</main></body></html>")
        await self._respond(writer, 401, body, html_body=True)

    @staticmethod
    async def _respond(writer: asyncio.StreamWriter, status: int, body: str, *,
                       headers: list[tuple[str, str]] | None = None, html_body: bool = False,
                       content_type: str | None = None) -> None:
        reason = {200: "OK", 303: "See Other", 400: "Bad Request", 401: "Unauthorized",
                  404: "Not Found", 431: "Request Header Fields Too Large"}.get(status, "Error")
        data = body.encode()
        ctype = content_type or ("text/html; charset=utf-8" if html_body
                                 else "text/plain; charset=utf-8")
        lines = [f"HTTP/1.1 {status} {reason}", f"Content-Type: {ctype}",
                 "Cache-Control: no-store", "Referrer-Policy: no-referrer",
                 "Connection: close", f"Content-Length: {len(data)}"]
        lines += [f"{k}: {v}" for k, v in headers or []]
        try:
            writer.write(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + data)
            await writer.drain()
        except ConnectionError:
            pass


__all__ = ["COOKIE_TTL_S", "NONCE_TTL_S", "PreviewGate", "body_framing", "cookie_name",
           "parse_head", "read_service_token", "split_cookie"]
