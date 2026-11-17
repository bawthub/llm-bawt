"""The sign-in gate in front of a house preview (TASK-2390)."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys

import pytest

from agent_bridge import house_preview as hp
from agent_bridge import house_preview_gate as hg
from agent_bridge import house_preview_supervisor as sup

TOKEN = "fixture-service-token"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Clock:
    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class Upstream:
    """A stand-in for Vite: records request heads and answers by path."""

    def __init__(self) -> None:
        self.port = _free_port()
        self.heads: list[str] = []
        self.bodies: list[bytes] = []
        self.release = asyncio.Event()
        self._server = None

    async def start(self):
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", self.port)
        return self

    async def stop(self):
        self._server.close()
        await self._server.wait_closed()

    async def _handle(self, reader, writer):
        try:
            while True:
                head = (await reader.readuntil(b"\r\n\r\n")).decode("latin-1")
                self.heads.append(head)
                req = hg.parse_head(head.encode("latin-1"))
                kind, length = hg.body_framing(req)
                body = b""
                if kind == "length":
                    body = await reader.readexactly(length)
                elif kind == "chunked":
                    while True:
                        size = int((await reader.readuntil(b"\r\n")).split(b";")[0], 16)
                        if size == 0:
                            await reader.readuntil(b"\r\n")
                            break
                        body += (await reader.readexactly(size + 2))[:-2]
                self.bodies.append(body)
                if req.is_upgrade and req.path == "/decline":
                    # A legal refusal: an ordinary answer, connection kept alive.
                    writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nno")
                    await writer.drain()
                    continue
                if req.path == "/ticks":
                    writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                                 b"Connection: close\r\n\r\n")
                    for i in range(30):
                        writer.write(b"data: tick %d\n\n" % i)
                        await writer.drain()
                        await asyncio.sleep(0.015)
                    return
                if req.is_upgrade:
                    writer.write(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                                 b"Connection: Upgrade\r\n\r\n")
                    await writer.drain()
                    while data := await reader.read(1024):
                        writer.write(b"echo:" + data)
                        await writer.drain()
                    return
                if req.path == "/events":
                    writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                                 b"Connection: close\r\n\r\ndata: one\n\n")
                    await writer.drain()
                    await self.release.wait()
                    writer.write(b"data: two\n\n")
                    await writer.drain()
                    return
                out = f"upstream {req.method} {req.path} {len(body)}".encode()
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n" % len(out) + out)
                await writer.drain()
                if (req.get("connection") or "").lower() == "close":
                    return
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()


async def _gate(house="h1", upstream_port=None, clock=None, token=TOKEN):
    gate = hg.PreviewGate(house, port=_free_port(), host="127.0.0.1",
                          upstream_port=upstream_port or _free_port(),
                          token=lambda: token, clock=clock or Clock())
    gate.page = "<p>fallback page</p>"
    await gate.start()
    return gate


async def _request(port, raw: bytes, *, read_all=True):
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(raw)
    await writer.drain()
    data = await asyncio.wait_for(reader.read(), 5) if read_all else b""
    writer.close()
    return data.decode("latin-1")


def _get(path="/", cookie=None, extra=""):
    lines = [f"GET {path} HTTP/1.1", "Host: preview.example.test"]
    if cookie:
        lines.append(f"Cookie: {cookie}")
    if extra:
        lines.append(extra)
    return ("\r\n".join(lines) + "\r\nConnection: close\r\n\r\n").encode()


async def _enter(gate) -> str:
    """Mint and open a link like the frontend and browser do; returns the Cookie pair."""
    minted = await _request(gate.port, (
        f"POST /__gate/mint HTTP/1.1\r\nHost: house-h1:5173\r\n"
        f"Authorization: Bearer {TOKEN}\r\nContent-Length: 0\r\n\r\n").encode())
    assert minted.startswith("HTTP/1.1 200")
    nonce = json.loads(minted.split("\r\n\r\n", 1)[1])["nonce"]
    entered = await _request(gate.port, _get(f"/__gate/enter?n={nonce}"))
    assert entered.startswith("HTTP/1.1 303")
    assert "\r\nLocation: /\r\n" in entered
    assert "Cache-Control: no-store" in entered and "Referrer-Policy: no-referrer" in entered
    set_cookie = next(line for line in entered.split("\r\n") if line.startswith("Set-Cookie:"))
    assert "HttpOnly" in set_cookie and "SameSite=Lax" in set_cookie
    assert "Secure" not in set_cookie
    return set_cookie.split(":", 1)[1].strip().split(";", 1)[0]


# --- parsing ----------------------------------------------------------------

def test_parse_and_framing_refuse_ambiguity():
    req = hg.parse_head(b"POST /x?a=1 HTTP/1.1\r\nContent-Length: 3\r\n\r\n")
    assert (req.method, req.path, req.query) == ("POST", "/x", {"a": ["1"]})
    assert hg.body_framing(req) == ("length", 3)
    for bad in (b"GET http://evil/ HTTP/1.1\r\n\r\n", b"GET / HTTP/2\r\n\r\n",
                b"GET / HTTP/1.1\r\n folded\r\n\r\n", b"GET / HTTP/1.1\r\nno colon\r\n\r\n"):
        with pytest.raises(hg.BadRequest):
            hg.parse_head(bad)
    for headers in (b"Content-Length: 3\r\nTransfer-Encoding: chunked",
                    b"Content-Length: 3\r\nContent-Length: 4",
                    b"Transfer-Encoding: gzip, chunked", b"Content-Length: -1"):
        with pytest.raises(hg.BadRequest):
            hg.body_framing(hg.parse_head(b"POST / HTTP/1.1\r\n" + headers + b"\r\n\r\n"))


def test_cookie_split_keeps_other_cookies():
    others, value = hg.split_cookie(["a=1; bawthub_preview_h1=v; b=2", "c=3"],
                                    "bawthub_preview_h1")
    assert value == "v" and others == ["a=1", "b=2", "c=3"]
    assert hg.cookie_name("H-2") == "bawthub_preview_h_2"


def test_nonce_is_one_use_and_expires():
    clock = Clock()
    gate = hg.PreviewGate("h1", clock=clock, token=lambda: TOKEN)
    n = gate.mint()
    assert gate.consume(n) and not gate.consume(n)
    n = gate.mint()
    clock.now += hg.NONCE_TTL_S + 1
    assert not gate.consume(n)


def test_cookie_binds_house_boot_and_expiry():
    clock = Clock()
    gate = hg.PreviewGate("h1", clock=clock, token=lambda: TOKEN)
    value = gate.make_cookie()
    assert gate.cookie_ok(value)
    assert not hg.PreviewGate("h2", clock=clock).cookie_ok(value)  # sibling house
    assert not hg.PreviewGate("h1", clock=clock).cookie_ok(value)  # after a restart
    exp, mac = value.split(".")
    assert not gate.cookie_ok(f"{int(exp) + 10}.{mac}")  # stretched expiry
    clock.now += hg.COOKIE_TTL_S + 1
    assert not gate.cookie_ok(value)
    assert not gate.cookie_ok("") and not gate.cookie_ok("garbage")


# --- over the wire ------------------------------------------------------------

@pytest.mark.asyncio
async def test_anonymous_requests_and_upgrades_get_401():
    up = await Upstream().start()
    gate = await _gate(upstream_port=up.port)
    gate.campus_url = "https://hub.example.test"
    try:
        body = await _request(gate.port, _get("/"))
        assert body.startswith("HTTP/1.1 401") and "/tools/campus" in body
        upgrade = await _request(gate.port, _get(
            "/", extra="Upgrade: websocket\r\nConnection: Upgrade"))
        assert upgrade.startswith("HTTP/1.1 401") and " 101 " not in upgrade
        forged = await _request(gate.port, _get("/", cookie="bawthub_preview_h1=1.abc"))
        assert forged.startswith("HTTP/1.1 401")
        assert up.heads == []  # nothing reached the preview
    finally:
        await gate.stop()
        await up.stop()


@pytest.mark.asyncio
async def test_mint_needs_the_service_token():
    gate = await _gate()
    try:
        for auth in ("", "Authorization: Bearer wrong\r\n", f"Authorization: Basic {TOKEN}\r\n"):
            out = await _request(gate.port, (
                f"POST /__gate/mint HTTP/1.1\r\nHost: x\r\n{auth}Content-Length: 0\r\n\r\n"
            ).encode())
            assert out.startswith("HTTP/1.1 401"), auth
        out = await _request(gate.port, _get("/__gate/mint", extra=f"Authorization: Bearer {TOKEN}"))
        assert out.startswith("HTTP/1.1 404")  # GET doesn't mint
    finally:
        await gate.stop()
    nobody = await _gate(token="")
    try:
        out = await _request(nobody.port, (
            "POST /__gate/mint HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer \r\n\r\n").encode())
        assert out.startswith("HTTP/1.1 401")  # no configured token mints nothing
    finally:
        await nobody.stop()


@pytest.mark.asyncio
async def test_link_is_one_use_and_expires_over_the_wire():
    clock = Clock()
    gate = await _gate(clock=clock)
    try:
        await _enter(gate)
        n = gate.mint()
        assert (await _request(gate.port, _get(f"/__gate/enter?n={n}"))).startswith("HTTP/1.1 303")
        replay = await _request(gate.port, _get(f"/__gate/enter?n={n}"))
        assert replay.startswith("HTTP/1.1 401") and "already used or has expired" in replay
        n = gate.mint()
        clock.now += hg.NONCE_TTL_S + 1
        assert (await _request(gate.port, _get(f"/__gate/enter?n={n}"))).startswith("HTTP/1.1 401")
        https = gate.mint()
        out = await _request(gate.port, _get(f"/__gate/enter?n={https}",
                                             extra="X-Forwarded-Proto: https"))
        assert "; Secure" in out
    finally:
        await gate.stop()


@pytest.mark.asyncio
async def test_sibling_house_and_restarted_gate_refuse_the_cookie():
    up = await Upstream().start()
    h1 = await _gate("h1", upstream_port=up.port)
    h2 = await _gate("h2", upstream_port=up.port)
    try:
        cookie = await _enter(h1)
        value = cookie.split("=", 1)[1]
        # Browsers send cookies per host, not port: h2 sees h1's cookie too.
        out = await _request(h2.port, _get("/", cookie=f"{cookie}; bawthub_preview_h2={value}"))
        assert out.startswith("HTTP/1.1 401")
        restarted = await _gate("h1", upstream_port=up.port)
        try:
            assert (await _request(restarted.port, _get("/", cookie=cookie))).startswith(
                "HTTP/1.1 401")
        finally:
            await restarted.stop()
        assert up.heads == []
    finally:
        await h1.stop()
        await h2.stop()
        await up.stop()


@pytest.mark.asyncio
async def test_signed_in_requests_reach_the_preview_without_the_gate_cookie():
    up = await Upstream().start()
    gate = await _gate(upstream_port=up.port)
    try:
        cookie = await _enter(gate)
        out = await _request(gate.port, _get("/src/main.tsx", cookie=f"app=1; {cookie}",
                                             extra="X-Forwarded-For: 203.0.113.9"))
        assert out.startswith("HTTP/1.1 200") and out.endswith("upstream GET /src/main.tsx 0")
        head = up.heads[-1]
        assert "bawthub_preview_h1" not in head and "Cookie: app=1\r\n" in head
        assert "X-Forwarded-For: 203.0.113.9, 127.0.0.1\r\n" in head
    finally:
        await gate.stop()
        await up.stop()


@pytest.mark.asyncio
async def test_every_request_on_a_kept_alive_connection_is_checked():
    up = await Upstream().start()
    gate = await _gate(upstream_port=up.port)
    try:
        cookie = await _enter(gate)
        reader, writer = await asyncio.open_connection("127.0.0.1", gate.port)
        body = b"x" * 70_000
        writer.write(b"POST /api/a HTTP/1.1\r\nHost: p\r\nCookie: " + cookie.encode()
                     + b"\r\nContent-Length: %d\r\n\r\n" % len(body) + body)
        writer.write(b"POST /api/b HTTP/1.1\r\nHost: p\r\nCookie: " + cookie.encode()
                     + b"\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n2\r\nde\r\n0\r\n\r\n")
        writer.write(b"GET /smuggled HTTP/1.1\r\nHost: p\r\n\r\n")  # no cookie
        await writer.drain()
        data = (await asyncio.wait_for(reader.read(), 5)).decode()
        writer.close()
        assert "upstream POST /api/a 70000" in data and "upstream POST /api/b 5" in data
        assert "smuggled" not in data
        assert [h.split(" ")[1] for h in up.heads] == ["/api/a", "/api/b"]
        assert up.bodies == [body, b"abcde"]
    finally:
        await gate.stop()
        await up.stop()


@pytest.mark.asyncio
async def test_event_streams_flow_through_unbuffered():
    up = await Upstream().start()
    gate = await _gate(upstream_port=up.port)
    try:
        cookie = await _enter(gate)
        reader, writer = await asyncio.open_connection("127.0.0.1", gate.port)
        writer.write(_get("/events", cookie=cookie))
        await writer.drain()
        first = b""
        while b"data: one" not in first:
            first += await asyncio.wait_for(reader.read(1024), 5)
        assert b"data: two" not in first  # arrived before the stream finished
        up.release.set()
        rest = await asyncio.wait_for(reader.read(), 5)
        assert b"data: two" in first + rest
        writer.close()
    finally:
        await gate.stop()
        await up.stop()


@pytest.mark.asyncio
async def test_signed_in_websocket_upgrade_is_piped_both_ways():
    up = await Upstream().start()
    gate = await _gate(upstream_port=up.port)
    try:
        cookie = await _enter(gate)
        reader, writer = await asyncio.open_connection("127.0.0.1", gate.port)
        writer.write((f"GET /?token=t HTTP/1.1\r\nHost: p\r\nCookie: {cookie}\r\n"
                      "Upgrade: websocket\r\nConnection: keep-alive, Upgrade\r\n\r\n").encode())
        await writer.drain()
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
        assert head.startswith(b"HTTP/1.1 101")
        writer.write(b"ping")
        await writer.drain()
        assert await asyncio.wait_for(reader.readexactly(9), 5) == b"echo:ping"
        writer.close()
    finally:
        await gate.stop()
        await up.stop()


@pytest.mark.asyncio
async def test_a_declined_upgrade_never_becomes_an_unchecked_tunnel():
    up = await Upstream().start()
    gate = await _gate(upstream_port=up.port)
    try:
        cookie = await _enter(gate)
        reader, writer = await asyncio.open_connection("127.0.0.1", gate.port)
        writer.write((f"GET /decline HTTP/1.1\r\nHost: p\r\nCookie: {cookie}\r\n"
                      "Upgrade: websocket\r\nConnection: Upgrade\r\n\r\n").encode())
        await writer.drain()
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
        assert head.startswith(b"HTTP/1.1 200")
        writer.write(b"GET /smuggled HTTP/1.1\r\nHost: p\r\n\r\n")  # no cookie
        await writer.drain()
        rest = (await asyncio.wait_for(reader.read(), 5)).decode()
        writer.close()
        assert rest == "no"  # the refusal's own body, then the connection closes
        assert [h.split(" ")[1] for h in up.heads] == ["/decline"]
    finally:
        await gate.stop()
        await up.stop()


@pytest.mark.asyncio
async def test_a_streaming_answer_outlives_the_idle_timer(monkeypatch):
    monkeypatch.setattr(hg, "IDLE_TIMEOUT_S", 0.1)
    up = await Upstream().start()
    gate = await _gate(upstream_port=up.port)
    try:
        cookie = await _enter(gate)
        reader, writer = await asyncio.open_connection("127.0.0.1", gate.port)
        writer.write(_get("/ticks", cookie=cookie).replace(b"Connection: close\r\n", b""))
        await writer.drain()
        data = await asyncio.wait_for(reader.read(), 5)
        writer.close()
        assert b"data: tick 29" in data  # ~0.45 s of events, well past the 0.1 s timer
    finally:
        await gate.stop()
        await up.stop()


@pytest.mark.asyncio
async def test_an_idle_connection_with_nothing_in_flight_still_times_out(monkeypatch):
    monkeypatch.setattr(hg, "IDLE_TIMEOUT_S", 0.1)
    monkeypatch.setattr(hg, "FIRST_HEAD_TIMEOUT_S", 0.1)
    gate = await _gate()
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", gate.port)
        assert await asyncio.wait_for(reader.read(), 5) == b""  # closed by the gate
        writer.close()
    finally:
        await gate.stop()


@pytest.mark.asyncio
async def test_page_is_behind_the_gate_while_no_preview_runs():
    gate = await _gate()  # nothing listens upstream
    try:
        assert (await _request(gate.port, _get("/"))).startswith("HTTP/1.1 401")
        cookie = await _enter(gate)
        out = await _request(gate.port, _get("/", cookie=cookie))
        assert out.startswith("HTTP/1.1 200") and "fallback page" in out
    finally:
        await gate.stop()


def test_service_token_file_wins_and_is_reread(tmp_path, monkeypatch):
    path = tmp_path / "token"
    path.write_text("one\n")
    monkeypatch.setenv(hg.TOKEN_FILE_ENV, str(path))
    monkeypatch.setenv(hg.TOKEN_ENV, "env")
    assert hg.read_service_token() == "one"
    path.write_text("two")
    assert hg.read_service_token() == "two"
    monkeypatch.delenv(hg.TOKEN_FILE_ENV)
    assert hg.read_service_token() == "env"


# --- house wiring -------------------------------------------------------------

def test_gate_is_off_unless_turned_on(monkeypatch):
    monkeypatch.delenv(hp.GATE_ENV, raising=False)
    assert not hp.gate_enabled() and hp.vite_port() == hp.PREVIEW_PORT
    monkeypatch.setenv(hp.GATE_ENV, "on")
    assert hp.gate_enabled() and hp.vite_port() == hp.INNER_PORT


def test_gated_vite_is_found_on_the_inner_port(tmp_path, monkeypatch):
    front = tmp_path / "wt" / "frontend"
    front.mkdir(parents=True)
    entry = tmp_path / "proc" / "13"
    entry.mkdir(parents=True)
    argv = ["node", "/w/node_modules/vite/bin/vite.js", "--host", "127.0.0.1",
            "--port", str(hp.INNER_PORT), "--strictPort"]
    (entry / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
    os.symlink(front, entry / "cwd")
    monkeypatch.delenv(hp.GATE_ENV, raising=False)
    assert hp.find_preview_cwd(tmp_path / "proc") is None
    monkeypatch.setenv(hp.GATE_ENV, "on")
    assert hp.find_preview_cwd(tmp_path / "proc") == front
    assert hp.describe(front, True)["gated"] is True
    monkeypatch.delenv(hp.GATE_ENV)
    assert "gated" not in hp.describe(front, True)


def test_gated_pair_keeps_vite_on_loopback():
    vite = sup.GATED_PAIR[4]
    assert "--host 127.0.0.1" in vite and f"--port {hp.INNER_PORT}" in vite
    assert sup.GATED_PAIR[:4] == sup.PAIR[:4] and sup.GATED_PAIR[5] == sup.PAIR[5]


@pytest.mark.asyncio
async def test_gated_supervisor_keeps_the_gate_on_the_port_while_the_preview_runs(
        tmp_path, monkeypatch):
    monkeypatch.setenv(hp.GATE_ENV, "on")
    monkeypatch.setattr(sup, "LOG_PATH", tmp_path / "preview.log")
    front = tmp_path / ".worktrees" / ".wt-bawthub-nova-TASK-7" / "frontend"
    front.mkdir(parents=True)
    (front / "package.json").write_text("{}")
    inner = _free_port()
    gate = hg.PreviewGate("h1", port=_free_port(), host="127.0.0.1", upstream_port=inner,
                          token=lambda: TOKEN)

    async def fetch():
        return {"houses": [{"id": "h1", "tenants": ["nova"]}]}

    async def house_of(bot):
        return "h1"

    async def step(self, cwd, argv):
        return True

    monkeypatch.setattr(sup.PreviewSupervisor, "_step", step)
    monkeypatch.setattr(sup, "GATED_PAIR", [sys.executable, "-m", "http.server", str(inner),
                                            "--bind", "127.0.0.1"])
    s = sup.PreviewSupervisor("h1", _FakeRedis(), fetch_campus=fetch,
                              fetch_house_of=house_of, dev_root=tmp_path, fallback=gate)
    assert s.gated
    try:
        await s.tick()
        for _ in range(100):
            if gate.serving:
                break
            await asyncio.sleep(0.02)
        cookie = await _enter(gate)
        for _ in range(100):
            await asyncio.sleep(0.05)
            out = await _request(gate.port, _get("/", cookie=cookie))
            if "Directory listing" in out:
                break
        else:
            pytest.fail("the preview never answered through the gate")
        assert gate.serving  # the gate never handed the port over
        assert (await _request(gate.port, _get("/"))).startswith("HTTP/1.1 401")
    finally:
        await s._stop_run()
        await gate.stop()


class _FakeRedis:
    def __init__(self):
        self.data = {}

    async def get(self, key):
        return self.data.get(key)

    async def delete(self, key):
        self.data.pop(key, None)

    async def eval(self, script, n, key, expected):
        if self.data.get(key) == expected:
            self.data.pop(key)
