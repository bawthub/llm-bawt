"""Preview inference egress proxy (TASK-1802; design: Codex DESIGN r2 cmvveuzgj00zq0mpdddj3j79c).

The preview bridge's networks are all internal; this CONNECT-only proxy is its
single path to model providers. It is deliberately small:

* CONNECT only, port 443 only, to a canonical host on a run-scoped allowlist
  snapshot (a read-only file rendered by preview-up from DB settings; the
  proxy has no database route or credential). IP literals, trailing dots,
  userinfo, wildcards and absolute-form requests are refused.
* Each CONNECT resolves the host once; if ANY address is not globally
  routable (private, loopback, link-local, ULA, CGNAT, IPv4-mapped, metadata,
  multicast, ...) the request is refused, otherwise the proxy dials the
  validated numeric address and never resolves the name again (no DNS
  rebinding window).
* The tunnel is opaque TLS: the proxy never sees the bearer. Logs carry
  host, port and verdict only, never headers or payload.
* Every resource is bounded: request head size and line count, head and
  connect timeouts, concurrent tunnels, idle and total tunnel lifetime.

    python -m llm_bawt.preview.egress
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import logging
import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("llm_bawt.preview.egress")

PREVIEW_FLAG_ENV = "LLM_BAWT_PREVIEW_ENABLED"
ALLOWLIST_ENV ="LLM_BAWT_PREVIEW_EGRESS_ALLOWLIST_FILE"
LISTEN_HOST_ENV = "LLM_BAWT_PREVIEW_EGRESS_LISTEN_HOST"
LISTEN_PORT_ENV = "LLM_BAWT_PREVIEW_EGRESS_LISTEN_PORT"
MAX_TUNNELS_ENV = "LLM_BAWT_PREVIEW_EGRESS_MAX_TUNNELS"

PORT = 443


@dataclass(frozen=True)
class Limits:
    head_bytes: int = 8192
    head_lines: int = 32
    head_timeout: float = 5.0
    connect_timeout: float = 3.0
    idle_timeout: float = 120.0
    lifetime: float = 3600.0
    max_tunnels: int = 32
    chunk: int = 65536


class Refused(Exception):
    def __init__(self, status: int, reason: str):
        super().__init__(reason)
        self.status, self.reason = status, reason


# --- allowlist ------------------------------------------------------------


def canonical_host(host: str) -> str:
    """Return ``host`` if it is already a canonical DNS name; raise ValueError otherwise."""
    if not host or host != host.strip() or host.endswith(".") or "*" in host or "@" in host:
        raise ValueError(f"not a canonical host: {host!r}")
    try:
        ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        pass
    else:
        raise ValueError(f"IP literal not allowed: {host!r}")
    try:
        ascii_host = host.encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise ValueError(f"not a valid IDNA host: {host!r}") from exc
    labels = ascii_host.split(".")
    if ascii_host != host or len(labels) < 2 or len(ascii_host) > 253:
        raise ValueError(f"not a canonical host: {host!r}")
    for label in labels:
        if not (0 < len(label) <= 63) or label[0] == "-" or label[-1] == "-":
            raise ValueError(f"not a canonical host: {host!r}")
        if not all(ch.isascii() and (ch.isalnum() or ch == "-") for ch in label):
            raise ValueError(f"not a canonical host: {host!r}")
    return host


def allowlist_digest(hosts: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(hosts)).encode()).hexdigest()


@dataclass(frozen=True)
class Allowlist:
    run_id: str
    hosts: frozenset[str]
    sha256: str


def load_allowlist(path: Path) -> Allowlist:
    """Validate the run-scoped snapshot; raise ValueError on anything off."""
    data = json.loads(path.read_text(encoding="utf-8"))
    run_id, hosts, digest = data.get("run_id"), data.get("hosts"), data.get("sha256")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("allowlist: run_id missing")
    if not isinstance(hosts, list) or not hosts or not all(isinstance(h, str) for h in hosts):
        raise ValueError("allowlist: hosts must be a non-empty list of names")
    for host in hosts:
        canonical_host(host)
    if len(set(hosts)) != len(hosts):
        raise ValueError("allowlist: duplicate host")
    if digest != allowlist_digest(hosts):
        raise ValueError("allowlist: sha256 does not match hosts")
    return Allowlist(run_id, frozenset(hosts), digest)


# --- addresses ------------------------------------------------------------


def is_public(address: str) -> bool:
    """True only for a globally routable unicast address."""
    ip = ipaddress.ip_address(address)
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None or ip.sixtofour is not None or ip.teredo is not None:
            return False  # an embedded IPv4 can smuggle a private target
    return ip.is_global and not ip.is_multicast and not ip.is_reserved


async def resolve_public(host: str, resolver=None) -> tuple[int, str]:
    """Resolve once; return ``(family, address)`` only if every address is public."""
    loop = asyncio.get_running_loop()
    infos = await (resolver or loop.getaddrinfo)(host, PORT, type=socket.SOCK_STREAM)
    addresses = [(family, sockaddr[0]) for family, _t, _p, _c, sockaddr in infos]
    if not addresses:
        raise Refused(502, "no address")
    if not all(is_public(address) for _family, address in addresses):
        raise Refused(403, "non-public address")
    return addresses[0]


# --- request --------------------------------------------------------------


def parse_connect(head: bytes, allowlist: Allowlist) -> str:
    """Return the allowlisted host of a valid ``CONNECT host:443`` head."""
    try:
        text = head.decode("ascii")
    except UnicodeDecodeError as exc:
        raise Refused(400, "non-ascii head") from exc
    request_line = text.split("\r\n", 1)[0]
    parts = request_line.split(" ")
    if len(parts) != 3 or parts[2] not in ("HTTP/1.1", "HTTP/1.0"):
        raise Refused(400, "malformed request line")
    method, authority, _version = parts
    if method != "CONNECT":
        raise Refused(405, "CONNECT only")
    if "/" in authority or "@" in authority or authority.startswith("["):
        raise Refused(400, "authority form only")
    host, sep, port = authority.rpartition(":")
    if not sep or port != str(PORT):
        raise Refused(403, "port 443 only")
    try:
        canonical_host(host)
    except ValueError as exc:
        raise Refused(403, "non-canonical host") from exc
    if host not in allowlist.hosts:
        raise Refused(403, "host not allowlisted")
    return host


async def read_head(reader: asyncio.StreamReader, limits: Limits) -> bytes:
    head, lines = b"", 0
    while True:
        line = await reader.readline()
        if not line:
            raise Refused(400, "incomplete head")
        head += line
        lines += 1
        if len(head) > limits.head_bytes or lines > limits.head_lines:
            raise Refused(431, "head too large")
        if line in (b"\r\n", b"\n"):
            return head


# --- server ---------------------------------------------------------------


class EgressProxy:
    def __init__(self, allowlist: Allowlist, limits: Limits = Limits(), resolver=None, opener=None):
        self.allowlist, self.limits = allowlist, limits
        self._resolver = resolver
        self._opener = opener or asyncio.open_connection
        self._slots = asyncio.Semaphore(limits.max_tunnels)

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        host = "-"
        try:
            if self._slots.locked():
                raise Refused(503, "tunnel limit")
            async with self._slots:
                head = await asyncio.wait_for(read_head(reader, self.limits), self.limits.head_timeout)
                host = parse_connect(head, self.allowlist)
                _family, address = await asyncio.wait_for(
                    resolve_public(host, self._resolver), self.limits.connect_timeout)
                up_reader, up_writer = await asyncio.wait_for(
                    self._opener(address, PORT), self.limits.connect_timeout)
                log.info("egress allow host=%s port=%d", host, PORT)
                writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                await writer.drain()
                await self._tunnel(reader, writer, up_reader, up_writer)
        except Refused as exc:
            log.info("egress refuse host=%s status=%d reason=%s", host, exc.status, exc.reason)
            await self._reply(writer, exc.status)
        except (asyncio.TimeoutError, OSError) as exc:
            log.info("egress fail host=%s error=%s", host, type(exc).__name__)
            await self._reply(writer, 504 if isinstance(exc, asyncio.TimeoutError) else 502)
        finally:
            writer.close()

    @staticmethod
    async def _reply(writer: asyncio.StreamWriter, status: int) -> None:
        try:
            writer.write(f"HTTP/1.1 {status} Refused\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".encode())
            await writer.drain()
        except OSError:
            pass

    async def _tunnel(self, reader, writer, up_reader, up_writer) -> None:
        async def pump(src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
            while data := await asyncio.wait_for(src.read(self.limits.chunk), self.limits.idle_timeout):
                dst.write(data)
                await dst.drain()  # bounded buffers: backpressure, never accumulate

        tasks = [asyncio.ensure_future(pump(reader, up_writer)), asyncio.ensure_future(pump(up_reader, writer))]
        try:
            await asyncio.wait(tasks, timeout=self.limits.lifetime, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            up_writer.close()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if os.environ.get(PREVIEW_FLAG_ENV) != "1":
        log.error("egress: preview disabled (%s != 1)", PREVIEW_FLAG_ENV)
        return 2
    path =os.environ.get(ALLOWLIST_ENV)
    if not path:
        log.error("egress: %s is required", ALLOWLIST_ENV)
        return 2
    try:
        allowlist = load_allowlist(Path(path))
    except (OSError, ValueError) as exc:
        log.error("egress: refusing to start: %s", exc)
        return 2
    limits = Limits(max_tunnels=int(os.environ.get(MAX_TUNNELS_ENV) or Limits.max_tunnels))
    log.info("egress allowlist loaded run_id=%s count=%d sha256=%s",
             allowlist.run_id, len(allowlist.hosts), allowlist.sha256)
    host = os.environ.get(LISTEN_HOST_ENV) or "0.0.0.0"
    port = int(os.environ.get(LISTEN_PORT_ENV) or 3128)

    async def serve() -> None:
        proxy = EgressProxy(allowlist, limits)
        server = await asyncio.start_server(proxy.handle, host, port, limit=limits.head_bytes)
        async with server:
            await server.serve_forever()

    asyncio.run(serve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
