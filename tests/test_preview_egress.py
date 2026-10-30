"""TASK-1802: the preview egress proxy's CONNECT policy, hermetically (no network)."""

from __future__ import annotations

import asyncio
import json
import socket

import pytest

from llm_bawt.preview import egress
from llm_bawt.preview.egress import (
    Allowlist,
    EgressProxy,
    Limits,
    Refused,
    allowlist_digest,
    canonical_host,
    is_public,
    load_allowlist,
    parse_connect,
)

HOSTS = ["api.example.com", "llm.example.org"]
ALLOW = Allowlist("run-1", frozenset(HOSTS), allowlist_digest(HOSTS))


def _snapshot(tmp_path, **override):
    data = {"run_id": "run-1", "hosts": HOSTS, "sha256": allowlist_digest(HOSTS), **override}
    path = tmp_path / "preview_egress_allowlist.json"
    path.write_text(json.dumps(data))
    return path


def test_snapshot_loads(tmp_path):
    assert load_allowlist(_snapshot(tmp_path)) == ALLOW


@pytest.mark.parametrize("override", [
    {"hosts": []},
    {"hosts": ["API.example.com"]},
    {"hosts": ["api.example.com."]},
    {"hosts": ["*.example.com"]},
    {"hosts": ["10.0.0.1"]},
    {"hosts": ["localhost"]},
    {"hosts": ["api.example.com", "api.example.com"]},
    {"sha256": "0" * 64},
    {"run_id": ""},
])
def test_bad_snapshot_is_refused(tmp_path, override):
    if "hosts" in override and "sha256" not in override:
        override["sha256"] = allowlist_digest(override["hosts"]) if all(isinstance(h, str) for h in override["hosts"]) else ""
    with pytest.raises(ValueError):
        load_allowlist(_snapshot(tmp_path, **override))


def test_idna_host_must_be_punycode():
    assert canonical_host("xn--bcher-kva.example") == "xn--bcher-kva.example"
    with pytest.raises(ValueError):
        canonical_host("bücher.example")


def _head(line: str) -> bytes:
    return f"{line}\r\nHost: x\r\n\r\n".encode()


def test_connect_to_allowlisted_host():
    assert parse_connect(_head("CONNECT api.example.com:443 HTTP/1.1"), ALLOW) == "api.example.com"


@pytest.mark.parametrize("line,status", [
    ("GET http://api.example.com/ HTTP/1.1", 405),
    ("CONNECT api.example.com:80 HTTP/1.1", 403),
    ("CONNECT api.example.com HTTP/1.1", 403),
    ("CONNECT other.example.com:443 HTTP/1.1", 403),
    ("CONNECT api.example.com.:443 HTTP/1.1", 403),
    ("CONNECT API.example.com:443 HTTP/1.1", 403),
    ("CONNECT 93.184.216.34:443 HTTP/1.1", 403),
    ("CONNECT [2606:2800::1]:443 HTTP/1.1", 400),
    ("CONNECT user@api.example.com:443 HTTP/1.1", 400),
    ("CONNECT http://api.example.com:443 HTTP/1.1", 400),
    ("CONNECT api.example.com:443 HTTP/2", 400),
    ("CONNECT api.example.com:443", 400),
])
def test_connect_refusals(line, status):
    with pytest.raises(Refused) as exc:
        parse_connect(_head(line), ALLOW)
    assert exc.value.status == status


@pytest.mark.parametrize("address", [
    "10.1.2.3", "172.18.0.1", "192.168.1.1", "127.0.0.1", "169.254.169.254", "100.64.0.1",
    "0.0.0.0", "224.0.0.1", "255.255.255.255", "::1", "fe80::1", "fd00::1",
    "::ffff:10.0.0.1", "::ffff:8.8.8.8", "2002:0a00:0001::1", "ff02::1",
])
def test_non_public_addresses(address):
    assert not is_public(address)


@pytest.mark.parametrize("address", ["8.8.8.8", "93.184.216.34", "2606:4700:4700::1111"])
def test_public_addresses(address):
    assert is_public(address)


def _infos(*addresses):
    return [(socket.AF_INET6 if ":" in a else socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, 443)) for a in addresses]


def _resolver(*addresses, calls=None):
    async def resolve(host, port, type=None):
        if calls is not None:
            calls.append(host)
        return _infos(*addresses)
    return resolve


async def _exchange(proxy: EgressProxy, request: bytes) -> bytes:
    reader = asyncio.StreamReader()
    reader.feed_data(request)
    reader.feed_eof()
    out = bytearray()

    class Writer:
        def write(self, data):
            out.extend(data)

        async def drain(self):
            pass

        def close(self):
            pass

    await proxy.handle(reader, Writer())
    return bytes(out)


class _Upstream:
    """Records the dial target; echoes nothing and closes at once."""

    def __init__(self):
        self.dialed = []
        self.writer_closed = False

    async def open(self, host, port):
        self.dialed.append((host, port))
        reader = asyncio.StreamReader()
        reader.feed_eof()
        upstream = self

        class Writer:
            def write(self, data):
                pass

            async def drain(self):
                pass

            def close(self):
                upstream.writer_closed = True

        return reader, Writer()


def test_mixed_resolution_is_refused_and_never_dialed():
    upstream = _Upstream()
    proxy = EgressProxy(ALLOW, resolver=_resolver("8.8.8.8", "10.0.0.5"), opener=upstream.open)
    reply = asyncio.run(_exchange(proxy, _head("CONNECT api.example.com:443 HTTP/1.1")))
    assert reply.startswith(b"HTTP/1.1 403") and upstream.dialed == []


def test_allowed_connect_dials_the_validated_address_once():
    upstream, calls = _Upstream(), []
    proxy = EgressProxy(ALLOW, resolver=_resolver("8.8.8.8", calls=calls), opener=upstream.open)
    reply = asyncio.run(_exchange(proxy, _head("CONNECT api.example.com:443 HTTP/1.1")))
    assert reply.startswith(b"HTTP/1.1 200")
    assert upstream.dialed == [("8.8.8.8", 443)]  # numeric, never the name
    assert calls == ["api.example.com"] and upstream.writer_closed


def test_oversized_head_is_refused():
    proxy = EgressProxy(ALLOW, Limits(head_lines=4), resolver=_resolver("8.8.8.8"), opener=_Upstream().open)
    request = b"CONNECT api.example.com:443 HTTP/1.1\r\n" + b"X: y\r\n" * 10 + b"\r\n"
    assert asyncio.run(_exchange(proxy, request)).startswith(b"HTTP/1.1 431")


def test_head_timeout():
    async def run():
        proxy = EgressProxy(ALLOW, Limits(head_timeout=0.01), resolver=_resolver("8.8.8.8"))
        reader = asyncio.StreamReader()  # never fed: a slowloris client
        out = bytearray()

        class Writer:
            write = out.extend

            async def drain(self):
                pass

            def close(self):
                pass

        await proxy.handle(reader, Writer())
        return bytes(out)

    assert asyncio.run(run()).startswith(b"HTTP/1.1 504")


def test_tunnel_cap():
    async def run():
        proxy = EgressProxy(ALLOW, Limits(max_tunnels=1), resolver=_resolver("8.8.8.8"), opener=_Upstream().open)
        await proxy._slots.acquire()
        return await _exchange(proxy, _head("CONNECT api.example.com:443 HTTP/1.1"))

    assert asyncio.run(run()).startswith(b"HTTP/1.1 503")


def test_main_is_inert_without_preview_flag(monkeypatch, tmp_path):
    monkeypatch.delenv(egress.PREVIEW_FLAG_ENV, raising=False)
    monkeypatch.setenv(egress.ALLOWLIST_ENV, str(_snapshot(tmp_path)))
    assert egress.main() == 2


def test_main_refuses_bad_snapshot(monkeypatch, tmp_path):
    monkeypatch.setenv(egress.PREVIEW_FLAG_ENV, "1")
    monkeypatch.setenv(egress.ALLOWLIST_ENV, str(_snapshot(tmp_path, hosts=[])))
    assert egress.main() == 2
    monkeypatch.delenv(egress.ALLOWLIST_ENV)
    assert egress.main() == 2
