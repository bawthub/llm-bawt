"""Peer links run over HTTPS pinned to each install's identity key (TASK-2517).

Two installs with their own databases and real TLS listeners pair through the
owner join route and then call each other; plain HTTP stays refused.
"""
from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest
import uvicorn
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import HTTPException

from llm_bawt.peer import client, identity, pairing, runtime, server, store, tls
from llm_bawt.service.routes import peers
from tests.test_peer_pairing_discovery import Install, _free_port


@pytest.fixture(autouse=True)
def _https_only(monkeypatch):
    monkeypatch.delenv(pairing.ALLOW_HTTP_ENV, raising=False)
    monkeypatch.setattr(identity, "_encrypt", lambda v: "x:" + v)
    monkeypatch.setattr(identity, "_decrypt", lambda v: v[2:])


class TlsInstall(Install):
    """An install whose peer listener serves HTTPS like ``runtime.serve_forever``."""

    def _start(self, *, as_identity: identity.Identity | None = None):
        self.url = f"https://127.0.0.1:{self.port}"
        app = server.build_app(lambda: self.engine, list_bots=self._list, admit=self._admit)
        config = uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning",
                                lifespan="on")
        tls.load_server_tls(config, as_identity or self.ident())
        self.server = uvicorn.Server(config)
        threading.Thread(target=self.server.run, daemon=True).start()
        deadline = time.time() + 10  # timing-ok: generous startup hang bound, not a speed claim
        while not self.server.started:
            assert time.time() < deadline, "peer listener did not start"  # timing-ok: hang bound
            time.sleep(0.05)


@pytest.fixture
def tls_installs():
    a = TlsInstall("Home", {"nova": "Nova"})
    b = TlsInstall("Lab", {"echo": "Echo"})
    yield a, b
    a.stop()
    b.stop()


def _as_owner(monkeypatch, inst: Install):
    monkeypatch.setattr(peers, "_require_enabled", lambda: SimpleNamespace(
        engine=lambda: inst.engine, public_url=lambda: inst.url))


def _join(monkeypatch, a: Install, b: Install, *, confirm=True):
    with b.engine.begin() as conn:
        invite, _ = pairing.create_invite(conn, b.ident(), b.url)
    _as_owner(monkeypatch, a)
    fp = invite.fingerprint if confirm else None
    return asyncio.run(peers.join(peers.JoinBody(code=invite.code(), confirm_fingerprint=fp)))


def test_pair_and_call_over_pinned_https(monkeypatch, tls_installs):
    a, b = tls_installs
    assert "paired" in _join(monkeypatch, a, b)

    b_on_a, a_on_b = a.link(b), b.link(a)
    assert tls.public_key_of(b_on_a.tls_cert) == b.ident().public_key
    assert tls.public_key_of(a_on_b.tls_cert) == a.ident().public_key
    with b.engine.begin() as conn:
        store.set_state(conn, a.ident().install_id, "active")
        store.set_export(conn, a.ident().install_id, ["echo"])
    got = asyncio.run(client.call(a.ident(), b_on_a, "peer_bots_list"))
    assert [bot["slug"] for bot in got["bots"]] == ["echo"]
    # And the other direction, with the certificate B pinned from the request.
    assert asyncio.run(client.call(b.ident(), b.link(a), "peer_bots_list"))["bots"] == []


def test_join_refuses_a_listener_with_another_key(monkeypatch, tls_installs):
    a, b = tls_installs
    b.stop()
    impostor = identity.Identity("impostor", "Lab", Ed25519PrivateKey.generate())
    b.port = _free_port()  # a fresh port: the stopped listener may still hold the old one
    b._start(as_identity=impostor)  # the invite's URL now serves someone else's cert
    with pytest.raises(HTTPException) as refused:
        _join(monkeypatch, a, b, confirm=False)
    assert refused.value.status_code == 409
    assert "certificate" in refused.value.detail
    assert a.link(b) is None


def test_calls_trust_only_the_pinned_certificate(monkeypatch, tls_installs):
    a, b = tls_installs
    _join(monkeypatch, a, b)
    with b.engine.begin() as conn:
        store.set_state(conn, a.ident().install_id, "active")
    link = a.link(b)
    other = tls.certificate_pem(identity.Identity("x", "X", Ed25519PrivateKey.generate()))
    with pytest.raises(client.PeerUnavailable):
        asyncio.run(client.call(a.ident(), store.PeerLink(**{**link.__dict__, "tls_cert": other}),
                                "peer_bots_list"))
    with pytest.raises(client.PeerUnavailable, match="pair it again"):
        asyncio.run(client.call(a.ident(), store.PeerLink(**{**link.__dict__, "tls_cert": None}),
                                "peer_bots_list"))


def test_pair_request_certificate_must_carry_the_presented_key():
    a = identity.Identity("a", "A", Ed25519PrivateKey.generate())
    other = identity.Identity("o", "O", Ed25519PrivateKey.generate())
    url = "https://peer.example:9443"
    assert pairing.peer_certificate(url, tls.certificate_pem(a), a.public_key)
    for bad in (tls.certificate_pem(other), "", None, "not a cert"):
        with pytest.raises(pairing.PairingError):
            pairing.peer_certificate(url, bad, a.public_key)


def test_certificate_is_stable_for_the_identity_key():
    own = identity.Identity("a", "A", Ed25519PrivateKey.generate())
    assert tls.certificate_pem(own) == tls.certificate_pem(own)
    assert tls.public_key_of(tls.certificate_pem(own)) == own.public_key


def test_listener_serves_plain_http_only_when_explicitly_allowed(monkeypatch):
    monkeypatch.setattr(runtime, "engine", lambda: pytest.fail("needs no identity"))
    monkeypatch.setenv(runtime.URL_ENV, "http://peer.example:9443")
    monkeypatch.setenv(pairing.ALLOW_HTTP_ENV, "1")
    config = uvicorn.Config(server.build_app(lambda: None), port=1)
    assert runtime._load_tls(config) is False
    assert config.ssl_keyfile is None


def test_listener_tls_is_1_3_and_leaves_no_key_file(monkeypatch):
    from sqlalchemy import create_engine
    eng = create_engine("sqlite://")
    with eng.begin() as conn:
        store.ensure_schema(conn)
        identity.load_or_create(conn)
    monkeypatch.setattr(runtime, "engine", lambda: eng)
    monkeypatch.setenv(runtime.URL_ENV, "https://peer.example:9443")
    import os
    import ssl
    import tempfile
    made = []
    real = tempfile.mkdtemp
    monkeypatch.setattr(tls.tempfile, "mkdtemp", lambda **k: made.append(real(**k)) or made[-1])
    config = uvicorn.Config(server.build_app(lambda: eng), port=1)
    assert runtime._load_tls(config) is True
    assert config.ssl.minimum_version == ssl.TLSVersion.TLSv1_3
    assert made and not os.path.exists(made[0])

    made.clear()  # a failed load removes the key file too
    broken = uvicorn.Config(server.build_app(lambda: eng), port=1)
    monkeypatch.setattr(broken, "load", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    with pytest.raises(RuntimeError):
        runtime._load_tls(broken)
    assert made and not os.path.exists(made[0])


def test_extra_certificates_never_reach_the_trust_store(monkeypatch, tls_installs):
    a, b = tls_installs
    _join(monkeypatch, a, b)
    with b.engine.begin() as conn:
        store.set_state(conn, a.ident().install_id, "active")
    link = a.link(b)
    extra = tls.certificate_pem(identity.Identity("x", "X", Ed25519PrivateKey.generate()))
    bundled = link.tls_cert + extra
    with pytest.raises(pairing.PairingError):
        pairing.peer_certificate("https://p", bundled, b.ident().public_key)
    with pytest.raises(tls.CertificateError):
        tls.client_context(bundled)
    with pytest.raises(Exception):
        asyncio.run(client.call(a.ident(), store.PeerLink(**{**link.__dict__,
                                                            "tls_cert": bundled}),
                                "peer_bots_list"))
    assert link.tls_cert.count("-----BEGIN") == 1
    assert tls.client_context(link.tls_cert).cert_store_stats()["x509"] == 1


def test_existing_link_table_gains_the_certificate_column():
    from sqlalchemy import create_engine, text
    eng = create_engine("sqlite://")
    with eng.begin() as conn:
        conn.execute(text(
            "CREATE TABLE peer_links (install_id VARCHAR(64) PRIMARY KEY, label VARCHAR(80), "
            "base_url TEXT, public_key TEXT, key_id VARCHAR(64), state VARCHAR(16), "
            "export_bots TEXT DEFAULT '[]', created_at DOUBLE PRECISION, "
            "updated_at DOUBLE PRECISION)"))
        store.ensure_schema(conn)
        store.ensure_schema(conn)  # idempotent
        store.upsert_pending(conn, install_id="p", label="P", base_url="https://p",
                             public_key="k", key_id="kid", tls_cert="pem")
        assert store.get_link(conn, "p").tls_cert == "pem"
