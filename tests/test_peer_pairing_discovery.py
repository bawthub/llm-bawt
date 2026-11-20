"""Two independent installs pair and discover shared bots (TASK-2517).

Each install has its own database and its own peer listener on a real port;
they only meet over HTTP, as two installations would.
"""
from __future__ import annotations

import asyncio
import socket
import threading
import time

import pytest
import uvicorn
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from llm_bawt.peer import client, identity, messaging, pairing, server, store


@pytest.fixture(autouse=True)
def _plain_key_storage(monkeypatch):
    monkeypatch.setenv(pairing.ALLOW_HTTP_ENV, "1")  # loopback test listeners
    monkeypatch.setattr(identity, "_encrypt", lambda v: "x:" + v)
    monkeypatch.setattr(identity, "_decrypt", lambda v: v[2:])


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Install:
    def __init__(self, label: str, bots: dict[str, str]):
        self.engine = create_engine("sqlite://", poolclass=StaticPool,
                                    connect_args={"check_same_thread": False})
        with self.engine.begin() as conn:
            store.ensure_schema(conn)
            messaging.ensure_schema(conn)
            identity.load_or_create(conn, default_label=label)
        self.admitted: dict[str, dict] = {}
        self.admit_reply: dict | None = None  # set to answer as a refusing receiver
        self.bots = bots
        self.undeliverable: set[str] = set()  # shared bots that lost queued delivery
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self._start()

    def _list(self, slugs, deliverable_only=True):
        return [{"slug": s, "name": self.bots[s], "description": ""}
                for s in slugs if s in self.bots
                and not (deliverable_only and s in self.undeliverable)]

    async def _admit(self, link, target, sender, message, key):
        """Stands in for DeliverySender: idempotent on the peer-scoped key."""
        if self.admit_reply is not None:
            return self.admit_reply
        k = f"peer:{link.install_id}:{key}"
        dup = k in self.admitted
        self.admitted.setdefault(k, {"target": target, "sender": sender, "message": message,
                                     "from": link.install_id, "id": f"d{len(self.admitted)}"})
        return {"accepted": True, "delivery_id": self.admitted[k]["id"], "duplicate": dup}

    def ident(self) -> identity.Identity:
        with self.engine.begin() as conn:
            return identity.load_or_create(conn)

    def link(self, other: "Install") -> store.PeerLink | None:
        with self.engine.begin() as conn:
            return store.get_link(conn, other.ident().install_id)

    def _start(self):
        app = server.build_app(lambda: self.engine, list_bots=self._list, admit=self._admit)
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port,
                                                    log_level="warning", lifespan="on"))
        threading.Thread(target=self.server.run, daemon=True).start()
        deadline = time.time() + 10  # timing-ok: generous startup hang bound, not a speed claim
        while not self.server.started:
            assert time.time() < deadline, "peer listener did not start"  # timing-ok: hang bound
            time.sleep(0.05)

    def restart(self):
        self._start()

    def stop(self):
        self.server.should_exit = True


@pytest.fixture
def pair_of_installs():
    a = Install("Home", {"nova": "Nova"})
    b = Install("Lab", {"echo": "Echo", "secret": "Secret"})
    yield a, b
    a.stop()
    b.stop()


def _pair(a: Install, b: Install) -> None:
    """B invites, A joins and confirms, B confirms. Mirrors the owner flow."""
    with b.engine.begin() as conn:
        invite, _ = pairing.create_invite(conn, b.ident(), b.url)
    code = invite.code()

    parsed = pairing.parse_code(code)
    challenge = pairing.new_challenge()
    remote = asyncio.run(client.fetch_identity(parsed.base_url, challenge))
    pairing.check_remote_identity(parsed, remote, challenge)
    body = pairing.pair_request_body(a.ident(), parsed, a.url)
    reply = asyncio.run(client.send_pair(parsed.base_url, body))
    pairing.check_pair_ack(a.ident(), parsed, remote["public_key"], reply)
    with a.engine.begin() as conn:
        store.upsert_pending(conn, install_id=remote["install_id"], label=remote["label"],
                             base_url=parsed.base_url, public_key=remote["public_key"],
                             key_id=identity.key_id(remote["public_key"]))
        store.set_state(conn, remote["install_id"], "active")
    assert b.link(a).state == "pending"
    with b.engine.begin() as conn:
        store.set_state(conn, a.ident().install_id, "active")
    return code, body


def _list_from(caller: Install, target: Install) -> dict:
    return asyncio.run(client.call(caller.ident(), caller.link(target), "peer_bots_list"))


def test_pair_then_discover_only_exported_bots(pair_of_installs):
    a, b = pair_of_installs
    _pair(a, b)

    assert _list_from(a, b)["bots"] == []  # nothing shared until B's owner chooses

    with b.engine.begin() as conn:
        store.set_export(conn, a.ident().install_id, ["echo"])
    got = _list_from(a, b)
    assert got["label"] == "Lab"
    assert [x["slug"] for x in got["bots"]] == ["echo"]

    # Each direction is granted separately.
    with a.engine.begin() as conn:
        store.set_export(conn, b.ident().install_id, ["nova"])
    assert [x["slug"] for x in _list_from(b, a)["bots"]] == ["nova"]


def test_invite_is_single_use_and_checks_identity(pair_of_installs):
    a, b = pair_of_installs
    code, body = _pair(a, b)
    with pytest.raises(client.PeerUnavailable, match="expired, used or unknown"):
        asyncio.run(client.send_pair(b.url, body))

    parsed = pairing.parse_code(code)
    challenge = pairing.new_challenge()
    wrong = asyncio.run(client.fetch_identity(a.url, challenge))  # an install that is not B
    with pytest.raises(pairing.PairingError, match="does not match"):
        pairing.check_remote_identity(parsed, wrong, challenge)

    # An endpoint echoing B's real public identity, without B's key, fails.
    real = asyncio.run(client.fetch_identity(b.url, pairing.new_challenge()))
    with pytest.raises(pairing.PairingError, match="did not prove"):
        pairing.check_remote_identity(parsed, real, challenge)


def test_plain_http_needs_explicit_opt_in(monkeypatch):
    monkeypatch.delenv(pairing.ALLOW_HTTP_ENV)
    with pytest.raises(pairing.PairingError, match="https"):
        pairing.normalize_base_url("http://peer.example:9000")
    assert pairing.normalize_base_url("https://peer.example:9000/") == "https://peer.example:9000"


def test_pair_ack_is_bound_to_the_invite(pair_of_installs):
    a, b = pair_of_installs
    with b.engine.begin() as conn:
        invite, _ = pairing.create_invite(conn, b.ident(), b.url)
        other, _ = pairing.create_invite(conn, b.ident(), b.url)
    body = pairing.pair_request_body(a.ident(), invite, a.url)
    reply = asyncio.run(client.send_pair(b.url, body))
    with pytest.raises(pairing.PairingError, match="another invite"):
        pairing.check_pair_ack(a.ident(), other, b.ident().public_key, reply)
    with pytest.raises(pairing.PairingError, match="not signed"):
        pairing.check_pair_ack(a.ident(), invite, a.ident().public_key, reply)


def test_call_assertion_cannot_pair(pair_of_installs):
    a, b = pair_of_installs
    with b.engine.begin() as conn:
        invite, _ = pairing.create_invite(conn, b.ident(), b.url)
    body = pairing.pair_request_body(a.ident(), invite, a.url)
    body["proof"] = a.ident().assertion(b.ident().install_id)
    with pytest.raises(client.PeerUnavailable, match="not valid"):
        asyncio.run(client.send_pair(b.url, body))


def test_owner_routes_require_the_service_token(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from llm_bawt.service.routes import peers

    app = FastAPI()
    app.include_router(peers.router)
    http = TestClient(app)
    for method, path in [("get", "/v1/peers"), ("post", "/v1/peers/invites"),
                         ("post", "/v1/peers/x/revoke"), ("put", "/v1/peers/x/export"),
                         ("post", "/v1/peers/x/confirm"), ("get", "/v1/peers/x/bots")]:
        resp = getattr(http, method)(path, headers={"Authorization": "Bearer not-the-token"})
        assert resp.status_code in (401, 403, 503), (path, resp.status_code)


def test_confirm_needs_the_matching_fingerprint(monkeypatch):
    from types import SimpleNamespace

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from fastapi import HTTPException

    from llm_bawt.service.routes import peers

    eng = create_engine("sqlite://")
    with eng.begin() as conn:
        store.ensure_schema(conn)
        identity.load_or_create(conn, default_label="Me")
        peer = identity.Identity("peer-1", "Peer", Ed25519PrivateKey.generate())
        store.upsert_pending(conn, install_id="peer-1", label="Peer", base_url="http://p",
                             public_key=peer.public_key, key_id=peer.key_id)
    monkeypatch.setattr(peers, "_require_enabled", lambda: SimpleNamespace(engine=lambda: eng))
    want = identity.fingerprint(peer.public_key)

    def confirm(fp):
        return asyncio.run(peers.confirm("peer-1", peers.ConfirmBody(confirm_fingerprint=fp)))

    with pytest.raises(HTTPException) as wrong:
        confirm("AAAA-BBBB")
    assert wrong.value.status_code == 409
    with eng.begin() as conn:
        assert store.get_link(conn, "peer-1").state == "pending"
    with pytest.raises(Exception):
        peers.ConfirmBody()  # the fingerprint is required
    with pytest.raises(Exception):
        peers.ConfirmBody(confirm_fingerprint="")
    with pytest.raises(Exception):
        peers.JoinBody(code="x", confirm_fingerprint="")  # empty is not "preview"
    assert peers.JoinBody(code="x").confirm_fingerprint is None
    assert confirm(f"  {want.lower()} ")["state"] == "active"
    with pytest.raises(HTTPException) as again:
        confirm(want)  # no longer pending
    assert again.value.status_code == 404


def test_sharing_changes_one_bot_at_a_time():
    eng = create_engine("sqlite://")
    with eng.begin() as conn:
        store.ensure_schema(conn)
        store.upsert_pending(conn, install_id="peer-1", label="Peer", base_url="http://p",
                             public_key="k", key_id="kid")
        store.set_export(conn, "peer-1", ["audit"])
        # Tab A unshares Audit; tab B, loaded before that, then shares Helper.
        assert store.set_shared(conn, "peer-1", "audit", False) == []
        assert store.set_shared(conn, "peer-1", "Helper", True) == ["helper"]
        assert store.set_shared(conn, "peer-1", "helper", True) == ["helper"]
        assert store.set_shared(conn, "nobody", "helper", True) is None


def test_only_bots_that_take_queued_delivery_are_shared(monkeypatch):
    from types import SimpleNamespace

    from fastapi import HTTPException

    from llm_bawt import bots, campus_events
    from llm_bawt.peer import server
    from llm_bawt.service.routes import peers

    fleet = {"agent": SimpleNamespace(slug="agent", name="Agent", description="",
                                      agent_backend="claude-code", harness="claude-code"),
             "chatty": SimpleNamespace(slug="chatty", name="Chatty", description="",
                                       agent_backend=None, harness="chat")}
    monkeypatch.setattr(bots, "get_bot", fleet.get)
    monkeypatch.setattr(campus_events, "visibility", lambda: (lambda slug: True))
    assert [b["slug"] for b in server.exported_bots(("agent", "chatty"))] == ["agent"]
    # Sending still reaches admission, which acknowledges a replay it already
    # admitted and refuses a fresh message itself.
    assert [b["slug"] for b in server.exported_bots(("agent", "chatty"), deliverable_only=False)] \
        == ["agent", "chatty"]

    eng = create_engine("sqlite://")
    with eng.begin() as conn:
        store.ensure_schema(conn)
        store.upsert_pending(conn, install_id="peer-1", label="Peer", base_url="http://p",
                             public_key="k", key_id="kid")
    monkeypatch.setattr(peers, "_require_enabled", lambda: SimpleNamespace(engine=lambda: eng))
    with pytest.raises(HTTPException) as refused:
        asyncio.run(peers.set_shared("peer-1", peers.ShareBody(bot="chatty", shared=True)))
    assert refused.value.status_code == 400
    got = asyncio.run(peers.set_shared("peer-1", peers.ShareBody(bot="agent", shared=True)))
    assert got["export_bots"] == ["agent"]
    # Unsharing never needs the bot to be deliverable.
    assert asyncio.run(peers.set_shared("peer-1", peers.ShareBody(bot="chatty", shared=False)))["export_bots"] == ["agent"]


def test_peer_listener_leaves_signals_to_the_app():
    import signal

    from llm_bawt.peer import runtime

    def sentinel(*_):
        pass

    previous = signal.signal(signal.SIGTERM, sentinel)
    try:
        srv = runtime._SecondaryServer(uvicorn.Config(lambda *a: None))
        with srv.capture_signals():  # main thread: the base class would replace it
            assert signal.getsignal(signal.SIGTERM) is sentinel
        base = uvicorn.Server(uvicorn.Config(lambda *a: None))
        with base.capture_signals():  # proves the check is meaningful here
            assert signal.getsignal(signal.SIGTERM) is not sentinel
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_expired_invite_is_refused(pair_of_installs):
    a, b = pair_of_installs
    with b.engine.begin() as conn:
        invite, _ = pairing.create_invite(conn, b.ident(), b.url,
                                          now=time.time() - pairing.INVITE_TTL - 1)
    body = pairing.pair_request_body(a.ident(), invite, a.url)
    with pytest.raises(client.PeerUnavailable, match="expired"):
        asyncio.run(client.send_pair(b.url, body))


def test_pending_and_revoked_peers_are_refused(pair_of_installs):
    a, b = pair_of_installs
    _pair(a, b)
    with b.engine.begin() as conn:
        store.set_export(conn, a.ident().install_id, ["echo"])
        store.set_state(conn, a.ident().install_id, "revoked")
    with pytest.raises(client.PeerUnavailable) as err:
        _list_from(a, b)
    assert err.value.status == 403, err.value.reason


def test_repairing_an_active_peer_says_why(pair_of_installs):
    a, b = pair_of_installs
    _pair(a, b)
    with pytest.raises(client.PeerUnavailable, match="already paired"):
        _pair(a, b)


def test_assertion_rules():
    eng = create_engine("sqlite://")
    with eng.begin() as conn:
        store.ensure_schema(conn)
        me = identity.load_or_create(conn, default_label="Me")
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        peer = identity.Identity("peer-1", "Peer", Ed25519PrivateKey.generate())
        store.upsert_pending(conn, install_id="peer-1", label="Peer", base_url="http://p",
                             public_key=peer.public_key, key_id=peer.key_id)
        store.set_state(conn, "peer-1", "active")

        tok = peer.assertion(me.install_id)
        with pytest.raises(identity.PeerAuthError, match="purpose"):
            identity.verify_assertion(conn, peer.sign(identity.PAIR, me.install_id),
                                      own_install_id=me.install_id)
        assert identity.verify_assertion(conn, tok, own_install_id=me.install_id).label == "Peer"
        with pytest.raises(identity.PeerAuthError, match="replayed"):
            identity.verify_assertion(conn, tok, own_install_id=me.install_id)
        with pytest.raises(identity.PeerAuthError, match="invalid"):
            identity.verify_assertion(conn, peer.assertion("someone-else"),
                                      own_install_id=me.install_id)
        old = peer.assertion(me.install_id, now=time.time() - 600)
        with pytest.raises(identity.PeerAuthError, match="expired"):
            identity.verify_assertion(conn, old, own_install_id=me.install_id)
        impostor = identity.Identity("peer-1", "Peer", Ed25519PrivateKey.generate())
        with pytest.raises(identity.PeerAuthError, match="invalid"):
            identity.verify_assertion(conn, impostor.assertion(me.install_id),
                                      own_install_id=me.install_id)

        # Rotation keeps the install ID.
        rotated = identity.rotate_key(conn, me)
        assert rotated.install_id == me.install_id and rotated.key_id != me.key_id
        assert identity.load_or_create(conn).key_id == rotated.key_id


def test_stored_http_link_obeys_current_transport_policy(pair_of_installs, monkeypatch):
    a, b = pair_of_installs
    _pair(a, b)
    monkeypatch.delenv(pairing.ALLOW_HTTP_ENV)
    opened = []
    monkeypatch.setattr(client, "session", lambda *args: opened.append(args))
    monkeypatch.setattr(client.AssertionAuth, "__init__",
                        lambda *args: opened.append("assertion"))
    with pytest.raises(client.PeerUnavailable, match="not allowed now"):
        _list_from(a, b)
    assert opened == []


def _send(a, b, target, *, key=None, sender="nova", message="hello"):
    with a.engine.begin() as conn:
        row, dup = messaging.enqueue(
            conn, target=messaging.RemoteTarget(target, b.ident().install_id), sender=sender,
            message=message, key=key)
    return row, dup


def _pump(a):
    asyncio.run(messaging.pump_once(a.engine))


def _state(a, row_id):
    with a.engine.begin() as conn:
        return messaging.get(conn, row_id)


def test_message_reply_and_dedupe(pair_of_installs):
    a, b = pair_of_installs
    _pair(a, b)
    with b.engine.begin() as conn:
        store.set_export(conn, a.ident().install_id, ["echo"])
    with a.engine.begin() as conn:
        store.set_export(conn, b.ident().install_id, ["nova"])

    row, dup = _send(a, b, "echo", key="k1")
    assert not dup and row["state"] == "QUEUED"
    _pump(a)
    sent = _state(a, row["id"])
    assert sent["state"] == "SENT" and sent["remote_delivery_id"]
    (got,) = b.admitted.values()
    assert got == {**got, "target": "echo", "sender": "nova", "from": a.ident().install_id}

    # Same key again: no second local row, no second remote delivery.
    again, dup = _send(a, b, "echo", key="k1")
    assert dup and again["id"] == row["id"]
    with pytest.raises(messaging.OutboundRefused, match="different message"):
        _send(a, b, "echo", key="k1", message="changed")
    # An uncertain retry with the same key is deduplicated by the receiver.
    with a.engine.begin() as conn:
        conn.execute(messaging.text("UPDATE peer_outbox SET state='QUEUED' WHERE id=:i"),
                     {"i": row["id"]})
    _pump(a)
    assert len(b.admitted) == 1
    # The recipient's harness changes after admission: discovery hides it, but
    # a replay still reaches admission and is acknowledged, not rejected.
    b.undeliverable.add("echo")
    assert _list_from(a, b)["bots"] == []
    with a.engine.begin() as conn:
        conn.execute(messaging.text("UPDATE peer_outbox SET state='QUEUED' WHERE id=:i"),
                     {"i": row["id"]})
    _pump(a)
    assert _state(a, row["id"])["state"] == "SENT" and len(b.admitted) == 1
    b.undeliverable.clear()

    # Reply travels back the same way, addressed to nova on A.
    reply, _ = _send(b, a, "nova", sender="echo", message="hi back", key="r1")
    _pump(b)
    assert _state(b, reply["id"])["state"] == "SENT"
    assert [v["target"] for v in a.admitted.values()] == ["nova"]


def test_unshared_target_and_revocation_are_rejected(pair_of_installs):
    a, b = pair_of_installs
    _pair(a, b)
    with b.engine.begin() as conn:
        store.set_export(conn, a.ident().install_id, ["echo"])
    row, _ = _send(a, b, "secret")
    _pump(a)
    assert _state(a, row["id"])["state"] == "REJECTED"
    assert "not shared" in _state(a, row["id"])["last_error"]

    with b.engine.begin() as conn:
        store.set_state(conn, a.ident().install_id, "revoked")
    row, _ = _send(a, b, "echo")
    _pump(a)
    final = _state(a, row["id"])
    assert final["state"] == "REJECTED" and b.admitted == {}


def test_offline_peer_retries_then_delivers(pair_of_installs):
    a, b = pair_of_installs
    _pair(a, b)
    with b.engine.begin() as conn:
        store.set_export(conn, a.ident().install_id, ["echo"])
    b.stop()
    time.sleep(0.5)
    row, _ = _send(a, b, "echo", key="off1")
    _pump(a)
    waiting = _state(a, row["id"])
    assert waiting["state"] == "QUEUED" and waiting["attempts"] == 1
    assert "unreachable" in waiting["last_error"]

    # Back online on the same port: the next due attempt delivers once.
    b.restart()
    asyncio.run(messaging.pump_once(a.engine, now=time.time() + 3600))
    assert _state(a, row["id"])["state"] == "SENT"
    assert len(b.admitted) == 1


def test_offline_gives_up_honestly(pair_of_installs):
    a, b = pair_of_installs
    _pair(a, b)
    b.stop()
    time.sleep(0.5)
    row, _ = _send(a, b, "echo", key="dead")
    t = time.time()
    for i in range(messaging.MAX_ATTEMPTS):
        asyncio.run(messaging.pump_once(a.engine, now=t + 10_000 * (i + 1)))
    final = _state(a, row["id"])
    assert final["state"] == "FAILED" and final["attempts"] == messaging.MAX_ATTEMPTS
    assert "not confirmed" in final["last_error"]


def test_remote_address_parsing_and_reply_note():
    t = messaging.parse_target("Echo@0f0e0d0c-0b0a-0908-0706-050403020100")
    assert t.slug == "echo" and t.qualified.startswith("echo@0f0e")
    assert messaging.parse_target("echo") is None
    with pytest.raises(ValueError):
        messaging.parse_target("echo@not an id")
    meta = {messaging.ORIGIN_KEY: {"install_id": "0f0e0d0c-0b0a", "sender": "nova", "label": "x"}}
    assert messaging.reply_address(meta) == "nova@0f0e0d0c-0b0a"
    assert messaging.reply_address({}) is None


def test_discovery_qualifies_remote_bots_and_names_offline_peers(pair_of_installs):
    from llm_bawt.peer import discovery

    a, b = pair_of_installs
    _pair(a, b)
    with b.engine.begin() as conn:
        store.set_export(conn, a.ident().install_id, ["echo"])
    bots, unavailable = asyncio.run(discovery.all_peer_bots(a.engine))
    b_id = b.ident().install_id
    assert [x["id"] for x in bots] == [f"echo@{b_id}"] and unavailable == []
    assert bots[0]["location"] == {"runtime": "peer", "peer": "Lab", "install_id": b_id}

    b.stop()
    time.sleep(0.5)
    bots, unavailable = asyncio.run(discovery.all_peer_bots(a.engine))
    assert bots == [] and [u["install_id"] for u in unavailable] == [b_id]


def test_wire_key_is_bounded_and_sender_scoped(pair_of_installs):
    a, b = pair_of_installs
    _pair(a, b)
    with b.engine.begin() as conn:
        store.set_export(conn, a.ident().install_id, ["echo"])
    for key in ("k", "x" * 196, "y" * 256):
        row, _ = _send(a, b, "echo", key=key, message=f"m{len(key)}")
        assert len(row["id"]) == 45
    _pump(a)
    assert len(b.admitted) == 3
    assert all(r["state"] == "SENT" for r in map(lambda k: _state(a, k), [
        messaging.wire_id("nova", k) for k in ("k", "x" * 196, "y" * 256)]))
    # The same caller key from another sender is a different message, not a conflict.
    other, dup = _send(a, b, "echo", key="k", sender="vex", message="m1")
    assert not dup and other["id"] != messaging.wire_id("nova", "k")


def test_temporary_refusal_retries_and_ack_needs_an_id(pair_of_installs):
    a, b = pair_of_installs
    _pair(a, b)
    with b.engine.begin() as conn:
        store.set_export(conn, a.ident().install_id, ["echo"])
    row, _ = _send(a, b, "echo", key="t1")
    t = time.time()
    b.admit_reply = {"accepted": False, "retry": True, "error": "admission unavailable"}
    asyncio.run(messaging.pump_once(a.engine, now=t))
    assert _state(a, row["id"])["state"] == "QUEUED"
    b.admit_reply = {"accepted": True}  # no delivery ID: not proof of admission
    asyncio.run(messaging.pump_once(a.engine, now=t + 1000))
    assert _state(a, row["id"])["state"] == "QUEUED"
    b.admit_reply = None
    asyncio.run(messaging.pump_once(a.engine, now=t + 5000))
    done = _state(a, row["id"])
    assert done["state"] == "SENT" and done["attempts"] == 3


def _submitted(status, *, duplicate, target_exists=True, last_error=None):
    """DeliverySender's real result shape, from submission_result."""
    from types import SimpleNamespace

    from llm_bawt.inter_bot_delivery import submission_result

    record = SimpleNamespace(status=status, to_api=lambda duplicate=False: {
        "delivery_id": "d1", "status": status, "last_error": last_error, "duplicate": duplicate})
    return submission_result(record, duplicate=duplicate, target_exists=target_exists)


_NOT_DURABLE = "Target bot 'echo' does not support durable when-idle delivery"


@pytest.mark.parametrize(("outcome", "expected"), [
    # A fresh admission and a duplicate of a since-cancelled row are both acknowledged.
    (lambda: _submitted("QUEUED", duplicate=False), {"accepted": True, "delivery_id": "d1"}),
    (lambda: _submitted("CANCELLED", duplicate=True), {"accepted": True, "delivery_id": "d1"}),
    (lambda: _submitted("FAILED", duplicate=True), {"accepted": True, "delivery_id": "d1"}),
    # A fresh row that failed on arrival is a permanent refusal.
    (lambda: _submitted("FAILED", duplicate=False, target_exists=False,
                        last_error="Unknown target bot"), {"accepted": False, "retry": False}),
    # A bot that can't take queued delivery refuses a fresh key, and its replay
    # after a lost response repeats the refusal instead of acknowledging it.
    (lambda: _submitted("FAILED", duplicate=False, last_error=_NOT_DURABLE),
     {"accepted": False, "retry": False}),
    (lambda: _submitted("FAILED", duplicate=True, last_error=_NOT_DURABLE),
     {"accepted": False, "retry": False}),
    # Temporary admission failures are retryable; validation failures are not.
    (503, {"accepted": False, "retry": True}),
    (409, {"accepted": False, "retry": False}),
])
def test_real_admission_edge_maps_outcomes(monkeypatch, outcome, expected):
    """server.admit_local over DeliverySender's own result and refusal shapes."""
    from llm_bawt import inter_bot_send

    seen = {}

    async def fake_admit_one(self, body, request=None):
        seen["body"] = body
        if isinstance(outcome, int):
            raise inter_bot_send.DeliveryRefused(status_code=outcome, detail="nope")
        return outcome()

    monkeypatch.setattr(inter_bot_send.DeliverySender, "_admit_one", fake_admit_one)
    link = store.PeerLink(install_id="0f0e0d0c-0b0a-0908-0706-050403020100", label="Lab",
                          base_url="https://lab.example", public_key="k", key_id="kid",
                          state="active", export_bots=("echo",))
    got = asyncio.run(server.admit_local(link, "echo", "nova", "hi", "peer:abc"))
    assert got == {**got, **expected}
    body = seen["body"]
    assert body.idempotency_key == f"peer:{link.install_id}:peer:abc"
    assert body.metadata[messaging.ORIGIN_KEY]["install_id"] == link.install_id
    assert body.session_policy == "continue"  # stable across a same-key replay
