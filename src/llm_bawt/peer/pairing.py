"""Single-use pairing invitations between two installs (TASK-2517).

Flow (B invites, A joins):

1. B's owner creates an invite: a code naming B's peer URL, install ID, key
   fingerprint and a random nonce. Only the nonce hash is stored; it expires
   after ``INVITE_TTL`` and can be used once.
2. A's owner pastes the code. A fetches B's public identity and refuses unless
   it matches the fingerprint and install ID in the code (so a wrong URL or a
   substituted server is caught), shows B's label and fingerprint, and on the
   owner's confirmation records B as ``active`` and sends ``/peer/pair``.
3. ``/peer/pair`` carries A's identity, the nonce and an assertion signed by
   A's key (proof that A holds the key it registers). B consumes the nonce and
   records A as ``pending``; B's owner then confirms A's fingerprint.

Neither side exports any bot until its owner chooses; each direction's grant is
set separately. A revoked peer can only come back through a new invite.
"""
from __future__ import annotations

import base64
import json
import os
import secrets
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

from sqlalchemy import text

from . import identity, store, tls

INVITE_TTL = 600
CODE_PREFIX = "bawthub-peer:"
# Peers talk over HTTPS. Plain HTTP is only for a link that is already
# encrypted and authenticated underneath (a WireGuard/Tailscale tunnel) or for
# tests, and the owner opts into it explicitly on both installs.
ALLOW_HTTP_ENV = "LLM_BAWT_PEER_ALLOW_HTTP"


class PairingError(Exception):
    def __init__(self, reason: str, status: int = 400):
        super().__init__(reason)
        self.reason = reason
        self.status = status


@dataclass(frozen=True)
class Invite:
    base_url: str
    install_id: str
    fingerprint: str
    nonce: str
    label: str = ""

    def code(self) -> str:
        body = json.dumps({"v": 1, "url": self.base_url, "id": self.install_id,
                           "fp": self.fingerprint, "n": self.nonce, "l": self.label},
                          separators=(",", ":"))
        return CODE_PREFIX + base64.urlsafe_b64encode(body.encode()).decode().rstrip("=")


def allow_http() -> bool:
    return os.environ.get(ALLOW_HTTP_ENV, "").strip().lower() in ("1", "true", "yes")


def normalize_base_url(url: str) -> str:
    parts = urlsplit((url or "").strip())
    schemes = ("https", "http") if allow_http() else ("https",)
    if parts.scheme not in schemes or not parts.netloc:
        raise PairingError("peer URL must be https://host[:port]")
    return f"{parts.scheme}://{parts.netloc}{parts.path.rstrip('/')}"


def parse_code(code: str) -> Invite:
    code = (code or "").strip()
    if not code.startswith(CODE_PREFIX):
        raise PairingError("not a BawtHub peer invite")
    raw = code[len(CODE_PREFIX):]
    try:
        body = json.loads(base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)))
        if body.get("v") != 1:
            raise ValueError
        return Invite(normalize_base_url(body["url"]), str(body["id"]), str(body["fp"]),
                      str(body["n"]), str(body.get("l") or "")[:80])
    except PairingError:
        raise
    except Exception as exc:  # noqa: BLE001 - any decode failure is a bad code
        raise PairingError("invite code is damaged") from exc


def create_invite(conn, ident: identity.Identity, base_url: str,
                  *, now: float | None = None) -> tuple[Invite, float]:
    now = time.time() if now is None else now
    nonce = identity.new_invite_nonce()
    expires = now + INVITE_TTL
    conn.execute(text("DELETE FROM peer_invites WHERE expires_at < :n"), {"n": now - 86400})
    conn.execute(text("INSERT INTO peer_invites (nonce_hash, expires_at) VALUES (:h, :e)"),
                 {"h": identity.nonce_hash(nonce), "e": expires})
    return Invite(normalize_base_url(base_url), ident.install_id, ident.fingerprint, nonce,
                  ident.label), expires


def peer_certificate(base_url: str, cert_pem, public_key: str) -> str | None:
    """The certificate to pin for an HTTPS peer: it must carry the peer's identity key.

    Plain HTTP (only when explicitly allowed) has no certificate to pin.
    """
    if not base_url.startswith("https://"):
        return None
    try:
        return tls.require_key(str(cert_pem or ""), public_key)
    except tls.CertificateError as exc:
        raise PairingError(f"peer TLS certificate refused: {exc}", 409) from exc


def new_challenge() -> str:
    return secrets.token_urlsafe(24)


def identity_proof(own: identity.Identity, challenge: str) -> str:
    return own.sign(identity.IDENTITY, f"challenge:{challenge}")


def check_remote_identity(invite: Invite, remote: dict, challenge: str) -> None:
    """A's check that the server it reached holds the key the invite names.

    The key must match the invite's fingerprint and the response must carry a
    fresh signature over A's random challenge made with that key, so an
    endpoint that merely echoes the real peer's public identity fails.
    """
    key = str(remote.get("public_key") or "")
    if str(remote.get("install_id")) != invite.install_id or \
            identity.fingerprint(key) != invite.fingerprint:
        raise PairingError("the peer's identity does not match the invite", 409)
    try:
        identity.verify_signed(str(remote.get("proof") or ""), key, typ=identity.IDENTITY,
                               issuer=invite.install_id, audience=f"challenge:{challenge}")
    except identity.PeerAuthError as exc:
        raise PairingError("the peer did not prove it holds the invite's key", 409) from exc


def accept_pair_request(conn, own: identity.Identity, body: dict,
                        *, now: float | None = None) -> store.PeerLink:
    """B's handler for ``/peer/pair``: consume the nonce, record A as pending."""
    now = time.time() if now is None else now
    try:
        nonce = str(body["nonce"])
        peer_id = str(body["install_id"])
        label = str(body.get("label") or "peer")[:80]
        base_url = normalize_base_url(str(body["base_url"]))
        public_key = str(body["public_key"])
        proof = str(body["proof"])
        identity.load_public_key(public_key)
    except PairingError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise PairingError("malformed pairing request") from exc
    tls_cert = peer_certificate(base_url, body.get("tls_cert"), public_key)
    if peer_id == own.install_id:
        raise PairingError("an install cannot pair with itself")
    res = conn.execute(text(
        "UPDATE peer_invites SET used_at=:n, used_by=:p "
        "WHERE nonce_hash=:h AND used_at IS NULL AND expires_at >= :n"),
        {"n": now, "p": peer_id, "h": identity.nonce_hash(nonce)})
    if res.rowcount != 1:
        raise PairingError("invite is expired, used or unknown", 403)
    try:
        claims = identity.verify_signed(proof, public_key, typ=identity.PAIR, issuer=peer_id,
                                        audience=own.install_id, now=now)
    except identity.PeerAuthError as exc:
        raise PairingError("pairing proof is not valid for the presented key", 403) from exc
    if claims.get("nonce") != identity.nonce_hash(nonce) or \
            claims.get("key") != identity.key_id(public_key):
        raise PairingError("pairing proof is bound to another invite or key", 403)
    try:
        return store.upsert_pending(conn, install_id=peer_id, label=label, base_url=base_url,
                                    public_key=public_key, key_id=identity.key_id(public_key),
                                    tls_cert=tls_cert, now=now)
    except ValueError:  # still active here: this owner must revoke before re-pairing
        raise PairingError("already paired with this install; its owner must revoke the "
                           "old link first", 409) from None


def pair_ack(own: identity.Identity, body: dict) -> str:
    """B's signed answer, bound to this invite and to A's presented key."""
    return own.sign(identity.PAIR_ACK, str(body["install_id"]),
                    nonce=identity.nonce_hash(str(body["nonce"])),
                    key=identity.key_id(str(body["public_key"])))


def pair_request_body(own: identity.Identity, invite: Invite, own_base_url: str) -> dict:
    return {"nonce": invite.nonce, "install_id": own.install_id, "label": own.label,
            "base_url": normalize_base_url(own_base_url), "public_key": own.public_key,
            "tls_cert": tls.certificate_pem(own),
            "proof": own.sign(identity.PAIR, invite.install_id,
                              nonce=identity.nonce_hash(invite.nonce), key=own.key_id)}


def check_pair_ack(own: identity.Identity, invite: Invite, remote_key: str, reply: dict) -> None:
    """A's check that the pairing answer came from the pinned key for this invite."""
    try:
        claims = identity.verify_signed(str(reply.get("ack") or ""), remote_key,
                                        typ=identity.PAIR_ACK, issuer=invite.install_id,
                                        audience=own.install_id)
    except identity.PeerAuthError as exc:
        raise PairingError("the peer's pairing answer is not signed by its key", 409) from exc
    if claims.get("nonce") != identity.nonce_hash(invite.nonce) or claims.get("key") != own.key_id:
        raise PairingError("the peer's pairing answer is for another invite", 409)
