"""This install's peer identity and signed peer assertions (TASK-2517).

Peers authenticate each call with a short-lived JWT signed by the caller's
Ed25519 installation key: the RFC 7523 JWT-bearer assertion shape (``iss`` =
``sub`` = caller install ID, ``aud`` = receiver install ID, ``exp``, ``jti``),
EdDSA via PyJWT and ``cryptography``. No shared secret ever crosses the wire.
The receiver checks the signature against the key it registered at pairing,
the audience, a two-minute lifetime and a one-time ``jti``, and then reads the
peer's current state and grants, so revocation stops tokens already issued.
"""
from __future__ import annotations

import base64
import hashlib
import secrets
import time
import uuid
from dataclasses import dataclass

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from . import store

ASSERTION_TTL = 120
CLOCK_SKEW = 30
_ALG = "EdDSA"


def public_key_text(key: Ed25519PublicKey) -> str:
    raw = key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def load_public_key(value: str) -> Ed25519PublicKey:
    raw = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    if len(raw) != 32:
        raise ValueError("not an Ed25519 public key")
    return Ed25519PublicKey.from_public_bytes(raw)


def fingerprint(public_key: str) -> str:
    """Short, human-comparable digest shown on both sides at pairing."""
    digest = hashlib.sha256(public_key.encode()).digest()
    code = base64.b32encode(digest[:10]).decode()
    return "-".join(code[i:i + 4] for i in range(0, 16, 4))


def key_id(public_key: str) -> str:
    return hashlib.sha256(public_key.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Identity:
    install_id: str
    label: str
    private_key: Ed25519PrivateKey

    @property
    def public_key(self) -> str:
        return public_key_text(self.private_key.public_key())

    @property
    def key_id(self) -> str:
        return key_id(self.public_key)

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.public_key)

    def sign(self, typ: str, audience: str, *, now: float | None = None, **extra) -> str:
        """A short-lived token of one purpose (``typ``) for one audience."""
        now = time.time() if now is None else now
        claims = {"iss": self.install_id, "sub": self.install_id, "aud": audience,
                  "iat": int(now), "exp": int(now) + ASSERTION_TTL,
                  "jti": uuid.uuid4().hex, "typ": typ, **extra}
        return jwt.encode(claims, self.private_key, algorithm=_ALG,
                          headers={"kid": self.key_id})

    def assertion(self, audience: str, *, now: float | None = None) -> str:
        return self.sign(CALL, audience, now=now)


CALL = "peer-call"
PAIR = "peer-pair"
PAIR_ACK = "peer-pair-ack"
IDENTITY = "peer-identity"


def verify_signed(token: str, public_key: str, *, typ: str, issuer: str, audience: str,
                  now: float | None = None) -> dict:
    """Check signature, purpose, issuer, audience and lifetime; return the claims."""
    now = time.time() if now is None else now
    try:
        claims = jwt.decode(token, load_public_key(public_key), algorithms=[_ALG],
                            audience=audience,
                            options={"require": ["iss", "sub", "aud", "exp", "iat", "jti", "typ"],
                                     "verify_exp": False, "verify_iat": False})
    except (jwt.PyJWTError, ValueError) as exc:
        raise PeerAuthError("invalid assertion") from exc
    if claims["typ"] != typ or claims["iss"] != issuer or claims["sub"] != issuer:
        raise PeerAuthError("assertion has the wrong purpose or issuer")
    exp, iat = int(claims["exp"]), int(claims["iat"])
    if exp < now - CLOCK_SKEW or iat > now + CLOCK_SKEW or exp - iat > ASSERTION_TTL:
        raise PeerAuthError("assertion expired or too long-lived")
    return claims


def _encrypt(value: str) -> str:
    from ..service.providers import crypto
    return crypto.encrypt(value)


def _decrypt(value: str) -> str:
    from ..service.providers import crypto
    return crypto.decrypt(value)


def _private_text(key: Ed25519PrivateKey) -> str:
    return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption()).decode()


def load_or_create(conn, *, default_label: str = "BawtHub") -> Identity:
    """Return this install's identity, creating it on first use."""
    row = conn.execute(text("SELECT * FROM peer_identity WHERE id = 1")).first()
    if row is None:
        key = Ed25519PrivateKey.generate()
        ident = Identity(store.new_install_id(), default_label[:80], key)
        conn.execute(text(
            "INSERT INTO peer_identity (id, install_id, label, signing_key, key_id, created_at) "
            "VALUES (1, :i, :l, :k, :kid, :n)"),
            {"i": ident.install_id, "l": ident.label, "k": _encrypt(_private_text(key)),
             "kid": ident.key_id, "n": time.time()})
        return ident
    m = row._mapping
    key = serialization.load_pem_private_key(_decrypt(m["signing_key"]).encode(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("stored peer key is not Ed25519")
    return Identity(m["install_id"], m["label"], key)


def rotate_key(conn, ident: Identity) -> Identity:
    """New signing key, same install ID. Peers must be told the new key."""
    key = Ed25519PrivateKey.generate()
    new = Identity(ident.install_id, ident.label, key)
    conn.execute(text("UPDATE peer_identity SET signing_key=:k, key_id=:kid WHERE id = 1"),
                 {"k": _encrypt(_private_text(key)), "kid": new.key_id})
    return new


class PeerAuthError(Exception):
    def __init__(self, reason: str, status: int = 401):
        super().__init__(reason)
        self.reason = reason
        self.status = status


def verify_assertion(conn, token: str, *, own_install_id: str,
                     now: float | None = None) -> store.PeerLink:
    """Authenticate a peer call; returns the peer's *current* link row."""
    now = time.time() if now is None else now
    try:
        unverified = jwt.decode(token, options={"verify_signature": False})
    except jwt.PyJWTError as exc:
        raise PeerAuthError("malformed assertion") from exc
    issuer = str(unverified.get("iss") or "")
    link = store.get_link(conn, issuer)
    if link is None:
        raise PeerAuthError("unknown peer")
    claims = verify_signed(token, link.public_key, typ=CALL, issuer=issuer,
                           audience=own_install_id, now=now)
    exp = int(claims["exp"])
    jti = f"{issuer}:{claims['jti']}"[:128]
    conn.execute(text("DELETE FROM peer_jti WHERE expires_at < :n"), {"n": now - CLOCK_SKEW})
    try:
        with conn.begin_nested():
            conn.execute(text("INSERT INTO peer_jti (jti, expires_at) VALUES (:j, :e)"),
                         {"j": jti, "e": exp + CLOCK_SKEW})
    except IntegrityError as exc:
        raise PeerAuthError("assertion replayed") from exc
    if link.state != "active":
        raise PeerAuthError(f"peer {link.state}", 403)
    return link


def new_invite_nonce() -> str:
    return secrets.token_urlsafe(24)


def nonce_hash(nonce: str) -> str:
    return hashlib.sha256(nonce.encode()).hexdigest()
