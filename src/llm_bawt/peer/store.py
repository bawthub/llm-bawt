"""Peer installation state (TASK-2517).

Three tables, all owned by this install:

* ``peer_identity``: this install's single identity row. ``install_id`` is a
  random UUID chosen once and never derived from a key, so rotating the
  signing key keeps the identity. The Ed25519 private key is stored encrypted
  with the existing provider-credential envelope.
* ``peer_links``: one row per paired installation, with its current public
  key, the base URL of its peer endpoint, its state and the grants this side
  gives it (``export_bots``: which of our bots it may see and message).
  Grants are read on every inbound call, so a revoke or a grant change takes
  effect for tokens already issued.
* ``peer_invites``: single-use, short-lived pairing invitations (hash only).
* ``peer_jti``: seen assertion IDs until they expire (replay protection).

Timestamps are Unix seconds so the schema is portable to the SQLite tests.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text

PEER_STATES = ("pending", "active", "revoked")

_DDL = (
    """CREATE TABLE IF NOT EXISTS peer_identity (
      id INTEGER PRIMARY KEY CHECK (id = 1),
      install_id VARCHAR(64) NOT NULL,
      label VARCHAR(80) NOT NULL,
      signing_key TEXT NOT NULL,
      key_id VARCHAR(64) NOT NULL,
      created_at DOUBLE PRECISION NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS peer_links (
      install_id VARCHAR(64) PRIMARY KEY,
      label VARCHAR(80) NOT NULL,
      base_url TEXT NOT NULL,
      public_key TEXT NOT NULL,
      key_id VARCHAR(64) NOT NULL,
      state VARCHAR(16) NOT NULL,
      export_bots TEXT NOT NULL DEFAULT '[]',
      created_at DOUBLE PRECISION NOT NULL,
      updated_at DOUBLE PRECISION NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS peer_invites (
      nonce_hash VARCHAR(64) PRIMARY KEY,
      expires_at DOUBLE PRECISION NOT NULL,
      used_at DOUBLE PRECISION NULL,
      used_by VARCHAR(64) NULL)""",
    """CREATE TABLE IF NOT EXISTS peer_jti (
      jti VARCHAR(128) PRIMARY KEY,
      expires_at DOUBLE PRECISION NOT NULL)""",
)


def ensure_schema(conn) -> None:
    for ddl in _DDL:
        conn.execute(text(ddl))


@dataclass(frozen=True)
class PeerLink:
    install_id: str
    label: str
    base_url: str
    public_key: str
    key_id: str
    state: str
    export_bots: tuple[str, ...] = field(default_factory=tuple)

    def public(self) -> dict[str, Any]:
        return {"install_id": self.install_id, "label": self.label, "base_url": self.base_url,
                "key_id": self.key_id, "state": self.state, "export_bots": list(self.export_bots)}


def _link(row) -> PeerLink:
    m = row._mapping
    bots = json.loads(m["export_bots"] or "[]")
    return PeerLink(m["install_id"], m["label"], m["base_url"], m["public_key"], m["key_id"],
                    m["state"], tuple(str(b) for b in bots))


def get_link(conn, install_id: str) -> PeerLink | None:
    row = conn.execute(text("SELECT * FROM peer_links WHERE install_id = :i"),
                       {"i": install_id}).first()
    return _link(row) if row else None


def list_links(conn) -> list[PeerLink]:
    return [_link(r) for r in conn.execute(text("SELECT * FROM peer_links ORDER BY label"))]


def upsert_pending(conn, *, install_id: str, label: str, base_url: str, public_key: str,
                   key_id: str, now: float | None = None) -> PeerLink:
    """Record a peer as ``pending``. A revoked peer must be re-invited, never revived here."""
    now = time.time() if now is None else now
    existing = get_link(conn, install_id)
    if existing and existing.state == "active":
        raise ValueError("peer already paired")
    if existing:
        conn.execute(text(
            "UPDATE peer_links SET label=:l, base_url=:u, public_key=:k, key_id=:kid, "
            "state='pending', export_bots='[]', updated_at=:n WHERE install_id=:i"),
            {"l": label, "u": base_url, "k": public_key, "kid": key_id, "n": now, "i": install_id})
    else:
        conn.execute(text(
            "INSERT INTO peer_links (install_id, label, base_url, public_key, key_id, state, "
            "export_bots, created_at, updated_at) VALUES (:i,:l,:u,:k,:kid,'pending','[]',:n,:n)"),
            {"i": install_id, "l": label, "u": base_url, "k": public_key, "kid": key_id, "n": now})
    link = get_link(conn, install_id)
    assert link is not None
    return link


def set_state(conn, install_id: str, state: str) -> bool:
    if state not in PEER_STATES:
        raise ValueError(f"unknown peer state {state!r}")
    res = conn.execute(text("UPDATE peer_links SET state=:s, updated_at=:n WHERE install_id=:i"),
                       {"s": state, "n": time.time(), "i": install_id})
    return res.rowcount == 1


def set_export(conn, install_id: str, bots: list[str]) -> bool:
    clean = sorted({b.strip().lower() for b in bots if b and b.strip()})
    res = conn.execute(text("UPDATE peer_links SET export_bots=:b, updated_at=:n "
                            "WHERE install_id=:i"),
                       {"b": json.dumps(clean), "n": time.time(), "i": install_id})
    return res.rowcount == 1


def set_shared(conn, install_id: str, bot: str, shared: bool) -> list[str] | None:
    """Share or unshare one bot, leaving the rest as stored now. None if unknown."""
    lock = " FOR UPDATE" if conn.dialect.name == "postgresql" else ""
    row = conn.execute(text(f"SELECT export_bots FROM peer_links WHERE install_id=:i{lock}"),
                       {"i": install_id}).first()
    if row is None:
        return None
    bots = set(json.loads(row[0] or "[]"))
    slug = bot.strip().lower()
    (bots.add if shared else bots.discard)(slug)
    set_export(conn, install_id, list(bots))
    return sorted(bots)


def new_install_id() -> str:
    return str(uuid.uuid4())
