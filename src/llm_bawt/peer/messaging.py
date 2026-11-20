"""Bot messages across paired installs (TASK-2517).

Remote bots are addressed as ``slug@<install_id>`` (``qualified``). They are
never local bots: nothing here creates a profile or grants a local privilege.

Outbound: ``bots_send_message`` to a qualified target is admitted into
``peer_outbox`` (idempotent on the caller's key or the generated delivery ID)
and a pump pushes it with the same key, so a retry after an uncertain result
is deduplicated by the receiver, never resent as a new message. Terminal
states are honest: ``SENT`` (the peer admitted it, with its delivery ID),
``REJECTED`` (the peer refused permanently: revoked, not shared, bad request)
and ``FAILED`` (admission still unconfirmed after ``MAX_ATTEMPTS``; an earlier
attempt may have landed with its answer lost). A temporary refusal is retried.

Inbound: the peer endpoint's ``peer_message_send`` admits an ordinary durable
delivery to an exported local bot. The author is the reserved ``system``
user; who it came from is bound to the authenticated peer in server-owned
``peer_origin`` metadata, so the reply note tells the recipient to answer
``slug@<peer install>``. A peer cannot address another peer through us: the
target must be one of our own exported bots.
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text

from . import store

log = logging.getLogger(__name__)

MAX_MESSAGE = 16_000
MAX_ATTEMPTS = 8
BACKOFF = (5, 15, 30, 60, 120, 300, 600, 900)
ORIGIN_KEY = "peer_origin"
_SLUG = re.compile(r"^[a-z0-9_][a-z0-9_-]{0,63}$")
_INSTALL = re.compile(r"^[0-9a-f-]{8,64}$")

QUEUED, SENT, REJECTED, FAILED = "QUEUED", "SENT", "REJECTED", "FAILED"

_DDL = """CREATE TABLE IF NOT EXISTS peer_outbox (
  id VARCHAR(256) PRIMARY KEY,
  install_id VARCHAR(64) NOT NULL,
  target_slug VARCHAR(64) NOT NULL,
  sender_slug VARCHAR(64) NOT NULL,
  message TEXT NOT NULL,
  state VARCHAR(16) NOT NULL,
  attempts INTEGER NOT NULL DEFAULT 0,
  next_at DOUBLE PRECISION NOT NULL,
  remote_delivery_id VARCHAR(128) NULL,
  last_error TEXT NULL,
  created_at DOUBLE PRECISION NOT NULL,
  updated_at DOUBLE PRECISION NOT NULL)"""


def ensure_schema(conn) -> None:
    conn.execute(text(_DDL))


@dataclass(frozen=True)
class RemoteTarget:
    slug: str
    install_id: str

    @property
    def qualified(self) -> str:
        return f"{self.slug}@{self.install_id}"


def parse_target(target: str) -> RemoteTarget | None:
    """``slug@install_id`` -> RemoteTarget; anything else is not remote."""
    if "@" not in (target or ""):
        return None
    slug, _, install = target.strip().lower().partition("@")
    if not _SLUG.match(slug) or not _INSTALL.match(install):
        raise ValueError(f"not a remote bot address: {target!r}")
    return RemoteTarget(slug, install)


class OutboundRefused(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def wire_id(sender: str, key: str | None) -> str:
    """The outbox ID, also the wire idempotency key: stable and always 45 chars.

    A caller key of any accepted length maps to the same bounded ID (scoped to
    the sender), so what is stored is always what the peer accepts.
    """
    if not key:
        return f"peer:{uuid.uuid4().hex}"
    return "peer:" + hashlib.sha256(f"{sender}\0{key}".encode()).hexdigest()[:40]


def enqueue(conn, *, target: RemoteTarget, sender: str, message: str, key: str | None,
            now: float | None = None) -> tuple[dict, bool]:
    """Admit one outbound message. Returns (row, duplicate).

    Conflict-safe: a concurrent same-key insert loses quietly and re-reads the
    winner, which is returned when it is the same message and refused when not.
    """
    now = time.time() if now is None else now
    link = store.get_link(conn, target.install_id)
    if link is None:
        raise OutboundRefused(404, f"no paired install {target.install_id}")
    if link.state != "active":
        raise OutboundRefused(409, f"install {link.label} is {link.state}")
    if not message.strip() or len(message) > MAX_MESSAGE:
        raise OutboundRefused(422, f"message must be 1-{MAX_MESSAGE} characters")
    row_id = wire_id(sender, key)
    inserted = conn.execute(text(
        "INSERT INTO peer_outbox (id, install_id, target_slug, sender_slug, message, state, "
        "attempts, next_at, created_at, updated_at) "
        "VALUES (:id, :i, :t, :s, :m, 'QUEUED', 0, :n, :n, :n) ON CONFLICT (id) DO NOTHING"),
        {"id": row_id, "i": target.install_id, "t": target.slug, "s": sender, "m": message,
         "n": now}).rowcount
    row = get(conn, row_id)
    if inserted:
        return row, False
    same = (row["install_id"], row["target_slug"], row["sender_slug"],
            row["message"]) == (target.install_id, target.slug, sender, message)
    if not same:
        raise OutboundRefused(409, "idempotency key reused for a different message")
    return row, True


def get(conn, row_id: str) -> dict | None:
    row = conn.execute(text("SELECT * FROM peer_outbox WHERE id = :id"), {"id": row_id}).first()
    return dict(row._mapping) if row else None


def receipt(row: dict, link: store.PeerLink | None = None) -> dict:
    """The sender's view. ``QUEUED`` means held here, not yet acknowledged by the peer."""
    return {"delivery_id": row["id"], "status": row["state"],
            "target_bot_id": f"{row['target_slug']}@{row['install_id']}",
            "peer": link.label if link else None, "remote_delivery_id": row["remote_delivery_id"],
            "attempts": row["attempts"], "last_error": row["last_error"], "remote": True}


def lookup(engine, row_id: str, *, senders: frozenset[str] | None = None) -> dict | None:
    """A receipt for ``row_id``; with ``senders``, only a row one of them sent."""
    with engine.begin() as conn:
        row = get(conn, row_id)
        if row is None or (senders is not None and row["sender_slug"] not in senders):
            return None
        return receipt(row, store.get_link(conn, row["install_id"]))


def _due(conn, now: float, limit: int) -> list[dict]:
    rows = conn.execute(text("SELECT * FROM peer_outbox WHERE state = 'QUEUED' AND next_at <= :n "
                             "ORDER BY created_at LIMIT :l"), {"n": now, "l": limit})
    return [dict(r._mapping) for r in rows]


def _settle(conn, row_id: str, **values: Any) -> None:
    values["updated_at"] = time.time()
    sets = ", ".join(f"{k} = :{k}" for k in values)
    conn.execute(text(f"UPDATE peer_outbox SET {sets} WHERE id = :id AND state = 'QUEUED'"),
                 {**values, "id": row_id})


async def pump_once(engine, *, now: float | None = None, limit: int = 20) -> int:
    """Push due outbound rows once. Returns how many were attempted."""
    from . import client, identity

    now = time.time() if now is None else now
    with engine.begin() as conn:
        own = identity.load_or_create(conn)
        rows = _due(conn, now, limit)
        links = {r["install_id"]: store.get_link(conn, r["install_id"]) for r in rows}
    for row in rows:
        link = links.get(row["install_id"])
        if link is None or link.state != "active":
            with engine.begin() as conn:
                _settle(conn, row["id"], state=REJECTED,
                        last_error=f"peer is {link.state if link else 'unknown'}")
            continue
        args = {"target": row["target_slug"], "sender": row["sender_slug"],
                "message": row["message"], "idempotency_key": row["id"]}
        try:
            got = await client.call(own, link, "peer_message_send", args)
        except client.PeerUnavailable as exc:
            got = {"retry": True, "error": exc.reason, "status": exc.status}
        delivery_id = str(got.get("delivery_id") or "")[:128]
        attempts = row["attempts"] + 1
        with engine.begin() as conn:
            if got.get("accepted") is True and delivery_id:
                _settle(conn, row["id"], state=SENT, attempts=attempts,
                        remote_delivery_id=delivery_id, last_error=None)
            elif got.get("status") in (401, 403) or (not got.get("retry")
                                                     and got.get("accepted") is False):
                _settle(conn, row["id"], state=REJECTED, attempts=attempts,
                        last_error=str(got.get("error") or "refused")[:500])
            elif attempts >= MAX_ATTEMPTS:
                # An earlier attempt may have landed with its answer lost: unconfirmed, not "never".
                _settle(conn, row["id"], state=FAILED, attempts=attempts,
                        last_error=("admission not confirmed after "
                                    f"{attempts} attempts: {got.get('error') or 'no answer'}")[:500])
            else:  # transient, or an answer without a delivery ID: same key next time
                _settle(conn, row["id"], state=QUEUED, attempts=attempts,
                        last_error=str(got.get("error") or "peer gave no delivery ID")[:500],
                        next_at=now + BACKOFF[min(attempts, len(BACKOFF)) - 1])
    return len(rows)


# --- inbound -------------------------------------------------------------

def check_inbound(link: store.PeerLink, args: dict) -> tuple[str, str, str, str]:
    """Validate a peer's send against its *current* grant. Returns fields or raises."""
    target = str(args.get("target") or "").strip().lower()
    sender = str(args.get("sender") or "").strip().lower()
    message = str(args.get("message") or "")
    key = str(args.get("idempotency_key") or "")
    if not _SLUG.match(target) or not _SLUG.match(sender):
        raise OutboundRefused(422, "target and sender must be bot slugs")
    if not message.strip() or len(message) > MAX_MESSAGE:
        raise OutboundRefused(422, f"message must be 1-{MAX_MESSAGE} characters")
    if not 1 <= len(key) <= 200:
        raise OutboundRefused(422, "idempotency_key is required")
    if target not in link.export_bots:
        raise OutboundRefused(403, "that bot is not shared with your install")
    return target, sender, message, key


def inbound_text(link: store.PeerLink, sender: str, message: str) -> str:
    return (f"[From {sender}@{link.label}: a bot on another BawtHub install, "
            f"as named by that install]\n\n{message}")


def origin(link: store.PeerLink, sender: str) -> dict:
    return {"install_id": link.install_id, "label": link.label, "sender": sender}


def reply_address(metadata: dict | None) -> str | None:
    """The qualified address a peer-originated delivery should be answered at."""
    o = (metadata or {}).get(ORIGIN_KEY)
    if not isinstance(o, dict):
        return None
    sender, install = str(o.get("sender") or ""), str(o.get("install_id") or "")
    if _SLUG.match(sender) and _INSTALL.match(install):
        return f"{sender}@{install}"
    return None
