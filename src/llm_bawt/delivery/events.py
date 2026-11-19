"""Append-only per-attempt execution evidence (``delivery_events``).

The evidence authority for MDv2 (TASK-1306 §4.1–4.2). Distinct from
``inter_bot_delivery_ledger`` (a ``system_events`` transition ledger that
keeps its user-notification role) and ``service/inter_bot_delivery_events``
(the wire/SSE shape).

Writes are idempotent on ``(delivery_id, attempt_no, kind)``. The same key
with a different body is logged as a conflict and the first body stays.
Every write joins the caller's connection so evidence commits with the
transition it explains.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Mapping

from sqlalchemy import text as sa_text

from .state import EventKind

logger = logging.getLogger(__name__)


def ensure_delivery_events_schema(conn) -> None:
    conn.execute(sa_text("""
        CREATE TABLE IF NOT EXISTS delivery_events (
            id BIGSERIAL PRIMARY KEY,
            delivery_id VARCHAR(96) NOT NULL,
            attempt_no INTEGER NOT NULL,
            kind VARCHAR(16) NOT NULL,
            evidence_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            bridge_epoch BIGINT,
            at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (delivery_id, attempt_no, kind)
        )
    """))


def record(conn, delivery_id: str, attempt_no: int, kind: EventKind | str,
           evidence: Mapping[str, Any] | None = None) -> bool:
    """Write one event; True when newly written. Same body twice is a no-op."""
    kind = EventKind(kind).value
    body = json.dumps(dict(evidence or {}), sort_keys=True)
    inserted = conn.execute(sa_text("""
        INSERT INTO delivery_events (delivery_id, attempt_no, kind, evidence_json)
        VALUES (:id, :n, :kind, CAST(:body AS jsonb))
        ON CONFLICT (delivery_id, attempt_no, kind) DO NOTHING
        RETURNING id
    """), {"id": delivery_id, "n": int(attempt_no), "kind": kind, "body": body}).scalar_one_or_none()
    if inserted is not None:
        return True
    existing = conn.execute(sa_text("""
        SELECT evidence_json FROM delivery_events
        WHERE delivery_id=:id AND attempt_no=:n AND kind=:kind
    """), {"id": delivery_id, "n": int(attempt_no), "kind": kind}).scalar_one_or_none()
    if existing is not None and json.dumps(existing, sort_keys=True) != body:
        logger.warning(
            "delivery_event conflict: %s attempt %s %s already recorded with a different body",
            delivery_id, attempt_no, kind,
        )
    return False


def record_event(conn, row: Mapping[str, Any], kind: EventKind | str,
                 evidence: Mapping[str, Any] | None = None) -> bool:
    """``record`` at the row's current attempt."""
    return record(conn, row["id"], int(row.get("attempt_no") or 1), kind, evidence)


def kinds_for(conn, delivery_id: str, attempt_no: int) -> list[EventKind]:
    rows = conn.execute(sa_text("""
        SELECT kind FROM delivery_events WHERE delivery_id=:id AND attempt_no=:n ORDER BY id
    """), {"id": delivery_id, "n": int(attempt_no)}).scalars().all()
    return [EventKind(k) for k in rows]


__all__ = ["ensure_delivery_events_schema", "kinds_for", "record"]
