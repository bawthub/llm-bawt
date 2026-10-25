"""Bot fan-out batch identity and rate limit (TASK-1431 design r3 §1.2, §1.3).

A batch is identified by ``metadata_json.batch_id`` on each of its rows. A bot
sender may start at most ``LIMIT`` new batches per ``WINDOW_MINUTES``; a retry
of a batch that already has rows is not a new batch. The count and the batch's
own inserts are serialized per sender by a transaction-scoped advisory lock
held on one dedicated connection for the whole batch, so two concurrent
batches cannot both read "2 so far".
"""
from __future__ import annotations

import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from sqlalchemy import text as sa_text

LIMIT = 3
WINDOW_MINUTES = 10
LOCK_TIMEOUT = "10s"
_LOCK_NAMESPACE = "inter-bot-fanout"
# Fixed namespace so a keyed batch id is stable across processes and restarts.
_BATCH_NAMESPACE = uuid.UUID("6f1b7c2e-1431-5a0e-9d3f-b4c1e2a7f00d")


class FanoutRateLimited(Exception):
    """The sender already started ``LIMIT`` batches in the window."""

    def __init__(self, sender: str, recent: int):
        self.sender = sender
        self.recent = recent
        super().__init__(
            f"{sender!r} started {recent} fan-out batches in the last {WINDOW_MINUTES} "
            f"minutes (limit {LIMIT}); retry later or reuse a batch's idempotency_key"
        )


def batch_id_for(sender: str, idempotency_key: str | None) -> str:
    """uuid5(sender, key) when keyed, so a retry names the same batch; else uuid4."""
    key = (idempotency_key or "").strip()
    if not key:
        return str(uuid.uuid4())
    return str(uuid.uuid5(_BATCH_NAMESPACE, f"{sender}\x00{key}"))


class FanoutLedger:
    """Reads a sender's batches on one connection."""

    def __init__(self, conn):
        self._conn = conn

    def recorded(self, sender: str, batch_id: str) -> Mapping[str, Any] | None:
        """The ``metadata_json.batch`` block of an existing row of this batch."""
        row = self._conn.execute(sa_text("""
            SELECT metadata_json->'batch' AS batch FROM inter_bot_deliveries
            WHERE sender_bot_id=:sender AND metadata_json->>'batch_id' = :batch_id
            ORDER BY ordinal LIMIT 1
        """), {"sender": sender, "batch_id": batch_id}).first()
        if row is None:
            return None
        return row.batch if isinstance(row.batch, Mapping) else {}

    def recent_batches(self, sender: str, batch_id: str) -> int:
        """Distinct other batches this bot sender started inside the window."""
        return int(self._conn.execute(sa_text("""
            SELECT count(DISTINCT metadata_json->>'batch_id') FROM inter_bot_deliveries
            WHERE sender_bot_id=:sender AND author_entity_type='bot'
              AND metadata_json ? 'batch_id'
              AND metadata_json->>'batch_id' <> :batch_id
              AND created_at > now() - (:window * interval '1 minute')
        """), {"sender": sender, "batch_id": batch_id, "window": WINDOW_MINUTES}).scalar() or 0)

    def check(self, sender: str, batch_id: str) -> None:
        recent = self.recent_batches(sender, batch_id)
        if recent >= LIMIT:
            raise FanoutRateLimited(sender, recent)


@contextmanager
def sender_batch_lock(engine, sender: str) -> Iterator[FanoutLedger]:
    """Hold the per-sender batch lock until the block (the whole batch) ends."""
    with engine.connect() as conn, conn.begin():
        conn.execute(sa_text(f"SET LOCAL lock_timeout = '{LOCK_TIMEOUT}'"))
        conn.execute(sa_text(
            "SELECT pg_advisory_xact_lock(hashtext(:namespace), hashtext(:sender))"
        ), {"namespace": _LOCK_NAMESPACE, "sender": sender})
        yield FanoutLedger(conn)


@contextmanager
def batch_reader(engine) -> Iterator[FanoutLedger]:
    """An unlocked ledger, for callers that are not rate limited."""
    with engine.connect() as conn:
        yield FanoutLedger(conn)
