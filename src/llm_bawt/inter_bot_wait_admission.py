"""One send-and-wait per target, and no 2-cycle waits (TASK-1253).

A waited delivery (``reply_mode="waited"``) blocks its sender's tool call
until the target's turn replies. Each target turn serves one waiter, so a
second waiter would only sit blocked behind the first until it timed out.
A waiter whose target is itself waiting on it can never be served.

``refuse_waited_conflict`` runs inside the enqueue transaction, after the
target's ``lock_target`` and the new row's insert. Concurrent waiters to one
target therefore serialize: exactly one commits, and the rest raise and roll
back. Async sends never reach it.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import text as sa_text
from sqlalchemy.engine import Connection

from .inter_bot_delivery_query import IN_FLIGHT_STATUSES

TARGET_WAIT_IN_PROGRESS = "target_wait_in_progress"
WAIT_WOULD_DEADLOCK = "wait_would_deadlock"

_ACTIVE_WAITED = """
    SELECT id, sender_bot_id, EXTRACT(EPOCH FROM (now() - created_at)) AS age_seconds
    FROM inter_bot_deliveries
    WHERE target_bot_id = :target AND sender_bot_id = :sender
      AND id <> :exclude
      AND status IN :statuses
      AND payload_json->>'reply_mode' = 'waited'
    ORDER BY ordinal LIMIT 1
"""
_ANY_SENDER = _ACTIVE_WAITED.replace("AND sender_bot_id = :sender", "")


class WaitAdmissionRefused(Exception):
    """A waited send refused at admission; the row was not committed."""

    status_code = 409

    def __init__(self, detail: dict[str, Any]) -> None:
        super().__init__(detail["message"])
        self.detail = detail


def _first(conn: Connection, sql: str, params: dict[str, Any]):
    from sqlalchemy import bindparam

    stmt = sa_text(sql).bindparams(bindparam("statuses", expanding=True))
    return conn.execute(stmt, {**params, "statuses": list(IN_FLIGHT_STATUSES)}).mappings().first()


def refuse_waited_conflict(conn: Connection, *, sender: str, target: str, delivery_id: str) -> None:
    """Raise ``WaitAdmissionRefused`` if this waited send must not be admitted."""
    reverse = _first(conn, _ACTIVE_WAITED, {"target": sender, "sender": target, "exclude": delivery_id})
    if reverse is not None:
        raise WaitAdmissionRefused({
            "error": WAIT_WOULD_DEADLOCK,
            "message": (
                f"{target} is waiting on a reply from {sender} (delivery {reverse['id']}), so "
                f"{sender} waiting on {target} would deadlock. Reply to {target} first, or send "
                "async without wait_for_reply."
            ),
            "waiting_delivery_id": reverse["id"],
        })
    current = _first(conn, _ANY_SENDER, {"target": target, "exclude": delivery_id})
    if current is not None:
        age = int(current["age_seconds"] or 0)
        raise WaitAdmissionRefused({
            "error": TARGET_WAIT_IN_PROGRESS,
            "message": (
                f"{current['sender_bot_id']} is already waiting on {target} (delivery "
                f"{current['id']}, {age}s old). Send async without wait_for_reply and ask "
                f"{target} to message you back."
            ),
            "current_waiter_bot_id": current["sender_bot_id"],
            "current_delivery_id": current["id"],
            "current_age_seconds": age,
        })
