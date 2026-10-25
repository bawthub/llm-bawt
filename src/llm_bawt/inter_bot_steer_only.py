"""``steer_only`` deliveries: steer a running turn, never start one (TASK-1431).

A steer_only row has nothing to do when its target has no executing turn. It
settles at claim as ``CANCELLED`` with ``delivery_mode='skipped'`` and a
machine-readable ``last_error='not_needed:<reason>'``: no reservation, no idle
fallback turn and no status CHECK change (design r3 §2).
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sqlalchemy import text as sa_text

from .inter_bot_delivery_record import DeliveryRecord

# The send modes every producer names (MCP ``bots_send_message``, the batch
# route). The one definition; the MCP resolver and ``inter_bot_send`` import it.
STEER_OR_IDLE = "steer_or_idle"
WHEN_IDLE = "when_idle"
STEER_ONLY = "steer_only"

STEER_ONLY_KEY = "steer_only"  # the stored payload flag
NOT_NEEDED_PREFIX = "not_needed:"
SKIPPED_MODE = "skipped"

NO_ACTIVE_RUN = "no_active_run"
UNSUPPORTED_BACKEND = "unsupported_backend"


def is_steer_only(payload: Mapping[str, Any] | None) -> bool:
    """Whether a stored delivery payload asks for steering only."""
    return bool(isinstance(payload, Mapping) and payload.get(STEER_ONLY_KEY))


def settle_not_needed(conn, delivery_id: str, reason: str) -> DeliveryRecord | None:
    """Settle a still-QUEUED steer_only row as not needed, on the caller's lock.

    Not an attempt: ``attempt_count`` is untouched. Returns the settled row, or
    None when the row was no longer QUEUED.
    """
    row = conn.execute(sa_text("""
        UPDATE inter_bot_deliveries SET status='CANCELLED', delivery_mode=:mode,
            last_error=:error, next_retry_at=NULL, updated_at=now()
        WHERE id=:id AND status='QUEUED'
        RETURNING *
    """), {"id": delivery_id, "mode": SKIPPED_MODE,
           "error": f"{NOT_NEEDED_PREFIX}{reason}"}).mappings().first()
    return DeliveryRecord.from_mapping(row) if row else None
