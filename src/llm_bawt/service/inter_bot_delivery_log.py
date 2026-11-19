"""One payload-free log line per delivery lifecycle event (TASK-1308).

A stuck queue must be traceable from the app log alone. The line carries
identifiers and state only: never message text, and never the raw
``last_error``, which can hold an exception string that echoes content.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ..inter_bot_delivery_record import DeliveryRecord

logger = logging.getLogger("llm_bawt.delivery_event")

#: Stable ``last_error`` prefixes the delivery code writes itself. Anything
#: else is reported as ``other``; its text stays out of this line.
_ERROR_KINDS: tuple[tuple[str, str], ...] = (
    ("Stopped by user", "user_stop"),
    ("delivery attempts exhausted", "attempts_exhausted"),
    ("delivery lease expired", "lease_expired"),
    ("inbox claim expired", "inbox_claim_expired"),
)


def error_kind(last_error: str | None) -> str | None:
    if not last_error:
        return None
    for prefix, kind in _ERROR_KINDS:
        if last_error.startswith(prefix):
            return kind
    return "other"


def _value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if hasattr(value, "isoformat"):
        value = value.isoformat()
    # JSON quoting escapes quotes, newlines and control characters, so the
    # entry can never span lines or forge another field.
    return json.dumps(str(value), ensure_ascii=True)


def delivery_event_line(record: DeliveryRecord, *, steer_uncertain: bool) -> str:
    fields = {
        "delivery_id": record.id,
        "target": record.target_bot_id,
        "sender": record.sender_bot_id,
        "author_type": record.author_entity_type,
        "author": record.author_entity_id,
        "status": record.status,
        "mode": record.delivery_mode,
        "attempt": record.attempt_count,
        "revision": record.revision,
        "next_retry_at": record.next_retry_at,
        "steer_uncertain": steer_uncertain,
        "has_error": bool(record.last_error),
        "error_kind": error_kind(record.last_error),
    }
    return "delivery_event " + " ".join(f"{k}={_value(v)}" for k, v in fields.items())


def log_delivery_event(record: DeliveryRecord) -> None:
    """Never raises: logging must not break the lifecycle it describes."""
    try:
        from ..inter_bot_delivery_steer_fence import steer_uncertain

        uncertain = steer_uncertain(record) is not None
        logger.info("%s", delivery_event_line(record, steer_uncertain=uncertain))
    except Exception:
        logger.warning("could not log delivery event", exc_info=True)
