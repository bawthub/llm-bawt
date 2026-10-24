"""The wire shape of a delivery lifecycle event (pure projection, no I/O)."""

from __future__ import annotations

import time
from typing import Any

from ..inter_bot_delivery import DELIVERED, DISPATCHING, FAILED, QUEUED, DeliveryRecord
from ..inter_bot_delivery_retry import retry_original_author
from ..inter_bot_delivery_steer_fence import steer_uncertain

_MAINTENANCE_STATES = {QUEUED: "queued", DISPATCHING: "started", DELIVERED: "completed",
                       FAILED: "refused", "CANCELLED": "cancelled"}


def delivery_event(record: DeliveryRecord, *, default_user: str) -> dict[str, Any]:
    """The ``inter_bot_delivery`` event for ``record``, routed to its owning user."""
    event = {
        "_type": "inter_bot_delivery",
        "delivery_id": record.id,
        "status": record.status,
        "sender_bot_id": record.sender_bot_id,
        "target_bot_id": record.target_bot_id,
        "bot_id": record.sender_bot_id,
        "user_id": default_user,
        "turn_id": record.turn_id,
        "target_turn_id": record.target_turn_id,
        "delivery_mode": record.delivery_mode,
        "reply_mode": record.reply_mode,
        "session_policy": record.session_policy,
        "reset_status": record.reset_status,
        "reset_reason": record.reset_reason,
        "old_session_id": record.old_session_id,
        "new_session_id": record.new_session_id,
        "reset_at": record.reset_at.isoformat() if record.reset_at else None,
        "overflow_recovery_count": record.overflow_recovery_count,
        "attempt_count": record.attempt_count,
        "next_retry_at": record.next_retry_at.isoformat() if record.next_retry_at else None,
        "last_error": record.last_error,
        "project_id": record.project_id,
        "task_id": record.task_id,
        "message_kind": record.message_kind,
        "retry_of_delivery_id": (record.metadata.get("manual_retry_of")
                                 or record.metadata.get("restart_retry_of")),
        "steer_uncertain": steer_uncertain(record),
        # TASK-1300: publishing is async, so two events for one row can
        # arrive out of order; consumers drop a lower revision.
        "revision": record.revision,
        "ts": time.time(),
    }
    if record.metadata.get("prompt_schedule"):
        event["user_id"] = record.author_entity_id
        event["bot_id"] = record.target_bot_id
        event["prompt_schedule"] = record.metadata["prompt_schedule"]
    chat_user = None
    if record.metadata.get("ordinary_chat"):
        # Ordinary queued sends belong to the selected user, not the
        # service's default user or the inter-bot sender namespace. A
        # system-authored restart child (TASK-1231B) belongs to the user
        # whose request it carries.
        if record.author_entity_type == "user":
            chat_user = record.author_entity_id
        else:
            original = retry_original_author(record)
            if original is not None and original.entity_type == "user":
                chat_user = original.entity_id
    if chat_user:
        event.update(
            bot_id=record.target_bot_id, user_id=chat_user,
            ordinary_chat=True, user_message_id=record.user_message_id,
            session_id=record.metadata["ordinary_chat"].get("session_id"),
        )
    if record.metadata.get("maintenance_user_queue"):
        event.update(bot_id=record.target_bot_id, user_id=record.author_entity_id)
    return event


def maintenance_steer_event(record: DeliveryRecord) -> dict[str, Any] | None:
    """The ``steer_lifecycle`` companion a maintenance user-queue row also publishes."""
    if not record.metadata.get("maintenance_user_queue"):
        return None
    state = _MAINTENANCE_STATES.get(record.status)
    if not state:
        return None
    return {"_type": "steer_lifecycle", "message_id": record.user_message_id,
            "bot_id": record.target_bot_id, "user_id": record.author_entity_id, "state": state}
