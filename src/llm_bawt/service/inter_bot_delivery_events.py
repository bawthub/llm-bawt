"""The wire shape of a delivery lifecycle event.

:func:`delivery_event` is a pure projection; :func:`lifecycle_event` adds the
one gate read a waiting chat row's ``wait_reason`` needs (TASK-1676).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from ..inter_bot_delivery import DELIVERED, DISPATCHING, FAILED, QUEUED, DeliveryRecord
from ..inter_bot_delivery_ledger import delivery_audience, proven_schedule
from ..inter_bot_delivery_steer_fence import steer_uncertain

logger = logging.getLogger(__name__)

_MAINTENANCE_STATES = {QUEUED: "queued", DISPATCHING: "started", DELIVERED: "completed",
                       FAILED: "refused", "CANCELLED": "cancelled"}


async def lifecycle_event(engine, record: DeliveryRecord, *, default_user: str) -> dict[str, Any]:
    """:func:`delivery_event` plus a waiting chat row's ``wait_reason``.

    A failed gate read publishes the event without it; the UI keeps its turn wording.
    """
    from .delivery_wait_reason import record_wait_reason

    try:
        reason = await asyncio.to_thread(record_wait_reason, engine, record)
    except Exception:  # noqa: BLE001 - the lifecycle event still goes out
        logger.warning("could not read a delivery wait reason", exc_info=True)
        reason = None
    return delivery_event(record, default_user=default_user, wait_reason=reason)


def delivery_event(record: DeliveryRecord, *, default_user: str,
                   wait_reason: dict[str, Any] | None = None) -> dict[str, Any]:
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
        "stopped_after_start": record.status == DELIVERED and record.metadata.get("stopped_after_start") is True,
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
    if wait_reason is not None:
        event["wait_reason"] = wait_reason
    if schedule := proven_schedule(record):
        event["prompt_schedule"] = record.metadata["prompt_schedule"]
        # TASK-1790: the relay graphic and bot-bar pulse read sender_bot_id; a
        # bot-created wake-up is that bot's message, not the scheduler's.
        event["sender_bot_id"] = schedule.origin
    # One audience rule for the live event and its durable receipt (TASK-1745):
    # ordinary sends belong to the selected user (a system restart child to
    # the user whose request it carries, TASK-1231B), not the default user.
    audience = delivery_audience(record, default_user=default_user)
    if audience.ordinary_chat_user:
        event.update(
            ordinary_chat=True, user_message_id=record.user_message_id,
            session_id=record.metadata["ordinary_chat"].get("session_id"),
        )
    event.update(bot_id=audience.bot_id, user_id=audience.user_id)
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
