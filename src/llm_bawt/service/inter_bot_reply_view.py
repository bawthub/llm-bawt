"""Read-only, sender-scoped projection of a durable bot delivery's answer.

The sender's tool-call record is the origin anchor. Never copy target text into
sender history or treat an accepted steer as an attributable answer.
"""
from __future__ import annotations

from typing import Any


class ReplyViewMismatch(ValueError):
    """The requested delivery is outside the originating bot/user scope."""


def _requester(record: Any) -> tuple[str | None, str | None]:
    """Who asked for this delivery: its author, or for a system-authored
    restart-retry child (TASK-1231B) the verified author of the original."""
    from ..inter_bot_delivery_retry import retry_original_author

    original = retry_original_author(record)
    if original is not None:
        return original.entity_type, original.entity_id
    return record.author_entity_type, record.author_entity_id


def _authored_by_row(incoming: dict[str, Any], record: Any) -> bool:
    """The persisted user message carries the delivery row's own author."""
    return (incoming.get("author_entity_type") == record.author_entity_type
            and incoming.get("author_entity_id") == record.author_entity_id)


def inter_bot_reply_view(record: Any, *, sender_bot_id: str, user_id: str,
                         turns: Any, memory_client: Any) -> dict[str, Any]:
    """Project only the canonical assistant message of this delivery's own turn."""
    if (record.sender_bot_id != sender_bot_id
            or _requester(record) != ("bot", sender_bot_id)
            or not user_id
            or any(key in record.metadata for key in (
                "ordinary_chat", "prompt_schedule", "maintenance_user_queue",
            ))
            or record.message_kind == "CONTEXT_MAINTENANCE"):
        raise ReplyViewMismatch("delivery does not belong to this bot conversation")

    view: dict[str, Any] = {
        "delivery_id": record.id,
        "status": record.status,
        "delivery_mode": record.delivery_mode,
        "reply_mode": record.reply_mode,
        "sender_bot_id": record.sender_bot_id,
        "target_bot_id": record.target_bot_id,
        "turn_id": record.turn_id,
        "target_turn_id": record.target_turn_id,
        "last_error": record.last_error if record.status == "FAILED" else None,
        "reply_status": "pending",
        "reply_text": None,
        "target_message_id": None,
        "target_session_id": None,
    }
    if record.status in {"FAILED", "CANCELLED"}:
        view["reply_status"] = "unavailable"
        return view
    if record.status != "DELIVERED":
        return view
    if record.delivery_mode != "turn":
        # STEERING/DELIVERED means transport acceptance into an active turn.
        # Its eventual reply may answer several steers (and a human), so it
        # cannot be attributed to this one delivery without a stronger signal.
        view["reply_status"] = "unattributed"
        incoming = memory_client.get_message_by_id(record.user_message_id) if memory_client else None
        if (incoming and incoming.get("id") == record.user_message_id
                and incoming.get("role") == "user"
                and _authored_by_row(incoming, record)
                and incoming.get("session_id")):
            view["target_message_id"] = record.user_message_id
            view["target_session_id"] = incoming["session_id"]
        return view

    turn = turns.get_turn(record.turn_id)
    if (turn is None or turn.ended_at is None
            or turn.status not in {"ok", "completed"} or turn.error_text
            or turn.bot_id != record.target_bot_id or turn.user_id != user_id
            or turn.trigger_message_id != record.user_message_id):
        view["reply_status"] = "unresolved"
        return view
    message_id = getattr(turn, "assistant_message_id", None)
    if not message_id or memory_client is None:
        view["reply_status"] = "unresolved"
        return view
    message = memory_client.get_message_by_id(message_id)
    incoming = memory_client.get_message_by_id(record.user_message_id)
    if (not message or message.get("id") != message_id
            or message.get("role") != "assistant"
            or message.get("author_entity_type") != "bot"
            or message.get("author_entity_id") != record.target_bot_id
            or not isinstance(message.get("content"), str)
            or not message.get("session_id")
            or not incoming or incoming.get("id") != record.user_message_id
            or incoming.get("role") != "user"
            or not _authored_by_row(incoming, record)
            or incoming.get("session_id") != message["session_id"]):
        view["reply_status"] = "unresolved"
        return view
    view.update(reply_status="answered", reply_text=message["content"],
                target_message_id=message_id,
                target_session_id=message["session_id"])
    return view
