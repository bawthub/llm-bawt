"""Resolve an ordinary queued chat receipt against its exact durable turn.

Consumers must not mistake DELIVERED (transport completion) for an answer until
that delivery's finished turn and original message identity agree.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

# Ordinary agent admission never answers inline: it returns one of these
# receipts. Every consumer recognizes them through this single definition.
QUEUED_RECEIPT = "chat.completion.queued"
IN_FLIGHT_RECEIPT = "chat.completion.in_flight"
ACCEPTED_RECEIPTS = frozenset({QUEUED_RECEIPT, IN_FLIGHT_RECEIPT})


def is_accepted_receipt(value: Any) -> bool:
    """True when a chat result is an admission receipt rather than a reply."""
    return isinstance(value, dict) and value.get("object") in ACCEPTED_RECEIPTS


class DeliveryReplyMismatch(ValueError):
    """The receipt does not belong to the supplied original request."""


def resolve_delivery_reply(
    deliveries: Any,
    turns: Any,
    delivery_id: str,
    *,
    bot_id: str,
    user_id: str,
    user_message_id: str,
) -> dict[str, Any] | None:
    """Return a pending/terminal receipt, or None for an unknown delivery.

    No partial output is exposed. A terminal delivery with missing or mismatched
    turn data remains explicitly unresolved instead of manufacturing a reply.
    """
    record = deliveries.get(delivery_id)
    if record is None:
        return None
    if (
        record.target_bot_id != bot_id
        or record.author_entity_type != "user"
        or record.author_entity_id != user_id
        or record.user_message_id != user_message_id
        or not (record.metadata or {}).get("ordinary_chat")
    ):
        raise DeliveryReplyMismatch("delivery receipt does not match the original request")
    receipt: dict[str, Any] = {
        "delivery_id": record.id,
        "status": record.status,
        "turn_id": record.turn_id,
        "user_message_id": record.user_message_id,
    }
    if record.status != "DELIVERED":
        if record.status in {"FAILED", "CANCELLED"}:
            receipt["error"] = record.last_error or record.status.lower()
        return receipt

    turn = turns.get_turn(record.turn_id)
    if (
        turn is None
        or turn.ended_at is None
        or turn.status not in {"ok", "completed"}
        or turn.error_text
        or turn.bot_id != bot_id
        or turn.user_id != user_id
        or turn.trigger_message_id != user_message_id
    ):
        receipt["status"] = "UNRESOLVED"
        return receipt
    receipt["response_text"] = turn.response_text or ""
    return receipt


def resolve_direct_turn_reply(
    turns: Any, turn_id: str, *, bot_id: str, user_id: str,
    user_message_id: str,
) -> dict[str, Any] | None:
    """Reattach to an exact direct-start turn without submitting it again."""
    turn = turns.get_turn(turn_id)
    if turn is None:
        return None
    if (turn.bot_id != bot_id or turn.user_id != user_id
            or turn.trigger_message_id != user_message_id):
        raise DeliveryReplyMismatch("direct turn does not match the original request")
    if turn.ended_at is None:
        return {"turn_id": turn_id, "status": "IN_FLIGHT"}
    if turn.status not in {"ok", "completed"} or turn.error_text:
        return {"turn_id": turn_id, "status": "FAILED"}
    return {"turn_id": turn_id, "status": "DELIVERED", "response_text": turn.response_text or ""}


async def _wait_exact_reply(
    resolve: Callable[..., dict[str, Any] | None], args: tuple[Any, ...],
    *, bot_id: str, user_id: str, user_message_id: str,
    timeout_seconds: float, pending: frozenset[str],
) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    while True:
        receipt = await asyncio.to_thread(
            resolve, *args, bot_id=bot_id, user_id=user_id,
            user_message_id=user_message_id,
        )
        if receipt is None:
            raise DeliveryReplyMismatch("original reply receipt disappeared")
        if receipt["status"] not in pending or loop.time() >= deadline:
            return receipt
        await asyncio.sleep(min(1.0, max(0.0, deadline - loop.time())))


async def wait_for_direct_turn_reply(
    turns: Any, turn_id: str, *, bot_id: str, user_id: str,
    user_message_id: str, timeout_seconds: float,
) -> dict[str, Any]:
    """Wait on the original turn only; never replay an ambiguous submission."""
    return await _wait_exact_reply(
        resolve_direct_turn_reply, (turns, turn_id), bot_id=bot_id,
        user_id=user_id, user_message_id=user_message_id,
        timeout_seconds=timeout_seconds, pending=frozenset({"IN_FLIGHT"}),
    )


async def wait_for_delivery_reply(
    deliveries: Any,
    turns: Any,
    delivery_id: str,
    *,
    bot_id: str,
    user_id: str,
    user_message_id: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Bounded reattachment to one exact receipt; never start a second turn."""
    return await _wait_exact_reply(
        resolve_delivery_reply, (deliveries, turns, delivery_id),
        bot_id=bot_id, user_id=user_id, user_message_id=user_message_id,
        timeout_seconds=timeout_seconds,
        pending=frozenset({"QUEUED", "STEERING", "DISPATCHING"}),
    )


async def wait_for_receipt_reply(
    deliveries: Any, turns: Any, receipt: dict[str, Any], *,
    bot_id: str, user_id: str, timeout_seconds: float,
) -> dict[str, Any]:
    """Wait on whichever exact identity an accepted receipt names."""
    user_message_id = str(receipt.get("user_message_id") or "")
    if receipt.get("object") == QUEUED_RECEIPT and receipt.get("delivery_id"):
        return await wait_for_delivery_reply(
            deliveries, turns, str(receipt["delivery_id"]), bot_id=bot_id,
            user_id=user_id, user_message_id=user_message_id,
            timeout_seconds=timeout_seconds,
        )
    if receipt.get("object") == IN_FLIGHT_RECEIPT and receipt.get("turn_id"):
        return await wait_for_direct_turn_reply(
            turns, str(receipt["turn_id"]), bot_id=bot_id, user_id=user_id,
            user_message_id=user_message_id, timeout_seconds=timeout_seconds,
        )
    raise DeliveryReplyMismatch("accepted receipt names no reply identity")
