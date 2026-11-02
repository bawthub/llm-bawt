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
    include_voice_binding: bool = False,
) -> dict[str, Any] | None:
    """Return a pending/terminal receipt, or None for an unknown delivery.

    No partial output is exposed. A terminal delivery with missing or mismatched
    turn data remains explicitly unresolved instead of manufacturing a reply.
    A FAILED original that was retried (a restart retry after its host shut
    down, TASK-1231B) reports its verified child's progress and reply instead;
    the child is system-authored, so it is matched to the original by
    provenance rather than by author.
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
    if record.status == "FAILED":
        from ..inter_bot_delivery_retry import verified_retry_child

        child = verified_retry_child(deliveries, record)
        if child is not None and child.target_bot_id == bot_id:
            receipt = _receipt(child, turns, bot_id=bot_id, user_id=user_id,
                               user_message_id=child.user_message_id)
            receipt["delivery_id"] = record.id  # the identity the caller waits on
            receipt["retry_delivery_id"] = child.id
            if include_voice_binding:
                receipt["voice_direct_reply"] = True
            return receipt
    if ((record.metadata.get("ordinary_chat") or {}).get("voice_input")
            and record.status == "DELIVERED" and record.delivery_mode == "steer"):
        return _voice_steer_reply(deliveries, record, turns, bot_id=bot_id, user_id=user_id,
                                  include_binding=include_voice_binding)
    receipt = _receipt(record, turns, bot_id=bot_id, user_id=user_id,
                       user_message_id=user_message_id)
    if include_voice_binding:
        receipt["voice_direct_reply"] = record.delivery_mode != "steer"
    return receipt


def _voice_steer_reply(deliveries: Any, record: Any, turns: Any, *, bot_id: str, user_id: str,
                       include_binding: bool) -> dict[str, Any]:
    """Observe the accepted injection's exact target, not a synthetic new turn.

    DELIVERED here acknowledges persisted transport acceptance, not a model
    answer. The injected message must still belong to the target's durable
    conversation. No heuristic latest-turn lookup or resubmission is allowed.
    """
    from sqlalchemy import text

    receipt = {"delivery_id": record.id, "status": "UNRESOLVED",
               "turn_id": record.target_turn_id, "user_message_id": record.user_message_id}
    engine = getattr(deliveries, "engine", None)
    turn = turns.get_turn(record.target_turn_id) if record.target_turn_id else None
    if (engine is None or record.transport_accepted_at is None or turn is None
            or turn.bot_id != bot_id or turn.user_id != user_id):
        return receipt
    with engine.connect() as conn:
        bound = conn.execute(text("""
            SELECT s.id FROM messages injected
            JOIN messages original ON original.bot_id=injected.bot_id
              AND original.id=:trigger AND original.session_id=injected.session_id
            JOIN sessions s ON s.id=injected.session_id AND s.bot_id=:bot AND s.user_id=:user
            WHERE injected.bot_id=:bot AND injected.id=:message AND injected.role='user'
              AND injected.author_entity_type='user' AND injected.author_entity_id=:user
              AND (injected.interrupt_source_message_id IS NULL
                   OR injected.interrupt_source_message_id=:trigger)
              AND s.status <> 'deleted'
        """), {"bot": bot_id, "user": user_id, "trigger": turn.trigger_message_id,
                 "message": record.user_message_id}).scalar_one_or_none()
    selected = (record.metadata.get("ordinary_chat") or {}).get("session_id")
    if not bound or (selected is not None and str(bound) != selected.strip()):
        return receipt
    from .inter_bot_dispatcher_steer import SteerGeneration

    # This binding proves persisted acceptance, NOT consumption or an answer.
    # Voice's next-step observer must separately match an ordered started frame.
    if include_binding:
        receipt["voice_binding"] = {
            "turn_id": turn.id, "session_id": str(bound),
            "run_request_id": turn.agent_request_id,
            "steer_rpc_id": SteerGeneration.of(record).request_id,
        }
    if turn.ended_at is None:
        receipt["status"] = "IN_FLIGHT"
    elif turn.status in {"ok", "completed"} and not turn.error_text:
        receipt.update(status="DELIVERED", response_text=turn.response_text or "")
    else:
        receipt.update(status="FAILED", error=turn.error_text or f"Steered turn ended with status={turn.status}")
    return receipt


def _receipt(
    record: Any, turns: Any, *, bot_id: str, user_id: str, user_message_id: str,
) -> dict[str, Any]:
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
        pending=frozenset({"QUEUED", "STEERING", "DISPATCHING", "IN_FLIGHT"}),
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
