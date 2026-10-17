"""Explicit user retry of a failed durable bot-to-bot delivery.

A retry is a *new* outbox row with new message/turn/bridge request IDs, not a
republication of the old Redis command. The original row and its failure remain
inspectable; one original delivery can create at most one direct retry.
"""

from __future__ import annotations

from sqlalchemy import text as sa_text

from .inter_bot_delivery import FAILED, InterBotDeliveryStore
from .inter_bot_delivery_record import DeliveryRecord


class DeliveryRetryConflict(ValueError):
    """The delivery cannot safely be resent as a new turn."""


def retry_failed_delivery(
    store: InterBotDeliveryStore, delivery_id: str,
) -> tuple[DeliveryRecord, bool] | None:
    """Atomically copy one failed bot callback into a new durable delivery.

    The row lock orders concurrent clicks with the first enqueue, and the
    deterministic key covers ambiguous HTTP outcomes and retries across tabs.
    Returns None for missing/hidden delivery IDs and raises for unsafe states.
    """
    if store.engine is None:
        raise RuntimeError("Inter-bot delivery database unavailable")
    with store.engine.begin() as conn:
        row = conn.execute(sa_text(
            "SELECT * FROM inter_bot_deliveries WHERE id=:id FOR UPDATE"
        ), {"id": delivery_id}).mappings().first()
        if row is None:
            return None
        original = DeliveryRecord.from_mapping(row)
        metadata = original.metadata
        if (any(key in metadata for key in (
                "prompt_schedule", "ordinary_chat", "maintenance_user_queue",
        )) or original.message_kind == "CONTEXT_MAINTENANCE"):
            return None
        if original.status != FAILED:
            raise DeliveryRetryConflict("Only failed deliveries can be retried")
        # A second reset can archive an entirely different session. The user
        # must submit an explicit new delivery if the old one requested a reset.
        if original.session_policy != "continue":
            raise DeliveryRetryConflict("Session-reset deliveries cannot be retried here")
        payload = row["payload_json"]
        if not isinstance(payload, dict):
            raise DeliveryRetryConflict("Original delivery payload is unavailable")
        # The old turn's compaction barrier is execution state, not a caller
        # instruction. enqueue stamps the new delivery's current barrier.
        retry_metadata = {key: value for key, value in metadata.items()
                          if key not in {"maintenance_barrier", "manual_retry_of"}}
        retry_metadata["manual_retry_of"] = original.id
        return store.enqueue(
            sender_bot_id=original.sender_bot_id,
            target_bot_id=original.target_bot_id,
            message=original.message,
            payload=payload,
            author=original.author,
            idempotency_key=f"manual-retry:{original.id}",
            project_id=original.project_id,
            task_id=original.task_id,
            message_kind=original.message_kind,
            metadata=retry_metadata,
            max_attempts=original.max_attempts,
            session_policy=original.session_policy,
            connection=conn,
            producer="retry",
        )
