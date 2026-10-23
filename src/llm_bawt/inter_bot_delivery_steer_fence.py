"""Settle a steered delivery from its bridge command's lifecycle (TASK-1300).

A delivery steer is DELIVERED once the bridge accepted the command and the
message was persisted, but the CLI only reports ``queued`` at that point. The
command can still be swept by "Interrupt now", discarded or refused before it
runs. The bridge reports each of those on ``steer_lifecycle`` with the RPC id
(``steer_delivery_<id>[_g<k>]``) and, for a rejection, the proof that command
``k`` never ran (:mod:`claude_code_bridge.active_run`, P1/P2).

Every write here is fenced on the row lock and on the event's own generation
and target turn, so a stale, replayed or out-of-order event cannot move a row
that has since moved on:

* ``started`` records ``steer_started={generation: k}``. A started command is
  never reopened, and a pending rejection for ``k`` is dropped with a warning.
* a proven rejection of ``k`` on a DELIVERED row reopens it: QUEUED,
  ``steer_rejected`` and generation ``k+1``. On a STEERING row (its claim is
  still settling) it writes ``steer_rejected_pending`` instead, and the claim's
  settlement (``mark_delivered``, every ``requeue``, lease recovery) takes the
  same reopen under that row lock.
* ``uncertain`` (the bridge cannot prove whether ``k`` ran) records
  ``steer_uncertain`` and leaves the row where it is. Nothing is re-sent.

Advancing past an accepted ``k`` is safe only because ``k`` is proven not to
have executed (see ``inter_bot_dispatcher_steer``).
"""
from __future__ import annotations

import json
import logging
from typing import Any, Mapping

from sqlalchemy import text as sa_text
from sqlalchemy.engine import Connection

from .inter_bot_delivery_record import DeliveryRecord

logger = logging.getLogger(__name__)

STEER_STARTED = "steer_started"
STEER_REJECTED_PENDING = "steer_rejected_pending"
STEER_REJECTION = "steer_rejection"
STEER_UNCERTAIN = "steer_uncertain"

REOPENED = "reopened"
PENDING = "pending"
IGNORED = "ignored"


def _generation(metadata: Mapping[str, Any]) -> int:
    try:
        return max(0, int(metadata.get("steer_generation") or 0))
    except (TypeError, ValueError):
        return 0


def _marker_generation(metadata: Mapping[str, Any], key: str) -> int | None:
    marker = metadata.get(key)
    if not isinstance(marker, dict):
        return None
    try:
        return int(marker.get("generation"))
    except (TypeError, ValueError):
        return None


def _metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata_json")
    return metadata if isinstance(metadata, dict) else {}


def rejection_merge(metadata: Mapping[str, Any], proof: str) -> dict[str, Any]:
    """The metadata merge that reopens generation ``k`` as proven not run."""
    generation = _generation(metadata)
    return {
        "steer_generation": generation + 1,
        STEER_REJECTED_PENDING: None,
        STEER_REJECTION: {"generation": generation, "proof": proof},
    }


def pending_rejection(row: Mapping[str, Any]) -> str | None:
    """The proof of a rejection recorded for this steer claim's own generation.

    Only a ``steer`` row's current generation counts, and never once that
    generation has started.
    """
    if row.get("delivery_mode") != "steer":
        return None
    metadata = _metadata(row)
    generation = _generation(metadata)
    if _marker_generation(metadata, STEER_REJECTED_PENDING) != generation:
        return None
    if _marker_generation(metadata, STEER_STARTED) == generation:
        return None
    marker = metadata[STEER_REJECTED_PENDING]
    return str(marker.get("proof") or "rejected")


def steer_uncertain(record: Any) -> dict[str, Any] | None:
    """The sender-facing uncertain marker for a delivery's current steer, if any.

    Set only while the row is still the ``steer`` of that generation and the
    command never reported started; a re-queue or a later start clears it.
    """
    if getattr(record, "delivery_mode", None) != "steer":
        return None
    metadata = getattr(record, "metadata", None) or {}
    generation = _generation(metadata)
    if _marker_generation(metadata, STEER_UNCERTAIN) != generation:
        return None
    if _marker_generation(metadata, STEER_STARTED) == generation:
        return None
    return {"generation": generation, "reason": str(metadata[STEER_UNCERTAIN].get("reason") or "uncertain")}


def rejection_error(proof: str, generation: int) -> str:
    return f"steer command was not run ({proof}, generation {generation}); re-queued"


class SteerLifecycleFence:
    """Row-locked lifecycle writes for one steered delivery."""

    def __init__(self, store):
        self.store = store

    def _locked_row(self, conn: Connection, delivery_id: str) -> Mapping[str, Any] | None:
        # Lock order matches claim_next and mark_transport_accepted: target, then row.
        target = conn.execute(sa_text(
            "SELECT target_bot_id FROM inter_bot_deliveries WHERE id=:id"
        ), {"id": delivery_id}).scalar_one_or_none()
        if target is None:
            return None
        from .service.turn_admission import lock_target

        lock_target(conn, target)
        return conn.execute(sa_text(
            "SELECT * FROM inter_bot_deliveries WHERE id=:id FOR UPDATE"
        ), {"id": delivery_id}).mappings().first()

    @staticmethod
    def _matches(row: Mapping[str, Any] | None, generation: int, target_turn_id: str) -> bool:
        return bool(
            row is not None
            and row.get("delivery_mode") == "steer"
            and target_turn_id
            and row.get("target_turn_id") == target_turn_id
            and _generation(_metadata(row)) == generation
        )

    @staticmethod
    def _merge(conn: Connection, delivery_id: str, merge: Mapping[str, Any]) -> DeliveryRecord | None:
        row = conn.execute(sa_text("""
            UPDATE inter_bot_deliveries
            SET metadata_json=metadata_json || CAST(:merge AS jsonb), updated_at=now()
            WHERE id=:id RETURNING *
        """), {"id": delivery_id, "merge": json.dumps(dict(merge))}).mappings().first()
        return DeliveryRecord.from_mapping(row) if row else None

    @staticmethod
    def reopen_delivered(conn: Connection, row: Mapping[str, Any], proof: str) -> DeliveryRecord | None:
        """DELIVERED steer -> QUEUED ``steer_rejected`` at ``k+1``, in the caller's lock."""
        metadata = _metadata(row)
        reopened = conn.execute(sa_text("""
            UPDATE inter_bot_deliveries SET
                status='QUEUED', delivery_mode='steer_rejected', transport_accepted_at=NULL,
                delivered_at=NULL, available_at=now(), next_retry_at=now(),
                attempt_count=GREATEST(attempt_count - 1, 0), last_error=:error,
                metadata_json=metadata_json || CAST(:merge AS jsonb), updated_at=now()
            WHERE id=:id AND status='DELIVERED' AND delivery_mode='steer'
            RETURNING *
        """), {
            "id": row["id"],
            "error": rejection_error(proof, _generation(metadata)),
            "merge": json.dumps(rejection_merge(metadata, proof)),
        }).mappings().first()
        return DeliveryRecord.from_mapping(reopened) if reopened else None

    def reopen_rejected_steer(
        self, delivery_id: str, generation: int, target_turn_id: str, proof: str,
    ) -> tuple[str, DeliveryRecord | None]:
        """Re-queue a steer whose command ``generation`` is proven not to have run."""
        if self.store.engine is None:
            return IGNORED, None
        with self.store.engine.begin() as conn:
            row = self._locked_row(conn, delivery_id)
            if not self._matches(row, generation, target_turn_id):
                return IGNORED, None
            if _marker_generation(_metadata(row), STEER_STARTED) == generation:
                logger.warning(
                    "steer rejection for delivery %s generation %s ignored: it already started",
                    delivery_id, generation,
                )
                return IGNORED, None
            if row["status"] == "DELIVERED":
                reopened = self.reopen_delivered(conn, row, proof)
                return (REOPENED, reopened) if reopened else (IGNORED, None)
            if row["status"] == "STEERING":
                # The claim is still settling; its settlement reopens it.
                pending = self._merge(conn, delivery_id, {
                    STEER_REJECTED_PENDING: {"generation": generation, "proof": proof},
                })
                return PENDING, pending
        return IGNORED, None

    def note_steer_started(self, delivery_id: str, generation: int, target_turn_id: str) -> bool:
        """Record that command ``generation`` started; it can never be reopened."""
        if self.store.engine is None:
            return False
        with self.store.engine.begin() as conn:
            row = self._locked_row(conn, delivery_id)
            if not self._matches(row, generation, target_turn_id):
                return False
            metadata = _metadata(row)
            merge: dict[str, Any] = {STEER_STARTED: {"generation": generation}}
            if _marker_generation(metadata, STEER_REJECTED_PENDING) == generation:
                logger.warning(
                    "delivery %s generation %s started after a rejection was recorded; "
                    "dropping the rejection", delivery_id, generation,
                )
                merge[STEER_REJECTED_PENDING] = None
            return self._merge(conn, delivery_id, merge) is not None

    def mark_steer_uncertain(
        self, delivery_id: str, generation: int, target_turn_id: str, reason: str,
    ) -> DeliveryRecord | None:
        """Fence a steer whose command may or may not have run. Never re-sent."""
        if self.store.engine is None:
            return None
        with self.store.engine.begin() as conn:
            row = self._locked_row(conn, delivery_id)
            if not self._matches(row, generation, target_turn_id):
                return None
            if _marker_generation(_metadata(row), STEER_STARTED) == generation:
                return None
            logger.warning(
                "delivery %s steer generation %s outcome is uncertain (%s); left %s, not re-sent",
                delivery_id, generation, reason, row["status"],
            )
            return self._merge(conn, delivery_id, {
                STEER_UNCERTAIN: {"generation": generation, "reason": reason},
            })


__all__ = [
    "IGNORED", "PENDING", "REOPENED", "STEER_REJECTED_PENDING", "STEER_REJECTION",
    "STEER_STARTED", "STEER_UNCERTAIN", "SteerLifecycleFence", "pending_rejection",
    "rejection_error", "rejection_merge", "steer_uncertain",
]
