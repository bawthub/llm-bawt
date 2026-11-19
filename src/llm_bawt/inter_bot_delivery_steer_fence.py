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

from .inter_bot_delivery_ledger import DeliveryTransitionEvents
from .inter_bot_delivery_record import DeliveryRecord

logger = logging.getLogger(__name__)

STEER_STARTED = "steer_started"
STEER_REJECTED_PENDING = "steer_rejected_pending"
STEER_REJECTION = "steer_rejection"
STEER_UNCERTAIN = "steer_uncertain"

REOPENED = "reopened"
PENDING = "pending"
IGNORED = "ignored"
CONFLICT = "conflict"


def _record(conn: Connection, row: Mapping[str, Any], kind: str, evidence: Mapping[str, Any],
            attempt: int | None = None) -> None:
    from .delivery import events
    from .delivery.state import EventKind

    events.record(conn, row["id"], attempt or int(row.get("attempt_no") or 1), EventKind(kind), dict(evidence))


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


def rejection_merge(metadata: Mapping[str, Any], proof: str,
                    row: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The metadata merge that reopens generation ``k`` as proven not run.

    TASK-2480: the rejection keeps its own binding (target turn and attempt)
    so a late STARTED for ``k`` is matched even after the row is re-claimed
    elsewhere (a fallback START rebinds ``target_turn_id``).
    """
    generation = _generation(metadata)
    rejection: dict[str, Any] = {"generation": generation, "proof": proof}
    if row is not None:
        rejection["turn_id"] = row.get("target_turn_id")
        rejection["attempt_no"] = int(row.get("attempt_no") or 1)
    return {
        **({"_inbox": None} if isinstance(metadata.get("_inbox"), dict) else {}),
        "steer_generation": generation + 1,
        STEER_REJECTED_PENDING: None,
        STEER_REJECTION: rejection,
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


def steer_started(row: Mapping[str, Any]) -> bool:
    """True when this steer row's current generation reported started.

    The only proof that the target turn actually received a steered message.
    """
    if row.get("delivery_mode") != "steer":
        return False
    metadata = _metadata(row)
    return _marker_generation(metadata, STEER_STARTED) == _generation(metadata)


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


def not_run_exhausted(row: Mapping[str, Any]) -> bool:
    """TASK-2480: a proven NOT_RUN re-queues only while execution attempts remain."""
    return int(row.get("attempt_no") or 1) >= int(row.get("max_attempts") or 1)


def fail_not_run_exhausted(conn: Connection, row: Mapping[str, Any], proof: str) -> DeliveryRecord | None:
    """Poison head: the last allowed attempt was proven not run. The FIFO continues."""
    record_not_run(conn, row, proof)
    failed = conn.execute(sa_text("""
        UPDATE inter_bot_deliveries SET status='FAILED', outcome='not_run',
            claim_token=NULL, claim_owner=NULL, lease_expires_at=NULL, next_retry_at=NULL,
            delivered_at=NULL, last_error=:error, updated_at=now()
        WHERE id=:id RETURNING *
    """), {"id": row["id"], "error": f"steer command was not run ({proof}); attempts exhausted"}
    ).mappings().first()
    DeliveryTransitionEvents.stage(conn, failed, reason_code="steer_rejected")
    return DeliveryRecord.from_mapping(failed) if failed else None


def record_not_run(conn: Connection, row: Mapping[str, Any], proof: str) -> None:
    from .delivery import events
    from .delivery.state import EventKind

    attempt = int(row.get("attempt_no") or 1)
    events.record(conn, row["id"], attempt, EventKind.REJECTED, {"proof": proof})
    events.record(conn, row["id"], attempt, EventKind.NOT_RUN, {"proof": proof})


STEER_CONFLICT = "steer_conflict"


def has_conflict(row: Mapping[str, Any]) -> bool:
    return isinstance(_metadata(row).get(STEER_CONFLICT), dict)


def fail_conflict(conn: Connection, row: Mapping[str, Any], detail: str) -> DeliveryRecord | None:
    """TASK-2480 §4.2: STARTED and a not-run proof for one attempt -> visible UNKNOWN.

    A conflict is never a success and never resent automatically. An already
    disposed row keeps its status but its error and outcome say so.
    """
    from .delivery.state import unknown_error

    status = "status" if row.get("status") in ("FAILED", "CANCELLED") else "'FAILED'"
    failed = conn.execute(sa_text(f"""
        UPDATE inter_bot_deliveries SET status={status}, outcome='unknown',
            claim_token=NULL, claim_owner=NULL, lease_expires_at=NULL, next_retry_at=NULL,
            last_error=:error,
            metadata_json=metadata_json || CAST(:merge AS jsonb), updated_at=now()
        WHERE id=:id RETURNING *
    """), {
        "id": row["id"], "error": unknown_error("evidence_conflict"),
        "merge": json.dumps({STEER_CONFLICT: {"detail": detail}}),
    }).mappings().first()
    logger.warning("delivery %s evidence conflict (%s): outcome unknown", row["id"], detail)
    DeliveryTransitionEvents.stage(conn, failed, reason_code="evidence_conflict")
    return DeliveryRecord.from_mapping(failed) if failed else None


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
            # Inbox lifecycle evidence also needs the original write token and
            # full binding. Legacy generation/turn-only events cannot settle it.
            and not isinstance(_metadata(row).get("_inbox"), dict)
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
        if not_run_exhausted(row):
            return fail_not_run_exhausted(conn, row, proof)
        record_not_run(conn, row, proof)
        metadata = _metadata(row)
        reopened = conn.execute(sa_text("""
            UPDATE inter_bot_deliveries SET
                status='QUEUED', delivery_mode='steer_rejected', transport_accepted_at=NULL,
                attempt_no=attempt_no + 1,
                delivered_at=NULL, available_at=now(), next_retry_at=now(),
                attempt_count=GREATEST(attempt_count - 1, 0), last_error=:error,
                metadata_json=metadata_json || CAST(:merge AS jsonb), updated_at=now()
            WHERE id=:id AND status='DELIVERED' AND delivery_mode='steer'
            RETURNING *
        """), {
            "id": row["id"],
            "error": rejection_error(proof, _generation(metadata)),
            "merge": json.dumps(rejection_merge(metadata, proof, row)),
        }).mappings().first()
        DeliveryTransitionEvents.stage(conn, reopened, reason_code="steer_rejected")
        return DeliveryRecord.from_mapping(reopened) if reopened else None

    @staticmethod
    def reopen_claimed(conn: Connection, row: Mapping[str, Any], proof: str) -> DeliveryRecord | None:
        """Reopen a row whose caller validated non-execution under its row lock."""
        if not_run_exhausted(row):
            return fail_not_run_exhausted(conn, row, proof)
        record_not_run(conn, row, proof)
        metadata = _metadata(row)
        reopened = conn.execute(sa_text("""
            UPDATE inter_bot_deliveries SET status='QUEUED', attempt_no=attempt_no + 1,
                delivery_mode='steer_rejected', transport_accepted_at=NULL,
                claim_token=NULL, claim_owner=NULL, lease_expires_at=NULL,
                available_at=now(), next_retry_at=now(),
                attempt_count=GREATEST(attempt_count - 1, 0), last_error=:error,
                metadata_json=metadata_json || CAST(:merge AS jsonb), updated_at=now()
            WHERE id=:id AND status='STEERING' AND delivery_mode='steer'
            RETURNING *
        """), {
            "id": row["id"],
            "error": rejection_error(proof, _generation(metadata)),
            "merge": json.dumps(rejection_merge(metadata, proof, row)),
        }).mappings().first()
        DeliveryTransitionEvents.stage(conn, reopened, reason_code="steer_rejected")
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
                # TASK-2480: STARTED plus a not-run proof is a conflict, not a win
                # for whichever arrived first. A claim still settling is marked;
                # its park/settlement fails it.
                _record(conn, row, "REJECTED", {"proof": proof, "generation": generation})
                if row["status"] == "STEERING" and row.get("claim_token") is not None:
                    return PENDING, self._merge(conn, delivery_id, {
                        STEER_CONFLICT: {"detail": f"rejected_after_started:{proof}"},
                    })
                return CONFLICT, fail_conflict(conn, row, f"rejected_after_started:{proof}")
            if row["status"] in ("FAILED", "CANCELLED"):
                return IGNORED, None
            if row["status"] == "DELIVERED":
                reopened = self.reopen_delivered(conn, row, proof)
                return (REOPENED, reopened) if reopened else (IGNORED, None)
            if row["status"] == "STEERING" and row.get("claim_token") is None:
                # TASK-2480: accepted and parked awaiting STARTED; proven not run.
                reopened = self.reopen_claimed(conn, row, proof)
                return (REOPENED, reopened) if reopened else (IGNORED, None)
            if row["status"] == "STEERING":
                # The claim is still settling; its settlement reopens it.
                pending = self._merge(conn, delivery_id, {
                    STEER_REJECTED_PENDING: {"generation": generation, "proof": proof},
                })
                return PENDING, pending
        return IGNORED, None

    def park_accepted(self, delivery_id: str, claim_token: str) -> DeliveryRecord | None:
        """TASK-2480: the run accepted and persisted the steer; await STARTED.

        The row stays STEERING with its claim and lease released, so lease
        recovery can't replay it and the FIFO head moves on while the target
        run is open. ``ACCEPTED`` is recorded. If ``started`` already arrived
        it settles DELIVERED now; a pending rejection re-queues as before.
        """
        from .delivery import events
        from .delivery.state import EventKind

        if self.store.engine is None or not claim_token:
            return None
        with self.store.engine.begin() as conn:
            row = self._locked_row(conn, delivery_id)
            if (row is None or row["status"] != "STEERING" or row["claim_token"] != claim_token
                    or row.get("delivery_mode") != "steer"
                    or isinstance(_metadata(row).get("_inbox"), dict)):
                return None
            if has_conflict(row):
                return fail_conflict(conn, row, str(_metadata(row)[STEER_CONFLICT].get("detail")))
            if pending_rejection(row) is not None:
                return self.store.requeue_claim(conn, delivery_id, claim_token, "", delay_seconds=1)
            attempt = int(row.get("attempt_no") or 1)
            events.record(conn, delivery_id, attempt, EventKind.ACCEPTED,
                          {"generation": _generation(_metadata(row)), "turn_id": row.get("target_turn_id")})
            parked = conn.execute(sa_text("""
                UPDATE inter_bot_deliveries SET kind='ADDITION', claim_token=NULL,
                    claim_owner=NULL, lease_expires_at=NULL, next_retry_at=NULL,
                    last_error=NULL, updated_at=now()
                WHERE id=:id RETURNING *
            """), {"id": delivery_id}).mappings().first()
            if steer_started(parked):
                return self._deliver(conn, parked)
            return DeliveryRecord.from_mapping(parked)

    @staticmethod
    def _deliver(conn: Connection, row: Mapping[str, Any]) -> DeliveryRecord | None:
        delivered = conn.execute(sa_text("""
            UPDATE inter_bot_deliveries SET status='DELIVERED', delivered_at=now(),
                response_chars=0, updated_at=now()
            WHERE id=:id AND status='STEERING' RETURNING *
        """), {"id": row["id"]}).mappings().first()
        DeliveryTransitionEvents.stage(conn, delivered)
        return DeliveryRecord.from_mapping(delivered) if delivered else None

    def note_steer_started(self, delivery_id: str, generation: int, target_turn_id: str
                           ) -> DeliveryRecord | bool:
        """Record that command ``generation`` started; it can never be reopened.

        TASK-2480: STARTED is the delivery proof. A parked (accepted) row goes
        DELIVERED here and the record is returned so the caller can announce it.
        """
        from .delivery import events
        from .delivery.state import EventKind

        if self.store.engine is None:
            return False
        with self.store.engine.begin() as conn:
            row = self._locked_row(conn, delivery_id)
            if not self._matches(row, generation, target_turn_id):
                return self._late_started(conn, row, generation, target_turn_id)
            metadata = _metadata(row)
            if row["status"] in ("FAILED", "CANCELLED"):
                # Disposed of as UNKNOWN/resent, then it turned out to have run.
                _record(conn, row, "STARTED", {"generation": generation, "turn_id": target_turn_id})
                return fail_conflict(conn, row, "started_after_disposal") or False
            merge: dict[str, Any] = {STEER_STARTED: {"generation": generation}}
            if _marker_generation(metadata, STEER_REJECTED_PENDING) == generation:
                # TASK-2480: both proofs for one attempt -> conflict, never a reopen.
                merge[STEER_REJECTED_PENDING] = None
                merge[STEER_CONFLICT] = {"detail": "started_after_rejection"}
            events.record(conn, delivery_id, int(row.get("attempt_no") or 1), EventKind.STARTED,
                          {"generation": generation, "turn_id": target_turn_id})
            merged = self._merge(conn, delivery_id, merge)
            if merged is None:
                return False
            if STEER_CONFLICT in merge and not (row["status"] == "STEERING" and row.get("claim_token")):
                return fail_conflict(conn, row, "started_after_rejection") or False
            if row["status"] == "STEERING" and row.get("claim_token") is None:
                return self._deliver(conn, row) or True
            return True

    def _late_started(self, conn: Connection, row: Mapping[str, Any] | None,
                      generation: int, target_turn_id: str) -> DeliveryRecord | bool:
        """STARTED for a generation already rejected (re-queued at k+1) -> conflict.

        Without this the late proof would be dropped by the generation fence
        and the row would run again as if k never executed.
        """
        if row is None or not target_turn_id:
            return False
        metadata = _metadata(row)
        if isinstance(metadata.get("_inbox"), dict):
            return False
        rejection = metadata.get(STEER_REJECTION)
        if _marker_generation(metadata, STEER_REJECTION) != generation:
            return False
        # The rejection's own binding; older markers fall back to the row's.
        bound_turn = rejection.get("turn_id") or row.get("target_turn_id")
        if bound_turn != target_turn_id:
            return False
        attempt = int(rejection.get("attempt_no") or max(1, int(row.get("attempt_no") or 1) - 1))
        _record(conn, row, "STARTED", {"generation": generation, "turn_id": target_turn_id}, attempt)
        detail = f"started_after_rejection:g{generation}"
        if row["status"] in ("STEERING", "DISPATCHING") and row.get("claim_token") is not None:
            # A later claim is settling; its settlement fails it as a conflict.
            return self._merge(conn, row["id"], {STEER_CONFLICT: {"detail": detail}}) or False
        # QUEUED, parked, DELIVERED (the k+1 run also happened) or already
        # disposed: the rejected attempt ran after all -> visible UNKNOWN.
        return fail_conflict(conn, row, detail) or False

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

    def settle_uncertain_steer(
        self, delivery_id: str, generation: int, target_turn_id: str, reason: str,
    ) -> tuple[DeliveryRecord | None, DeliveryRecord | None]:
        """TASK-2480: an accepted (parked) steer goes through delivery/recovery.

        Its outcome becomes a noted resend or UNKNOWN, never DELIVERED. A row
        still settling its claim keeps the TASK-1300 marker path: its claim
        settlement or lease recovery decides it.
        """
        if self.store.engine is None:
            return None, None
        with self.store.engine.begin() as conn:
            row = self._locked_row(conn, delivery_id)
            if not self._matches(row, generation, target_turn_id):
                return None, None
            if _marker_generation(_metadata(row), STEER_STARTED) == generation:
                return None, None
            if row["status"] in ("FAILED", "CANCELLED"):
                return None, None  # already disposed of; never resent twice
            if row["status"] == "STEERING" and row.get("claim_token") is None:
                from .delivery.recovery import settle_uncertain

                marked = self._merge(conn, delivery_id, {
                    STEER_UNCERTAIN: {"generation": generation, "reason": reason},
                })
                locked = conn.execute(sa_text(
                    "SELECT * FROM inter_bot_deliveries WHERE id=:id"
                ), {"id": delivery_id}).mappings().first()
                if marked is None or locked is None:
                    return None, None
                return settle_uncertain(self.store, conn, locked, reason)
        return self.mark_steer_uncertain(delivery_id, generation, target_turn_id, reason), None


__all__ = [
    "IGNORED", "PENDING", "REOPENED", "STEER_REJECTED_PENDING", "STEER_REJECTION",
    "STEER_STARTED", "STEER_UNCERTAIN", "SteerLifecycleFence", "pending_rejection",
    "rejection_error", "rejection_merge", "steer_started", "steer_uncertain",
]
