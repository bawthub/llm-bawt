"""Settle a bot steer whose execution is UNCERTAIN (design §4.4, C1 = TASK-2480).

One function, called inside the transaction (and target lock) of the event
that ended the attempt: an ``uncertain`` lifecycle from the bridge, or the
target run ending while the accepted steer never reported STARTED.

Order (first match decides):

a. explicit user Stop of the target run -> UNKNOWN "stopped"; never resent.
b. excluded from automatic resend (session reset, reset-only / typed
   follow-up, CONTEXT_MAINTENANCE, RESTART_RETRY's producer exclusions other
   than ``restart_retry_of``) -> UNKNOWN with that reason. MANUAL_RETRY's set
   is not used: it excludes ordinary chat, which automatic resend allows.
c. lineage already used its one resend (``restart_retry_of`` on the row, an
   existing same-producer child, or a cross-producer sibling) -> UNKNOWN
   "already resent".
d. otherwise one ``system`` resend carrying the may-already-have-run note
   (``restart_retry`` extended from HOST_SHUTDOWN to unfinished teardown).

Proven NOT_RUN never reaches here: the steer fence reopens it at k+1.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

from sqlalchemy import text as sa_text

from ..inter_bot_delivery_ledger import DeliveryTransitionEvents
from ..inter_bot_delivery_record import DeliveryRecord
from . import events
from .state import EventKind, unknown_error

logger = logging.getLogger(__name__)

RESENT = "resent"
UNKNOWN = "unknown"


def _exclusion(row: Mapping[str, Any]) -> str | None:
    from ..inter_bot_delivery_retry import RESTART_RETRY

    metadata = row.get("metadata_json") or {}
    payload = row.get("payload_json") or {}
    if (row.get("session_policy") or "continue") != "continue":
        return "session_reset"
    if row.get("message_kind") == "CONTEXT_MAINTENANCE":
        return "context_maintenance"
    if payload.get("inter_bot_reset_only") or payload.get("inter_bot_typed_follow_up"):
        return "reset_or_follow_up"
    for key in sorted(RESTART_RETRY.refused_metadata - {RESTART_RETRY.marker}):
        if key in metadata:
            return f"excluded:{key}"
    return None


def _already_resent(conn, row: Mapping[str, Any]) -> bool:
    from ..inter_bot_delivery_retry import RESTART_RETRY, _existing_child

    if RESTART_RETRY.marker in (row.get("metadata_json") or {}):
        return True
    same = conn.execute(sa_text("""
        SELECT 1 FROM inter_bot_deliveries
        WHERE target_bot_id=:target AND idempotency_key=:key LIMIT 1
    """), {"target": row["target_bot_id"], "key": RESTART_RETRY.key(row["id"])}).first()
    if same is not None:
        return True
    return _existing_child(conn, DeliveryRecord.from_mapping(row), RESTART_RETRY) is not None


def _fail(conn, row: Mapping[str, Any], error: str, outcome: str) -> Mapping[str, Any]:
    failed = conn.execute(sa_text("""
        UPDATE inter_bot_deliveries SET status='FAILED', outcome=:outcome,
            claim_token=NULL, claim_owner=NULL, lease_expires_at=NULL,
            next_retry_at=NULL, last_error=:error, updated_at=now()
        WHERE id=:id RETURNING *
    """), {"id": row["id"], "error": error[:4000], "outcome": outcome}).mappings().one()
    DeliveryTransitionEvents.stage(conn, failed, reason_code="outcome_uncertain")
    return failed


def settle_uncertain(store, conn, row: Mapping[str, Any], cause: str
                     ) -> tuple[DeliveryRecord, DeliveryRecord | None]:
    """Dispose of one UNCERTAIN steer attempt; caller holds target + row locks.

    Returns ``(original, child)``; ``child`` is the annotated resend, if any.
    """
    attempt = int(row.get("attempt_no") or 1)
    events.record(conn, row["id"], attempt, EventKind.UNCERTAIN, {"cause": cause})
    turn_id = row.get("target_turn_id")
    if turn_id and store._explicit_stop_requested(conn, turn_id):
        reason = "stopped"
    else:
        reason = _exclusion(row) or ("already_resent" if _already_resent(conn, row) else None)
    if reason is not None:
        failed = _fail(conn, row, unknown_error(reason), UNKNOWN)
        return DeliveryRecord.from_mapping(failed), None

    from ..inter_bot_delivery_retry import RESTART_RETRY, _retry

    failed = _fail(conn, row, f"outcome_uncertain ({cause}); resent once with a note", RESENT)
    try:
        with conn.begin_nested():
            result = _retry(store, row["id"], RESTART_RETRY, connection=conn)
    except Exception as error:  # a refused/failed resend leaves it visibly UNKNOWN
        logger.warning("uncertain steer %s resend refused: %s", row["id"], error)
        result = None
    if result is None:
        failed = _fail(conn, row, unknown_error("resend_refused"), UNKNOWN)
        return DeliveryRecord.from_mapping(failed), None
    child, _created = result
    conn.execute(sa_text(
        "UPDATE inter_bot_deliveries SET resent_from_id=:orig WHERE id=:id AND resent_from_id IS NULL"
    ), {"orig": row["id"], "id": child.id})
    logger.info("uncertain steer %s (%s) resent as %s", row["id"], cause, child.id)
    return DeliveryRecord.from_mapping(failed), child


def settle_parked_after_run(store, target_turn_id: str) -> list[tuple[DeliveryRecord, DeliveryRecord | None]]:
    """The target run ended: every accepted steer into it that never STARTED is UNCERTAIN."""
    if store.engine is None or not target_turn_id:
        return []
    from ..service.turn_admission import lock_target

    settled = []
    with store.engine.begin() as conn:
        target = conn.execute(sa_text(
            "SELECT bot_id FROM turn_logs WHERE id=:id"
        ), {"id": target_turn_id}).scalar_one_or_none()
        if target is None:
            return []
        lock_target(conn, target)
        rows = conn.execute(sa_text("""
            SELECT * FROM inter_bot_deliveries
            WHERE target_turn_id=:turn AND status='STEERING' AND kind='ADDITION'
              AND claim_token IS NULL
            ORDER BY ordinal FOR UPDATE
        """), {"turn": target_turn_id}).mappings().all()
        from ..inter_bot_delivery_steer_fence import steer_started

        for row in rows:
            if steer_started(row):
                continue  # STARTED already settles DELIVERED; never re-sent
            settled.append(settle_uncertain(store, conn, row, "run_ended_without_start"))
    return settled


def settle_ended_parked(store) -> list[DeliveryRecord]:
    """Dispatcher-sweep entry: settle accepted steers whose target run has ended.

    Such a row is the FIFO head again (it no longer has the open-run
    exemption), so nothing later can claim before this disposes of it.
    """
    if store.engine is None:
        return []
    with store.engine.connect() as conn:
        turn_ids = conn.execute(sa_text("""
            SELECT DISTINCT d.target_turn_id FROM inter_bot_deliveries d
            JOIN turn_logs t ON t.id=d.target_turn_id
            WHERE d.status='STEERING' AND d.kind='ADDITION' AND d.claim_token IS NULL
              AND t.ended_at IS NOT NULL
        """)).scalars().all()
    settled: list[DeliveryRecord] = []
    for turn_id in turn_ids:
        for original, child in settle_parked_after_run(store, turn_id):
            settled.append(original)
            if child is not None:
                settled.append(child)
    return settled


__all__ = ["RESENT", "UNKNOWN", "settle_ended_parked", "settle_parked_after_run", "settle_uncertain"]
