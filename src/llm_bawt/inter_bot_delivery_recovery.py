"""Lease recovery for durable deliveries whose claim no live dispatcher works.

Extracted from ``inter_bot_delivery`` (TASK-1286 D0, monolith gate) without a
behavior change; ``InterBotDeliveryStore.recover_expired`` delegates here.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from sqlalchemy import text as sa_text
from sqlalchemy.engine import Connection

from .inter_bot_delivery_claim import QUEUED
from .inter_bot_delivery_ledger import DeliveryTransitionEvents
from .inter_bot_delivery_record import DeliveryRecord
from .inter_bot_delivery_steer_fence import pending_rejection, rejection_error, rejection_merge


class DeliveryLeaseRecovery:
    """Settle or re-arm expired and orphaned claims against a store."""

    @staticmethod
    def expire_own_claims(store, *, claim_owner: str, turn_ids: list[str]) -> list[str]:
        """Hand this process's confirmed-stopped turn claims to its successor (TASK-1286).

        Only ``lease_expires_at`` changes, so the next leader's first
        ``recover_expired`` sends each row down the unchanged lease-expired
        branch. The caller passes only turns whose worker and stream thread
        have provably exited. Steer rows never qualify: nothing proves a steer
        RPC has gone quiet, so they keep their lease.
        """
        if store.engine is None or not turn_ids:
            return []
        with store.engine.begin() as conn:
            conn.exec_driver_sql("SET LOCAL lock_timeout = '2s'")
            conn.exec_driver_sql("SET LOCAL statement_timeout = '2s'")
            released = conn.execute(sa_text("""
                UPDATE inter_bot_deliveries SET lease_expires_at=now(), updated_at=now()
                WHERE claim_owner=:owner AND status='DISPATCHING'
                  AND delivery_mode='turn' AND turn_id = ANY(:turn_ids)
                RETURNING id
            """), {"owner": claim_owner, "turn_ids": list(turn_ids)}).scalars().all()
        return list(released)

    @staticmethod
    def _settle_stopped_turn(conn: Connection, turn_id: str) -> None:
        """Settle a stopped delivery's open turn in the caller's transaction."""
        from sqlmodel import Session, select

        from .service.turn_logs import TurnLog
        from .service.turn_termination import settle_open_turn

        with Session(bind=conn) as session:
            row = session.exec(select(TurnLog).where(TurnLog.id == turn_id).with_for_update()).first()
            if row is not None:
                settle_open_turn(session, row, audit_key="user_stop", now=datetime.now(timezone.utc), audit={
                    "bridge_state": "lease_expired", "confirmed": False, "by": "recover_expired",
                })

    @staticmethod
    def recover_expired(store, *, live_owner: str | None = None) -> list[DeliveryRecord]:
        """Recover expired claims; singleton fencing excludes live peers.

        TASK-1286 #2: a user-stopped ordinary turn whose claim no live
        dispatcher owns (claim_owner != live_owner, so no in-process _dispatch
        settles it) is cancelled now. Other foreign claims, accepted resets
        included (their ambiguity check precedes Stop), wait out the lease.
        """
        if store.engine is None:
            return []
        from .service.session_lifecycle.recovery import NewSessionRecovery
        recovered: list[DeliveryRecord] = []
        orphan = (" OR (d.delivery_mode='turn' AND t.ended_at IS NOT NULL"
                  " AND d.claim_owner IS DISTINCT FROM :live_owner)") if live_owner else ""
        with store.engine.begin() as conn:
            rows = conn.execute(sa_text(f"""
                SELECT d.*, t.status AS turn_status, t.ended_at AS turn_ended_at,
                       t.error_text AS turn_error_text,
                       d.lease_expires_at < now() AS lease_expired
                FROM inter_bot_deliveries d
                LEFT JOIN turn_logs t ON t.id=d.turn_id
                WHERE d.status IN ('STEERING','DISPATCHING')
                  AND (d.lease_expires_at < now(){orphan})
                ORDER BY d.ordinal FOR UPDATE OF d
            """), {"live_owner": live_owner} if orphan else {}).mappings().all()
            for row in rows:
                is_turn = row.get("delivery_mode") != "steer"
                accepted = row.get("transport_accepted_at") is not None
                if not row["lease_expired"] and ((accepted and NewSessionRecovery.is_reset(row))
                                                 or not store._explicit_stop_requested(conn, row["turn_id"])):
                    continue
                if is_turn and accepted:
                    if NewSessionRecovery.is_reset(row) and (
                        row.get("turn_ended_at") is None
                        or row.get("turn_status") not in ("ok", "completed")
                        or row.get("turn_error_text")
                    ):
                        ambiguous = NewSessionRecovery.mark_ambiguous(
                            conn, delivery_id=row["id"], claim_token=row["claim_token"],
                            status=row["status"],
                            reason="accepted reset outcome is ambiguous; never replay provider reset",
                        )
                        if ambiguous:
                            recovered.append(ambiguous)
                        continue
                if is_turn and store._explicit_stop_requested(conn, row["turn_id"]):
                    assignments = "status='CANCELLED', claim_token=NULL, claim_owner=NULL, lease_expires_at=NULL, next_retry_at=NULL, last_error='Stopped by user'"
                elif is_turn and row.get("turn_ended_at") is not None:
                    success = (
                        row.get("turn_status") in ("ok", "completed")
                        and not row.get("turn_error_text")
                    )
                    assignments = (
                        "status='DELIVERED', delivered_at=COALESCE(delivered_at, now()), claim_token=NULL, claim_owner=NULL, lease_expires_at=NULL, last_error=NULL"
                        if success else
                        "status='FAILED', claim_token=NULL, claim_owner=NULL, lease_expires_at=NULL, last_error=COALESCE(last_error, 'target turn ended unsuccessfully')"
                    )
                elif not is_turn and (proof := pending_rejection(row)) is not None:
                    # TASK-1300: this claim's command is proven not to have run;
                    # leave its target at k+1 instead of replaying the frozen k.
                    metadata = row.get("metadata_json") or {}
                    updated = conn.execute(sa_text("""
                        UPDATE inter_bot_deliveries SET status='QUEUED',
                            delivery_mode='steer_rejected', transport_accepted_at=NULL,
                            claim_token=NULL, claim_owner=NULL, lease_expires_at=NULL,
                            available_at=now(), next_retry_at=now(),
                            attempt_count=GREATEST(attempt_count - 1, 0), last_error=:error,
                            metadata_json=metadata_json || CAST(:merge AS jsonb), updated_at=now()
                        WHERE id=:id RETURNING *
                    """), {
                        "id": row["id"],
                        "error": rejection_error(proof, int(metadata.get("steer_generation") or 0)),
                        "merge": json.dumps(rejection_merge(metadata, proof)),
                    }).mappings().one()
                    DeliveryTransitionEvents.stage(conn, updated, reason_code="steer_rejected")
                    recovered.append(DeliveryRecord.from_mapping(updated))
                    continue
                elif (
                    row.get("transport_accepted_at") is not None
                    and row["transport_accepted_at"]
                    < datetime.now(timezone.utc) - timedelta(days=7)
                ):
                    assignments = "status='FAILED', claim_token=NULL, claim_owner=NULL, lease_expires_at=NULL, last_error='accepted bridge run exceeded the seven-day recovery window; outcome is ambiguous and was not replayed'"
                elif int(row["attempt_count"] or 0) < int(row["max_attempts"] or 0):
                    assignments = "status='QUEUED', claim_token=NULL, claim_owner=NULL, lease_expires_at=NULL, available_at=now(), next_retry_at=now(), last_error='delivery lease expired; retrying same logical message'"
                else:
                    assignments = "status='FAILED', claim_token=NULL, claim_owner=NULL, lease_expires_at=NULL, last_error='delivery attempts exhausted before completion'"
                updated = conn.execute(sa_text(
                    f"UPDATE inter_bot_deliveries SET {assignments}, updated_at=now() WHERE id=:id RETURNING *"
                ), {"id": row["id"]}).mappings().one()
                DeliveryTransitionEvents.stage(conn, updated, reason_code="lease_recovery")
                if updated["status"] == "CANCELLED" and is_turn:
                    # TASK-1232 P4: a user Stop ends the turn too. Cancelling
                    # only the delivery left the row open, and admission
                    # blocked every later message (the 2026-10-21 deadlock).
                    DeliveryLeaseRecovery._settle_stopped_turn(conn, row["turn_id"])
                if updated["status"] == QUEUED and is_turn:
                    conn.execute(sa_text("""
                        UPDATE turn_logs SET status='reserved', ended_at=NULL,
                            end_reason=NULL, error_text=NULL
                        WHERE id=:turn_id AND ended_at IS NULL
                    """), {"turn_id": row["turn_id"]})
                recovered.append(DeliveryRecord.from_mapping(updated))
        return recovered
