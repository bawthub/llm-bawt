"""Step 3: reservations whose producer will never consume them (TASK-1232 C-B3).

A reservation is released only after its producer's authority is revoked, in
one transaction in the admission lock order (target lock → producer row →
turn row). A stalled claimant that resumes afterwards then fails its fence
(``AdmissionFenceLost`` at ``save_turn``) instead of consuming the row.

- Approval: the continuation claim behind the reservation stopped renewing.
  ``turn_admission.requeue_stale_reservation`` takes it back, through the same
  lock protocol as a user Stop: every lock before any write, so a reservation
  the claimant consumed since detect keeps its live claim (Loopy B1).
- Delivery: the delivery is FAILED or CANCELLED, or ``recover_expired`` already
  nulled its claim. Released through the evidence-checked
  ``release_reservation(delivery_id=…)``. A QUEUED delivery's reservation is
  left alone: ``recover_expired`` re-arms it on purpose, so the requeued
  delivery keeps its place ahead of later chats, and its next claim reuses it
  (Vex S2).
- An unowned ordinary reservation is reported, never released.

The released row is deleted, so the run's stamp goes on its producer's row
(``recovery.audit``): the approval's ``continuation_last_error``, or the
delivery's ``metadata_json.recovery``. A token no approval row holds has no
producer row to stamp; the report and the log are its only trace.
"""
from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from typing import Any

from sqlalchemy import text as sa_text

from ...approval_models import CONT_DISPATCHING, CONTINUATION_LEASE_SECONDS
from ..turn_admission import lock_target, release_reservation, requeue_stale_reservation
from .base import Plan, RecoveryStep, SweepScope

_RESERVED = """
    SELECT t.id AS turn_id, t.bot_id, t.trigger_message_id, t.created_at,
           CAST(t.request_json AS jsonb)->>'reservation_owner' AS owner,
           CAST(t.request_json AS jsonb)->>'inter_bot_delivery_id' AS delivery_id
    FROM turn_logs t WHERE t.status = 'reserved' AND t.ended_at IS NULL
    ORDER BY t.created_at
"""


class OrphanedReservationsStep(RecoveryStep):
    name = "reservations"

    async def detect(self, scope: SweepScope) -> list[dict[str, Any]]:
        stale_before = scope.now - timedelta(seconds=CONTINUATION_LEASE_SECONDS)

        def read() -> list[dict[str, Any]]:
            with scope.engine.connect() as conn:
                rows = conn.execute(sa_text(_RESERVED)).mappings().all()
                found = []
                for row in rows:
                    if not scope.includes(row["bot_id"]):
                        continue
                    entry = {"turn_id": row["turn_id"], "bot_id": row["bot_id"],
                             "trigger_message_id": row["trigger_message_id"]}
                    if row["owner"]:
                        producer = conn.execute(sa_text(
                            "SELECT id, continuation_state,"
                            " continuation_next_attempt_at <= :stale_before AS lease_stale"
                            " FROM tool_approval_requests WHERE continuation_claim_token=:token"
                        ), {"token": row["owner"], "stale_before": stale_before}).mappings().first()
                        entry.update(kind="approval", owner=row["owner"],
                                     approval_id=producer["id"] if producer else None,
                                     continuation_state=producer["continuation_state"] if producer else None,
                                     lease_stale=bool(producer and producer["lease_stale"]))
                    elif row["delivery_id"]:
                        delivery = conn.execute(sa_text(
                            "SELECT status, claim_token FROM inter_bot_deliveries WHERE id=:id AND turn_id=:turn"
                        ), {"id": row["delivery_id"], "turn": row["turn_id"]}).mappings().first()
                        entry.update(kind="delivery", delivery_id=row["delivery_id"],
                                     delivery_status=delivery["status"] if delivery else None,
                                     claimed=bool(delivery and delivery["claim_token"]))
                    else:
                        entry.update(kind="ordinary")
                    found.append(entry)
                return found

        return await asyncio.to_thread(read)

    async def plan(self, scope: SweepScope, candidate: dict[str, Any]) -> Plan:
        kind = candidate["kind"]
        if kind == "approval":
            if candidate["approval_id"] is None or candidate["continuation_state"] != "dispatching":
                # No live claim holds this token: its owner can't pass the fence.
                return Plan("release_unowned_claim")
            if not candidate["lease_stale"]:
                return Plan(None)  # a live continuation, not a candidate
            return Plan("revoke_and_release")
        if kind == "delivery":
            if candidate["delivery_status"] == "QUEUED":
                return Plan(None)  # recover_expired's ordering fence, reused by the next claim
            if candidate["delivery_status"] in ("FAILED", "CANCELLED") or not candidate["claimed"]:
                return Plan("release")
            return Plan(None)  # the claim is live or recover_expired's to recover
        return Plan.leave("unowned ordinary reservation (report only)")

    async def apply(self, scope: SweepScope, candidate: dict[str, Any], plan: Plan) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._apply, scope, candidate, plan.action)

    def _apply(self, scope: SweepScope, candidate: dict[str, Any], action: str) -> dict[str, Any] | None:
        turn_id = candidate["turn_id"]
        reason = scope.stamp.text(f"recovery: released reservation {turn_id}")
        with scope.engine.begin() as conn:
            lock_target(conn, candidate["bot_id"])
            if candidate["kind"] == "approval":
                owner = candidate["owner"]
                if action == "revoke_and_release":
                    # The shared revoke primitive (Loopy B1): every lock before
                    # any write, and a reservation the claimant consumed since
                    # detect is left with its live claim.
                    released = requeue_stale_reservation(
                        conn, turn_id, owner=owner, approval_id=candidate["approval_id"],
                        stale_before=scope.now - timedelta(seconds=CONTINUATION_LEASE_SECONDS),
                        reason=scope.stamp.text(
                            f"recovery: revoked a stale continuation claim (reservation {turn_id})"))
                else:
                    held = conn.execute(sa_text(
                        "SELECT 1 FROM tool_approval_requests WHERE continuation_claim_token=:token"
                        " AND continuation_state=:state"
                    ), {"token": owner, "state": CONT_DISPATCHING}).first()
                    if held is not None:
                        return None
                    # No live claim holds this token, so there is nothing to
                    # revoke and the release is the only write.
                    released = release_reservation(scope.engine, turn_id, owner=owner, connection=conn)
                if released and action != "revoke_and_release" and candidate["approval_id"] is not None:
                    # Only the stamp is new here: keep the row's own last error.
                    conn.execute(sa_text("""
                        UPDATE tool_approval_requests SET continuation_last_error =
                            :reason || COALESCE(' (was: ' || continuation_last_error || ')', '')
                        WHERE id=:id
                    """), {"id": candidate["approval_id"], "reason": reason})
            else:
                lock = " FOR UPDATE" if conn.dialect.name == "postgresql" else ""
                delivery = conn.execute(sa_text(
                    "SELECT status, claim_token FROM inter_bot_deliveries WHERE id=:id" + lock
                ), {"id": candidate["delivery_id"]}).mappings().first()
                if delivery is not None and (delivery["status"] == "QUEUED" or (
                        delivery["status"] not in ("FAILED", "CANCELLED") and delivery["claim_token"])):
                    return None  # requeued or re-claimed since detect
                released = release_reservation(
                    scope.engine, turn_id, delivery_id=candidate["delivery_id"],
                    bot_id=candidate["bot_id"], trigger_message_id=candidate["trigger_message_id"], connection=conn)
                if released:
                    conn.execute(sa_text("""
                        UPDATE inter_bot_deliveries
                        SET metadata_json = metadata_json || jsonb_build_object('recovery', CAST(:stamp AS jsonb))
                        WHERE id=:id
                    """), {"id": candidate["delivery_id"],
                           "stamp": json.dumps({**scope.stamp.as_dict(), "released_reservation": turn_id})})
            if not released:
                return None
        return {"released": True}
