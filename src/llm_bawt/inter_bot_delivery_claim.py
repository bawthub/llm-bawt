"""Target-locked FIFO claim and reset precondition for durable deliveries."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import text as sa_text

from .agent_context import SessionPolicy
from .inter_bot_delivery_record import DeliveryRecord
from .inter_bot_delivery_user import resolve_delivery_user
from .maintenance_delivery import MaintenanceBarrier

QUEUED = "QUEUED"
STEERING = "STEERING"
DISPATCHING = "DISPATCHING"


class DeliveryClaim:
    """Choose the next target action against a locked store transaction."""

    def claim_next(
        store,
        target_bot_id: str,
        *,
        claim_owner: str,
        steer_capable: bool,
        lease_seconds: int = 300,
    ) -> DeliveryRecord | None:
        """Atomically choose STEERING for an active Claude turn or safe dispatch."""
        if store.engine is None:
            return None
        target = target_bot_id.strip().lower()
        claim_token = f"claim-{uuid.uuid4().hex}"
        from .service.turn_admission import lock_target
        from .service.turn_gates import GateActor, gate_blocks, live_gate_for_owner
        from .service.user_stop_hold import hold_owner

        with store.engine.begin() as conn:
            # The one per-bot admission lock, so a claim linearizes with gate
            # activation and every other admission edge (TASK-1233).
            lock_target(conn, target)
            from .inter_bot_preclaim import load_claim_head, load_user_bypass

            # A user-Stop hold lets the user's own chat run ahead of the held
            # backlog; every other gate still applies to it (TASK-1327).
            head, exempt_owner = None, None
            if live_gate_for_owner(conn, hold_owner(target), fresh=True) is not None:
                head = load_user_bypass(conn, target, store.config)
                exempt_owner = hold_owner(target) if head else None
            if head is None:
                head = load_claim_head(conn, target, store.config)
            if (
                not head
                or head["status"] != QUEUED
                or head["available_at"] > datetime.now(timezone.utc)
            ):
                return None
            if store._explicit_stop_requested(conn, head["turn_id"]):
                conn.execute(sa_text("""
                    UPDATE inter_bot_deliveries SET status='CANCELLED',
                        next_retry_at=NULL, last_error='Stopped by user', updated_at=now()
                    WHERE id=:id AND status='QUEUED'
                """), {"id": head["id"]})
                return None
            if gate_blocks(conn, target, GateActor("delivery_claim", head["turn_id"]),
                           exclude_owner_id=exempt_owner) is not None:
                # A live system gate holds the head: it stays QUEUED with no
                # attempt spent and nothing written, and the target's FIFO
                # waits behind it. The first sweep after release claims it.
                # An explicit Stop above still settles under a gate.
                return None
            from .service.session_lifecycle.recovery import NewSessionRecovery

            if NewSessionRecovery.refuse_accepted_replay(conn, head):
                return None
            if (head.get("metadata_json") or {}).get("new_session_intent"):
                from .service.session_lifecycle.lifecycle import NewSessionLifecycle

                if NewSessionLifecycle.skip_stale_claim(conn, head):
                    return None
            payload = head["payload_json"] if isinstance(head["payload_json"], dict) else {}
            policy = SessionPolicy(head.get("session_policy") or SessionPolicy.CONTINUE.value)
            prefer_steer = bool(payload.get("prefer_steer", True))
            from .inter_bot_steer_only import (
                NO_ACTIVE_RUN, UNSUPPORTED_BACKEND, is_steer_only, settle_not_needed,
            )

            steer_only = is_steer_only(payload)
            if steer_only and not steer_capable:
                # TASK-1431: nothing on this backend can be steered, so the row
                # is not needed; it never waits for or starts a turn.
                return settle_not_needed(conn, head["id"], UNSUPPORTED_BACKEND)
            try:
                delivery_user_id = resolve_delivery_user(store.config, payload.get("user"))
            except ValueError as exc:
                # A recovered legacy claim may still own an open reservation.
                # Release it under this target lock only when the delivery and
                # turn evidence prove it was never accepted or started.
                if head.get("delivery_mode") == "turn" and head.get("transport_accepted_at") is None:
                    from .service.turn_admission import release_reservation
                    release_reservation(
                        store.engine, head["turn_id"], connection=conn,
                        delivery_id=head["id"], bot_id=target,
                        trigger_message_id=head["user_message_id"],
                    )
                conn.execute(sa_text("""
                    UPDATE inter_bot_deliveries SET status='FAILED',
                        last_error=:error, next_retry_at=NULL, updated_at=now()
                    WHERE id=:id AND status='QUEUED'
                """), {"id": head["id"], "error": str(exc)})
                return None
            from .service.turn_admission import classify_target, write_reservation
            # Same classified view save_turn admits against (TASK-1082). Only an
            # unambiguous executing row is a steer target; a fenced reservation
            # or several open rows fail closed (wait), never "newest row wins".
            state = classify_target(conn, target, exclude_turn_id=head["turn_id"])
            busy = not state.idle
            active = state.executing
            if active is None and state.kind == "reserved" and len(state.rows) == 1:
                # TASK-1067: a fallback reservation whose own delivery the target
                # already accepted is a committed start, not an idle fence. Later
                # deliveries steer into it (the bridge answers "not ready" until
                # the run registers) instead of queueing behind it.
                committed = conn.execute(sa_text("""
                    SELECT 1 FROM inter_bot_deliveries
                    WHERE turn_id=:turn_id AND status='DISPATCHING'
                      AND transport_accepted_at IS NOT NULL
                """), {"turn_id": state.rows[0]["id"]}).first()
                if committed:
                    active = state.rows[0]
            if any(str(row.get("user_id") or "") != delivery_user_id for row in state.rows):
                return None
            if busy and policy != SessionPolicy.CONTINUE:
                # Reset is an ordered precondition, never an interrupt. Wait until
                # the authoritative active turn ends; do not steer or rotate under it.
                return None
            existing_steer = (
                head.get("delivery_mode") == "steer"
                and head.get("target_turn_id")
                and prefer_steer
            )
            rejected_same_turn = (
                head.get("delivery_mode") == "steer_rejected"
                and head.get("target_turn_id")
                and active
                and active["id"] == head["target_turn_id"]
            )
            if existing_steer:
                # A steer RPC may have reached the bridge even if the app timed
                # out before seeing its result. Always retry the SAME deterministic
                # RPC/message against the SAME turn until it proves not accepted;
                # only then may the dispatcher clear steer intent for idle fallback.
                status = STEERING
                target_turn_id = head["target_turn_id"]
                delivery_mode = "steer"
            elif MaintenanceBarrier(conn, target).prepare(head) is not None:
                return None
            elif busy:
                if active is None:
                    # Reserved by another producer, or ambiguous open rows.
                    return None
                if rejected_same_turn:
                    # This exact active row definitively rejected the deterministic
                    # steer RPC. Wait for it to end rather than hot-looping or
                    # starting an overlapping fallback turn. A newer active row
                    # may still be steered on the next claim.
                    return None
                if not steer_capable or not prefer_steer:
                    return None
                status = STEERING
                target_turn_id = active["id"]
                delivery_mode = "steer"
            elif steer_only:
                # Idle target: a steer_only row settles as not needed under this
                # lock, with no reservation and no fallback turn (TASK-1431).
                return settle_not_needed(conn, head["id"], NO_ACTIVE_RUN)
            else:
                status = DISPATCHING
                target_turn_id = head["turn_id"]
                delivery_mode = "turn"
                user_id = delivery_user_id
                from .service.session_lifecycle.lifecycle import NewSessionLifecycle

                payload, _, _ = NewSessionLifecycle.rotate_delivery_claim(
                    conn, head=head, payload=payload,
                    target_bot_id=target, user_id=user_id,
                )
                NewSessionLifecycle.bind_wait_predecessor(
                    conn, head, bot_id=target, user_id=user_id,
                )
                write_reservation(
                    conn,
                    turn_id=head["turn_id"],
                    bot_id=target,
                    user_id=user_id,
                    user_prompt=head["message"],
                    request_json=payload,
                    trigger_message_id=head["user_message_id"],
                )
            row = conn.execute(sa_text("""
                UPDATE inter_bot_deliveries SET
                    status=:status, attempt_count=attempt_count+1,
                    dispatch_started_at=now(),
                    lease_expires_at=now() + (:lease * interval '1 second'),
                    claim_token=:claim_token, claim_owner=:claim_owner,
                    target_turn_id=:target_turn_id, delivery_mode=:delivery_mode,
                    next_retry_at=NULL, last_error=NULL, updated_at=now()
                WHERE id=:id AND status='QUEUED'
                RETURNING *
            """), {
                "id": head["id"],
                "status": status,
                "lease": max(60, lease_seconds),
                "claim_token": claim_token,
                "claim_owner": claim_owner,
                "target_turn_id": target_turn_id,
                "delivery_mode": delivery_mode,
            }).mappings().first()
        return DeliveryRecord.from_mapping(row) if row else None
