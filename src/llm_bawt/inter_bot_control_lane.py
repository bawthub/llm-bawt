"""The control lane: system steers that pass a held FIFO head (TASK-1431 S2).

A control row is an ordinary durable delivery with ``lane='control'``. It steers
one pinned run and nothing else, and it never reserves or starts a turn. A
quiesce gate holds every normal row of its bot at the claim; a control row is
the gate owner's own request that the pinned run stop, so the claim takes it
ahead of the held head (design §2.1). Normal rows never see control rows: they
don't head the normal FIFO and don't fence the user-Stop bypass
(``inter_bot_preclaim``).

Authority is never stored on the row. At claim (``ControlClaim``) and again at
the steer grant (``admit_delivery_grant``) the row is admitted only while some
live gate member links it through ``gate_members.control_delivery_id``, in
``steering`` state, for the same bot, turn and pinned request (design §3).

The pinned run (``metadata_json.control.target_run``) is written once at enqueue
and never rewritten. Turn rows are only read here, never locked or written.

Terminal "nothing to steer" is ``CANCELLED`` with ``delivery_mode='skipped'``
and ``last_error='not_needed:<reason>'``; an ask that could not be confirmed is
``FAILED`` with ``last_error='unconfirmed:<reason>'``.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Mapping

from sqlalchemy import text as sa_text
from sqlalchemy.engine import Connection

from .bot_identities import SYSTEM_BOT_ID
from .inter_bot_delivery_record import DeliveryRecord
from .message_authorship import AuthorReference
from .utils.schema import ensure_relations

CONTROL_LANE = "control"
NORMAL_LANE = "normal"
STEERABLE_BACKEND = "claude-code"
TARGET_RUN_KEYS = ("turn_id", "agent_request_id", "backend", "session_key")

# One ask per pinned run, retried until its turn ends or this many attempts
# have been spent (the pump's budget: about a minute of backoff).
CONTROL_MAX_ATTEMPTS = 6
BACKOFF_BASE_SECONDS = 2.0
BACKOFF_CAP_SECONDS = 30.0
GRANT_REFUSED_RETRY_SECONDS = 1.0
GRANT_REFUSED = "control_grant_refused"

NOT_NEEDED = "not_needed"
UNCONFIRMED = "unconfirmed"

_CONTROL_RELATIONS = {
    "ix_inter_bot_delivery_control_head": """
        CREATE INDEX IF NOT EXISTS ix_inter_bot_delivery_control_head
        ON inter_bot_deliveries (target_bot_id, ordinal)
        WHERE lane = 'control' AND status = 'QUEUED'
    """,
    "ix_inter_bot_delivery_control_run": """
        CREATE INDEX IF NOT EXISTS ix_inter_bot_delivery_control_run
        ON inter_bot_deliveries (target_bot_id, ((metadata_json->'control'->>'run_key')))
        WHERE lane = 'control'
    """,
}


def ensure_control_lane_schema(conn: Connection) -> None:
    """Add ``lane`` and its indexes when the catalog lacks them (store bootstrap).

    One catalog read decides, so a migrated table takes no ACCESS EXCLUSIVE
    lock. Old code ignores the column; every existing row is ``normal``.
    """
    present = conn.execute(sa_text("""
        SELECT 1 FROM pg_attribute
        WHERE attrelid = to_regclass('inter_bot_deliveries') AND attname = 'lane'
          AND attnum > 0 AND NOT attisdropped
    """)).first()
    if present is None:
        conn.execute(sa_text(
            "ALTER TABLE inter_bot_deliveries"
            " ADD COLUMN IF NOT EXISTS lane VARCHAR(16) NOT NULL DEFAULT 'normal'"
        ))
    ensure_relations(conn, _CONTROL_RELATIONS)


def control_metadata(record_or_row: Any) -> Mapping[str, Any]:
    """The immutable ``metadata_json.control`` block, or {} for a normal row."""
    metadata = getattr(record_or_row, "metadata", None)
    if metadata is None and isinstance(record_or_row, Mapping):
        metadata = record_or_row.get("metadata_json")
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    control = (metadata or {}).get("control")
    return control if isinstance(control, Mapping) else {}


def _pin(row: Any) -> Mapping[str, Any]:
    target_run = control_metadata(row).get("target_run")
    return target_run if isinstance(target_run, Mapping) else {}


def _payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = row.get("payload_json")
    if isinstance(payload, str):
        payload = json.loads(payload)
    return payload if isinstance(payload, Mapping) else {}


def enqueue_control(
    store,
    conn: Connection,
    *,
    target_bot_id: str,
    target_run: Mapping[str, Any],
    run_key: str,
    idempotency_key: str,
    message: str,
    user: str,
) -> tuple[DeliveryRecord, bool]:
    """Admit one control row in the caller's transaction (``DeliverySender`` only).

    The caller is the system principal and holds ``lock_target`` (re-entrant).
    The row is authored by ``system``, steer-only, and waits at the claim like
    any ``hold`` producer. ``lane`` is set in the same transaction, so no claim
    ever sees the row as normal. A duplicate is returned as stored: the
    producer's link step checks it before linking (design §3 step 4).
    """
    pin = {key: target_run.get(key) for key in TARGET_RUN_KEYS}
    if not pin["turn_id"] or not pin["agent_request_id"]:
        raise ValueError("a control row needs a pinned turn and request")
    if not run_key or not idempotency_key or not user:
        raise ValueError("a control row needs run_key, idempotency_key and user")
    target = target_bot_id.strip().lower()
    record, duplicate = store.enqueue(
        sender_bot_id=SYSTEM_BOT_ID,
        target_bot_id=target,
        message=message,
        payload={
            "messages": [{"role": "user", "content": message}],
            "bot_id": target,
            "user": user,
            "prefer_steer": True,
            "steer_only": True,
            "extract_memory": False,
            "stream": False,
        },
        author=AuthorReference.bot(SYSTEM_BOT_ID),
        idempotency_key=idempotency_key,
        metadata={CONTROL_LANE: {"target_run": pin, "run_key": run_key}},
        max_attempts=CONTROL_MAX_ATTEMPTS,
        connection=conn,
        producer="bot",
        on_gate="hold",
    )
    if duplicate:
        return record, True
    row = conn.execute(sa_text("""
        UPDATE inter_bot_deliveries SET lane='control'
        WHERE id=:id AND status='QUEUED' RETURNING *
    """), {"id": record.id}).mappings().one()
    return DeliveryRecord.from_mapping(row), False


# A live member that authorizes control row :delivery for its pinned run.
_LINKED_MEMBER = (
    "m.control_delivery_id = :delivery AND m.control_state = 'steering'"
    " AND m.bot_id = :bot AND m.turn_id = :turn AND m.agent_request_id = :request"
)


def linked_member(conn: Connection, *, delivery_id: str, bot_id: str,
                  pin: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The oldest live member linking ``delivery_id`` for its pin; reads only."""
    from .service.system_gate_admission import LIVE_MEMBER_SQL

    if not pin.get("turn_id") or not pin.get("agent_request_id"):
        return None
    return conn.execute(sa_text(
        LIVE_MEMBER_SQL.format(where=_LINKED_MEMBER) + " ORDER BY m.id LIMIT 1"
    ), {"delivery": delivery_id, "bot": bot_id, "turn": pin["turn_id"],
        "request": pin["agent_request_id"]}).mappings().first()


def _pinned_target(member: Mapping[str, Any], pin: Mapping[str, Any]):
    """The member's ``control_target`` when it is exactly the pin, else None."""
    from .service.system_gate_admission import control_target

    target = control_target(member)
    if target is None or target.request_id != pin.get("agent_request_id"):
        return None
    if target.backend != pin.get("backend") or target.session_key != pin.get("session_key"):
        return None
    return target


class ControlClaim:
    """The claim-side control path, called from ``claim_next`` under ``lock_target``."""

    @staticmethod
    def load_head(conn: Connection, target: str) -> Mapping[str, Any] | None:
        """The oldest ready QUEUED control row for ``target``, locked; else None.

        Unlike the user-Stop bypass it does not wait for claimed rows to
        finish: a control steer reserves nothing and only adds a message to a
        run that is already executing.
        """
        return conn.execute(sa_text("""
            SELECT * FROM inter_bot_deliveries
            WHERE target_bot_id=:target AND lane='control' AND status='QUEUED'
              AND available_at <= clock_timestamp()
            ORDER BY ordinal ASC LIMIT 1
            FOR UPDATE
        """), {"target": target}).mappings().first()

    @classmethod
    def claim(
        cls, conn: Connection, target: str, *, claim_owner: str, lease_seconds: int,
    ) -> DeliveryRecord | None:
        """Claim or settle the target's control head; None when there is none.

        Returns a STEERING record to steer, or the terminal record the head
        was settled to (the dispatcher emits it and drains on). There is no
        "leave it QUEUED" outcome: whether a run is ready is the bridge's
        answer to the steer, retried with backoff (``ControlSteerFailure``).
        """
        head = cls.load_head(conn, target)
        if head is None:
            return None
        pin = _pin(head)
        member = linked_member(conn, delivery_id=head["id"], bot_id=target, pin=pin)
        if member is None:
            return cls._settle(conn, head, "CANCELLED", f"{NOT_NEEDED}:control_revoked")
        if member["turn_present"] is None or member["turn_ended"]:
            return cls._settle(conn, head, "CANCELLED", f"{NOT_NEEDED}:run_ended")
        if not pin.get("backend") or not pin.get("session_key") or _pinned_target(member, pin) is None:
            return cls._settle(conn, head, "FAILED", f"{UNCONFIRMED}:pinned_run_unaddressable")
        if pin["backend"] != STEERABLE_BACKEND:
            return cls._settle(conn, head, "CANCELLED", f"{NOT_NEEDED}:unsupported_backend")
        if str(_payload(head).get("user") or "") != str(member["turn_user_id"] or ""):
            # steer_active_turn refuses an ownership mismatch on every attempt.
            return cls._settle(conn, head, "FAILED", f"{UNCONFIRMED}:user_mismatch")
        row = conn.execute(sa_text("""
            UPDATE inter_bot_deliveries SET
                status='STEERING', attempt_count=attempt_count+1,
                dispatch_started_at=now(),
                lease_expires_at=now() + (:lease * interval '1 second'),
                claim_token=:claim_token, claim_owner=:claim_owner,
                target_turn_id=:turn, delivery_mode='steer',
                next_retry_at=NULL, last_error=NULL, updated_at=now()
            WHERE id=:id AND status='QUEUED' AND lane='control'
            RETURNING *
        """), {
            "id": head["id"], "lease": max(60, lease_seconds),
            "claim_token": f"claim-{uuid.uuid4().hex}", "claim_owner": claim_owner,
            "turn": pin["turn_id"],
        }).mappings().one()
        return DeliveryRecord.from_mapping(row)

    @staticmethod
    def _settle(conn: Connection, head: Mapping[str, Any], status: str, reason: str) -> DeliveryRecord:
        row = conn.execute(sa_text("""
            UPDATE inter_bot_deliveries SET status=:status,
                delivery_mode=CASE WHEN :status='CANCELLED' THEN 'skipped' ELSE delivery_mode END,
                last_error=:reason, next_retry_at=NULL, claim_token=NULL, claim_owner=NULL,
                lease_expires_at=NULL, updated_at=now()
            WHERE id=:id AND status='QUEUED' AND lane='control'
            RETURNING *
        """), {"id": head["id"], "status": status, "reason": reason}).mappings().one()
        return DeliveryRecord.from_mapping(row)


@dataclass(frozen=True)
class DeliveryControlGrant:
    """A control row's claim, carried by its steer (design §2.3). Internal only.

    No HTTP field can carry one. It authorizes nothing by itself:
    ``check_steer`` re-reads the row and its linking member under
    ``lock_target`` (``admit_delivery_grant``).
    """

    bot_id: str
    turn_id: str
    delivery_id: str
    claim_token: str

    @classmethod
    def of(cls, record: DeliveryRecord) -> "DeliveryControlGrant | None":
        """The grant a claimed control record steers under; None for a normal row."""
        if getattr(record, "lane", NORMAL_LANE) != CONTROL_LANE:
            return None
        return cls(
            bot_id=record.target_bot_id,
            turn_id=str(_pin(record).get("turn_id") or ""),
            delivery_id=record.id,
            claim_token=record.claim_token or "",
        )


def admit_delivery_grant(conn: Connection, grant: DeliveryControlGrant):
    """The grant's pinned run, under the caller's ``lock_target``; else refuse.

    Admits only a live claim: the row is a STEERING control row under this
    claim token with an unexpired lease (read on the current clock, after the
    lock wait), for the grant's bot and pinned turn, and a live member still
    links it for that pin. Reads only.
    """
    from .service.system_gate_admission import ControlGrantRefused

    row = conn.execute(sa_text("""
        SELECT * FROM inter_bot_deliveries
        WHERE id=:id AND lane='control' AND status='STEERING' AND claim_token=:token
          AND lease_expires_at > clock_timestamp() AND target_bot_id=:bot
    """), {"id": grant.delivery_id, "token": grant.claim_token,
           "bot": grant.bot_id}).mappings().first()
    pin = _pin(row) if row is not None else {}
    if row is None or not grant.turn_id or pin.get("turn_id") != grant.turn_id:
        raise ControlGrantRefused(grant)
    member = linked_member(conn, delivery_id=grant.delivery_id, bot_id=grant.bot_id, pin=pin)
    target = _pinned_target(member, pin) if member is not None else None
    if target is None:
        raise ControlGrantRefused(grant)
    return target


class ControlSteerFailure:
    """Settle a control steer that did not land; never a fallback or a reservation.

    Under the delivery row lock, against the pinned turn row (read only):
    * the grant refused → back to QUEUED, no attempt spent; the next claim
      re-admits it or settles it ``control_revoked``;
    * the pinned turn ended or gone → ``not_needed:run_ended``;
    * attempts spent → ``unconfirmed:<error>``;
    * otherwise → QUEUED with backoff, spending the attempt. The steer
      generation advances only on a definitive rejection before transport
      acceptance (TASK-1251), so an accepted ask is never re-sent under a new id.
    """

    @classmethod
    def settle(cls, store, record: DeliveryRecord, error: str) -> DeliveryRecord | None:
        from .service.inter_bot_dispatcher_steer import DEFINITIVE_NOT_ACCEPTED, SteerGeneration

        token = record.claim_token or ""
        if store.engine is None or not token:
            return None
        with store.engine.begin() as conn:
            row = conn.execute(sa_text("""
                SELECT * FROM inter_bot_deliveries
                WHERE id=:id AND lane='control' AND status='STEERING' AND claim_token=:token
                FOR UPDATE
            """), {"id": record.id, "token": token}).mappings().first()
            if row is None:
                return None
            if error == GRANT_REFUSED:
                return store.requeue_claim(
                    conn, record.id, token, error,
                    delay_seconds=GRANT_REFUSED_RETRY_SECONDS, refund_attempt=True,
                )
            turn = conn.execute(sa_text(
                "SELECT ended_at IS NOT NULL AS ended FROM turn_logs WHERE id=:id"
            ), {"id": _pin(row).get("turn_id")}).mappings().first()
            if turn is None or turn["ended"]:
                return cls._settle_claimed(conn, row, "CANCELLED", f"{NOT_NEEDED}:run_ended")
            attempts = int(row["attempt_count"] or 0)
            if attempts >= int(row["max_attempts"] or 0):
                return cls._settle_claimed(conn, row, "FAILED", f"{UNCONFIRMED}:{error}"[:4000])
            accepted = row["transport_accepted_at"] is not None
            advance = not accepted and error in DEFINITIVE_NOT_ACCEPTED
            return store.requeue_claim(
                conn, record.id, token, error,
                delay_seconds=cls.backoff_seconds(attempts),
                metadata_merge=SteerGeneration.of(DeliveryRecord.from_mapping(row)).advance()
                if advance else None,
            )

    @staticmethod
    def backoff_seconds(attempts: int) -> float:
        """2, 4, 8, 16, then 30 s between asks of a still-open run."""
        return min(BACKOFF_CAP_SECONDS, BACKOFF_BASE_SECONDS * 2 ** max(0, attempts - 1))

    @staticmethod
    def _settle_claimed(conn: Connection, row: Mapping[str, Any], status: str,
                        reason: str) -> DeliveryRecord | None:
        settled = conn.execute(sa_text("""
            UPDATE inter_bot_deliveries SET status=:status,
                delivery_mode=CASE WHEN :status='CANCELLED' THEN 'skipped' ELSE delivery_mode END,
                last_error=:reason, next_retry_at=NULL, claim_token=NULL, claim_owner=NULL,
                lease_expires_at=NULL, updated_at=now()
            WHERE id=:id AND status='STEERING' AND claim_token=:token
            RETURNING *
        """), {"id": row["id"], "token": row["claim_token"], "status": status,
               "reason": reason}).mappings().first()
        return DeliveryRecord.from_mapping(settled) if settled else None


__all__ = [
    "CONTROL_LANE", "CONTROL_MAX_ATTEMPTS", "ControlClaim", "ControlSteerFailure",
    "DeliveryControlGrant", "GRANT_REFUSED", "NORMAL_LANE", "admit_delivery_grant",
    "control_metadata", "enqueue_control", "ensure_control_lane_schema", "linked_member",
]
