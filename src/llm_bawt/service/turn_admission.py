"""Single backend admission authority for agent turns (TASK-1082).

Every producer that opens a turn row goes through ``TurnLogStore.save_turn``
(streaming, nonstream and background callers alike) under the per-bot advisory
lock ``pg_advisory_xact_lock(hashtext(bot_id))``. The durable delivery claim
(``InterBotDeliveryStore.claim_next``) takes the same lock. This module owns
the decisions both need, so neither re-implements them:

* ``classify_target`` — the one classified view of the bot's open turn rows:
  idle / reserved / exactly-one-executing / ambiguous. Nothing here picks
  "the latest open row" as a target; ambiguity fails closed.
* ``TurnAdmission`` — whether a request may start beside an open turn.
* ``reserve_turn`` — the fenced reservation that claims an idle target BEFORE
  a producer performs external side effects (bridge grants, session rotation).

Producers never decide busy/idle themselves. Server-originated work that must
not overlap the target's running turn is IDLE_ONLY; losing the race raises
``TargetBusy`` under the lock, and the producer's own durable outbox re-queues
without counting an attempt. Steering and held-tool results are separate
transport operations (``steer_active_turn`` / ``send_tool_result``) and never
pass through here.
"""

from __future__ import annotations

import json
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Mapping

from sqlalchemy import bindparam, text as sa_text, update
from sqlmodel import Session, select

from ..approval_mcp_store import requeued_continuation_state, returned_claim
from ..approval_models import CONT_DISPATCHING, CONT_SUPPRESSED, ToolApprovalRequest
from ..inter_bot_delivery import RESERVATION_KEPT_RESET_STATUSES, InterBotDeliveryStore
from ..request_identity import AdmissionUnavailable as IdentityAdmissionUnavailable
from .turn_gates import (
    ACCEPTED,
    REVOKED_UNSTARTED,
    RUNNING,
    GateActor,
    GateBlock,
    GateMember,
    GateSpec,
    bots_in_scope,
    carry_forward,
    gate_blocks,
    has_execution_evidence,
    initial_control_state,
    insert_gate,
    insert_member,
)
from .turn_termination import rearm_request_payload, settle_open_turn

log = logging.getLogger(__name__)

ADMISSION_KEY = "admission"
RESERVED = "reserved"
# request_json key fencing a reservation to the producer claim that wrote it.
RESERVATION_OWNER_KEY = "reservation_owner"


class TargetBusy(RuntimeError):
    """A request lost the target's admission race (nothing was started)."""


class SystemGateRefused(TargetBusy, IdentityAdmissionUnavailable):
    """A live system gate refused this admission; nothing was started or written.

    One refusal honoured by every existing handler (TASK-1233 C4): durable
    producers already treat ``TargetBusy`` as "not an attempt" and defer with
    a refund, and the HTTP edges map the identity ``AdmissionUnavailable`` to
    ``status_code``: 503 ``system_frozen`` under a deny gate, 409
    ``system_held`` under a hold gate.
    """

    def __init__(self, block: GateBlock):
        self.block = block
        self.code = block.refusal
        self.status_code = 503 if self.code == "system_frozen" else 409
        detail = self.code + (f": {block.reason}" if block.reason else "")
        super().__init__(f"{detail} (until {block.expires_at.isoformat()})")


def refuse_if_gated(conn, bot_id: str, actor: GateActor) -> None:
    """Raise ``SystemGateRefused`` when a live gate blocks ``actor`` (under the lock)."""
    block = gate_blocks(conn, bot_id, actor)
    if block is not None:
        raise SystemGateRefused(block)


class AdmissionFenceLost(RuntimeError):
    """The producer no longer owns the work it tried to reserve for."""


class AdmissionUnavailable(RuntimeError):
    """The admission authority could not decide; the turn must not start.

    Admission is mandatory, not best-effort telemetry: a store failure or an
    unavailable configured database fails closed for every producer.
    """


@dataclass(frozen=True)
class AdmissionClaim:
    """Trusted, Python-only producer authority over one deterministic turn id.

    ``owner`` is the producer's claim token; approval reservations stamp it
    on their turn rows, while delivery reservations are unowned until consumed.
    ``fence(conn)`` proves inside the caller's transaction that the producer
    still holds its durable claim. Delivery claims also require an open reserved
    row; neither permission is accepted from public request JSON.
    """

    owner: str
    fence: Callable[[Any], bool]
    # Delivery claim_next writes its reservation before the claim is consumed;
    # unlike an approval reservation, that row has no reservation_owner. Only
    # the trusted delivery-claim builder may opt into consuming such a row.
    allow_unowned_reservation: bool = False
    require_reservation: bool = False
    reservation_delivery_id: str | None = None


class TurnAdmission(str, Enum):
    # Ordinary ingress: loses only to a durable reservation. Concurrent START
    # turns are still serialized by the bridge SessionQueue (TASK-1082 slice 2
    # replaces this with backend queueing).
    START = "start"
    # Start only when the target is idle; else ``TargetBusy``.
    IDLE_ONLY = "idle_only"

    @classmethod
    def for_request(cls, request: Any) -> "TurnAdmission":
        """Derive admission from trusted server-side authority only.

        Both markers are claim-validated before execution (see
        ``inter_bot_claims.validate_inter_bot_claim``); the approval claim is a
        Python-only private attribute that public JSON cannot set.
        """
        if getattr(request, "inter_bot_delivery_id", None):
            return cls.IDLE_ONLY
        if getattr(request, "_internal_approval_claim", None) is not None:
            return cls.IDLE_ONLY
        return cls.START

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "TurnAdmission":
        if not payload:
            return cls.START
        delivery_id = payload.get("inter_bot_delivery_id")
        if isinstance(delivery_id, str) and delivery_id.strip():
            return cls.IDLE_ONLY
        try:
            return cls(payload.get(ADMISSION_KEY) or cls.START.value)
        except ValueError:
            return cls.START


@dataclass(frozen=True)
class TargetState:
    """Classified open-turn rows for one bot (excluding the caller's own id)."""

    rows: tuple[Mapping[str, Any], ...]

    @property
    def kind(self) -> str:
        if not self.rows:
            return "idle"
        if any(row["status"] == RESERVED for row in self.rows):
            return "reserved"
        return "executing" if len(self.rows) == 1 else "ambiguous"

    @property
    def idle(self) -> bool:
        return not self.rows

    @property
    def executing(self) -> Mapping[str, Any] | None:
        """The one unambiguous running turn, or None (never a guess)."""
        return self.rows[0] if self.kind == "executing" else None

    def blocking_id(self) -> str | None:
        return str(self.rows[0]["id"]) if self.rows else None

    def admits(self, admission: TurnAdmission) -> bool:
        if admission is TurnAdmission.IDLE_ONLY:
            return self.idle
        return self.kind != "reserved"

    def describe(self) -> str:
        ids = ", ".join(str(row["id"]) for row in self.rows[:3])
        return f"{self.kind} ({ids})" if ids else self.kind


def lock_target(conn, bot_id: str) -> None:
    """The shared per-bot admission lock (no-op off PostgreSQL)."""
    if conn.dialect.name == "postgresql":
        conn.execute(
            sa_text("SELECT pg_advisory_xact_lock(hashtext(:target))"),
            {"target": bot_id},
        )


_OPEN_TARGET_ROWS = sa_text(
    "SELECT bot_id, id, status, agent_session_key, agent_request_id, user_id, created_at"
    " FROM turn_logs WHERE bot_id IN :targets AND ended_at IS NULL"
    " AND id != :turn_id ORDER BY created_at DESC"
).bindparams(bindparam("targets", expanding=True))


def classify_target(conn, bot_id: str, *, exclude_turn_id: str | None) -> TargetState:
    """All other open turn rows for ``bot_id``, newest first (call under the lock)."""
    return classify_targets(conn, (bot_id,), exclude_turn_id=exclude_turn_id)[bot_id]


def classify_targets(
    conn, bot_ids, *, exclude_turn_id: str | None = None,
) -> dict[str, TargetState]:
    """``classify_target`` for many bots in one query (TASK-1400).

    The single-bot form delegates here, so a batched readout and admission
    classify the same rows the same way. Without each bot's lock this is a
    ``peek_target``-grade advisory view.
    """
    targets = tuple(dict.fromkeys(bot_ids))
    grouped: dict[str, list[dict[str, Any]]] = {bot: [] for bot in targets}
    if targets:
        rows = conn.execute(
            _OPEN_TARGET_ROWS, {"targets": list(targets), "turn_id": exclude_turn_id or ""},
        ).mappings().all()
        for row in rows:
            values = dict(row)
            grouped[values.pop("bot_id")].append(values)
    return {bot: TargetState(tuple(rows)) for bot, rows in grouped.items()}


def peek_target(engine, bot_id: str, *, exclude_turn_id: str | None) -> TargetState:
    """Unlocked, read-only ``classify_target`` for cheap pre-checks.

    Advisory only: it lets a producer skip expensive preparation while the
    target is plainly busy. It never admits anything — ``reserve_turn`` stays
    the authority and re-classifies under the per-bot lock.
    """
    with engine.connect() as conn:
        return classify_target(conn, bot_id, exclude_turn_id=exclude_turn_id)


def peek_targets(engine, bot_ids) -> dict[str, TargetState]:
    """Unlocked, read-only ``classify_targets``: advisory, admits nothing."""
    with engine.connect() as conn:
        return classify_targets(conn, bot_ids)


def _reservation_payload(request_json: Any) -> Mapping[str, Any]:
    if not request_json:
        return {}
    try:
        payload = json.loads(request_json) if isinstance(request_json, str) else request_json
    except (TypeError, ValueError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def fence_consumption(conn, *, turn_id: str, claim: AdmissionClaim | None) -> str | None:
    """Prove the caller may turn ``turn_id`` into a running turn (call under the lock).

    The one consumption fence for every producer, in the caller's transaction:

    * a presented ``claim`` must still hold (``AdmissionFenceLost`` otherwise),
      whether or not a row exists yet;
    * an open reservation is consumable only by the exact owner that wrote it:
      a fenced reservation needs the matching claim. An unowned delivery
      reservation requires a live delivery claim; ordinary unclaimed requests
      remain compatible with legacy unowned reservations until all producers
      have migrated.

    Returns the existing row's status (row locked on PostgreSQL), or None.
    """
    if claim is not None and not claim.fence(conn):
        raise AdmissionFenceLost(f"claim for turn '{turn_id}' is no longer held")
    lock = " FOR UPDATE" if conn.dialect.name == "postgresql" else ""
    row = conn.execute(
        sa_text("SELECT status, ended_at, request_json FROM turn_logs WHERE id=:id" + lock),
        {"id": turn_id},
    ).mappings().first()
    if row is None:
        if claim is not None and claim.require_reservation:
            raise AdmissionFenceLost(f"reservation for turn '{turn_id}' disappeared")
        return None
    if claim is not None and claim.require_reservation and (
        row["status"] != RESERVED or row["ended_at"] is not None
    ):
        raise AdmissionFenceLost(f"reservation for turn '{turn_id}' is no longer open")
    if row["status"] == RESERVED and row["ended_at"] is None:
        payload = _reservation_payload(row["request_json"])
        owner = payload.get(RESERVATION_OWNER_KEY)
        owner = owner if isinstance(owner, str) and owner else None
        expected = (
            None if claim is None or claim.allow_unowned_reservation else claim.owner
        )
        if owner != expected or (
            claim is None and payload.get("inter_bot_delivery_id")
        ) or (
            claim is not None and claim.reservation_delivery_id is not None
            and payload.get("inter_bot_delivery_id") != claim.reservation_delivery_id
        ):
            raise AdmissionFenceLost(
                f"reservation for turn '{turn_id}' belongs to another claim"
            )
    return row["status"]


def write_reservation(
    conn,
    *,
    turn_id: str,
    bot_id: str,
    user_id: str,
    user_prompt: str,
    request_json: dict[str, Any],
    trigger_message_id: str | None,
    only_unstarted: bool = False,
) -> bool:
    """Insert or re-arm a reserved row (caller holds the lock and decided idle).

    ``only_unstarted`` re-arms an existing row of the same id only while it is
    still an open reservation; a started or finished row is never rewritten
    and ``False`` is returned. Without it (durable delivery retries) an ended
    row is deliberately re-armed for the next attempt.
    """
    guard = (
        " WHERE turn_logs.status='reserved' AND turn_logs.ended_at IS NULL"
        if only_unstarted else ""
    )
    # A re-arm starts a new attempt on the same row: carry the prior attempt's
    # audits and history forward (TASK-1232 r2-M1, TASK-1234 Fix 2).
    lock = " FOR UPDATE" if conn.dialect.name == "postgresql" else ""
    prior = conn.execute(sa_text(
        "SELECT request_json, status, end_reason, error_text, created_at, ended_at"
        " FROM turn_logs WHERE id=:id" + lock
    ), {"id": turn_id}).mappings().first()
    merged_request_json = rearm_request_payload(prior, request_json)
    result = conn.execute(sa_text("""
        INSERT INTO turn_logs (
            id, created_at, path, stream, bot_id, user_id, status,
            user_prompt, request_json, response_text,
            trigger_message_id, ended_at
        ) VALUES (
            :turn_id, now(), '/v1/chat/completions', false, :target,
            :user_id, 'reserved', :message, CAST(:request_json AS jsonb)::text,
            '', :message_id, NULL
        )
        ON CONFLICT (id) DO UPDATE SET
            status='reserved', ended_at=NULL, error_text=NULL, created_at=EXCLUDED.created_at,
            request_json=EXCLUDED.request_json, user_prompt=EXCLUDED.user_prompt
    """ + guard), {
        "turn_id": turn_id,
        "target": bot_id,
        "user_id": user_id,
        "message": user_prompt,
        "request_json": merged_request_json,
        "message_id": trigger_message_id,
    })
    return bool(result.rowcount)


def reserve_turn(
    engine,
    *,
    turn_id: str,
    bot_id: str,
    user_id: str,
    user_prompt: str,
    request_json: dict[str, Any],
    trigger_message_id: str | None,
    owner: str,
    fence: Callable[[Any], bool] | None = None,
) -> None:
    """Atomically claim an idle target with a fenced reservation, else TargetBusy.

    The reservation is its own short transaction; callers then perform async
    side effects without holding a DB lock. While it exists every other
    admission loses (``TargetState.admits``) and steering does not target it.
    ``save_turn`` for the same id upserts it into the running turn.

    ``owner`` (the producer's claim token) fences ``release_reservation``: a
    worker whose claim was taken over cannot drop the reclaimer's re-armed
    reservation. A same-id row that already started or finished is never
    re-armed; that is reported as ``TargetBusy`` so the outbox re-evaluates.

    ``fence(conn)`` proves, inside this same transaction, that the producer
    still owns its durable claim; otherwise ``AdmissionFenceLost``. A live
    system gate covering the bot refuses with ``SystemGateRefused`` (a
    ``TargetBusy``, so the producer defers without spending an attempt).
    """
    with engine.begin() as conn:
        lock_target(conn, bot_id)
        if fence is not None and not fence(conn):
            raise AdmissionFenceLost(f"claim for turn '{turn_id}' is no longer held")
        refuse_if_gated(conn, bot_id, GateActor("reserve", turn_id))
        state = classify_target(conn, bot_id, exclude_turn_id=turn_id)
        if not state.idle:
            raise TargetBusy(f"target '{bot_id}' is {state.describe()}")
        if not write_reservation(
            conn, turn_id=turn_id, bot_id=bot_id, user_id=user_id,
            user_prompt=user_prompt,
            request_json={**request_json, RESERVATION_OWNER_KEY: owner},
            trigger_message_id=trigger_message_id, only_unstarted=True,
        ):
            raise TargetBusy(f"turn '{turn_id}' already started")


def release_reservation(
    engine, turn_id: str, *, owner: str | None = None,
    fence: Callable[[Any], bool] | None = None,
    delivery_id: str | None = None, bot_id: str | None = None,
    trigger_message_id: str | None = None, connection=None,
) -> bool:
    """Release a provably unstarted reservation, fenced to its producer.

    Approval reservations carry ``owner``. Delivery reservations are unowned;
    their persisted delivery and trigger IDs must match, and any acceptance,
    execution, side-effect or ambiguous open-turn evidence prevents release.
    The delivery caller holds the target lock and supplies its transaction.
    """
    if bool(owner) == bool(delivery_id):
        raise ValueError("exactly one reservation owner or delivery_id is required")
    with (nullcontext(connection) if connection is not None else engine.begin()) as conn:
        if fence is not None and not fence(conn):
            return False
        if delivery_id is None:
            result = conn.execute(
                sa_text(
                    "DELETE FROM turn_logs WHERE id=:id AND status='reserved'"
                    " AND ended_at IS NULL"
                    " AND CAST(request_json AS jsonb)->>:key = :owner"
                ),
                {"id": turn_id, "key": RESERVATION_OWNER_KEY, "owner": owner},
            )
        else:
            if not bot_id or not trigger_message_id:
                return False
            lock = " FOR UPDATE" if conn.dialect.name == "postgresql" else ""
            row = conn.execute(sa_text(
                "SELECT status, ended_at, bot_id, trigger_message_id, request_json,"
                " stream, agent_session_key, agent_request_id, response_text"
                " FROM turn_logs WHERE id=:id" + lock
            ), {"id": turn_id}).mappings().first()
            if row is None:
                return False
            payload = _reservation_payload(row["request_json"])
            if (row["status"] != RESERVED or row["ended_at"] is not None
                    or row["bot_id"] != bot_id or row["stream"] is not False
                    or row["trigger_message_id"] != trigger_message_id
                    or payload.get("inter_bot_delivery_id") != delivery_id
                    or payload.get(RESERVATION_OWNER_KEY)
                    or "inter_bot_claim_token" in payload
                    or row["agent_session_key"] is not None
                    or row["agent_request_id"] is not None
                    or row["response_text"]
                    or not classify_target(conn, bot_id, exclude_turn_id=turn_id).idle
                    or conn.execute(sa_text(
                        "SELECT 1 FROM tool_call_records WHERE turn_id=:id LIMIT 1"
                    ), {"id": turn_id}).first()):
                return False
            result = conn.execute(sa_text(
                "DELETE FROM turn_logs WHERE id=:id AND bot_id=:bot"
                " AND trigger_message_id=:message AND status='reserved'"
                " AND ended_at IS NULL AND stream IS FALSE AND request_json=:request_json"
                " AND agent_session_key IS NULL AND agent_request_id IS NULL"
                " AND COALESCE(response_text, '') = ''"
            ), {"id": turn_id, "bot": bot_id, "message": trigger_message_id,
                "request_json": row["request_json"]})
    return bool(result.rowcount)


REVOKE_DISPOSITIONS = frozenset({"user_stop", "hold"})


def revoke_reservation(
    engine, turn_id: str, *, disposition: str, audit: Mapping[str, Any], connection=None,
) -> dict[str, Any] | None:
    """End an open reservation by first revoking its producer's authority.

    One transaction in the admission lock order (target lock → producer row →
    turn row), so a stalled claimant can never consume the row —
    ``fence_consumption`` checks the claim before the row exists or not.

    * ``user_stop`` (TASK-1232 Stop on a ``reserved`` row): the producer
      loses its claim (delivery CANCELLED, approval continuation
      ``suppressed``), then the row settles through the shared settle.
    * ``hold`` (TASK-1233 B4, from ``activate_hold``): the producer gets its
      claim back without spending an attempt — the delivery returns to QUEUED
      through ``requeue_claim(refund_attempt)``, the approval continuation to
      pending through ``returned_claim`` (cancelled → suppressed, S1) — then
      the row is released, so the held work re-claims after thaw.

    Returns the producer outcome, or None, with nothing written, when
    ``turn_id`` is not (or is no longer) an open reservation. ``connection``
    runs inside the caller's transaction (``activate_hold``).

    Every lock is taken, and the turn row re-checked, BEFORE any write (Vex
    B1): a claimant holding the target lock may consume the row between the
    unlocked head read and our lock, and revoking a consumed row's producer
    would cancel a live run's delivery.
    """
    if disposition not in REVOKE_DISPOSITIONS:
        raise ValueError(f"unsupported reservation disposition: {disposition!r}")
    for _ in range(_REVOKE_ATTEMPTS):
        with (nullcontext(connection) if connection is not None else engine.begin()) as conn, \
                Session(bind=conn) as session:
            outcome = _revoke_reservation_once(conn, session, turn_id, disposition=disposition, audit=audit)
        if outcome is not _PRODUCER_CHANGED:
            return outcome
    log.warning("reservation %s changed producer on every revoke attempt; left open", turn_id)
    return None


# The head read is unlocked, so the row's producer can change before it is
# locked (recover_expired re-arms without the target lock). Retry, bounded.
_REVOKE_ATTEMPTS = 3
_PRODUCER_CHANGED = object()


@dataclass(frozen=True)
class _LockedReservation:
    row: Any  # the TurnLog row, held FOR UPDATE
    delivery_id: str | None
    owner: str | None


def _lock_open_reservation(conn, session: Session, turn_id: str, *, owner: str | None = None):
    """Lock an open reservation in the admission lock order; write nothing.

    The one lock protocol for every revoke disposition: target lock → producer
    row → turn row, with the turn re-checked under its own lock. A writer that
    revokes before this can take a live claimant's authority (Vex B1, Loopy B1).
    Returns the locked reservation; None when ``turn_id`` is not (or no longer)
    an open reservation, or isn't ``owner``'s; ``_PRODUCER_CHANGED`` when its
    producer changed between the unlocked head read and the lock.
    """
    from .turn_logs import TurnLog  # turn_logs imports this module

    head = conn.execute(sa_text(
        "SELECT bot_id, request_json FROM turn_logs"
        " WHERE id=:id AND status='reserved' AND ended_at IS NULL"
    ), {"id": turn_id}).mappings().first()
    if head is None or not head["bot_id"]:
        return None
    lock_target(conn, head["bot_id"])
    payload = _reservation_payload(head["request_json"])
    delivery_id = payload.get("inter_bot_delivery_id")
    head_owner = payload.get(RESERVATION_OWNER_KEY)
    if owner is not None and head_owner != owner:
        return None
    lock = " FOR UPDATE" if conn.dialect.name == "postgresql" else ""
    if delivery_id:
        conn.execute(sa_text("SELECT 1 FROM inter_bot_deliveries WHERE id=:id" + lock), {"id": delivery_id})
    elif head_owner:
        conn.execute(sa_text(
            "SELECT 1 FROM tool_approval_requests WHERE continuation_claim_token=:owner" + lock
        ), {"owner": head_owner})
    row = session.exec(select(TurnLog).where(TurnLog.id == turn_id).with_for_update()).first()
    if row is None or row.status != RESERVED or row.ended_at is not None:
        return None  # consumed or ended since the head read
    locked = _reservation_payload(row.request_json)
    if (locked.get("inter_bot_delivery_id"), locked.get(RESERVATION_OWNER_KEY)) != (delivery_id, head_owner):
        return _PRODUCER_CHANGED
    return _LockedReservation(row, delivery_id, head_owner)


def _revoke_reservation_once(conn, session: Session, turn_id: str, *, disposition: str,
                             audit: Mapping[str, Any]):
    locked = _lock_open_reservation(conn, session, turn_id)
    if not isinstance(locked, _LockedReservation):
        return locked
    if disposition == "hold":
        return _hold_producer(conn, turn_id, locked, reason=str(audit.get("reason") or "system gate hold"))
    producer = _revoke_producer(conn, turn_id, delivery_id=locked.delivery_id, owner=locked.owner)
    settle_open_turn(session, locked.row, audit_key="user_stop", now=datetime.now(timezone.utc),
                     audit={**audit, "reservation_producer": producer})
    return producer


def _hold_producer(conn, turn_id: str, locked: _LockedReservation, *, reason: str) -> dict[str, Any]:
    """Hand the reservation's producer its claim back (not an attempt), release the row.

    Rows already locked by ``_lock_open_reservation``. A delivery still
    DISPATCHING returns to QUEUED through the one requeue authority, which also
    releases its reservation (kept after a rotated /new reset, by requeue's
    rule); a delivery already QUEUED (an expired lease re-armed it) spent
    nothing, so only its reservation is released, under the same rule.
    """
    if locked.delivery_id:
        claim = conn.execute(sa_text(
            "SELECT status, claim_token, reset_status FROM inter_bot_deliveries"
            " WHERE id=:id AND turn_id=:turn_id"
        ), {"id": locked.delivery_id, "turn_id": turn_id}).mappings().first()
        record = None
        if claim and claim["status"] == "DISPATCHING" and claim["claim_token"]:
            # Releases the reservation by requeue's own rule. The dispatcher
            # stamps transport_accepted_at before save_turn, but a still-reserved
            # row was never admitted, so nothing reached the bridge: clear that
            # stamp, or claim_next's refuse_accepted_replay fails a held reset.
            record = InterBotDeliveryStore.requeue_claim(
                conn, locked.delivery_id, claim["claim_token"], reason,
                delay_seconds=0, refund_attempt=True, clear_delivery_mode=True,
            )
        elif claim is None or claim["reset_status"] not in RESERVATION_KEPT_RESET_STATUSES:
            _release_unstarted(conn, turn_id)
        return {"kind": "delivery", "id": locked.delivery_id,
                "claim": claim["claim_token"] if claim else None,
                "held": record is not None and record.status == "QUEUED",
                "status": record.status if record else None}
    if locked.owner:
        where, values = returned_claim(claim_token=locked.owner, reason=reason, delay_seconds=0)
        held = conn.execute(
            update(ToolApprovalRequest).where(*where).values(**values)
            .returning(ToolApprovalRequest.id)
        ).scalars().all()
        if len(held) > 1:
            # The token column is not unique; never return two claims at once.
            raise RuntimeError(f"claim token for reservation {turn_id} matched {len(held)} approvals")
        if not release_reservation(None, turn_id, owner=locked.owner, connection=conn):
            # Unreachable while the row is held and re-checked; raising rolls
            # the claim return back with the caller's transaction.
            raise RuntimeError(f"locked reservation {turn_id} was not released")
        return {"kind": "approval", "id": held[0] if held else None, "claim": locked.owner,
                "held": bool(held)}
    _release_unstarted(conn, turn_id)
    return {"kind": "ordinary", "id": None, "claim": None, "held": False}


def _release_unstarted(conn, turn_id: str) -> bool:
    """Delete a reservation that provably never ran (no bridge request bound)."""
    return bool(conn.execute(sa_text(
        "DELETE FROM turn_logs WHERE id=:id AND status='reserved' AND ended_at IS NULL"
        " AND agent_request_id IS NULL"
    ), {"id": turn_id}).rowcount)


@dataclass(frozen=True)
class ActivatedGate:
    """``activate_hold``'s result: the gate (id = generation) and its roster."""

    gate: GateBlock
    members: tuple[GateMember, ...]


_ROW_COLUMNS = "id, status, agent_session_key, agent_request_id, agent_backend, request_json"
_OPEN_ROWS = f"SELECT {_ROW_COLUMNS} FROM turn_logs WHERE bot_id=:bot AND ended_at IS NULL ORDER BY created_at, id"
_LOCK_ROWS = f"SELECT {_ROW_COLUMNS} FROM turn_logs WHERE id = ANY(:ids) ORDER BY id FOR UPDATE"


def activate_hold(
    conn, spec: GateSpec, *, predecessor: int | None = None, fresh_clock: bool = False,
) -> ActivatedGate:
    """Open a gate and capture its roster in the caller's transaction (TASK-1233 B2/B4).

    1. ``lock_target`` every in-scope agent bot, in sorted bot_id order.
    2. Insert the gate row (its id is the generation).
    3. Capture EVERY open turn row of those bots (``ended_at IS NULL``).
    4. Disposition each: a reservation with no execution evidence has its
       producer's claim returned and the row released (``revoked_unstarted``,
       drained); a reservation with evidence is ``accepted`` (drains on
       exit; re-attaching to it is exempt from this gate); any other open row
       is ``running``.

    The caller commits. Every admission evaluates ``gate_blocks`` after the
    same per-bot lock, so either it committed first and its row is captured
    here, or it sees this gate. Lock order: sorted target locks, then every
    producer row a revoke will lock (``_lock_producers``), then the captured
    turn rows (``_lock_rows``), then member rows.
    No path locks a producer row before its target lock, and the one
    multi-row delivery locker without a target lock (``recover_expired``)
    also locks in ordinal order, so no cycle forms. Only a job-owned gate's
    members are steered; an operator gate's roster exists for revocation and
    the C1 exemption (``control_state='not_needed'``).

    ``predecessor`` (a rebuild after gate loss) names the same owner's earlier
    generation: its undrained members are carried into this roster in this
    transaction (``carry_forward``), so a rebuild is a union, never a drop.
    Scope governs admission only: carried members are kept regardless of
    ``spec.scope``, and are not target-locked (their admission isn't gated).
    Their turn rows are locked with the captured ones (``_lock_rows``), so a
    concurrent bind is never lost between the predecessor read and the insert.

    ``fresh_clock`` starts the lease on the wall clock read after the target
    locks (``insert_gate(fresh=True)``), not the transaction start (TASK-1327 IR1).
    """
    if conn.dialect.name != "postgresql":
        raise AdmissionUnavailable("system gates require PostgreSQL")
    if predecessor is not None:
        prior_owner = conn.execute(sa_text("SELECT owner_id FROM system_gates WHERE id=:id"),
                                   {"id": predecessor}).scalar_one_or_none()
        if prior_owner != spec.owner_id:
            raise ValueError(f"gate {predecessor} is not a generation of owner {spec.owner_id!r}")
    bots = bots_in_scope(conn, spec.scope)
    for bot_id in bots:
        lock_target(conn, bot_id)
    # Under every target lock no row can be reserved or consumed, only ended.
    rows = [(bot_id, row) for bot_id in bots
            for row in conn.execute(sa_text(_OPEN_ROWS), {"bot": bot_id}).mappings().all()]
    _lock_producers(conn, [row for _, row in rows])
    carried_turns = [] if predecessor is None else list(conn.execute(sa_text(
        "SELECT DISTINCT turn_id FROM gate_members WHERE gate_id=:old"), {"old": predecessor}).scalars())
    rows = _lock_rows(conn, rows, also=carried_turns)
    gate = insert_gate(conn, spec, fresh=fresh_clock)
    members: list[GateMember] = []
    for bot_id, row in rows:
        member = _capture_member(conn, gate, spec, bot_id, row)
        if member is not None:
            insert_member(conn, member)
            members.append(member)
    if predecessor is not None:
        members.extend(carry_forward(conn, predecessor=predecessor, gate=gate))
    return ActivatedGate(gate, tuple(members))


def _lock_producers(conn, rows) -> None:
    """Lock every producer row the capture will revoke, one table per statement in its canonical order.

    Per-member locking would take deliveries in bot/created order, while
    ``recover_expired`` locks lease-expired deliveries in ordinal order with
    no target lock: two such rows in opposite orders deadlock. Each producer
    is picked exactly as ``_lock_open_reservation`` picks it.
    """
    payloads = [_reservation_payload(row["request_json"]) for row in rows
                if row["status"] == RESERVED and not has_execution_evidence(row)]
    deliveries = sorted({p["inter_bot_delivery_id"] for p in payloads if p.get("inter_bot_delivery_id")})
    owners = sorted({p[RESERVATION_OWNER_KEY] for p in payloads
                     if p.get(RESERVATION_OWNER_KEY) and not p.get("inter_bot_delivery_id")})
    if deliveries:
        conn.execute(sa_text("SELECT 1 FROM inter_bot_deliveries WHERE id = ANY(:ids)"
                             " ORDER BY ordinal FOR UPDATE"), {"ids": deliveries})
    if owners:
        conn.execute(sa_text("SELECT 1 FROM tool_approval_requests WHERE continuation_claim_token = ANY(:owners)"
                             " ORDER BY id FOR UPDATE"), {"owners": owners})


def _lock_rows(conn, rows, *, also=()):
    """Lock the captured turn rows after their producers (the admission lock order) and re-read them.

    ``bind_agent_bridge`` holds only the turn row, so this lock orders every
    bind against the capture (test_Z (a)): one committed first is read here,
    one that waited pins the committed member (``pin_member_request``). A
    member never lacks the request its run was sent with. A row deleted
    meanwhile is no longer captured; one that ended still is, as before.

    ``also`` is a rebuild's predecessor roster (``carry_forward``): those turn
    rows are locked in the same id-ordered statement, so a bind is ordered
    against the carried member too, including a bot outside the new scope
    with no target lock here (xp-z-r1). A bind committed first has pinned the
    predecessor member that ``carry_forward`` reads; one that waited pins the
    carried row. One statement keeps two rebuilds from locking each other's
    rows in opposite orders.
    """
    ids = sorted({row["id"] for _, row in rows} | set(also))
    if not ids:
        return rows
    locked = {row["id"]: row for row in conn.execute(sa_text(_LOCK_ROWS), {"ids": ids}).mappings().all()}
    return [(bot_id, locked[row["id"]]) for bot_id, row in rows if row["id"] in locked]


def _capture_member(conn, gate: GateBlock, spec: GateSpec, bot_id: str, row) -> GateMember | None:
    payload = _reservation_payload(row["request_json"])
    delivery_id = payload.get("inter_bot_delivery_id")
    owner = payload.get(RESERVATION_OWNER_KEY)
    producer = {"kind": "delivery" if delivery_id else "approval" if owner else "ordinary",
                "id": delivery_id, "claim": owner}
    status = row["status"]
    if status == RESERVED and not has_execution_evidence(row):
        outcome = revoke_reservation(
            None, row["id"], disposition="hold", connection=conn,
            audit={"reason": f"held by system gate {gate.id}"},
        )
        if outcome is not None:
            disposition, producer = REVOKED_UNSTARTED, outcome
        else:
            # No longer an open reservation (nothing else can consume it under
            # our locks; an unlocked re-arm or end may have): re-read it.
            current = conn.execute(sa_text(
                "SELECT status, ended_at FROM turn_logs WHERE id=:id"
            ), {"id": row["id"]}).mappings().first()
            if current is None or current["ended_at"] is not None:
                return None
            disposition, status = RUNNING, current["status"]
    elif status == RESERVED:
        disposition = ACCEPTED
    else:
        disposition = RUNNING
    if producer["kind"] == "delivery" and producer.get("claim") is None and disposition != REVOKED_UNSTARTED:
        producer["claim"] = conn.execute(sa_text(
            "SELECT claim_token FROM inter_bot_deliveries WHERE id=:id"
        ), {"id": delivery_id}).scalar_one_or_none()
    return GateMember(
        gate_id=gate.id, bot_id=bot_id, turn_id=row["id"], turn_status=status,
        disposition=disposition, backend=row["agent_backend"],
        agent_session_key=row["agent_session_key"], agent_request_id=row["agent_request_id"],
        producer_kind=producer["kind"], producer_id=producer.get("id"),
        producer_claim=producer.get("claim"),
        control_state=initial_control_state(spec.owner_kind, disposition),
    )


def requeue_stale_reservation(
    conn, turn_id: str, *, owner: str, approval_id: str, stale_before: datetime, reason: str,
) -> bool:
    """Take a stalled continuation's claim back, then release its reservation.

    The recovery sweep's disposition (TASK-1232 C-B3), in the caller's
    transaction and through ``revoke_reservation``'s lock protocol. The claim
    is taken back by a CAS on the observed approval row and token whose lease
    went stale before ``stale_before`` (the token column is not unique, so
    the row id pins it); ``requeued_continuation_state`` suppresses a
    cancelled request (Vex S1). Only then is the row deleted. False, with
    nothing written, when the row is no longer ``owner``'s open reservation
    (a claimant that started the turn keeps its claim) or the claim was
    renewed.
    """
    with Session(bind=conn) as session:
        locked = _lock_open_reservation(conn, session, turn_id, owner=owner)
        if not isinstance(locked, _LockedReservation):
            return False
        revoked = conn.execute(
            update(ToolApprovalRequest)
            .where(ToolApprovalRequest.id == approval_id,
                   ToolApprovalRequest.continuation_claim_token == owner,
                   ToolApprovalRequest.continuation_state == CONT_DISPATCHING,
                   ToolApprovalRequest.continuation_next_attempt_at <= stale_before)
            .values(continuation_state=requeued_continuation_state(), continuation_claim_token=None,
                    continuation_last_error=reason)
        )
        if revoked.rowcount != 1:
            return False  # renewed since detect; nothing written
        if not release_reservation(None, turn_id, owner=owner, connection=conn):
            # Unreachable while the row is held and re-checked. Raising rolls
            # the revoke back with the caller's transaction: a revoke never
            # commits without its release.
            raise RuntimeError(f"locked reservation {turn_id} was not released")
    return True


def _revoke_producer(conn, turn_id: str, *, delivery_id: str | None, owner: str | None) -> dict[str, Any]:
    """Take the claim back from the reservation's producer (rows already locked)."""
    producer: dict[str, Any] = {"kind": "ordinary", "revoked": False}
    if delivery_id:
        revoked = conn.execute(sa_text("""
            UPDATE inter_bot_deliveries SET status='CANCELLED', claim_token=NULL,
                claim_owner=NULL, lease_expires_at=NULL, next_retry_at=NULL,
                last_error='Stopped by user', updated_at=now()
            WHERE id=:id AND turn_id=:turn_id AND status IN ('QUEUED','DISPATCHING')
            RETURNING transport_accepted_at
        """), {"id": delivery_id, "turn_id": turn_id}).mappings().first()
        producer = {"kind": "delivery", "id": delivery_id, "revoked": revoked is not None,
                    "accepted": bool(revoked and revoked["transport_accepted_at"])}
    elif owner:
        revoked = conn.execute(sa_text("""
            UPDATE tool_approval_requests SET continuation_state=:suppressed,
                continuation_claim_token=NULL
            WHERE continuation_claim_token=:owner AND continuation_state=:dispatching
            RETURNING id
        """), {"owner": owner, "suppressed": CONT_SUPPRESSED,
               "dispatching": CONT_DISPATCHING}).scalar_one_or_none()
        producer = {"kind": "approval", "id": revoked, "revoked": revoked is not None}
    return producer


def turn_request_extensions(request: Any) -> dict[str, Any]:
    """Request fields persisted on the turn row, shared by stream and nonstream."""
    continuation = getattr(request, "continuation_payload", None)
    admission = TurnAdmission.for_request(request)
    values = {
        "inter_bot_delivery_id": getattr(request, "inter_bot_delivery_id", None),
        "inter_bot_turn_id": getattr(request, "inter_bot_turn_id", None),
        "inter_bot_bridge_request_id": getattr(request, "inter_bot_bridge_request_id", None),
        "inter_bot_timeout_seconds": getattr(request, "inter_bot_timeout_seconds", None),
        "inter_bot_session_policy": getattr(request, "inter_bot_session_policy", None),
        "inter_bot_seed_session_id": getattr(request, "inter_bot_seed_session_id", None),
        "continuation_payload": continuation.model_dump() if continuation else None,
        ADMISSION_KEY: admission.value if admission is not TurnAdmission.START else None,
    }
    return {key: value for key, value in values.items() if value}


def turn_start_extensions(
    request: Any, *, is_agent_backend: bool, admission_claim: AdmissionClaim | None,
) -> dict[str, Any]:
    """Extensions for a directly started turn row (stream and nonstream).

    A direct agent start without a durable claim comes from a harness that
    cannot reattach an ambiguous run, so it may only start on an idle target.
    """
    extensions = turn_request_extensions(request)
    if is_agent_backend and admission_claim is None:
        extensions[ADMISSION_KEY] = TurnAdmission.IDLE_ONLY.value
    return extensions


__all__ = [
    "ADMISSION_KEY",
    "ActivatedGate",
    "AdmissionClaim",
    "AdmissionFenceLost",
    "AdmissionUnavailable",
    "RESERVATION_OWNER_KEY",
    "SystemGateRefused",
    "TargetBusy",
    "TargetState",
    "TurnAdmission",
    "activate_hold",
    "classify_target",
    "fence_consumption",
    "lock_target",
    "refuse_if_gated",
    "release_reservation",
    "requeue_stale_reservation",
    "reserve_turn",
    "revoke_reservation",
    "turn_request_extensions",
    "turn_start_extensions",
    "write_reservation",
]
