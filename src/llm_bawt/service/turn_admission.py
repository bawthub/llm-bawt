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
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping

from sqlalchemy import text as sa_text

ADMISSION_KEY = "admission"
RESERVED = "reserved"
# request_json key fencing a reservation to the producer claim that wrote it.
RESERVATION_OWNER_KEY = "reservation_owner"


class TargetBusy(RuntimeError):
    """A request lost the target's admission race (nothing was started)."""


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


def classify_target(conn, bot_id: str, *, exclude_turn_id: str | None) -> TargetState:
    """All other open turn rows for ``bot_id``, newest first (call under the lock)."""
    rows = conn.execute(
        sa_text(
            "SELECT id, status, agent_session_key, agent_request_id, user_id"
            " FROM turn_logs WHERE bot_id=:target AND ended_at IS NULL"
            " AND id != :turn_id ORDER BY created_at DESC"
        ),
        {"target": bot_id, "turn_id": exclude_turn_id or ""},
    ).mappings().all()
    return TargetState(tuple(dict(row) for row in rows))


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
            status='reserved', ended_at=NULL, error_text=NULL,
            request_json=EXCLUDED.request_json, user_prompt=EXCLUDED.user_prompt
    """ + guard), {
        "turn_id": turn_id,
        "target": bot_id,
        "user_id": user_id,
        "message": user_prompt,
        "request_json": json.dumps(request_json, ensure_ascii=False, default=str),
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
    still owns its durable claim; otherwise ``AdmissionFenceLost``.
    """
    with engine.begin() as conn:
        lock_target(conn, bot_id)
        if fence is not None and not fence(conn):
            raise AdmissionFenceLost(f"claim for turn '{turn_id}' is no longer held")
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
    engine, turn_id: str, *, owner: str, fence: Callable[[Any], bool] | None = None,
) -> bool:
    """Drop this owner's reservation if it never became a running turn.

    Fenced twice in one transaction: the row must carry ``owner`` and, when
    given, ``fence(conn)`` must prove the producer still holds its claim.
    """
    with engine.begin() as conn:
        if fence is not None and not fence(conn):
            return False
        result = conn.execute(
            sa_text(
                "DELETE FROM turn_logs WHERE id=:id AND status='reserved'"
                " AND ended_at IS NULL"
                " AND CAST(request_json AS jsonb)->>:key = :owner"
            ),
            {"id": turn_id, "key": RESERVATION_OWNER_KEY, "owner": owner},
        )
    return bool(result.rowcount)


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
    "AdmissionClaim",
    "AdmissionFenceLost",
    "AdmissionUnavailable",
    "RESERVATION_OWNER_KEY",
    "TargetBusy",
    "TargetState",
    "TurnAdmission",
    "classify_target",
    "fence_consumption",
    "lock_target",
    "release_reservation",
    "reserve_turn",
    "turn_request_extensions",
    "turn_start_extensions",
    "write_reservation",
]
