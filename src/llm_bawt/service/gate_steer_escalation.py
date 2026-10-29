"""Steers let past, or pushed through, a live system gate (TASK-1749, TASK-1748).

A live gate (an ops quiesce hold, an operator freeze) holds every queued row
at the delivery claim and refuses every steer (``system_gate_admission``).
Two narrow exceptions exist, and both are recorded in one place,
``gate_escalations`` (one row per gate, kind and subject), next to the
gate's own roster:

* ``user_send_now`` (TASK-1749): the user forces their OWN queued chat
  message past the gates holding it. Only a QUEUED ordinary-chat row the
  same user authored qualifies; bot deliveries, scheduled prompts and
  callbacks never do. The stamp names the exact gate generations it was
  forced past, so a later generation holds again. A busy target takes it as
  a ``now`` steer (it interrupts the running tool); an idle target starts a
  turn that is rostered as an ``accepted`` member of each forced gate, so a
  quiesce waits for it by design.
* ``quiesce_escalate`` (TASK-1748): the quiesce driver pushes its own
  stuck control steer through a blocking tool. Its authority stays the live
  member link (``DeliveryControlGrant``); it only records here.

Every decision runs under ``lock_target(conn, bot_id)`` in the caller's
transaction, like every other gate edge.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from fastapi import HTTPException
from sqlalchemy import text as sa_text

KIND_USER_SEND_NOW = "user_send_now"
KIND_QUIESCE_ESCALATE = "quiesce_escalate"
ESCALATION_KINDS = frozenset({KIND_USER_SEND_NOW, KIND_QUIESCE_ESCALATE})
SEND_NOW_KEY = "send_now"


def record_gate_escalation(
    conn, *, gate_id: int, bot_id: str, kind: str, subject: str, actor: str,
    turn_id: str | None = None, outcome: str | None = None,
) -> bool:
    """Insert the one record for (gate, kind, subject); False when it exists.

    The insert is the one-shot stamp: a caller that must act at most once
    per subject acts only when this returns True.
    """
    if kind not in ESCALATION_KINDS:
        raise ValueError(f"unknown gate escalation kind {kind!r}")
    return bool(conn.execute(sa_text("""
        INSERT INTO gate_escalations (gate_id, bot_id, kind, subject, actor, turn_id, outcome)
        VALUES (:gate, :bot, :kind, :subject, :actor, :turn, :outcome)
        ON CONFLICT (gate_id, kind, subject) DO NOTHING
    """), {"gate": gate_id, "bot": bot_id, "kind": kind, "subject": subject,
           "actor": actor, "turn": turn_id, "outcome": outcome}).rowcount)


def set_escalation_outcome(conn, *, gate_id: int, kind: str, subject: str, outcome: str,
                           turn_id: str | None = None) -> bool:
    """Record what happened after the stamp (e.g. ``escalated``, ``refused:<detail>``)."""
    return bool(conn.execute(sa_text("""
        UPDATE gate_escalations SET outcome=:outcome, turn_id=COALESCE(:turn, turn_id),
            updated_at=now()
        WHERE gate_id=:gate AND kind=:kind AND subject=:subject
    """), {"gate": gate_id, "kind": kind, "subject": subject, "outcome": outcome,
           "turn": turn_id}).rowcount)


def gate_escalations(conn, gate_id: int) -> list[dict[str, Any]]:
    """A gate's escalation records, oldest first (roster/audit readout)."""
    rows = conn.execute(sa_text("""
        SELECT kind, subject, bot_id, actor, turn_id, outcome, created_at, updated_at
        FROM gate_escalations WHERE gate_id=:gate ORDER BY id
    """), {"gate": gate_id}).mappings().all()
    return [{**row, "created_at": row["created_at"].isoformat(),
             "updated_at": row["updated_at"].isoformat()} for row in rows]


def blocking_gates(conn, bot_id: str, actor) -> list:
    """Every live gate refusing ``actor`` on ``bot_id``, strictest first."""
    from .turn_gates import gate_blocks
    from .user_stop_hold import hold_owner

    found, excluded = [], frozenset()
    while True:
        # The user-Stop hold never holds the user's own chat (TASK-1327).
        block = gate_blocks(conn, bot_id, actor, exclude_owner_id=hold_owner(bot_id),
                            exclude_gate_ids=excluded)
        if block is None:
            return found
        found.append(block)
        excluded = excluded | {block.id}


@dataclass(frozen=True)
class SendNow:
    """A user's Send now stamp on one delivery: the gates it was forced past."""

    user_id: str
    gate_ids: frozenset[int]

    @classmethod
    def of_metadata(cls, metadata: Any) -> "SendNow | None":
        if isinstance(metadata, str):
            metadata = json.loads(metadata or "{}")
        raw = (metadata or {}).get(SEND_NOW_KEY) if isinstance(metadata, Mapping) else None
        if not isinstance(raw, Mapping) or not raw.get("user_id"):
            return None
        ids = frozenset(int(gate) for gate in raw.get("gate_ids") or ())
        return cls(user_id=str(raw["user_id"]), gate_ids=ids) if ids else None

    @classmethod
    def of(cls, record: Any) -> "SendNow | None":
        """The stamp on a delivery record or row mapping, else None."""
        metadata = getattr(record, "metadata", None)
        if metadata is None and isinstance(record, Mapping):
            metadata = record.get("metadata_json")
        return cls.of_metadata(metadata)

    @staticmethod
    def priority_for(caps: frozenset[str]) -> str:
        """``now`` interrupts the running tool; a run that can't take it gets ``next``."""
        from ..steer_capabilities import NEXT, NOW

        return NOW if NOW in caps else NEXT


class SendNowRefused(HTTPException):
    def __init__(self, status_code: int, detail: str):
        super().__init__(status_code=status_code, detail=detail)


def _user_chat_row(row: Mapping[str, Any], user_id: str) -> bool:
    """Only the user's own ordinary chat: never bot work, schedules or callbacks."""
    metadata = row.get("metadata_json") or {}
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return (
        row.get("author_entity_type") == "user"
        and (row.get("author_entity_id") or "").lower() == user_id
        and metadata.get("ordinary_chat") is not None
        and metadata.get("prompt_schedule") is None
        and row.get("lane") == "normal"
    )


def stamp_send_now(engine, delivery_id: str, user_id: str) -> dict[str, Any]:
    """Force the user's own held chat row past the gates holding it now.

    One transaction under ``lock_target``: the row must be QUEUED, the user's
    own ordinary chat, and held by at least one live gate. The stamp names
    those generations, and each gets a ``user_send_now`` record. Idempotent:
    a second click adds only gates that started holding since.
    """
    from .turn_admission import lock_target
    from .turn_gates import GateActor

    user_id = (user_id or "").strip().lower()
    if not user_id:
        raise SendNowRefused(400, "user_id is required")
    with engine.begin() as conn:
        bot = conn.execute(sa_text(
            "SELECT target_bot_id FROM inter_bot_deliveries WHERE id=:id"
        ), {"id": delivery_id}).scalar_one_or_none()
        if bot is None:
            raise SendNowRefused(404, "Delivery not found")
        lock_target(conn, bot)
        row = conn.execute(sa_text(
            "SELECT * FROM inter_bot_deliveries WHERE id=:id FOR UPDATE"
        ), {"id": delivery_id}).mappings().first()
        if row is None or row["status"] != "QUEUED":
            raise SendNowRefused(409, f"Delivery is {row['status'] if row else 'gone'}, not queued")
        if not _user_chat_row(row, user_id):
            raise SendNowRefused(403, "Send now is only for your own chat message")
        blocks = blocking_gates(conn, bot, GateActor("delivery_claim", row["turn_id"]))
        if not blocks:
            raise SendNowRefused(409, "not_held")
        prior = SendNow.of(row)
        gate_ids = sorted({block.id for block in blocks} | (prior.gate_ids if prior else set()))
        stamp = {"user_id": user_id, "gate_ids": gate_ids}
        conn.execute(sa_text("""
            UPDATE inter_bot_deliveries SET
                metadata_json=jsonb_set(COALESCE(metadata_json, '{}'::jsonb), '{send_now}',
                                        CAST(:stamp AS jsonb)),
                available_at=LEAST(available_at, now()), next_retry_at=NULL, updated_at=now()
            WHERE id=:id
        """), {"id": delivery_id, "stamp": json.dumps(stamp)})
        for block in blocks:
            record_gate_escalation(
                conn, gate_id=block.id, bot_id=bot, kind=KIND_USER_SEND_NOW,
                subject=delivery_id, actor=f"user:{user_id}", outcome="stamped",
            )
    return {"ok": True, "delivery_id": delivery_id, "bot_id": bot, "gate_ids": gate_ids}


def send_now_exemption(conn, head: Mapping[str, Any]) -> frozenset[int]:
    """The gate generations a claimed head may pass (empty unless stamped)."""
    stamp = SendNow.of(head)
    if stamp is None or not _user_chat_row(head, stamp.user_id):
        return frozenset()
    return stamp.gate_ids


def roster_send_now_turn(conn, *, bot_id: str, turn_id: str, delivery_id: str,
                         gate_ids: frozenset[int]) -> None:
    """Roster a forced turn on each live gate it passed (caller holds lock_target).

    ``accepted``: the restart waits for it (it drains on exit) and its own
    ``save_turn`` re-attach is exempt (C1). A job gate steers it like any
    member; an operator gate only records it.
    """
    from .turn_gates import ACCEPTED, GateMember, gate_is_live, initial_control_state, insert_member

    for gate_id in sorted(gate_ids):
        if not gate_is_live(conn, gate_id, fresh=True):
            continue
        owner_kind = conn.execute(sa_text("SELECT owner_kind FROM system_gates WHERE id=:id"),
                                  {"id": gate_id}).scalar_one()
        insert_member(conn, GateMember(
            gate_id=gate_id, bot_id=bot_id, turn_id=turn_id, turn_status="reserved",
            disposition=ACCEPTED, producer_kind="delivery", producer_id=delivery_id,
            control_state=initial_control_state(owner_kind, ACCEPTED),
        ), skip_existing=True)
        set_escalation_outcome(conn, gate_id=gate_id, kind=KIND_USER_SEND_NOW,
                               subject=delivery_id, outcome="turn_started", turn_id=turn_id)


def admit_send_now_steer(conn, bot_id: str, *, turn_id: str, delivery_id: str):
    """The gate still blocking a stamped steer, else None (caller holds lock_target).

    Re-reads the stamp from the row: a caller's claim is an address, never
    authority. Records the steer on each passed gate.
    """
    from .turn_gates import GateActor, gate_blocks
    from .user_stop_hold import hold_owner

    row = conn.execute(sa_text("SELECT * FROM inter_bot_deliveries WHERE id=:id"),
                       {"id": delivery_id}).mappings().first()
    gate_ids = send_now_exemption(conn, row) if row and row["target_bot_id"] == bot_id else frozenset()
    block = gate_blocks(conn, bot_id, GateActor("steer", turn_id),
                        exclude_owner_id=hold_owner(bot_id), exclude_gate_ids=gate_ids)
    if block is None:
        for gate_id in sorted(gate_ids):
            set_escalation_outcome(conn, gate_id=gate_id, kind=KIND_USER_SEND_NOW,
                                   subject=delivery_id, outcome="steer_admitted", turn_id=turn_id)
    return block


__all__ = [
    "ESCALATION_KINDS",
    "KIND_QUIESCE_ESCALATE",
    "KIND_USER_SEND_NOW",
    "SendNow",
    "SendNowRefused",
    "admit_send_now_steer",
    "blocking_gates",
    "gate_escalations",
    "record_gate_escalation",
    "roster_send_now_turn",
    "send_now_exemption",
    "set_escalation_outcome",
    "stamp_send_now",
]
