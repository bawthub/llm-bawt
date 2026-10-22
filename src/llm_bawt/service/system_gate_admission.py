"""Edge policy for the producers and steers that ask a system gate (TASK-1233 C3).

``turn_gates`` owns the gate rows and ``gate_blocks``; ``turn_admission`` owns
``lock_target`` and ``SystemGateRefused``. This module holds the per-edge
decisions that are more than a plain refusal. Every check runs after
``lock_target(conn, bot_id)`` in the caller's transaction, so it is linearized
with ``activate_hold``: either the producer commits first and the roster
captures it, or the gate commits first and the check sees it.

What a live gate does at each edge:

* enqueue: a deny gate (freeze) refuses ``deny``-disposition producers
  (external sends, composer starts) with ``system_frozen``. ``hold`` producers
  (internal callbacks, scheduled work, restart retries, a trusted self-model
  continuation) are accepted and wait at the claim. A hold gate (quiesce)
  refuses nothing at enqueue.
* delivery claim: any live gate leaves the head QUEUED with no attempt spent
  (``inter_bot_delivery_claim``).
* scheduled prompt: any live gate leaves the occurrence due, a held no-op.
* steer: any live gate refuses a steer into a running turn.
"""
from __future__ import annotations

from fastapi import HTTPException

from .turn_admission import SystemGateRefused, lock_target
from .turn_gates import DENY, GateActor, GateBlock, gate_blocks

ON_GATE_DENY = "deny"
ON_GATE_HOLD = "hold"
GATE_DISPOSITIONS = frozenset({ON_GATE_DENY, ON_GATE_HOLD})


def admit_enqueue(conn, bot_id: str, *, on_gate: str) -> None:
    """Refuse a ``deny`` producer under a deny gate (call under lock_target)."""
    if on_gate not in GATE_DISPOSITIONS:
        raise ValueError(f"on_gate must be one of {sorted(GATE_DISPOSITIONS)}")
    if on_gate == ON_GATE_HOLD:
        return
    block = gate_blocks(conn, bot_id, GateActor("enqueue"))
    if block is not None and block.mode == DENY:
        raise SystemGateRefused(block)


def schedule_held(conn, bot_id: str) -> bool:
    """True when a live gate holds a due scheduled occurrence (under lock_target)."""
    return gate_blocks(conn, bot_id, GateActor("schedule")) is not None


class SteerGateRefused(HTTPException):
    """A live system gate refused a steer before any bridge RPC.

    An HTTPException so the public steer route reports it unchanged (503
    ``system_frozen`` / 409 ``system_held``); typed so the durable dispatcher
    can wait it out without spending an attempt.
    """

    def __init__(self, block: GateBlock):
        refused = SystemGateRefused(block)
        self.block = block
        self.code = refused.code
        super().__init__(status_code=refused.status_code, detail=str(refused))


def check_steer(engine, bot_id: str, *, turn_id: str) -> None:
    """Raise ``SteerGateRefused`` when a live gate covers ``bot_id``.

    Its own short transaction under ``lock_target``. Only this check is
    linearized with gate activation; the bridge RPC that follows is not. A
    gate activated after the check is harmless: its roster already holds the
    running turn, and stopping that turn is the quiesce owner's job (S1).
    """
    if engine is None or engine.dialect.name != "postgresql":
        return  # system gates are PostgreSQL-only
    with engine.begin() as conn:
        lock_target(conn, bot_id)
        block = gate_blocks(conn, bot_id, GateActor("steer", turn_id))
    if block is not None:
        raise SteerGateRefused(block)


__all__ = [
    "GATE_DISPOSITIONS",
    "ON_GATE_DENY",
    "ON_GATE_HOLD",
    "SteerGateRefused",
    "admit_enqueue",
    "check_steer",
    "schedule_held",
]
