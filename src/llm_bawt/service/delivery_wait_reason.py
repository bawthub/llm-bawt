"""Why a QUEUED ordinary chat delivery is still waiting (TASK-1676).

A queued chat send waits for one of two things: the target's running turn,
or a live system gate (an ops quiesce hold, or an operator freeze). The chat
UI labels the wait, so every QUEUED surface (the composer receipt, the
hydrate rows, the lifecycle event) carries the same readout::

    {"kind": "turn_running" | "system_hold", "gate": "quiesce" | "freeze" | None}

It asks the gate exactly as the delivery claim does for a user's own chat:
the ``delivery_claim`` actor, with the target's user-Stop hold exempt, since
that hold lets the user's chat run ahead (TASK-1327). A readout only: it
takes no lock and decides nothing.
"""

from __future__ import annotations

from typing import Any, Iterable

from .turn_gates import DENY, GateActor, GateBlock, gate_blocks
from .user_stop_hold import hold_owner

TURN_RUNNING = "turn_running"
SYSTEM_HOLD = "system_hold"
GATE_QUIESCE = "quiesce"
GATE_FREEZE = "freeze"


def reason_for(block: GateBlock | None) -> dict[str, Any]:
    """The wire reason for a delivery the gate ``block`` holds (None = no gate)."""
    if block is None:
        return {"kind": TURN_RUNNING, "gate": None}
    return {"kind": SYSTEM_HOLD, "gate": GATE_FREEZE if block.mode == DENY else GATE_QUIESCE}


def wait_reason(conn, target_bot_id: str) -> dict[str, Any]:
    """The reason one ordinary chat row queued for ``target_bot_id`` waits."""
    block = gate_blocks(conn, target_bot_id, GateActor("delivery_claim"),
                        exclude_owner_id=hold_owner(target_bot_id))
    return reason_for(block)


def wait_reasons(conn, target_bot_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    """:func:`wait_reason` for each distinct target, one gate read per target."""
    return {bot: wait_reason(conn, bot) for bot in sorted(set(target_bot_ids))}


def is_waiting_chat(record: Any) -> bool:
    """Does ``record`` get a wait reason: a QUEUED ordinary chat row?"""
    return record.status == "QUEUED" and bool((record.metadata or {}).get("ordinary_chat"))


def annotate_rows(engine, records: list[Any], views: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add ``wait_reason`` to each view whose record is a waiting chat row.

    ``views`` are the records' projections, in the same order. A gate read
    failure leaves the views unannotated: the UI then keeps its turn wording.
    """
    targets = [record.target_bot_id for record in records if is_waiting_chat(record)]
    if not targets or engine is None:
        return views
    with engine.connect() as conn:
        reasons = wait_reasons(conn, targets)
    for record, view in zip(records, views):
        if is_waiting_chat(record):
            view["wait_reason"] = reasons[record.target_bot_id]
    return views


def record_wait_reason(engine, record: Any) -> dict[str, Any] | None:
    """The reason for one record, or None when it isn't a waiting chat row."""
    if engine is None or not is_waiting_chat(record):
        return None
    with engine.connect() as conn:
        return wait_reason(conn, record.target_bot_id)
