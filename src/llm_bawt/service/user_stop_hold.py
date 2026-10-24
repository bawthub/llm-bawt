"""The user-Stop queue hold (TASK-1327 D1).

A user Stop also holds the stopped bot's queued work for a bounded time, so
the backlog cannot start a new turn the moment the stopped one settles. The
hold is an ordinary operator HOLD gate (``turn_gates``) owned by exactly
``user_stop:<bot>``: its refusals, roster and expiry are the gate system's.

* Held: delivery claims (the bot-to-bot backlog and every other queued row,
  which stay QUEUED with no attempt spent), enqueue is not refused, and due
  scheduled prompts wait. ``save_turn``, ``steer`` and ``reserve`` are allowed:
  the held backlog never reaches them because it is held at the claim.
* Not held: the user's own chat. ``claim_next`` claims the earliest ready
  user-authored ordinary-chat row past the hold (``load_user_bypass``), while
  every other gate still applies in full.
* Ends on Resume (release), Cancel all (cancel the held rows, then release)
  or expiry (``user_stop_queue_hold_seconds``; 0 disables the hold).

Open-or-extend, Resume and Cancel all each run in one transaction under
``lock_target(bot)``, and every liveness read and new lease after that lock
reads the wall clock (``fresh``), so an expiry that passed during the lock
wait is never revived (IR1). A hold failure never fails the Stop.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import text as sa_text

from ..inter_bot_delivery_record import DeliveryRecord
from ..utils.db import is_postgresql
from .turn_admission import activate_hold, lock_target
from .turn_gates import (
    HOLD,
    OWNER_OPERATOR,
    GateBlock,
    GateScope,
    GateSpec,
    extend_gate,
    live_gate_for_owner,
    release_gates,
)

log = logging.getLogger(__name__)

# The ChatAbortRequest sources that are a human pressing Stop.
USER_STOP_SOURCES = frozenset({"chat_stop", "bot_list_stop"})
HOLD_SETTING = "user_stop_queue_hold_seconds"
HOLD_REASON = "user_stop"
HOLD_CANCEL_REASON = "Cancelled by user (queue hold)"
HOLD_ALLOW_ACTORS = frozenset({"save_turn", "steer", "reserve"})

# The rows a hold keeps waiting: queued for the bot, neither a scheduled
# prompt (owned by the scheduling API) nor the user's own chat (bypassed).
# The one predicate behind both the snapshot and Cancel all. NULL-safe: a row
# with no metadata or no author is held, never silently dropped.
_HELD_ROWS = """
    FROM inter_bot_deliveries
    WHERE target_bot_id=:bot AND status='QUEUED'
      AND metadata_json->'prompt_schedule' IS NULL
      AND NOT (author_entity_type IS NOT DISTINCT FROM 'user'
               AND metadata_json->'ordinary_chat' IS NOT NULL)
"""


def hold_owner(bot_id: str) -> str:
    return f"user_stop:{bot_id}"


def stop_hold_ttl(engine, config: Any, source: str) -> int | None:
    """The hold's lease for a Stop from ``source``; None when no hold applies.

    None off PostgreSQL (gates are PostgreSQL-only), for a non-user source,
    for a setting of 0, or when the setting can't be read: Stop never fails
    on the hold.
    """
    if source not in USER_STOP_SOURCES or not is_postgresql(engine):
        return None
    from ..runtime_setting_resolution import resolve_global_runtime_setting

    try:
        ttl = int(resolve_global_runtime_setting(config, HOLD_SETTING) or 0)
    except Exception:
        log.exception("Unreadable %s; user-Stop queue hold skipped", HOLD_SETTING)
        return None
    return ttl if ttl > 0 else None


def hold_spec(bot_id: str, ttl_seconds: int) -> GateSpec:
    return GateSpec(
        owner_kind=OWNER_OPERATOR, owner_id=hold_owner(bot_id), mode=HOLD,
        scope=GateScope.for_bots(bot_id), ttl_seconds=ttl_seconds,
        reason=HOLD_REASON, allow_actors=HOLD_ALLOW_ACTORS,
    )


def open_or_extend(conn, bot_id: str, ttl_seconds: int) -> GateBlock:
    """Extend the bot's live hold, or open a new generation (caller's transaction).

    Never stacks: one live ``user_stop:<bot>`` gate at a time. A lapsed gate
    is never revived; its successor is a new generation.
    """
    owner = hold_owner(bot_id)
    lock_target(conn, bot_id)
    live = live_gate_for_owner(conn, owner, fresh=True)
    if live is not None and extend_gate(conn, live.id, owner_id=owner, ttl_seconds=ttl_seconds, fresh=True):
        return live_gate_for_owner(conn, owner, fresh=True) or live
    return activate_hold(conn, hold_spec(bot_id, ttl_seconds), fresh_clock=True).gate


def hold_quietly(conn, bot_id: str | None, ttl_seconds: int | None) -> GateBlock | None:
    """``open_or_extend`` in a savepoint: a failed hold is logged, never raised.

    Stop always stops; the caller's transaction (a reservation revoke) commits
    whatever happens here.
    """
    if not bot_id or not ttl_seconds:
        return None
    try:
        with conn.begin_nested():
            return open_or_extend(conn, bot_id, ttl_seconds)
    except Exception:
        log.exception("user-Stop queue hold failed for %s; Stop proceeds without it", bot_id)
        return None


@dataclass(frozen=True)
class QueueHold:
    """A bot's live user-Stop hold and the rows it is holding, in FIFO order."""

    gate_id: int
    expires_at: datetime
    queued: tuple[dict[str, Any], ...]

    def summary(self) -> dict[str, Any]:
        return {"gate_id": self.gate_id, "expires_at": self.expires_at.isoformat(),
                "queued_count": len(self.queued)}

    def to_api(self) -> dict[str, Any]:
        return {**self.summary(), "queued": [dict(row) for row in self.queued]}


def snapshot(engine, bot_id: str) -> QueueHold | None:
    """The bot's live hold, else None (read-only; gates are PostgreSQL-only)."""
    if not is_postgresql(engine):
        return None
    with engine.connect() as conn:
        gate = live_gate_for_owner(conn, hold_owner(bot_id), fresh=True)
        if gate is None:
            return None
        rows = conn.execute(sa_text(
            "SELECT id, sender_bot_id, message_kind, task_id, created_at"
            f" {_HELD_ROWS} ORDER BY ordinal ASC"
        ), {"bot": bot_id}).mappings().all()
    queued = tuple({**row, "created_at": row["created_at"].isoformat() if row["created_at"] else None}
                   for row in rows)
    return QueueHold(gate_id=gate.id, expires_at=gate.expires_at, queued=queued)


def resume(engine, bot_id: str) -> list[int]:
    """Release the bot's hold; the gate ids released (empty when none was live)."""
    with engine.begin() as conn:
        lock_target(conn, bot_id)
        return release_gates(conn, owner_id=hold_owner(bot_id))


def cancel_all(engine, bot_id: str) -> list[DeliveryRecord] | None:
    """Cancel every held row, then release the hold; None when no hold is live.

    Each row goes through the store's one request-cancel rule
    (``cancel_queued_locked``), so a row that rule refuses (routed work after
    acceptance) is left QUEUED. Lock order: target lock, then the held rows
    in ordinal order, as every other multi-row delivery locker.
    """
    from ..inter_bot_delivery import InterBotDeliveryStore

    with engine.begin() as conn:
        lock_target(conn, bot_id)
        if live_gate_for_owner(conn, hold_owner(bot_id), fresh=True) is None:
            return None
        rows = conn.execute(sa_text(
            f"SELECT * {_HELD_ROWS} ORDER BY ordinal ASC FOR UPDATE"
        ), {"bot": bot_id}).mappings().all()
        cancelled = []
        for row in rows:
            done = InterBotDeliveryStore.cancel_queued_locked(conn, dict(row), HOLD_CANCEL_REASON)
            if done is not None:
                cancelled.append(DeliveryRecord.from_mapping(done))
        release_gates(conn, owner_id=hold_owner(bot_id))
    return cancelled
