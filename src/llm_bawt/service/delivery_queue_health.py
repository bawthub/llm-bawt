"""``deliveries.queue`` health: name what waits, flag only waits nothing explains (TASK-1458).

A QUEUED delivery older than ``STALE_QUEUED`` is classified with the same
authorities the dispatcher's claim uses (``inter_bot_delivery_claim``):

* target busy: ``turn_admission.classify_targets`` sees another open turn for
  the target (the delivery's own turn id excluded, as admission does). Waiting
  for it is normal, so it is reported as OK with the delivery named;
* held: ``turn_gates.gate_blocks`` refuses the ``delivery_claim`` actor, i.e. a
  live quiesce/restart/stop gate is deliberately holding the target. Also OK;
* otherwise the target is idle and ungated, so the dispatcher should already
  have claimed it: Degraded.

No watchdog ends a long agent turn while the app runs: ``recovery.orphan_turns``
settles stranded turns only at startup, and the turn reaper skips ordinary agent
starts. A wait behind a turn open longer than ``BUSY_TURN_CEILING`` is therefore
reported Degraded too, so a stuck turn can't hide behind "target busy".

An active (STEERING/DISPATCHING) delivery whose lease has expired is Degraded.
The check reads only; it locks and admits nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from .health_checks import DEGRADED, OK, _check
from .turn_admission import TargetState, classify_targets
from .turn_gates import GateActor, gate_blocks

NAME = "deliveries.queue"
STALE_QUEUED = timedelta(minutes=2)
# Agent turns, 14 days to 2026-10-25: p99.9 73 min, max 99 min.
BUSY_TURN_CEILING = timedelta(hours=2)
SHOWN = 3  # deliveries named per group in the reason line
MAX_ITEMS = 50  # structured rows returned; ``item_count`` keeps the total

_WAITING = text("""
    SELECT id, status, target_bot_id, sender_bot_id, author_entity_type,
           author_entity_id, task_id, turn_id, created_at, lease_expires_at
    FROM inter_bot_deliveries
    WHERE (status = 'QUEUED' AND available_at <= :now AND created_at < :stale)
       OR (status IN ('STEERING', 'DISPATCHING') AND lease_expires_at < :now)
    ORDER BY created_at, ordinal
""")


def _utc(value) -> datetime:
    if isinstance(value, str):  # SQLite returns text timestamps
        value = datetime.fromisoformat(value)
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def _span(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    hours, minutes = divmod(seconds // 60, 60)
    return f"{hours}h{minutes}m" if minutes else f"{hours}h"


@dataclass(frozen=True)
class WaitingDelivery:
    """One delivery that has waited past its threshold, and why."""

    id: str
    status: str
    target: str
    sender: str
    task_id: str | None
    age_seconds: int
    wait: str  # busy | held | idle | long_turn | lease_expired
    detail: str

    @property
    def degraded(self) -> bool:
        return self.wait in ("idle", "long_turn", "lease_expired")

    def describe(self) -> str:
        context = ", ".join(filter(None, (self.task_id, _span(self.age_seconds), self.detail)))
        return f"{self.target} ← {self.sender} ({context})"

    def as_dict(self) -> dict:
        return {
            "id": self.id, "status": self.status, "target": self.target,
            "sender": self.sender, "task_id": self.task_id,
            "age_seconds": self.age_seconds, "wait": self.wait,
            "detail": self.detail, "degraded": self.degraded,
        }


class DeliveryQueueHealth:
    def __init__(self, engine):
        self.engine = engine

    def check(self, now: datetime) -> dict:
        with self.engine.connect() as conn:
            rows = conn.execute(_WAITING, {"now": now, "stale": now - STALE_QUEUED}).mappings().all()
            waits = self._classify(conn, rows, now)
        stuck = [w for w in waits if w.degraded]
        normal = [w for w in waits if not w.degraded]
        parts = []
        if stuck:
            parts.append(_group(stuck, "delivery stuck", "deliveries stuck"))
        if normal:
            parts.append(_group(normal, "queued behind busy or held bots",
                                "queued behind busy or held bots"))
        result = _check(NAME, DEGRADED if stuck else OK, "; ".join(parts) or None)
        if waits:
            result["items"] = [w.as_dict() for w in (stuck + normal)[:MAX_ITEMS]]
            result["item_count"] = len(waits)
        return result

    @staticmethod
    def _classify(conn, rows, now: datetime) -> list[WaitingDelivery]:
        queued = [row for row in rows if row["status"] == "QUEUED"]
        states = classify_targets(conn, {row["target_bot_id"] for row in queued})
        waits = []
        for row in rows:
            age = (now - _utc(row["created_at"])).total_seconds()
            if row["status"] != "QUEUED":
                expired = (now - _utc(row["lease_expires_at"])).total_seconds()
                waits.append(_waiting(row, age, "lease_expired", f"lease expired {_span(expired)} ago"))
                continue
            # Admission classifies the target without the head's own turn id.
            state = TargetState(tuple(
                turn for turn in states[row["target_bot_id"]].rows if turn["id"] != row["turn_id"]
            ))
            if not state.idle:
                turn_age = max((now - _utc(turn["created_at"])).total_seconds() for turn in state.rows)
                if turn_age > BUSY_TURN_CEILING.total_seconds():
                    waits.append(_waiting(row, age, "long_turn", f"behind a turn open {_span(turn_age)}"))
                else:
                    waits.append(_waiting(row, age, "busy", f"target busy: {state.kind}"))
                continue
            gate = gate_blocks(conn, row["target_bot_id"], GateActor("delivery_claim", row["turn_id"]))
            if gate is not None:
                waits.append(_waiting(row, age, "held",
                                      f"held by {gate.mode} gate ({gate.owner_kind} {gate.owner_id})"))
            else:
                waits.append(_waiting(row, age, "idle", "target idle, not picked up"))
        return waits


def _waiting(row, age: float, wait: str, detail: str) -> WaitingDelivery:
    sender = row["author_entity_id"] if row["author_entity_type"] == "user" else row["sender_bot_id"]
    return WaitingDelivery(
        id=row["id"], status=row["status"], target=row["target_bot_id"], sender=sender,
        task_id=row["task_id"], age_seconds=int(age), wait=wait, detail=detail,
    )


def _group(waits: list[WaitingDelivery], one: str, many: str) -> str:
    named = ", ".join(w.describe() for w in waits[:SHOWN])
    more = f" +{len(waits) - SHOWN} more" if len(waits) > SHOWN else ""
    return f"{len(waits)} {one if len(waits) == 1 else many}: {named}{more}"
