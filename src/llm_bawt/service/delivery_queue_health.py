"""``deliveries.queue`` health: name what waits, flag only waits nothing explains (TASK-1458).

The dispatcher acts on one row per target, so each target's waits are judged
by the row its claim (``DeliveryClaim.claim_next``) would take next, read with
the claim's own selectors and authorities:

* the candidate is the FIFO head (``inter_bot_preclaim.head_candidate``), or,
  under a live user-Stop hold, the user's ready chat row the hold lets run
  ahead (``user_bypass_candidate``), judged with that hold excluded exactly as
  the claim excludes it;
* a candidate already STEERING/DISPATCHING is in flight (OK) until its lease
  expires (Degraded); one in retry backoff is OK;
* target busy: ``turn_admission.classify_targets`` sees another open turn for
  the target (the candidate's own turn id excluded, as the claim does). OK;
* held: ``turn_gates.gate_blocks`` refuses the ``delivery_claim`` actor, i.e. a
  live quiesce/restart/stop gate is deliberately holding the target. OK;
* otherwise the target is idle and ungated, so the dispatcher should have
  claimed the candidate once it was due for ``STALE_QUEUED``: Degraded.

A row behind the candidate waits on it, so it inherits the candidate's verdict
and names it. No watchdog ends a long agent turn while the app runs:
``recovery.orphan_turns`` settles stranded turns only at startup, and the turn
reaper skips ordinary agent starts. A wait behind a turn open longer than
``BUSY_TURN_CEILING`` is therefore Degraded too, so a stuck turn can't hide
behind "target busy". The check reads only; it locks and admits nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from ..inter_bot_preclaim import head_candidate, user_bypass_candidate
from .health_checks import DEGRADED, OK, _check
from .turn_admission import TargetState, classify_targets
from .turn_gates import GateActor, gate_blocks, live_gate_for_owner
from .user_stop_hold import hold_owner

NAME = "deliveries.queue"
STALE_QUEUED = timedelta(minutes=2)
# Agent turns, 14 days to 2026-10-25: p99.9 73 min, max 99 min.
BUSY_TURN_CEILING = timedelta(hours=2)
SHOWN = 3  # deliveries named per group in the reason line
MAX_ITEMS = 50  # structured rows returned; ``item_count`` keeps the total
DEGRADED_WAITS = frozenset({"idle", "long_turn", "lease_expired"})

# Due (and created) over STALE_QUEUED ago, so a row just out of backoff isn't
# reported before a sweep could claim it.
_WAITING = text("""
    SELECT id, status, lane, target_bot_id, sender_bot_id, author_entity_type,
           author_entity_id, task_id, turn_id, created_at, available_at, lease_expires_at
    FROM inter_bot_deliveries
    WHERE (status = 'QUEUED' AND available_at < :stale AND created_at < :stale)
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
class Verdict:
    wait: str  # busy | held | idle | pending | backoff | in_flight | long_turn | lease_expired
    detail: str


@dataclass(frozen=True)
class WaitingDelivery:
    """One delivery that has waited past its threshold, and why."""

    id: str
    status: str
    target: str
    sender: str
    task_id: str | None
    age_seconds: int
    wait: str
    detail: str
    behind: str | None = None  # the claim candidate this row queues behind

    @property
    def degraded(self) -> bool:
        return self.wait in DEGRADED_WAITS

    def describe(self) -> str:
        context = ", ".join(filter(None, (self.task_id, _span(self.age_seconds), self.detail)))
        return f"{self.target} ← {self.sender} ({context})"

    def as_dict(self) -> dict:
        return {
            "id": self.id, "status": self.status, "target": self.target,
            "sender": self.sender, "task_id": self.task_id,
            "age_seconds": self.age_seconds, "wait": self.wait,
            "detail": self.detail, "degraded": self.degraded, "behind": self.behind,
        }


class DeliveryQueueHealth:
    def __init__(self, engine):
        self.engine = engine

    def check(self, now: datetime) -> dict:
        with self.engine.connect() as conn:
            rows = conn.execute(_WAITING, {"now": now, "stale": now - STALE_QUEUED}).mappings().all()
            waits = _classify(conn, rows, now)
        stuck = [w for w in waits if w.degraded]
        normal = [w for w in waits if not w.degraded]
        parts = []
        if stuck:
            parts.append(_group(stuck, "delivery stuck", "deliveries stuck"))
        if normal:
            parts.append(_group(normal, "waiting as expected", "waiting as expected"))
        result = _check(NAME, DEGRADED if stuck else OK, "; ".join(parts) or None)
        if waits:
            result["items"] = [w.as_dict() for w in (stuck + normal)[:MAX_ITEMS]]
            result["item_count"] = len(waits)
        return result


def _classify(conn, rows, now: datetime) -> list[WaitingDelivery]:
    queued = [row for row in rows if row["status"] == "QUEUED"]
    lanes = sorted({row["target_bot_id"] for row in queued if row["lane"] == "normal"})
    states = classify_targets(conn, {row["target_bot_id"] for row in queued})
    # Per target: candidate id -> verdict; the FIFO head is listed last.
    candidates = {target: _candidates(conn, target, states[target], now) for target in lanes}
    waits = []
    for row in rows:
        age = (now - _utc(row["created_at"])).total_seconds()
        if row["status"] != "QUEUED":
            expired = (now - _utc(row["lease_expires_at"])).total_seconds()
            waits.append(_waiting(row, age, Verdict("lease_expired", f"lease expired {_span(expired)} ago")))
            continue
        if row["lane"] != "normal":
            # Control rows steer a pinned run outside the FIFO (TASK-1431).
            waits.append(_waiting(row, age, _verdict(conn, row, states[row["target_bot_id"]], None, now)))
            continue
        own = candidates[row["target_bot_id"]]
        if row["id"] in own or not own:
            verdict = own.get(row["id"]) or _verdict(conn, row, states[row["target_bot_id"]], None, now)
            waits.append(_waiting(row, age, verdict))
            continue
        # Everything else waits on the FIFO head (listed last), not on itself.
        head_id, verdict = list(own.items())[-1]
        waits.append(_waiting(row, age, Verdict(verdict.wait, f"behind {_short(head_id)}: {verdict.detail}"),
                              behind=head_id))
    return waits


def _candidates(conn, target: str, state: TargetState, now: datetime) -> dict[str, Verdict]:
    """What ``claim_next`` would try for ``target``: candidate id -> verdict."""
    found = {}
    owner = hold_owner(target)
    if live_gate_for_owner(conn, owner, fresh=True) is not None:
        bypass = user_bypass_candidate(conn, target)
        if bypass is not None:
            found[bypass["id"]] = _verdict(conn, bypass, state, owner, now)
    head = head_candidate(conn, target)
    if head is not None and head["id"] not in found:
        found[head["id"]] = _verdict(conn, head, state, None, now)
    return found


def _verdict(conn, row, state: TargetState, exempt_owner: str | None, now: datetime) -> Verdict:
    if row["status"] != "QUEUED":
        lease = row["lease_expires_at"]
        if lease is not None and _utc(lease) < now:
            return Verdict("lease_expired", f"{row['status'].lower()} lease expired "
                                            f"{_span((now - _utc(lease)).total_seconds())} ago")
        return Verdict("in_flight", f"{row['status'].lower()} in flight")
    due = _utc(row["available_at"])
    if due > now:
        return Verdict("backoff", f"retry due in {_span((due - now).total_seconds())}")
    # The claim classifies the target without the candidate's own turn id.
    others = TargetState(tuple(turn for turn in state.rows if turn["id"] != row["turn_id"]))
    if not others.idle:
        turn_age = max((now - _utc(turn["created_at"])).total_seconds() for turn in others.rows)
        if turn_age > BUSY_TURN_CEILING.total_seconds():
            return Verdict("long_turn", f"behind a turn open {_span(turn_age)}")
        return Verdict("busy", f"target busy: {others.kind}")
    gate = gate_blocks(conn, row["target_bot_id"], GateActor("delivery_claim", row["turn_id"]),
                       exclude_owner_id=exempt_owner)
    if gate is not None:
        return Verdict("held", f"held by {gate.mode} gate ({gate.owner_kind} {gate.owner_id})")
    if now - max(due, _utc(row["created_at"])) > STALE_QUEUED:
        return Verdict("idle", "target idle, not picked up")
    return Verdict("pending", "target idle, claim due")


def _short(delivery_id: str) -> str:
    return delivery_id[:17]  # "delivery-" + 8 hex


def _waiting(row, age: float, verdict: Verdict, *, behind: str | None = None) -> WaitingDelivery:
    sender = row["author_entity_id"] if row["author_entity_type"] == "user" else row["sender_bot_id"]
    return WaitingDelivery(
        id=row["id"], status=row["status"], target=row["target_bot_id"], sender=sender,
        task_id=row["task_id"], age_seconds=int(age), wait=verdict.wait, detail=verdict.detail,
        behind=behind,
    )


def _group(waits: list[WaitingDelivery], one: str, many: str) -> str:
    named = ", ".join(w.describe() for w in waits[:SHOWN])
    more = f" +{len(waits) - SHOWN} more" if len(waits) > SHOWN else ""
    return f"{len(waits)} {one if len(waits) == 1 else many}: {named}{more}"
