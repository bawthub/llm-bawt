"""The quiesce control producer: one control-lane ask per held running run (TASK-1431 S3).

A quiesce job's gate roster (``gate_members``) says which runs must be asked
to stop. The producer turns each member that needs asking into a link to a
control-lane delivery (``inter_bot_control_lane``). From then on the delivery
dispatcher owns the ask: claim, steer under ``DeliveryControlGrant``, retry and
settle. The producer itself has no claim, lease, retry, backoff, attempts or
CAS. It runs in the app on the dispatcher leader's sweep (``LeaderSidePumps``).

Per live member (``LIVE_MEMBER_SQL``) that is ``pending`` with no link, or
``steering`` with a link to a dead row (design §3, r2 R1a), in one transaction:

* ``lock_target(bot)``, then the member ``FOR UPDATE SKIP LOCKED``, re-read
  with the same selector. Turn rows are only read, never locked:
  ``bind_agent_bridge`` locks the turn row and then updates its members, so
  locking a turn row while holding a member could deadlock with a bind (Caid,
  df30f61).
* Member-local decisions: revoked, turn gone or ended -> ``not_needed``; the
  job's caller's own turn -> ``self``; no pinned request yet -> skipped,
  nothing written; a pinned run on a backend that cannot steer ->
  ``unsupported``; a binding that is unknown -> ``unconfirmed``; a pinned
  turn with no user -> ``unconfirmed`` (``turn_user_missing``: the steer's
  user must be the turn's own, and there is no fallback).
* The link step (design §3). The run's rows are found by ``run_key``
  (``control_request_id``). The newest absorbing row is linked; otherwise
  successor ``k`` (``run_key`` for 0, ``run_key:r<k>`` after) is enqueued
  and linked. A duplicate key is linked only when it is that run's absorbing
  control row; anything else is ``unconfirmed`` (``control_key_collision``).

A run that was asked, or may have been, is never asked again; a run that
needs asking always gets a live row.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Mapping

from sqlalchemy import text as sa_text

from ..inter_bot_control_lane import (
    ABSORBING_SQL,
    CONTROL_LANE,
    STEERABLE_BACKEND,
    control_metadata,
    enqueue_control,
)
from .system_gate_admission import LIVE_MEMBER_SQL, control_target
from .turn_admission import lock_target
from .turn_gates import REVOKED_UNSTARTED, ensure_gate_schema

log = logging.getLogger(__name__)

# A member that needs a (re)link: never linked, or linked to a dead row.
NEEDS_LINK_SQL = (
    "((m.control_state = 'pending' AND m.control_delivery_id IS NULL)"
    " OR (m.control_state = 'steering' AND NOT EXISTS ("
    "SELECT 1 FROM inter_bot_deliveries d WHERE d.id = m.control_delivery_id AND "
    + ABSORBING_SQL + ")))"
)
_TABLES = (
    "SELECT to_regclass('ops_jobs') IS NOT NULL AND to_regclass('gate_members') IS NOT NULL"
    " AND to_regclass('inter_bot_deliveries') IS NOT NULL"
)
# Linked members wait on their rows; backing-off rows wake the dispatcher.
_NEXT_CONTROL_RETRY_SQL = """
    SELECT EXTRACT(EPOCH FROM (min(available_at) - clock_timestamp()))
    FROM inter_bot_deliveries
    WHERE lane = 'control' AND status = 'QUEUED' AND available_at > clock_timestamp()
"""

LINKED = "linked"  # to an existing absorbing row
ENQUEUED = "enqueued"  # a new row, linked
UNPINNED = "unpinned"  # no pinned request yet: retried on a later pass
SETTLED = "settled"  # a member-local decision; nothing to steer


def control_request_id(turn_id: str, agent_request_id: str) -> str:
    """The run key of one pinned run: the same for every job and gate generation (Caid)."""
    return f"steer_quiesce_{turn_id}_{agent_request_id}"


def successor_key(run_key: str, k: int) -> str:
    """The idempotency key of the run's ``k``-th row."""
    return run_key if k == 0 else f"{run_key}:r{k}"


def successor_index(run_key: str, idempotency_key: str | None) -> int | None:
    """``k`` for one of the run's own keys, else None."""
    if idempotency_key == run_key:
        return 0
    match = re.fullmatch(re.escape(run_key) + r":r(\d+)", idempotency_key or "")
    return int(match.group(1)) if match else None


@dataclass(frozen=True)
class ProducerPass:
    """One pass: members linked (to new or existing rows), and when to look again."""

    linked: int = 0
    enqueued: int = 0
    retry_in: float | None = None


class QuiesceControlProducer:
    """Links each live member that needs asking to its run's control row."""

    UNPINNED_RECHECK_SECONDS = 2.0  # a pin arrives with bind_agent_bridge, which wakes nothing
    BATCH = 32

    def __init__(self, store):
        self.store = store
        self.engine = store.engine

    # --- reads -------------------------------------------------------------

    def due(self) -> list[tuple[int, str]]:
        """(member id, bot id) of live members needing a link: pinned first, then oldest. No locks.

        An unpinned member writes nothing and stays due, so it is ordered
        after every pinned one: a batch full of unpinned members would
        otherwise starve the pinned members behind them (Codex S3 S1).
        """
        with self.engine.connect() as conn:
            if not conn.execute(sa_text(_TABLES)).scalar():
                return []
            rows = conn.execute(sa_text(
                LIVE_MEMBER_SQL.format(where=NEEDS_LINK_SQL)
                + " ORDER BY (m.agent_request_id IS NULL), m.id LIMIT :limit"
            ), {"limit": self.BATCH}).mappings().all()
        return [(int(row["id"]), row["bot_id"]) for row in rows]

    def next_control_retry(self) -> float | None:
        """Seconds until the earliest backing-off QUEUED control row is ready, if any.

        Ready rows are left out: the sweep that ran this pass already saw them.
        """
        with self.engine.connect() as conn:
            seconds = conn.execute(sa_text(_NEXT_CONTROL_RETRY_SQL)).scalar()
        return None if seconds is None else max(0.0, float(seconds))

    # --- one member --------------------------------------------------------

    def produce(self, member_id: int, bot_id: str, message: str) -> str | None:
        """Decide or link one member under its bot's lock; None when there is nothing to do."""
        with self.engine.begin() as conn:
            lock_target(conn, bot_id)
            row = conn.execute(sa_text(
                LIVE_MEMBER_SQL.format(where=f"m.id = :id AND m.bot_id = :bot AND {NEEDS_LINK_SQL}")
                + " FOR UPDATE OF m SKIP LOCKED"
            ), {"id": member_id, "bot": bot_id}).mappings().first()
            if row is None:
                return None  # released, linked meanwhile, or held by another producer
            return self._decide(conn, row, message)

    def _decide(self, conn, row: Mapping[str, Any], message: str) -> str:
        if row["disposition"] == REVOKED_UNSTARTED or row["turn_present"] is None or row["turn_ended"]:
            return self._set_state(conn, row["id"], "not_needed", "turn_not_running")
        if row["caller_turn_id"] and row["caller_turn_id"] == row["turn_id"]:
            return self._set_state(conn, row["id"], "self")
        target = control_target(row)
        if target is None:
            return UNPINNED
        if target.backend and target.backend != STEERABLE_BACKEND:
            return self._set_state(conn, row["id"], "unsupported", f"backend:{target.backend}")
        if not target.backend or not target.session_key:
            return self._set_state(conn, row["id"], "unconfirmed", "pinned_run_unaddressable")
        if not row["turn_user_id"]:
            return self._set_state(conn, row["id"], "unconfirmed", "turn_user_missing")
        pin = {"turn_id": row["turn_id"], "agent_request_id": target.request_id,
               "backend": target.backend, "session_key": target.session_key}
        return self._link(conn, row, pin, message)

    def _link(self, conn, row: Mapping[str, Any], pin: Mapping[str, Any], message: str) -> str:
        """Design §3 link step, under the caller's ``lock_target(bot)``."""
        run_key = control_request_id(pin["turn_id"], pin["agent_request_id"])
        rows = conn.execute(sa_text(f"""
            SELECT d.id, d.idempotency_key, {ABSORBING_SQL} AS absorbing
            FROM inter_bot_deliveries d
            WHERE d.target_bot_id = :bot AND d.lane = 'control'
              AND d.metadata_json->'control'->>'run_key' = :run_key
        """), {"bot": row["bot_id"], "run_key": run_key}).mappings().all()
        indexed = sorted(((successor_index(run_key, r["idempotency_key"]), r) for r in rows),
                         key=lambda pair: -1 if pair[0] is None else pair[0], reverse=True)
        absorbing = next((r for _, r in indexed if r["absorbing"]), None)
        if absorbing is not None:
            return self._set_link(conn, row["id"], absorbing["id"], LINKED)
        k = 1 + max((index for index, _ in indexed if index is not None), default=-1)
        record, duplicate = enqueue_control(
            self.store, conn, target_bot_id=row["bot_id"], target_run=pin, run_key=run_key,
            idempotency_key=successor_key(run_key, k), message=message, user=row["turn_user_id"],
        )
        if duplicate and not self._absorbs_as(conn, record, run_key, pin):
            log.warning("quiesce control key collision: member=%s run_key=%s row=%s",
                        row["id"], run_key, record.id)
            return self._set_state(conn, row["id"], "unconfirmed", "control_key_collision")
        return self._set_link(conn, row["id"], record.id, LINKED if duplicate else ENQUEUED)

    @staticmethod
    def _absorbs_as(conn, record, run_key: str, pin: Mapping[str, Any]) -> bool:
        """A duplicate is this run's own absorbing control row (r2 S8)."""
        control = control_metadata(record)
        if getattr(record, "lane", None) != CONTROL_LANE or control.get("run_key") != run_key:
            return False
        if dict(control.get("target_run") or {}) != dict(pin):
            return False
        return bool(conn.execute(sa_text(
            f"SELECT {ABSORBING_SQL} FROM inter_bot_deliveries d WHERE d.id = :id"
        ), {"id": record.id}).scalar())

    @staticmethod
    def _set_link(conn, member_id: int, delivery_id: str, outcome: str) -> str:
        conn.execute(sa_text("""
            UPDATE gate_members SET control_delivery_id = :delivery, control_state = 'steering',
                control_changed_at = now(), control_error = NULL
            WHERE id = :id
        """), {"id": member_id, "delivery": delivery_id})
        return outcome

    @staticmethod
    def _set_state(conn, member_id: int, state: str, error: str | None = None) -> str:
        """A member-local decision; stamps ``control_changed_at``."""
        conn.execute(sa_text("""
            UPDATE gate_members SET control_state = :state, control_changed_at = now(),
                control_error = :error
            WHERE id = :id
        """), {"id": member_id, "state": state, "error": error})
        return SETTLED

    # --- one pass ----------------------------------------------------------

    def run_pass(self) -> ProducerPass:
        """Link every due member. Sync: the caller runs it off the event loop."""
        from ..ops.quiesce_prompts import QUIESCE_REQUEST_KEY, resolve

        ensure_gate_schema(self.engine)  # once per engine; the delivery store already ran it
        messages: dict[str, str] = {}
        linked = enqueued = 0
        unpinned = False
        for member_id, bot_id in self.due():
            if bot_id not in messages:  # resolved outside the bot's lock, once per pass
                messages[bot_id] = resolve(QUIESCE_REQUEST_KEY, bot_id)
            outcome = self.produce(member_id, bot_id, messages[bot_id])
            linked += outcome in (LINKED, ENQUEUED)
            enqueued += outcome == ENQUEUED
            unpinned = unpinned or outcome == UNPINNED
            if outcome in (LINKED, ENQUEUED):
                log.info("quiesce control member=%s bot=%s %s", member_id, bot_id, outcome)
        retry = self.next_control_retry()
        candidates = [s for s in (retry, self.UNPINNED_RECHECK_SECONDS if unpinned else None) if s is not None]
        return ProducerPass(linked=linked, enqueued=enqueued,
                            retry_in=min(candidates) if candidates else None)


def build_producer(store) -> QuiesceControlProducer | None:
    """The app's producer, or None off PostgreSQL (gates are PostgreSQL-only)."""
    engine = getattr(store, "engine", None)
    if getattr(getattr(engine, "dialect", None), "name", None) != "postgresql":
        return None
    return QuiesceControlProducer(store)


__all__ = [
    "ENQUEUED",
    "LINKED",
    "NEEDS_LINK_SQL",
    "ProducerPass",
    "QuiesceControlProducer",
    "SETTLED",
    "UNPINNED",
    "build_producer",
    "control_request_id",
    "successor_index",
    "successor_key",
]
