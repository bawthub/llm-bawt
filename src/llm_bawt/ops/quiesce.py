"""Quiesce driver: hold a scope's agent turns, wait for them to drain, then
dispatch the job inside a protected interval (TASK-1233 C).

A quiesce job is created QUIESCING and is never claimed from QUEUED
(:meth:`.store.OpsStore.claim_job`). This driver, run by the ops reconciler,
moves it forward one bounded step at a time:

1. **Lease.** Every step runs in ONE transaction that first locks the
   operation row (the same order as a slot claim), then claims or renews the
   job's driver lease on the DB clock. A step whose claim matches no row
   (another driver holds a live lease, or the job left QUIESCING) rolls back
   having written nothing, so a stale driver can never open an orphan gate.
2. **Open.** No gate yet: a disabled operation fails the job (no action
   submitted); a job still unopened ``quiesce_timeout_seconds`` after
   submission fails ``quiesce_timeout`` (a preflight that keeps failing, noted
   by :meth:`QuiesceDriver.note_unopened`, or a slot that never frees); a
   reached concurrency cap waits with a visible note; otherwise
   ``activate_hold`` opens a job-owned hold gate (id = generation) and the
   quiesce deadline starts on the DB clock.
3. **Hold.** Each step renews the gate. A gate that lapsed (the driver was
   away longer than its TTL) is rebuilt as a new generation with its
   predecessor's undrained members carried forward, or fails the job
   ``quiesce_lost`` past the deadline. Then the roster's drain status: all
   drained → dispatch; past the deadline → fail ``quiesce_timeout``.
4. **Dispatch (CAS).** The gate is extended to cover the start delay, the
   action's timeout and two margins, and the worker is given
   ``start_not_after = expires_at - timeout - margin``: it refuses to start
   later (:func:`.worker.fence_expired`), so the action always finishes
   inside the hold. The job moves to DISPATCHING in the same transaction;
   the caller then submits it exactly like a claimed job.

Any terminal job's gates are released, and their members' pending steers
cancelled, by :meth:`QuiesceDriver.release_terminal` (reconciler sweep).
PostgreSQL only.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from sqlalchemy import text

from ..service.turn_admission import activate_hold
from ..service.turn_gates import (GateScope, GateSpec, _LIVE, ensure_gate_schema, extend_gate, gate_drain_status,
                                  release_gates)
from .models import (JOB_DISPATCHING, JOB_FAILED, JOB_QUIESCING, JOB_TERMINAL_STATES)
from .store import slot_cap_reached, slot_holders

logger = logging.getLogger(__name__)

LEASE_SECONDS = 30
GATE_TTL_SECONDS = 60
MARGIN_SECONDS = 30
ACK_STATUSES = ("stopping", "cannot_stop")


class QuiesceUnavailable(RuntimeError):
    """Quiesce needs the canonical PostgreSQL database."""


@dataclass(frozen=True)
class StepResult:
    """What one driver step did. ``dispatch``: the job is now DISPATCHING
    with a fence and the caller must submit it."""

    outcome: str
    dispatch: bool = False


class TurnGates:
    """The admission-side gate operations the driver uses (injectable for tests)."""

    def activate(self, conn, spec: GateSpec, predecessor: int | None):
        return activate_hold(conn, spec, predecessor=predecessor)

    def drain_status(self, conn, gate_id: int) -> list[Any]:
        return gate_drain_status(conn, gate_id)


def cancel_pending_controls(conn, *, gate_ids: Iterable[int]) -> int:
    """Cancel every not-yet-delivered steer of these generations' members."""
    ids = sorted({int(i) for i in gate_ids})
    if not ids:
        return 0
    return conn.execute(text(
        "UPDATE gate_members SET control_state='cancelled', control_claim_token=NULL,"
        " control_lease_expires_at=NULL"
        " WHERE gate_id = ANY(:ids) AND control_state IN ('pending','steering')"
    ), {"ids": ids}).rowcount


def release_job_gates(conn, job_id: str) -> list[int]:
    """Release all of a job's gates and cancel their members' pending steers."""
    released = release_gates(conn, owner_id=job_id)
    every = [row[0] for row in conn.execute(text(
        "SELECT id FROM system_gates WHERE owner_kind='job' AND owner_id=:job"
    ), {"job": job_id}).all()]
    cancel_pending_controls(conn, gate_ids=every)
    return released


def record_ack(engine, *, bot_id: str, turn_id: str, status: str, note: str | None = None) -> int | None:
    """Record a held turn's answer to a quiesce steer; the gate id, or ``None``
    when the turn is no member of the current live generation of a job that is
    still QUIESCING. Once the job dispatches the drain decision is made, so a
    later answer has nothing to inform and is refused, not silently stored."""
    if status not in ACK_STATUSES:
        raise ValueError(f"ack status must be one of {list(ACK_STATUSES)}")
    if engine is None or engine.dialect.name != "postgresql":
        raise QuiesceUnavailable("quiesce requires PostgreSQL")
    with engine.begin() as conn:
        row = conn.execute(text(
            "UPDATE gate_members m SET ack_status=:status, ack_note=:note, acked_at=now()"
            " FROM system_gates g JOIN ops_jobs j"
            "   ON j.id=g.owner_id AND g.owner_kind='job' AND j.quiesce_gate_id=g.id"
            " WHERE g.id=m.gate_id AND g.released_at IS NULL AND g.expires_at > now()"
            "   AND j.state=:quiescing AND m.bot_id=:bot AND m.turn_id=:turn"
            " RETURNING m.gate_id"
        ), {"status": status, "note": (note or None) and note[:2000], "bot": bot_id,
            "turn": turn_id, "quiescing": JOB_QUIESCING}).first()
    return int(row[0]) if row else None


def scope_covers_bot(conn, snapshot: dict[str, Any], bot_id: str) -> bool:
    """Whether a quiesce snapshot's scope covers ``bot_id``'s current profile."""
    row = conn.execute(text("SELECT bot_type, agent_backend FROM bot_profiles WHERE slug=:slug"),
                       {"slug": bot_id}).mappings().first()
    if row is None:
        return False
    scope = GateScope.from_json(snapshot["quiesce"]["scope"])
    return scope.covers(bot_id, bot_type=row["bot_type"], agent_backend=row["agent_backend"])


def caller_instruction(engine, job_id: str, bot_id: str | None) -> dict[str, Any] | None:
    """What the job's own caller must do while the job is QUIESCING and holds it.

    The caller's running turn is in the hold roster like any other member
    (Snark ruling): the job dispatches only once that turn exits. Polling
    ops_job_status in the same turn keeps the turn alive until the quiesce
    deadline fails the job, so the caller is told to end its turn. Its terminal
    receipt always arrives as a new turn (TASK-1233 e): an approved job's
    continuation delivers it, and a direct job carries the receipt callback
    recorded at creation exactly when its scope covered the caller. ``None``
    for any other state, any other bot, or a caller the job does not hold.
    """
    if not bot_id or engine is None or engine.dialect.name != "postgresql":
        return None
    with engine.connect() as conn:
        job = conn.execute(text(
            "SELECT state, invocation_snapshot_json, approval_request_id, caller_bot_id,"
            " receipt_callback_json FROM ops_jobs WHERE id=:id"
        ), {"id": job_id}).mappings().first()
        if job is None or job["state"] != JOB_QUIESCING or job["caller_bot_id"] != bot_id:
            return None
        if job["approval_request_id"]:
            held = scope_covers_bot(conn, json.loads(job["invocation_snapshot_json"]), bot_id)
        else:
            callback = json.loads(job["receipt_callback_json"]) if job["receipt_callback_json"] else {}
            held = callback.get("bot_id") == bot_id
    if not held:
        return None
    return {
        "action": "end_turn",
        "held_bot": bot_id,
        "result_delivery": "new_turn",
        "message": ("This operation waits for your current turn to exit before it runs. End your turn "
                    "now and do not poll in this turn: polling keeps the turn alive until "
                    "quiesce_timeout fails the job with no action. The terminal receipt arrives "
                    "as a new turn."),
    }


def _status_row(status: Any) -> dict[str, Any]:
    return {"turn_id": status.turn_id, "drained": bool(status.drained), "reason": status.reason}


class QuiesceDriver:
    def __init__(self, engine, *, driver_id: str | None = None, gates: TurnGates | None = None,
                 lease_seconds: float = LEASE_SECONDS, gate_ttl_seconds: float = GATE_TTL_SECONDS,
                 margin_seconds: float = MARGIN_SECONDS):
        if engine is None or engine.dialect.name != "postgresql":
            raise QuiesceUnavailable("quiesce requires PostgreSQL")
        ensure_gate_schema(engine)
        self.engine = engine
        self.driver_id = driver_id or uuid.uuid4().hex
        self.gates = gates or TurnGates()
        self.lease_seconds = lease_seconds
        self.gate_ttl_seconds = gate_ttl_seconds
        self.margin_seconds = margin_seconds

    # -- one step ---------------------------------------------------------

    def step(self, job_id: str) -> StepResult:
        with self.engine.begin() as conn:
            slug = conn.execute(text("SELECT operation_slug FROM ops_jobs WHERE id=:id"),
                                {"id": job_id}).scalar()
            if slug is None:
                return StepResult("missing")
            op = conn.execute(text(
                "SELECT enabled, soft_deleted_at FROM ops_operations WHERE slug=:slug FOR UPDATE"
            ), {"slug": slug}).mappings().first()
            job = self._claim(conn, job_id)
            if job is None:
                return StepResult("not_owner")  # wrote nothing
            snapshot = json.loads(job["invocation_snapshot_json"])
            if op is None or not op["enabled"] or op["soft_deleted_at"] is not None:
                return self._fail(conn, job, "operation disabled before dispatch; no action submitted")
            if job["quiesce_gate_id"] is None:
                return self._open(conn, job, snapshot)
            return self._hold(conn, job, snapshot)

    def _claim(self, conn, job_id: str):
        return conn.execute(text(
            "UPDATE ops_jobs SET quiesce_driver_id=:me,"
            " quiesce_driver_lease_expires_at=now() + (:lease * interval '1 second'),"
            " last_reconcile_at=now()"
            " WHERE id=:id AND state=:quiescing AND (quiesce_driver_id IS NULL"
            "   OR quiesce_driver_id=:me OR quiesce_driver_lease_expires_at <= now())"
            " RETURNING id, operation_slug, invocation_snapshot_json, quiesce_gate_id,"
            "   quiesce_json, error_text, submitted_at, now() >= quiesce_deadline_at AS past_deadline"
        ), {"me": self.driver_id, "lease": float(self.lease_seconds), "id": job_id,
            "quiescing": JOB_QUIESCING}).mappings().first()

    def _spec(self, job, snapshot) -> GateSpec:
        scope = GateScope.from_json(snapshot["quiesce"]["scope"])
        return GateSpec("job", job["id"], "hold", scope, self.gate_ttl_seconds,
                        reason=f"ops job {job['id']} ({job['operation_slug']}) is quiescing")

    def _receipt(self, job) -> dict[str, Any]:
        return json.loads(job["quiesce_json"]) if job["quiesce_json"] else {"generations": []}

    def _write(self, conn, job_id: str, values: dict[str, Any], *, deadline_from: int | None = None):
        sets = [f"{key}=:{key}" for key in values]
        if deadline_from is not None:
            sets.append("quiesce_deadline_at=COALESCE(quiesce_deadline_at,"
                        " now() + (:timeout * interval '1 second'))")
        conn.execute(text(f"UPDATE ops_jobs SET {', '.join(sets)} WHERE id=:id"),
                     {**values, "id": job_id, "timeout": deadline_from})

    @staticmethod
    def _open_expired(conn, job, snapshot) -> bool:
        """A job that has not opened its hold within ``quiesce_timeout_seconds``
        of submission fails: nothing else bounds a preflight that keeps failing
        or a slot that never frees, and its caller waits for the receipt."""
        return bool(conn.execute(text(
            "SELECT now() >= CAST(:submitted AS timestamptz) + (:timeout * interval '1 second')"
        ), {"submitted": job["submitted_at"],
            "timeout": float(snapshot["quiesce"]["timeout_seconds"])}).scalar())

    def _fail_unopened(self, conn, job, note: str | None) -> StepResult:
        detail = f"; last: {note}" if note else ""
        return self._fail(conn, job, "quiesce_timeout: the hold did not open within quiesce_timeout_seconds"
                                     f" of submission; no action submitted{detail}")

    def note_unopened(self, job_id: str, error: str) -> StepResult:
        """Record why a not-yet-opened job cannot open (a preflight failure),
        failing it once its open deadline has passed."""
        with self.engine.begin() as conn:
            job = self._claim(conn, job_id)
            if job is None:
                return StepResult("not_owner")
            if job["quiesce_gate_id"] is not None:
                return StepResult("opened")
            if self._open_expired(conn, job, json.loads(job["invocation_snapshot_json"])):
                return self._fail_unopened(conn, job, error)
            self._write(conn, job["id"], {"error_text": error})
            return StepResult("preflight_blocked")

    def _open(self, conn, job, snapshot) -> StepResult:
        if self._open_expired(conn, job, snapshot):
            return self._fail_unopened(conn, job, job["error_text"])
        holders = slot_holders(conn, job["operation_slug"], exclude_job_id=job["id"])
        if slot_cap_reached(holders, snapshot["execution"].get("max_concurrent")):
            self._write(conn, job["id"], {"error_text": "waiting for a concurrency slot before quiescing"})
            return StepResult("waiting_slot")
        activated = self.gates.activate(conn, self._spec(job, snapshot), None)
        receipt = self._receipt(job)
        receipt["generations"].append(activated.gate.id)
        receipt["roster"] = [{"turn_id": m.turn_id, "drained": None, "reason": None}
                             for m in activated.members]
        self._write(conn, job["id"], {"quiesce_gate_id": activated.gate.id, "error_text": None,
                                      "quiesce_json": json.dumps(receipt)},
                    deadline_from=int(snapshot["quiesce"]["timeout_seconds"]))
        return StepResult("opened")

    def _hold(self, conn, job, snapshot) -> StepResult:
        gate_id = job["quiesce_gate_id"]
        receipt = self._receipt(job)
        if not extend_gate(conn, gate_id, owner_id=job["id"], ttl_seconds=self.gate_ttl_seconds):
            if job["past_deadline"]:
                return self._fail(conn, job, "quiesce_lost: the hold lapsed and the deadline has passed")
            activated = self.gates.activate(conn, self._spec(job, snapshot), gate_id)
            gate_id = activated.gate.id
            receipt["generations"].append(gate_id)
            self._write(conn, job["id"], {"quiesce_gate_id": gate_id, "quiesce_json": json.dumps(receipt)})
        statuses = list(self.gates.drain_status(conn, gate_id))
        receipt["roster"] = [_status_row(s) for s in statuses]
        if all(s.drained for s in statuses):
            return self._dispatch(conn, job, snapshot, gate_id, receipt)
        if job["past_deadline"]:
            return self._fail(conn, job, "quiesce_timeout: held turns did not drain before the deadline",
                              receipt=receipt)
        self._write(conn, job["id"], {"quiesce_json": json.dumps(receipt)})
        return StepResult("holding")

    def _dispatch(self, conn, job, snapshot, gate_id: int, receipt) -> StepResult:
        execution = snapshot["execution"]
        timeout = int(execution["timeout_seconds"])
        cover = int(execution["start_delay_seconds"]) + timeout + 2 * self.margin_seconds
        row = conn.execute(text(
            "UPDATE system_gates SET expires_at=GREATEST(expires_at, now() + (:cover * interval '1 second'))"
            f" WHERE id=:id AND owner_id=:owner AND {_LIVE}"
            "   AND expires_at > now() + (:margin * interval '1 second')"
            " RETURNING expires_at - ((:timeout + :margin) * interval '1 second') AS start_not_after"
        ), {"cover": float(cover), "id": gate_id, "owner": job["id"],
            "margin": float(self.margin_seconds), "timeout": float(timeout)}).first()
        if row is None:
            # Renewed above, so only a concurrent release lands here; retry next step.
            self._write(conn, job["id"], {"quiesce_json": json.dumps(receipt)})
            return StepResult("cas_retry")
        fence = {"generation": gate_id, "start_not_after": row[0].isoformat()}
        receipt["outcome"] = "drained"
        self._write(conn, job["id"], {
            "state": JOB_DISPATCHING, "host_unit_name": f"llm-bawt-ops-{job['id']}",
            "quiesce_fence_json": json.dumps(fence, sort_keys=True),
            "quiesce_json": json.dumps(receipt), "error_text": None,
        })
        conn.execute(text("UPDATE ops_jobs SET dispatched_at=now() WHERE id=:id"), {"id": job["id"]})
        return StepResult("dispatched", dispatch=True)

    def _fail(self, conn, job, error: str, *, receipt=None) -> StepResult:
        receipt = receipt or self._receipt(job)
        receipt["outcome"] = error.split(":", 1)[0] if ":" in error else "failed"
        self._write(conn, job["id"], {"state": JOB_FAILED, "error_text": error,
                                      "quiesce_json": json.dumps(receipt)})
        conn.execute(text("UPDATE ops_jobs SET finished_at=now() WHERE id=:id"), {"id": job["id"]})
        release_job_gates(conn, job["id"])
        return StepResult("failed")

    # -- reconciler sweep -------------------------------------------------

    def release_terminal(self) -> list[str]:
        """Release every unreleased gate owned by a terminal job."""
        with self.engine.begin() as conn:
            owners = [row[0] for row in conn.execute(text(
                "SELECT DISTINCT g.owner_id FROM system_gates g JOIN ops_jobs j ON j.id=g.owner_id"
                " WHERE g.owner_kind='job' AND g.released_at IS NULL AND j.state = ANY(:terminal)"
                " ORDER BY g.owner_id"
            ), {"terminal": list(JOB_TERMINAL_STATES)}).all()]
            for owner in owners:
                release_job_gates(conn, owner)
        return owners


def build_quiesce_driver(engine, *, factory: Callable[..., QuiesceDriver] = QuiesceDriver):
    """The reconciler's driver, or ``None`` off PostgreSQL (quiesce jobs then
    stay visibly QUIESCING until a PostgreSQL-backed reconciler runs)."""
    try:
        return factory(engine)
    except QuiesceUnavailable:
        logger.warning("quiesce driver unavailable: the ops store is not PostgreSQL")
        return None


__all__ = [
    "ACK_STATUSES", "GATE_TTL_SECONDS", "LEASE_SECONDS", "MARGIN_SECONDS", "QuiesceDriver",
    "QuiesceUnavailable", "StepResult", "TurnGates", "build_quiesce_driver",
    "caller_instruction", "cancel_pending_controls", "record_ack", "release_job_gates",
    "scope_covers_bot",
]
