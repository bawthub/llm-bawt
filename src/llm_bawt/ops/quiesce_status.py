"""Read-only operator views of quiesce holds (TASK-1233 (f)).

* :func:`attach_members` adds ``quiesce.members`` to job API dicts: one row
  per turn enrolled in the job's CURRENT gate generation, with its
  ``control_state``/``ack_status`` and the driver's last drain verdict for
  that turn from ``quiesce_json.roster``. One batched query serves a whole
  page of jobs. A member linked to a control-lane delivery (TASK-1431 S3)
  reports that delivery's state, attempts and error (``_CONTROL_SQL``).
* :func:`live_holds` lists every live gate (job-owned and operator-owned)
  with member counts; ``job`` is null for operator gates.

Both read PostgreSQL only; on any other engine they return nothing.
"""
from __future__ import annotations

import json
from typing import Any

from sqlalchemy import bindparam, text

from ..service.turn_gates import ensure_gate_schema

# Design §5: a ``steering`` member reports its linked delivery's state. A
# DELIVERED row means the run was asked, not that it stopped: the drain never
# reads this. Every other stored state is reported as it is.
_CONTROL_SQL = """
    CASE WHEN m.control_state <> 'steering' OR d.id IS NULL THEN m.control_state
         WHEN d.status IN ('QUEUED', 'STEERING') THEN 'steering'
         WHEN d.status = 'DELIVERED' THEN 'steered'
         WHEN d.status = 'CANCELLED' AND d.last_error LIKE 'not\\_needed:%' THEN 'not_needed'
         WHEN d.status = 'CANCELLED' THEN 'cancelled'
         WHEN d.status = 'FAILED' THEN 'unconfirmed'
         ELSE m.control_state END
"""

_MEMBER_SQL = text(f"""
    SELECT m.gate_id, m.bot_id, m.turn_id, m.backend, m.disposition,
           {_CONTROL_SQL} AS control_state,
           COALESCE(d.attempt_count, 0) AS control_attempts,
           CASE WHEN d.id IS NULL THEN m.control_error ELSE d.last_error END AS control_error,
           m.ack_status, m.ack_note, m.acked_at, m.created_at AS enrolled_at,
           COALESCE(m.control_changed_at, m.created_at) AS control_changed_at
    FROM gate_members m LEFT JOIN inter_bot_deliveries d ON d.id = m.control_delivery_id
    WHERE m.gate_id IN :gate_ids ORDER BY m.gate_id, m.id
""").bindparams(bindparam("gate_ids", expanding=True))

# Same predicate as turn_gates._LIVE: a gate holds until released or expired.
_HOLDS_SQL = text("""
    SELECT g.id, g.owner_kind, g.owner_id, g.mode, g.scope, g.reason, g.allow_actors,
           g.created_at, g.expires_at,
           j.id AS job_id, j.operation_slug, j.state AS job_state, j.quiesce_json
    FROM system_gates g
    LEFT JOIN ops_jobs j ON g.owner_kind = 'job' AND j.id = g.owner_id
    WHERE g.released_at IS NULL AND g.expires_at > now()
    ORDER BY g.id
""")

_COUNTS_SQL = text(f"""
    SELECT m.gate_id, m.disposition, {_CONTROL_SQL} AS control_state, count(*) AS n
    FROM gate_members m LEFT JOIN inter_bot_deliveries d ON d.id = m.control_delivery_id
    WHERE m.gate_id IN :gate_ids
    GROUP BY 1, 2, 3
""").bindparams(bindparam("gate_ids", expanding=True))


def _is_postgres(engine) -> bool:
    return getattr(getattr(engine, "dialect", None), "name", None) == "postgresql"


def _iso(value) -> str | None:
    return value.isoformat() if value is not None else None


def _member_api(row, drain: dict[str, Any]) -> dict[str, Any]:
    verdict = drain.get(row["turn_id"]) or {}
    return {
        "bot_id": row["bot_id"], "turn_id": row["turn_id"], "backend": row["backend"],
        "disposition": row["disposition"], "control_state": row["control_state"],
        "control_attempts": row["control_attempts"], "control_error": row["control_error"],
        "control_changed_at": _iso(row["control_changed_at"]),
        "ack_status": row["ack_status"], "ack_note": row["ack_note"],
        "acked_at": _iso(row["acked_at"]), "enrolled_at": _iso(row["enrolled_at"]),
        "drained": verdict.get("drained"), "drain_reason": verdict.get("reason"),
    }


def attach_members(engine, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Set ``quiesce.members`` on each job API dict with a quiesce block, in place.

    A job with no current gate gets ``[]``. ``drained`` is null for a turn the
    driver has not yet judged (enrolled after its last step)."""
    blocks = [job["quiesce"] for job in jobs if isinstance(job.get("quiesce"), dict)]
    for block in blocks:
        block["members"] = []
    gate_ids = sorted({block["gate_id"] for block in blocks if block.get("gate_id") is not None})
    if not gate_ids or not _is_postgres(engine):
        return jobs
    ensure_gate_schema(engine)
    with engine.connect() as conn:
        rows = conn.execute(_MEMBER_SQL, {"gate_ids": gate_ids}).mappings().all()
    by_gate: dict[int, list] = {}
    for row in rows:
        by_gate.setdefault(row["gate_id"], []).append(row)
    for block in blocks:
        if block.get("gate_id") is None:
            continue
        receipt = block.get("receipt") if isinstance(block.get("receipt"), dict) else {}
        drain = {m.get("turn_id"): m for m in receipt.get("roster") or [] if isinstance(m, dict)}
        block["members"] = [_member_api(row, drain) for row in by_gate.get(block["gate_id"], [])]
    return jobs


def _json(value):
    return json.loads(value) if isinstance(value, str) else value


def _job_phase(raw: str | None) -> str | None:
    try:
        phase = (json.loads(raw) if raw else {}).get("phase")
    except (json.JSONDecodeError, TypeError, AttributeError):
        return None
    return phase.get("name") if isinstance(phase, dict) else None


def live_holds(engine) -> list[dict[str, Any]]:
    """Every live gate with member counts. ``job`` is null for operator gates."""
    if not _is_postgres(engine):
        return []
    ensure_gate_schema(engine)
    with engine.connect() as conn:
        gates = conn.execute(_HOLDS_SQL).mappings().all()
        ids = [gate["id"] for gate in gates]
        counts = conn.execute(_COUNTS_SQL, {"gate_ids": ids}).mappings().all() if ids else []
    members: dict[int, dict[str, Any]] = {
        gate_id: {"total": 0, "by_disposition": {}, "by_control_state": {}} for gate_id in ids}
    for row in counts:
        tally = members[row["gate_id"]]
        tally["total"] += row["n"]
        for key, value in (("by_disposition", row["disposition"]), ("by_control_state", row["control_state"])):
            tally[key][value] = tally[key].get(value, 0) + row["n"]
    return [{
        "gate_id": gate["id"], "owner_kind": gate["owner_kind"], "owner_id": gate["owner_id"],
        "mode": gate["mode"], "scope": _json(gate["scope"]), "reason": gate["reason"],
        "allow_actors": _json(gate["allow_actors"]), "created_at": _iso(gate["created_at"]),
        "expires_at": _iso(gate["expires_at"]), "members": members[gate["id"]],
        "job": None if gate["job_id"] is None else {
            "id": gate["job_id"], "operation": gate["operation_slug"], "state": gate["job_state"],
            "phase": _job_phase(gate["quiesce_json"])},
    } for gate in gates]


__all__ = ["attach_members", "live_holds"]
