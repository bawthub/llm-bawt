"""Indexed strict-plan history for the dormant first-attempt gate claim.

The key is derived by trusted store code, never an admission argument or grant.
Existing rows are decoded once, in bounded pages, in the normal additive schema
transaction. A claim must not scan/decode the general ops ledger.
"""
from __future__ import annotations

import json

from sqlalchemy import inspect, text

from .gate_plan import GATE_PLAN_SCHEMA, GatePlan

GATE_HISTORY_INDEX = "ix_ops_jobs_gate_attempt"
GATE_HISTORY_PREDICATE = (
    "gate_plan_key IS NOT NULL AND (state <> 'queued' OR dispatched_at IS NOT NULL)"
)
_BACKFILL_PAGE_SIZE = 128


def plan_key(plan: GatePlan) -> str:
    """Collision-free bounded identity of the closed v1 workload, not JSON bytes."""
    return f"{GATE_PLAN_SCHEMA}:{plan.repository}:{plan.source_sha}:{plan.target}"


def gate_plan_key(args_json: str) -> str | None:
    """Keep ordinary/malformed args out of gate history; accept legacy JSON layout."""
    try:
        return plan_key(GatePlan.from_dict(json.loads(args_json)))
    except (ValueError, TypeError):
        return None


def migrate_gate_history(conn) -> None:
    """One additive migration, never a per-claim or repeated startup backfill.

    PostgreSQL's ADD COLUMN holds the table lock through commit, so old writes
    cannot interleave with the backfill. A failure rolls back column and keys
    together. Existing args, states, snapshots and events are not rewritten.
    Future gate execution still requires separately authorized source/worker
    activation; this does not create or enable an operation.
    """
    if "gate_plan_key" not in {c["name"] for c in inspect(conn).get_columns("ops_jobs")}:
        conn.execute(text("ALTER TABLE ops_jobs ADD COLUMN gate_plan_key VARCHAR(128)"))
        after = ""
        while True:
            rows = conn.execute(text(
                "SELECT id, args_json FROM ops_jobs WHERE id > :after ORDER BY id LIMIT :limit"
            ), {"after": after, "limit": _BACKFILL_PAGE_SIZE}).all()
            if not rows:
                break
            keys = [{"id": row.id, "key": key} for row in rows
                    if (key := gate_plan_key(row.args_json)) is not None]
            if keys:
                conn.execute(text("UPDATE ops_jobs SET gate_plan_key=:key WHERE id=:id"), keys)
            after = rows[-1].id
        # Fresh databases get the index from create_all. Existing databases
        # create it only with this column migration: even IF NOT EXISTS takes
        # SHARE on PostgreSQL and would queue writers on ordinary restarts.
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS {GATE_HISTORY_INDEX} ON ops_jobs (gate_plan_key) "
                          f"WHERE {GATE_HISTORY_PREDICATE}"))


def has_prior_attempt(conn, *, job_id: str, plan: GatePlan) -> bool:
    """One indexed existence lookup, under the caller's repo+SHA advisory lock.

    Preserve the original predicate, including never-dispatched cancellations,
    LOST/timeouts and a dispatched row whose state was changed back to queued.
    No terminal state is positive retirement or permission for a second attempt.
    """
    return conn.execute(text(
        "SELECT 1 FROM ops_jobs WHERE gate_plan_key=:key AND id<>:id AND "
        f"{GATE_HISTORY_PREDICATE} LIMIT 1"
    ), {"key": plan_key(plan), "id": job_id}).first() is not None
