"""The worker slot bound on process-mode children (TASK-1169 S4b).

A process-mode job runs as a child of the ops service, so live children share
its memory and file descriptors. ``LLM_BAWT_OPS_WORKER_CONCURRENCY`` (default
4) bounds them across every dispatcher. It is a DB invariant, checked in the
same transaction as each transition that commits a job to execution:

- :meth:`.store.OpsStore.claim_job` (QUEUED -> DISPATCHING);
- the quiesce driver's open step, before any hold opens (``waiting_slot``).

A job that takes a slot records ``worker_slot_at``; it holds the slot while it
is QUIESCING, DISPATCHING, ACCEPTED or RUNNING, and frees it by going terminal.
A respawn (S4d) is the same job and takes no new slot. Isolated jobs are not in
the ops service's process tree and are never counted; per-operation
``max_concurrent`` applies in both modes, unchanged.

On PostgreSQL every slot decision takes one transaction-scoped advisory lock,
after the operation-row lock (the order every slot claim uses), so two
dispatchers can never both see the last free slot. Other engines are
single-writer (SQLite's write lock is already held by the operation-row
update) and need no extra lock.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Mapping

from sqlalchemy import text

from .execution_mode import EXECUTION_PROCESS, snapshot_execution_mode
from .models import (EXECUTOR_DOCKER, JOB_ACCEPTED, JOB_DISPATCHING, JOB_QUIESCING, JOB_RUNNING,
                     JOB_TERMINAL_STATES)

logger = logging.getLogger(__name__)

WORKER_CONCURRENCY_ENV = "LLM_BAWT_OPS_WORKER_CONCURRENCY"
DEFAULT_WORKER_CONCURRENCY = 4
WORKER_SLOT_LOCK = "ops-worker-slots"
# The executor kinds whose process mode starts a child of the ops service.
# Release orchestration never starts a worker (its mode is inert).
CHILD_PROCESS_EXECUTORS = frozenset({EXECUTOR_DOCKER})
WORKER_SLOT_STATES = (JOB_QUIESCING, JOB_DISPATCHING, JOB_ACCEPTED, JOB_RUNNING)
# TASK-1865: a slot-reserved gate keeps counting in ANY state until a durable
# positive retirement stamp. Ordinary terminal non-gate rows stay excluded.
UNRETIRED_GATE_SLOT = ("(host_unit_name LIKE 'gate-custody:%' AND worker_slot_at IS NOT NULL"
                       " AND gate_retired_at IS NULL)")
WAITING_WORKER_SLOT = "waiting for a process worker slot"


def worker_concurrency() -> int:
    """The configured bound; an unset, invalid or non-positive value is the default."""
    raw = os.environ.get(WORKER_CONCURRENCY_ENV, "").strip()
    if not raw:
        return DEFAULT_WORKER_CONCURRENCY
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        logger.warning("%s=%r is not a positive integer; using %d",
                       WORKER_CONCURRENCY_ENV, raw, DEFAULT_WORKER_CONCURRENCY)
        return DEFAULT_WORKER_CONCURRENCY
    return value


def needs_worker_slot(snapshot: Mapping[str, Any]) -> bool:
    """Does this approved snapshot run as a child process of the ops service?"""
    kind = (snapshot.get("execution") or {}).get("executor_kind")
    if kind not in CHILD_PROCESS_EXECUTORS:
        return False
    return snapshot_execution_mode(snapshot, kind) == EXECUTION_PROCESS


def _lock_worker_slots(conn) -> None:
    if conn.dialect.name == "postgresql":
        conn.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:k, 0))"), {"k": WORKER_SLOT_LOCK})


def process_spawn_held(conn) -> bool:
    """Plain ledger read: no advisory lock, usable by a spawn-side flock holder.

    Legacy dormant holds end on terminal state. A prepared recovery-policy
    hold stays closed across terminalization until explicitly verified released.
    Unknown nonterminal states fail closed; this read never takes a DB lock.
    """
    terminal = sorted(JOB_TERMINAL_STATES)
    states = ", ".join(f":t{i}" for i in range(len(terminal)))
    return conn.execute(text(
        "SELECT 1 FROM ops_jobs WHERE process_spawn_hold_at IS NOT NULL"
        f" AND (state NOT IN ({states}) OR"
        " (process_spawn_hold_target_json IS NOT NULL AND process_spawn_hold_release_json IS NULL)) LIMIT 1"
    ), {f"t{i}": state for i, state in enumerate(terminal)}).first() is not None


def process_spawn_hold_available(conn, *, child_count: int | None) -> bool:
    """Check an isolated claim's drain evidence under the worker-slot lock.

    The future self-config caller must supply a fresh child count observed
    under its exclusive receipt-volume flock, retained through this claim.
    No Docker I/O belongs in this transaction. Unknown counts never permit a
    hold; this ledger prerequisite alone is not complete spawn/drain safety.
    """
    if type(child_count) is not int or child_count != 0:
        return False
    active_slots = live_worker_slots(conn)
    return active_slots == 0 and not process_spawn_held(conn)


def live_worker_slots(conn, *, exclude_job_id: str | None = None) -> int:
    """How many jobs hold a worker slot now, under the slot lock on PostgreSQL."""
    _lock_worker_slots(conn)
    states = ", ".join(f":s{i}" for i in range(len(WORKER_SLOT_STATES)))
    params: dict[str, Any] = {f"s{i}": state for i, state in enumerate(WORKER_SLOT_STATES)}
    params["exclude"] = exclude_job_id or ""
    return int(conn.execute(text(
        f"SELECT count(*) FROM ops_jobs WHERE worker_slot_at IS NOT NULL"
        f" AND (state IN ({states}) OR {UNRETIRED_GATE_SLOT}) AND id <> :exclude"), params).scalar_one())


def worker_slot_free(conn, snapshot: Mapping[str, Any], *, job_id: str) -> bool | None:
    """None when the job needs no worker slot; else whether one is free for it.

    A job that already holds a slot (a quiesce job past its open step) keeps
    it unless a spawn hold is present (an inconsistent ledger fails closed).
    Call inside the transition's own transaction, after the operation-row
    lock, and set ``worker_slot_at`` in that same transaction.
    """
    if not needs_worker_slot(snapshot):
        return None
    _lock_worker_slots(conn)
    if process_spawn_held(conn):
        return False
    held = conn.execute(text("SELECT worker_slot_at IS NOT NULL FROM ops_jobs WHERE id=:id"),
                        {"id": job_id}).scalar()
    if held:
        return True
    return live_worker_slots(conn, exclude_job_id=job_id) < worker_concurrency()
