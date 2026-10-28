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
from .models import EXECUTOR_DOCKER, JOB_ACCEPTED, JOB_DISPATCHING, JOB_QUIESCING, JOB_RUNNING

logger = logging.getLogger(__name__)

WORKER_CONCURRENCY_ENV = "LLM_BAWT_OPS_WORKER_CONCURRENCY"
DEFAULT_WORKER_CONCURRENCY = 4
WORKER_SLOT_LOCK = "ops-worker-slots"
# The executor kinds whose process mode starts a child of the ops service.
# Release orchestration never starts a worker (its mode is inert).
CHILD_PROCESS_EXECUTORS = frozenset({EXECUTOR_DOCKER})
WORKER_SLOT_STATES = (JOB_QUIESCING, JOB_DISPATCHING, JOB_ACCEPTED, JOB_RUNNING)
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


def live_worker_slots(conn, *, exclude_job_id: str | None = None) -> int:
    """How many jobs hold a worker slot now, under the slot lock on PostgreSQL."""
    if conn.dialect.name == "postgresql":
        conn.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:k, 0))"), {"k": WORKER_SLOT_LOCK})
    states = ", ".join(f":s{i}" for i in range(len(WORKER_SLOT_STATES)))
    params: dict[str, Any] = {f"s{i}": state for i, state in enumerate(WORKER_SLOT_STATES)}
    params["exclude"] = exclude_job_id or ""
    return int(conn.execute(text(
        f"SELECT count(*) FROM ops_jobs WHERE worker_slot_at IS NOT NULL AND state IN ({states})"
        " AND id <> :exclude"), params).scalar_one())


def worker_slot_free(conn, snapshot: Mapping[str, Any], *, job_id: str) -> bool | None:
    """None when the job needs no worker slot; else whether one is free for it.

    A job that already holds a slot (a quiesce job past its open step) always
    keeps it. Call inside the transition's own transaction, after the
    operation-row lock, and set ``worker_slot_at`` in that same transaction.
    """
    if not needs_worker_slot(snapshot):
        return None
    held = conn.execute(text("SELECT worker_slot_at IS NOT NULL FROM ops_jobs WHERE id=:id"),
                        {"id": job_id}).scalar()
    if held:
        return True
    return live_worker_slots(conn, exclude_job_id=job_id) < worker_concurrency()
