"""Unused trusted-request claim; fresh DB authority, never historic permission.

No service/store API, scheduler, route, executor or runner calls this adapter.
It only commits QUEUED -> DISPATCHING in the existing ledger, not an execution
or publication permit. Runtime qualification and dispatch/recovery wiring remain
absent; ordinary and legacy gate claims still refuse enriched snapshots.
"""
from __future__ import annotations

import hashlib

from . import gate_trust as trust
from .gate_claim import _claim_first_attempt, _current_grant, _lock_claim_rows, _snapshot
from .models import OpsJob
from .validation import canonical_json


def claim_trusted_gate_job(store, job_id: str) -> bool:
    """Claim one authenticated stored request under live operation/grant/config locks.

    The durable server MAC supplies caller identity, never caller/body overrides
    or an expired turn token. Permission is re-established from current DB rows.
    PostgreSQL only; no worker slot, launch, callback, retry or terminal receipt.
    """
    store._require()
    if store.engine.dialect.name != "postgresql":
        return False
    with store.engine.begin() as conn:
        return _claim_trusted(conn, job_id, events=getattr(store, "events", None))


def _claim_trusted(conn, job_id: str, *, events=None) -> bool:
    """Same transaction for proof, live revalidation and the existing transition.

    Lock order extends the legacy claim: operation -> job -> bot advisory ->
    profile -> tag catalog -> assignment -> exact-source config -> repo+SHA.
    Locks remain held through commit/rollback. Config edits/revocations committed
    before the winning reads deny; later edits wait and are prospective only.
    No terminal state is positive runner retirement: still FIRST attempts only.
    """
    locked = _lock_claim_rows(conn, job_id)
    if locked is None:
        return False
    operation, row = locked
    try:
        request = trust.read_trusted_gate_request(OpsJob(**row))
        snapshot = trust._json(row["invocation_snapshot_json"], trust._MAX_SNAPSHOT)
    except (ValueError, TypeError):
        return False

    # The MAC authenticates the historic snapshot, not the operation today.
    # Canonical equality (not Python bool/int equality) binds its whole original
    # operation, caller, plan and execution limits to the freshly locked catalog.
    expected = _snapshot(operation, request.plan, request.caller)
    expected.pop("snapshot_hash")
    if (canonical_json({key: snapshot[key] for key in expected}) != canonical_json(expected)
            or operation.script_hash != hashlib.sha256(operation.command_script.encode()).hexdigest()):
        return False
    if not _current_grant(conn, bot_id=request.caller.bot_id, tag=operation.execution_tag):
        return False
    try:
        configured = trust._configured_environment(conn, request.plan, lock=True)
        environment, digest = trust._environment(configured, request.plan)
    except (ValueError, TypeError):
        return False
    if (environment, digest) != (request.environment_json, request.environment_sha256):
        return False
    return _claim_first_attempt(conn, job_id, operation, request.plan, events=events)
