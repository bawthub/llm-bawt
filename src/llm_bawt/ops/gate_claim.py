"""Dormant exact-plan authorization at the existing ops ledger's claim boundary.

No caller, scheduler, executor, catalog seed or callback attacher uses this yet.
An internal adapter must obtain the opaque turn capability from trusted request
context, not model/body arguments. A snapshot is integrity data, not approval
or permission to execute. Only a current DB-owned tag authorizes this claim.

This slice admits FIRST attempts only. A prior nonqueued job for the same
repo+SHA blocks forever, even LOST/timed_out/success: there is no retirement
fence implementation or attempt-2 escape hatch. Future retry/publication code
must positively retire runner/sidecar and fence the active attempt; callbacks
and terminal ledger states are not that proof.
"""
from __future__ import annotations

import hashlib
import json

from sqlalchemy import text

from ..bot_tags.catalog import lock_bot
from ..bot_tags.vocabulary import BehaviorTag, TAG_SLUG
from ..task_turn_context import TaskTurnContext, TaskTurnContextError, open_task_turn_context
from .events import append_job_transition_by_id
from .gate_plan import GatePlan
from .gate_history import has_prior_attempt, plan_key
from .models import JOB_DISPATCHING, JOB_QUEUED, OpsOperation
from .store import slot_cap_reached, slot_holders
from .validation import canonical_json

GATE_CLAIM_SCHEMA = "llm-bawt.ops.gate-claim/v1"


def _caller(context: TaskTurnContext) -> dict:
    return {key: getattr(context, key) for key in ("bot_id", "user_id", "turn_id", "session_id")}


def _snapshot(operation: OpsOperation, plan: GatePlan, context: TaskTurnContext) -> dict:
    snapshot = {"gate_claim_schema": GATE_CLAIM_SCHEMA, "operation": operation.to_api(),
                "gate_plan": plan.to_dict(), "caller": _caller(context),
                "execution": {"max_concurrent": operation.max_concurrent}}
    snapshot["snapshot_hash"] = hashlib.sha256(canonical_json(snapshot).encode()).hexdigest()
    return snapshot


def gate_claim_snapshot(operation: OpsOperation, plan: GatePlan, *, turn_token: str) -> dict:
    """Internal detached snapshot only; does not create or authorize any job.

    Includes the whole operation (execution_tag included) in the content hash.
    The opaque capability is verified, never stored in the snapshot/ledger.
    """
    return _snapshot(operation, GatePlan.from_dict(plan.to_dict()), open_task_turn_context(turn_token))


def claim_gate_job(store, job_id: str, *, turn_token: str) -> bool:
    """Atomic gate-only QUEUED -> DISPATCHING, no execution or runner slot.

    Fail closed off PostgreSQL, for untrusted identity, or any changed input.
    Normal nongate claims are untouched. The returned bool is not SHIP proof.
    """
    store._require()
    if store.engine.dialect.name != "postgresql":
        return False
    try:
        caller = open_task_turn_context(turn_token)
    except TaskTurnContextError:
        return False
    with store.engine.begin() as conn:
        return _claim(conn, job_id, caller, events=getattr(store, "events", None))


def _claim(conn, job_id: str, caller: TaskTurnContext, *, events=None) -> bool:
    """One transaction; split out so PG tests can hold the real winning claim.

    Lock order: operation -> job -> bot-tags advisory -> profile -> tag catalog
    -> assignment -> repo+SHA advisory. Existing tag writers use lock_bot;
    row locks also serialize profile deletion and direct assignment deletion.
    All inputs are read again after the locks; no pre-lock grant cache is used.
    """
    locked = _lock_claim_rows(conn, job_id)
    if locked is None:
        return False
    operation, job = locked
    if ((job["caller_bot_id"], job["caller_user_id"], job["caller_turn_id"])
            != (caller.bot_id, caller.user_id, caller.turn_id)):
        return False
    try:
        plan = GatePlan.from_dict(json.loads(job["args_json"]))
        snapshot = json.loads(job["invocation_snapshot_json"])
    except (ValueError, TypeError):
        return False
    if (canonical_json(snapshot) != canonical_json(_snapshot(operation, plan, caller))
            or job["gate_plan_key"] != plan_key(plan)
            or job["operation_slug"] != operation.slug
            or job["operation_version"] != operation.version
            or job["operation_script_hash"] != operation.script_hash
            or operation.script_hash != hashlib.sha256(operation.command_script.encode()).hexdigest()):
        return False
    if not _current_grant(conn, bot_id=caller.bot_id, tag=operation.execution_tag):
        return False
    return _claim_first_attempt(conn, job_id, operation, plan, events=events)


def _lock_claim_rows(conn, job_id: str):
    """Common operation -> job locks, with the same QUEUED-only fence."""
    job = conn.execute(text("SELECT operation_slug FROM ops_jobs WHERE id=:id"),
                       {"id": job_id}).mappings().first()
    if job is None:
        return None
    op_row = conn.execute(text("SELECT * FROM ops_operations WHERE slug=:slug FOR UPDATE"),
                          {"slug": job["operation_slug"]}).mappings().first()
    if op_row is None:
        return None
    operation = OpsOperation(**op_row)
    job = conn.execute(text("SELECT * FROM ops_jobs WHERE id=:id FOR UPDATE"),
                       {"id": job_id}).mappings().first()
    tag = operation.execution_tag
    if (job is None or job["state"] != JOB_QUEUED or job["dispatched_at"] is not None
            or not operation.enabled or operation.soft_deleted_at is not None
            or operation.quiesce_enabled or not isinstance(tag, str) or not TAG_SLUG.fullmatch(tag)):
        return None
    return operation, job


def _current_grant(conn, *, bot_id: str, tag: str) -> bool:
    """Lock current caller/profile/catalog/assignment, never trust historic tags."""
    lock_bot(conn, bot_id)
    profile = conn.execute(text("SELECT slug FROM bot_profiles WHERE slug=:bot FOR SHARE"),
                           {"bot": bot_id}).first()
    if profile is None:
        return False
    # Catalog rename/delete also changes assignments without lock_bot. Lock the
    # catalog BEFORE its assignment, matching that writer's order.
    known = conn.execute(text("SELECT slug FROM bot_tag_catalog WHERE slug=:tag FOR SHARE"),
                         {"tag": tag}).first()
    if known is None:
        return False
    assignment = conn.execute(text("SELECT tag_slug FROM bot_profile_tags "
                                   "WHERE bot_slug=:bot AND tag_slug=:tag FOR SHARE"),
                              {"bot": bot_id, "tag": tag}).first()
    restricted = conn.execute(text("SELECT 1 FROM bot_profile_tags "
                                   "WHERE bot_slug=:bot AND tag_slug IN (:private, :inactive)"),
                              {"bot": bot_id, "private": str(BehaviorTag.PRIVATE),
                               "inactive": str(BehaviorTag.INACTIVE)}).first()
    return assignment is not None and restricted is None


def _claim_first_attempt(conn, job_id: str, operation: OpsOperation, plan: GatePlan, *, events=None) -> bool:
    """Same ledger transition/fence for both dormant gate-only claim adapters."""
    # All claimers for this immutable workload serialize, even across operation
    # rows. No terminal status is a runner-retirement fence in this slice.
    conn.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
                 {"key": f"ops-gate:{plan.repository}:{plan.source_sha}"})
    if has_prior_attempt(conn, job_id=job_id, plan=plan):
        return False
    if slot_cap_reached(slot_holders(conn, operation.slug), operation.max_concurrent):
        return False
    claimed = conn.execute(text("UPDATE ops_jobs SET state=:dispatching, dispatched_at=now(), "
                                "error_text=NULL WHERE id=:id AND state=:queued "
                                "AND dispatched_at IS NULL"),
                           {"id": job_id, "queued": JOB_QUEUED, "dispatching": JOB_DISPATCHING})
    if claimed.rowcount == 1:
        append_job_transition_by_id(events, conn, job_id)
        return True
    return False
