"""Operator purge of every in-flight ops action and its callback (TASK-1232 B1).

Reached only through an explicit ``RecoverySweep.run(purge_ops=True)``: never
at startup, never periodically, never over MCP. No HTTP route calls it yet;
the operator route lands with the admin sweep endpoint.

Correctness depends on every other writer of these rows being a compare-and-set
on the current row (``approval_mcp_store``): ``enqueue_continuation``,
``complete_mcp_execution``, ``claim_continuation`` and ``execution_fence`` refuse
a cancelled request, and the dispatching→pending writers suppress it. So a
writer that read the row before the purge committed cannot undo it.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from sqlalchemy import and_, or_
from sqlmodel import Session, select

from ...agent_backends.agent_bridge import get_agent_subscriber
from ...approval_models import (
    CONT_DELIVERED,
    CONT_DISPATCHING,
    CONT_NOT_NEEDED,
    CONT_PENDING,
    CONT_SUPPRESSED,
    EXEC_SKIPPED,
    KIND_MCP,
    OPS_RUN_TOOL,
    REQ_APPROVED,
    REQ_CANCELLED,
    REQ_PENDING,
    ToolApprovalRequest,
    _utcnow,
    awaits_ops_admission,
)
from ...ops.models import JOB_CANCELLED, JOB_TERMINAL_STATES, OpsJob
from ..approval_events import publish_approval_resolved
from ..tool_call_store import ToolCallStore
from .audit import RunStamp

log = logging.getLogger(__name__)

PURGED_BY_OPERATOR = "purged by operator"
# The stored result of a purged request that had none: the same terminal shape
# a user cancel writes (routes/approval_policies.py), so the card resolves.
PURGED_RESULT = {"status": "cancelled", "message": PURGED_BY_OPERATOR}


def in_flight_ops_run():
    """The ``ops_run`` approvals a purge wipes.

    An undecided card, or an approved one whose work or callback isn't done:
    no result yet; a result whose continuation is not enqueued yet (what
    ``find_stranded_mcp_results`` re-enqueues); or a continuation still
    queued or dispatching.
    """
    return and_(
        ToolApprovalRequest.request_kind == KIND_MCP,
        ToolApprovalRequest.tool_name == OPS_RUN_TOOL,
        or_(
            ToolApprovalRequest.status == REQ_PENDING,
            and_(
                ToolApprovalRequest.status == REQ_APPROVED,
                or_(
                    ToolApprovalRequest.result_json.is_(None),
                    and_(ToolApprovalRequest.continuation_capable.is_(True),
                         ToolApprovalRequest.continuation_state == CONT_NOT_NEEDED),
                    ToolApprovalRequest.continuation_state.in_([CONT_PENDING, CONT_DISPATCHING]),
                ),
            ),
        ),
    )


def purge_ops(engine, *, stamp: RunStamp, dry_run: bool = False) -> dict[str, Any]:
    """Cancel in-flight ``ops_run`` approvals and every non-terminal ops job.

    One transaction, in the lock order the job-creating path also takes
    (approval row through ``execution_fence``, then its job):
    1. lock the in-flight approval rows;
    2. cancel them (an undecided card is cancelled, not denied: a deny writes
       a receipt turn), suppress the continuation and revoke an unfinished
       execution claim. A ``dispatching`` or ``delivered`` continuation can't
       be recalled, so it is reported as ``unrecallable``; its
       dispatching→pending writers suppress it when the claimant returns;
    3. cancel every non-terminal ops job.

    A request with no result gets the cancelled result a user cancel writes
    (Vex C1), and after commit its tool card is resolved, best-effort, so the
    card no longer renders as actionable or awaiting a result. Every row
    written carries ``stamp`` (``recovery.audit``). The caller publishes each
    cancelled request's ``approval_resolved`` (``publish_purge_resolutions``).

    A worker Docker already accepted or started can't be un-run: the purge
    stops tracking it and suppresses its callback. ``dry_run`` reports the
    same rows and changes nothing.
    """
    report: dict[str, Any] = {"dry_run": dry_run, **stamp.as_dict(),
                              "approvals": [], "unrecallable": [], "jobs": []}
    now = _utcnow()
    reason = stamp.text(PURGED_BY_OPERATOR)
    cards: list[ToolApprovalRequest] = []
    with Session(engine) as session:
        approvals = select(ToolApprovalRequest).where(in_flight_ops_run()).order_by(ToolApprovalRequest.id)
        for row in session.exec(approvals if dry_run else approvals.with_for_update()).all():
            recallable = row.continuation_state not in (CONT_DISPATCHING, CONT_DELIVERED)
            report["approvals" if recallable else "unrecallable"].append({
                "id": row.id, "status": row.status, "bot_id": row.bot_id,
                "execution_state": row.execution_state, "continuation_state": row.continuation_state,
            })
            if dry_run:
                continue
            if row.status == REQ_PENDING:
                row.resolved_at, row.resolved_by = now, stamp.actor
                row.resolution_message = reason
            row.status = REQ_CANCELLED
            row.continuation_last_error = reason
            if recallable:
                row.continuation_state = CONT_SUPPRESSED
                row.continuation_claim_token = None
                row.continuation_next_attempt_at = None
            if row.result_json is None:
                row.execution_state, row.execution_claim_token = EXEC_SKIPPED, None
                row.execution_error = reason
                row.result_json, row.result_is_error = json.dumps(PURGED_RESULT), False
                cards.append(row)
            session.add(row)
        session.flush()
        jobs = select(OpsJob).where(OpsJob.state.not_in(JOB_TERMINAL_STATES)).order_by(OpsJob.id)
        for job in session.exec(jobs if dry_run else jobs.with_for_update()).all():
            report["jobs"].append({"id": job.id, "state": job.state, "operation": job.operation_slug,
                                   "approval_request_id": job.approval_request_id})
            if not dry_run:
                job.state, job.finished_at, job.error_text = JOB_CANCELLED, now, reason
                session.add(job)
        if dry_run:
            session.rollback()
            return report
        resolved = [(row.id, row.tool_use_id) for row in cards]
        session.commit()
    _resolve_cards(engine, resolved)
    return report


def _resolve_cards(engine, cards: list[tuple[str, str | None]]) -> None:
    """Replace each purged card's placeholder with its cancelled result.

    Best-effort and after commit, as the decide route does: the stored
    resolution is already durable, so a failure here only delays the card
    until the next read of the request.
    """
    store = ToolCallStore(engine)
    for request_id, tool_use_id in cards:
        try:
            store.resolve_approval_result(tool_use_id=tool_use_id or "", approval_request_id=request_id,
                                          approval_status=REQ_CANCELLED, result=PURGED_RESULT, is_error=False)
        except Exception:
            log.debug("could not resolve purged approval card %s", request_id, exc_info=True)


def purged_resolutions(engine, request_ids: list[str]) -> list[dict[str, Any]]:
    """Each purged request's ``approval_resolved`` arguments, read back after commit.

    The payload the decide route publishes for an MCP request
    (``routes/approval_policies.py``): its stored result, error flag and
    execution state.
    """
    if not request_ids:
        return []
    with Session(engine) as session:
        rows = session.exec(select(ToolApprovalRequest)
                            .where(ToolApprovalRequest.id.in_(request_ids))
                            .order_by(ToolApprovalRequest.id)).all()
        events = []
        for row in rows:
            result: Any = None
            if row.result_json:
                try:
                    result = json.loads(row.result_json)
                except json.JSONDecodeError:
                    result = row.result_json
            events.append({
                "bot_id": row.bot_id, "user_id": row.user_id, "request_id": row.id,
                "turn_id": row.turn_id, "status": row.status, "tool_use_id": row.tool_use_id,
                "result": result, "is_error": bool(row.result_is_error),
                "execution_state": row.execution_state, "has_result": not awaits_ops_admission(row),
            })
        return events


async def publish_purge_resolutions(engine, request_ids: list[str]) -> None:
    """Tell open clients each purged card is resolved.

    Best-effort: the purge is already durable, so a lost event only leaves an
    open card stale until its next read of the request. Never fails the purge.
    """
    try:
        subscriber = get_agent_subscriber()
        if subscriber is None:
            return
        for event in await asyncio.to_thread(purged_resolutions, engine, request_ids):
            await publish_approval_resolved(subscriber, **event)
    except Exception:  # noqa: BLE001
        log.warning("could not publish approval_resolved for purged requests %s", request_ids, exc_info=True)
