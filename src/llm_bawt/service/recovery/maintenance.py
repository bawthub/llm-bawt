"""Small recovery steps (TASK-1232 steps 4, 6, 9 and Vex S1's sweep rule)."""
from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

from sqlalchemy import text as sa_text
from sqlmodel import Session, select

from ...approval_models import (
    CONT_DISPATCHING,
    CONT_FAILED,
    CONT_SUPPRESSED,
    CONTINUATION_LEASE_SECONDS,
    REQ_CANCELLED,
)
from ..turn_admission import peek_target
from ..turn_logs import TurnLog
from ..turn_termination import interrupt_unresolved_tools, merge_audit
from .base import Plan, RecoveryStep, SweepScope


# ``failed`` continuations are history, not work: report a count and the
# newest few, never every row on every pass (Vex S1).
FAILED_SAMPLE_LIMIT = 10


class CancelledContinuationsStep(RecoveryStep):
    """A cancelled (purged) request's continuation stuck ``dispatching``.

    Its claimant went quiet past the lease, so no dispatching→pending writer
    will ever suppress it (Vex S1). Suppress it here, by CAS on the observed
    token. ``failed`` continuations are reported as one bounded summary.
    """

    name = "continuations"

    async def detect(self, scope: SweepScope) -> list[dict[str, Any]]:
        def read() -> list[dict[str, Any]]:
            with scope.engine.connect() as conn:
                rows = conn.execute(sa_text("""
                    SELECT id, bot_id, status, continuation_state, continuation_claim_token AS token
                    FROM tool_approval_requests
                    WHERE continuation_state=:dispatching AND status=:cancelled
                      AND continuation_next_attempt_at <= :stale_before
                    ORDER BY id
                """), {"dispatching": CONT_DISPATCHING, "cancelled": REQ_CANCELLED,
                       "stale_before": scope.now - timedelta(seconds=CONTINUATION_LEASE_SECONDS)}).mappings().all()
                bots = sorted(scope.bots) if scope.bots is not None else None
                failed = conn.execute(sa_text(
                    "SELECT id, bot_id, count(*) OVER () AS total FROM tool_approval_requests"
                    " WHERE continuation_state=:failed"
                    + (" AND bot_id = ANY(:bots)" if bots is not None else "")
                    + " ORDER BY resolved_at DESC NULLS LAST, id DESC LIMIT :limit"
                ), {"failed": CONT_FAILED, "bots": bots, "limit": FAILED_SAMPLE_LIMIT}).mappings().all()
            found = [dict(row) for row in rows if scope.includes(row["bot_id"])]
            if failed:
                found.append({"continuation_state": CONT_FAILED, "count": int(failed[0]["total"]),
                              "newest": [row["id"] for row in failed]})
            return found

        return await asyncio.to_thread(read)

    async def plan(self, scope: SweepScope, candidate: dict[str, Any]) -> Plan:
        if candidate["continuation_state"] == "failed":
            return Plan.leave("continuations failed after their retries (report only)")
        return Plan("suppress")

    async def apply(self, scope: SweepScope, candidate: dict[str, Any], plan: Plan) -> dict[str, Any] | None:
        def write() -> dict[str, Any] | None:
            with scope.engine.begin() as conn:
                result = conn.execute(sa_text("""
                    UPDATE tool_approval_requests SET continuation_state=:suppressed, continuation_claim_token=NULL,
                        continuation_last_error=:reason
                    WHERE id=:id AND status=:cancelled AND continuation_state=:dispatching
                      AND continuation_claim_token IS NOT DISTINCT FROM :token
                """), {"id": candidate["id"], "token": candidate["token"], "suppressed": CONT_SUPPRESSED,
                       "cancelled": REQ_CANCELLED, "dispatching": CONT_DISPATCHING,
                       "reason": scope.stamp.text("recovery: suppressed a cancelled request's stale continuation")})
            return {"suppressed": True} if result.rowcount == 1 else None

        return await asyncio.to_thread(write)


class UnresolvedToolsStep(RecoveryStep):
    """Step 4: tool calls still open on a turn that has ended → interrupted."""

    name = "unresolved_tools"

    async def detect(self, scope: SweepScope) -> list[dict[str, Any]]:
        def read() -> list[dict[str, Any]]:
            with scope.engine.connect() as conn:
                rows = conn.execute(sa_text("""
                    SELECT r.turn_id, t.bot_id, count(*) AS open_tools
                    FROM tool_call_records r JOIN turn_logs t ON t.id = r.turn_id
                    WHERE r.ended_at IS NULL AND r.result_text IS NULL AND t.ended_at IS NOT NULL
                    GROUP BY r.turn_id, t.bot_id ORDER BY r.turn_id
                """)).mappings().all()
            return [dict(row) for row in rows if scope.includes(row["bot_id"])]

        return await asyncio.to_thread(read)

    async def plan(self, scope: SweepScope, candidate: dict[str, Any]) -> Plan:
        return Plan("interrupt")

    async def apply(self, scope: SweepScope, candidate: dict[str, Any], plan: Plan) -> dict[str, Any] | None:
        def write() -> dict[str, Any] | None:
            with Session(scope.engine) as session:
                turn = session.exec(select(TurnLog).where(TurnLog.id == candidate["turn_id"]).with_for_update(key_share=True)).first()
                if turn is None or turn.ended_at is None:
                    return None
                interrupt_unresolved_tools(session, turn.id, turn.ended_at)
                # The stamp goes under ``recovery`` without replacing a settle's cause.
                turn.request_json = merge_audit(turn.request_json, "recovery", {"tools_interrupted": {
                    "count": candidate["open_tools"], **scope.stamp.as_dict()}})
                session.add(turn)
                session.commit()
            return {"interrupted": candidate["open_tools"]}

        return await asyncio.to_thread(write)


class PostCheckStep(RecoveryStep):
    """Step 9: each bot with open rows, as admission sees it now (read-only)."""

    name = "post_check"

    async def detect(self, scope: SweepScope) -> list[dict[str, Any]]:
        def read() -> list[dict[str, Any]]:
            with scope.engine.connect() as conn:
                bots = [row[0] for row in conn.execute(sa_text(
                    "SELECT DISTINCT bot_id FROM turn_logs WHERE ended_at IS NULL AND bot_id IS NOT NULL ORDER BY bot_id"
                ))]
            if scope.bots is not None:
                bots = sorted(set(bots) | scope.bots)
            states = []
            for bot in bots:
                state = peek_target(scope.engine, bot, exclude_turn_id=None)
                states.append({"bot_id": bot, "state": state.kind, "blocking_turn_id": state.blocking_id()})
            return states

        return await asyncio.to_thread(read)

    async def plan(self, scope: SweepScope, candidate: dict[str, Any]) -> Plan:
        return Plan(None)
