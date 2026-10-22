"""Step 1d/1e: open turns that no live execution owns (TASK-1232).

A candidate is an open, started turn older than ``ORPHAN_MIN_AGE_SECONDS`` with
no owner the app can see: no live ``TurnExecution`` in this process, no
delivery claim (a claimed delivery is ``recover_expired``'s, two-phase r2-M4)
and no freshly leased approval continuation (Vex S5).

Only the bridge can say the run is not executing. The step settles a turn
(``recovery.cause = orphaned_by_restart``) only when a ``RunProbe`` answers
``not_found`` from the backend the turn was addressed to. Every other answer
is ambiguous: the turn is left and reported. A user Stop still settles it
(``turn_abort``). A finished run's final text, when the probe returns it, is
kept in the audit as ``stranded_result``, not projected into history
(TASK-1242).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Protocol

from sqlalchemy import text as sa_text
from sqlmodel import Session, select

from ...approval_models import CONTINUATION_LEASE_SECONDS
from ..turn_admission import lock_target
from ..turn_execution import turn_executions
from ..turn_logs import TurnLog
from ..turn_termination import settle_open_turn
from .base import Plan, RecoveryStep, SweepScope

# Past the 300s delivery lease, so a crashed delivery is recover_expired's first.
ORPHAN_MIN_AGE_SECONDS = 300.0

_OPEN_STARTED = """
    SELECT t.id, t.bot_id, t.status, t.agent_session_key, t.agent_request_id, t.created_at
    FROM turn_logs t
    WHERE t.ended_at IS NULL AND t.status <> 'reserved' AND t.created_at < :min_created
      AND NOT EXISTS (
          SELECT 1 FROM inter_bot_deliveries d
          WHERE d.turn_id = t.id AND d.claim_token IS NOT NULL)
      AND NOT EXISTS (
          SELECT 1 FROM tool_approval_requests a
          WHERE 'turn-' || a.continuation_id = t.id AND a.continuation_state = 'dispatching'
            AND a.continuation_claim_token IS NOT NULL AND a.continuation_next_attempt_at > :lease_floor)
"""


@dataclass(frozen=True)
class ProbeResult:
    """A bridge's ``run.status`` answer for one request id."""

    state: str
    backend_matched: bool = False
    backend: str | None = None
    boot_id: str | None = None
    detail: str = ""
    stranded_result: dict[str, Any] | None = field(default=None)

    def audit(self) -> dict[str, Any]:
        return {"state": self.state, "backend": self.backend, "boot_id": self.boot_id, "detail": self.detail}


class RunProbe(Protocol):
    async def probe(self, turn: dict[str, Any]) -> ProbeResult: ...


class UnavailableRunProbe:
    """No bridge answers ``run.status`` yet: every orphan is reported, none settled."""

    async def probe(self, turn: dict[str, Any]) -> ProbeResult:
        return ProbeResult(state="unsupported", detail="run.status is not available")


class OrphanTurnsStep(RecoveryStep):
    name = "orphan_turns"

    def __init__(self, probe: RunProbe | None = None, *, min_age_seconds: float = ORPHAN_MIN_AGE_SECONDS):
        self.probe = probe or UnavailableRunProbe()
        self.min_age_seconds = min_age_seconds

    def _params(self, scope: SweepScope) -> dict[str, Any]:
        return {"min_created": scope.now - timedelta(seconds=self.min_age_seconds),
                "lease_floor": scope.now - timedelta(seconds=CONTINUATION_LEASE_SECONDS)}

    async def detect(self, scope: SweepScope) -> list[dict[str, Any]]:
        def read() -> list[dict[str, Any]]:
            with scope.engine.connect() as conn:
                rows = conn.execute(sa_text(_OPEN_STARTED + " ORDER BY t.created_at"), self._params(scope)).mappings().all()
            live = set(turn_executions.active_ids())
            return [{"turn_id": row["id"], "bot_id": row["bot_id"], "status": row["status"],
                     "agent_session_key": row["agent_session_key"], "agent_request_id": row["agent_request_id"],
                     "created_at": row["created_at"].isoformat() if row["created_at"] else None}
                    for row in rows if row["id"] not in live and scope.includes(row["bot_id"])]

        return await asyncio.to_thread(read)

    async def plan(self, scope: SweepScope, candidate: dict[str, Any]) -> Plan:
        if not candidate["agent_session_key"] or not candidate["agent_request_id"]:
            return Plan.leave("no bridge request id to probe")
        result = await self.probe.probe(candidate)
        if result.state == "not_found" and result.backend_matched:
            return Plan("settle", detail={"probe": result.audit(), "stranded": result.stranded_result is not None},
                        carry=result)
        return Plan.leave(f"run.status {result.state}", probe=result.audit())

    async def apply(self, scope: SweepScope, candidate: dict[str, Any], plan: Plan) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._settle, scope, candidate, plan.carry)

    def _settle(self, scope: SweepScope, candidate: dict[str, Any], result: ProbeResult) -> dict[str, Any] | None:
        turn_id = candidate["turn_id"]
        with scope.engine.begin() as conn:
            lock_target(conn, candidate["bot_id"])
            # Re-check every exclusion under the admission lock and row lock.
            lock = " FOR UPDATE OF t" if conn.dialect.name == "postgresql" else ""
            still = conn.execute(sa_text(_OPEN_STARTED + " AND t.id = :id" + lock),
                                 {**self._params(scope), "id": turn_id}).first()
            if still is None or turn_id in set(turn_executions.active_ids()):
                return None
            with Session(bind=conn) as session:
                row = session.exec(select(TurnLog).where(TurnLog.id == turn_id).with_for_update()).first()
                audit = {"cause": "orphaned_by_restart", **scope.stamp.as_dict(), "probe": result.audit()}
                if result.stranded_result is not None:
                    audit["stranded_result"] = result.stranded_result
                if row is None or not settle_open_turn(session, row, audit_key="recovery", audit=audit, now=scope.now):
                    return None
        return {"settled": True}
