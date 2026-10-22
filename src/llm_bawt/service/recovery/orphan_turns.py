"""Step 1d/1e: open turns that no live execution owns (TASK-1232).

A candidate is an open, started turn created before this process started (the
sweep's ``started_at``), with no owner the app can see: no live
``TurnExecution`` in this process, no delivery claim (a claimed delivery is
``recover_expired``'s, two-phase r2-M4) and no freshly leased approval
continuation (Vex S5). A turn created since startup belongs to this process's
own workers and is never a candidate.

Only the bridge can say the run is not executing. The step settles a turn
(``recovery.cause = orphaned_by_restart``) only when a ``RunProbe`` answers
``not_found`` from the backend the turn was published to
(``turn_logs.agent_backend``). Every other answer is ambiguous: the turn is
left and reported. A user Stop still settles it (``turn_abort``). A finished
run's final text, when a probe returns it, is kept in the audit as
``stranded_result``, not projected into history (TASK-1242).

Cutoff limits. Turn rows carry no owner stamp, so "created before this process
started" is the only exact "not this process's". Consequences:

- A turn the check leaves (no answer within the probe timeout, any state but
  ``not_found``) gets no second look from this process. The turn reaper never
  clears an ordinary agent START (``reap_unowned_turns`` skips rows carrying
  ``ordinary_request_digest``), so it stays open until a user Stop or the next
  startup check.
- A process that takes leadership late (it was passive and the leader died)
  never settles turns the dead leader opened after this process started; they
  stay open the same way. This app runs as one process, so the case is a
  failover we don't deploy.
- A rolling restart is safe: the new process leads only once the old one
  releases the leadership lock, which it holds until it exits, so every turn
  the old process opened is dead by the time it is probed.
- A row with no ``agent_backend`` (bound before the column existed, i.e. in
  flight at the first restart after it shipped) is never settled here.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol

from sqlalchemy import text as sa_text
from sqlmodel import Session, select

from ...approval_models import CONTINUATION_LEASE_SECONDS
from ..turn_admission import lock_target
from ..turn_execution import turn_executions
from ..turn_logs import TurnLog
from ..turn_termination import settle_open_turn
from .base import Plan, RecoveryStep, SweepScope

logger = logging.getLogger(__name__)

_OPEN_STARTED = """
    SELECT t.id, t.bot_id, t.status, t.agent_backend, t.agent_session_key, t.agent_request_id, t.created_at
    FROM turn_logs t
    WHERE t.ended_at IS NULL AND t.status <> 'reserved' AND t.created_at < :started_at
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
    """No transport to ask: every orphan is reported, none settled."""

    async def probe(self, turn: dict[str, Any]) -> ProbeResult:
        return ProbeResult(state="unsupported", detail="run.status is not available")


class BridgeRunProbe:
    """Asks the bridge a turn was published to whether its request still exists.

    One ``run.status`` RPC, addressed to the turn's recorded ``agent_backend``
    only: a bridge that never received the request would also answer
    ``not_found``, so a turn without a recorded backend is never probed. The
    reply parser (``RunStatusReply.from_wire``) refuses a reply from any other
    backend, which is what ``backend_matched`` asserts.

    The timeout covers a bridge restarting alongside the app: its consumer
    group reads the command once it is back and answers from its new boot.
    """

    TIMEOUT_SECONDS = 60.0

    def __init__(self, subscriber: Callable[[], Any] | None = None, *, timeout_seconds: float = TIMEOUT_SECONDS):
        if subscriber is None:
            from ...agent_backends.agent_bridge import get_agent_subscriber

            subscriber = get_agent_subscriber
        self._subscriber = subscriber
        self.timeout_seconds = timeout_seconds

    async def probe(self, turn: dict[str, Any]) -> ProbeResult:
        _reply, result = await self.status(
            backend=turn.get("agent_backend"), session_key=turn.get("agent_session_key"),
            request_id=turn.get("agent_request_id"), label=turn.get("turn_id"),
        )
        return result

    async def status(self, *, backend: str | None, session_key: str | None, request_id: str | None,
                     label: Any = None) -> tuple[Any, ProbeResult]:
        """The one ``run.status`` call: ``(RunStatusReply | None, ProbeResult)``.

        The parsed reply is returned only when the addressed backend answered
        with a v1 reply; every ambiguous outcome is ``None`` plus its
        ``ProbeResult`` state. The quiesce exit sweep (TASK-1233) needs the
        reply itself (its ``pid1``); recovery needs only the result.
        """
        from agent_bridge.run_status import RUN_STATUS_METHOD, RunStatusReply, RunStatusReplyError

        if not backend:
            return None, ProbeResult(state="unaddressed", detail="turn has no recorded bridge backend")
        subscriber = self._subscriber()
        if subscriber is None:
            return None, ProbeResult(state="bridge_unreachable", backend=backend, detail="no agent subscriber")
        try:
            wire = await subscriber.send_rpc(
                RUN_STATUS_METHOD, {"sessionKey": session_key, "requestId": request_id},
                f"run_status_{uuid.uuid4().hex}", timeout_s=self.timeout_seconds, backend=backend,
            )
        except TimeoutError:
            return None, ProbeResult(state="timeout", backend=backend,
                                     detail=f"no reply within {self.timeout_seconds:.0f}s")
        except Exception as exc:
            logger.warning("run.status for %s failed: %s", label, exc)
            return None, ProbeResult(state="rpc_error", backend=backend, detail=type(exc).__name__)
        try:
            reply = RunStatusReply.from_wire(wire, backend=backend)
        except RunStatusReplyError as exc:
            return None, ProbeResult(state="malformed_reply", backend=backend, detail=str(exc))
        return reply, ProbeResult(state=reply.state.value, backend_matched=True, backend=reply.backend,
                                  boot_id=reply.boot_id, detail=reply.detail)


class OrphanTurnsStep(RecoveryStep):
    name = "orphan_turns"
    # Probes wait on bridges, so they run concurrently; settles stay serial.
    MAX_CONCURRENT_PROBES = 8

    def __init__(self, probe: RunProbe | None = None, *, started_at: datetime):
        self.probe = probe or UnavailableRunProbe()
        self.started_at = started_at

    def _params(self, scope: SweepScope) -> dict[str, Any]:
        return {"started_at": self.started_at,
                "lease_floor": scope.now - timedelta(seconds=CONTINUATION_LEASE_SECONDS)}

    async def detect(self, scope: SweepScope) -> list[dict[str, Any]]:
        def read() -> list[dict[str, Any]]:
            with scope.engine.connect() as conn:
                rows = conn.execute(sa_text(_OPEN_STARTED + " ORDER BY t.created_at"), self._params(scope)).mappings().all()
            live = set(turn_executions.active_ids())
            return [{"turn_id": row["id"], "bot_id": row["bot_id"], "status": row["status"],
                     "agent_backend": row["agent_backend"], "agent_session_key": row["agent_session_key"],
                     "agent_request_id": row["agent_request_id"],
                     "created_at": row["created_at"].isoformat() if row["created_at"] else None}
                    for row in rows if row["id"] not in live and scope.includes(row["bot_id"])]

        return await asyncio.to_thread(read)

    async def plan_all(self, scope: SweepScope, candidates: list[dict[str, Any]]) -> list[Plan]:
        gate = asyncio.Semaphore(self.MAX_CONCURRENT_PROBES)

        async def bounded(candidate: dict[str, Any]) -> Plan:
            async with gate:
                return await self.plan(scope, candidate)

        return list(await asyncio.gather(*(bounded(candidate) for candidate in candidates)))

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
