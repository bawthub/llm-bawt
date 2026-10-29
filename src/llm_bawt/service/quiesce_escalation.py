"""Escalate a quiesce control stuck behind a long-running tool (TASK-1748).

A quiesce control row steers its pinned run at ``priority=next``: it lands at
the next tool boundary. A run inside one long tool (a ten-minute ``gh run
watch``) has no boundary until the tool ends, so the hold waits on it. Once a
control row has been DELIVERED (queued on the pinned run) for
``ESCALATE_AFTER_SECONDS`` and the member has not drained, this pass asks the
bridge to escalate that queued command: interrupt the tool and run the request
now, inside the registry frame ``ops.quiesce_interrupt_frame``.

Rules:

* One attempt per member, stamped before the RPC and never retried: the
  bridge's own rule is that an unconfirmed escalation is never retried.
* Only a run whose pinned capabilities include ``escalate``.
* The bridge refuses (``escalation_other_queued``) when any other command is
  queued on the run, because the sweep would mark it started and the quiesce
  would then drop it. The member then waits for the job's existing deadline.
* Authority is the member's live link to its control row, the same link
  ``DeliveryControlGrant`` admits on; the row and pin are re-read under the
  bot's lock before the stamp.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

from sqlalchemy import text as sa_text

from ..steer_capabilities import ESCALATE
from .gate_steer_escalation import KIND_QUIESCE_ESCALATE, record_gate_escalation, set_escalation_outcome
from .system_gate_admission import LIVE_MEMBER_SQL, control_target
from .turn_admission import lock_target

log = logging.getLogger(__name__)

# Long enough for any ordinary tool call to reach its boundary; short against
# the quiesce deadline (minutes), so a stuck run stops holding a restart.
ESCALATE_AFTER_SECONDS = 60

ESCALATED = "escalated"
REFUSED = "refused"
UNCERTAIN = "uncertain"

# A member ready to escalate: linked to its control row, that row queued on
# the run for the grace period, no attempt yet, and its turn still running.
_DUE_SQL = (
    "m.control_state = 'steering'"
    " AND NOT EXISTS (SELECT 1 FROM gate_escalations e WHERE e.gate_id = m.gate_id"
    "   AND e.kind = 'quiesce_escalate' AND e.subject = CAST(m.id AS TEXT))"
    " AND t.id IS NOT NULL AND t.ended_at IS NULL"
    " AND EXISTS (SELECT 1 FROM inter_bot_deliveries d"
    "   WHERE d.id = m.control_delivery_id AND d.lane = 'control' AND d.status = 'DELIVERED'"
    "     AND d.delivered_at <= clock_timestamp() - (:grace * interval '1 second'))"
)


@dataclass(frozen=True)
class EscalationAsk:
    member_id: int
    gate_id: int
    bot_id: str
    message_id: str
    session_key: str
    backend: str
    request_id: str


@dataclass
class EscalationPass:
    attempted: int = 0
    escalated: int = 0


class QuiesceEscalator:
    """Interrupts each stuck held run once, with the quiesce request."""

    BATCH = 16

    def __init__(self, store, *, grace_seconds: int = ESCALATE_AFTER_SECONDS):
        self.store = store
        self.engine = store.engine
        self.grace_seconds = grace_seconds

    def due(self) -> list[tuple[int, str]]:
        with self.engine.connect() as conn:
            rows = conn.execute(sa_text(
                LIVE_MEMBER_SQL.format(where=_DUE_SQL) + " ORDER BY m.id LIMIT :limit"
            ), {"grace": float(self.grace_seconds), "limit": self.BATCH}).mappings().all()
        return [(int(row["id"]), row["bot_id"]) for row in rows]

    def claim(self, member_id: int, bot_id: str) -> EscalationAsk | None:
        """Re-check one member under its bot's lock and stamp the one attempt."""
        with self.engine.begin() as conn:
            lock_target(conn, bot_id)
            row = conn.execute(sa_text(
                LIVE_MEMBER_SQL.format(where=f"m.id = :id AND m.bot_id = :bot AND {_DUE_SQL}")
                + " FOR UPDATE OF m SKIP LOCKED"
            ), {"id": member_id, "bot": bot_id, "grace": float(self.grace_seconds)}).mappings().first()
            if row is None:
                return None
            ask = self._ask(conn, row)
            if ask is None:
                return None
            if not self._stamp(conn, row):
                return None
            return ask

    def _ask(self, conn, row: Mapping[str, Any]) -> EscalationAsk | None:
        target = control_target(row)
        if target is None or not target.backend or not target.session_key:
            return None
        if ESCALATE not in target.steer_caps:
            self._stamp(conn, row, outcome=f"{REFUSED}:escalate_unsupported")
            return None
        control = conn.execute(sa_text(
            "SELECT d.user_message_id, d.metadata_json->'control'->'target_run'->>'agent_request_id' AS pinned"
            " FROM gate_members m JOIN inter_bot_deliveries d ON d.id = m.control_delivery_id"
            " WHERE m.id = :id"
        ), {"id": row["id"]}).mappings().first()
        if control is None or control["pinned"] != target.request_id or not control["user_message_id"]:
            return None
        return EscalationAsk(int(row["id"]), int(row["gate_id"]), row["bot_id"], str(control["user_message_id"]),
                             target.session_key, target.backend, target.request_id)

    # --- the record (one attempt, then its outcome) ------------------------

    # The shared gate_escalations row (TASK-1749): its insert is the one-shot
    # stamp, keyed by member id; the outcome is filled in after the RPC.
    @staticmethod
    def _stamp(conn, row: Mapping[str, Any], *, outcome: str = "attempting") -> bool:
        return record_gate_escalation(
            conn, gate_id=int(row["gate_id"]), bot_id=row["bot_id"], kind=KIND_QUIESCE_ESCALATE,
            subject=str(row["id"]), actor="system:quiesce", turn_id=row.get("turn_id"),
            outcome=outcome,
        )

    def record(self, ask: EscalationAsk, outcome: str) -> None:
        with self.engine.begin() as conn:
            set_escalation_outcome(conn, gate_id=ask.gate_id, kind=KIND_QUIESCE_ESCALATE,
                                   subject=str(ask.member_id), outcome=outcome[:500])

    # --- one pass ----------------------------------------------------------

    async def run_pass(self, subscriber) -> EscalationPass:
        import asyncio

        from ..ops.quiesce_prompts import render_quiesce_interrupt_frame

        result = EscalationPass()
        if subscriber is None:
            return result
        for member_id, bot_id in await asyncio.to_thread(self.due):
            ask = await asyncio.to_thread(self.claim, member_id, bot_id)
            if ask is None:
                continue
            result.attempted += 1
            outcome = await self._escalate(subscriber, ask, render_quiesce_interrupt_frame)
            result.escalated += outcome == ESCALATED
            log.info("quiesce escalation member=%s bot=%s %s", ask.member_id, ask.bot_id, outcome)
            await asyncio.to_thread(self.record, ask, outcome)
        return result

    @staticmethod
    async def _escalate(subscriber, ask: EscalationAsk, render_frame) -> str:
        import asyncio

        try:
            frame = await asyncio.to_thread(render_frame, ask.bot_id)
            reply = await subscriber.send_steer(
                session_key=ask.session_key, message="", message_id=ask.message_id,
                backend=ask.backend, target_request_id=ask.request_id,
                request_id=f"quiesce_escalate_{ask.member_id}",
                origin="system", escalate=True, interrupt_frame=frame,
            )
        except TimeoutError as exc:
            return f"{UNCERTAIN}:{exc or 'timeout'}"
        except Exception as exc:  # never fails the sweep; the deadline still applies
            return f"{UNCERTAIN}:{exc.__class__.__name__}:{exc}"
        if reply.get("ok"):
            return ESCALATED
        return f"{REFUSED}:{reply.get('error') or 'rejected'}"


def build_escalator(store) -> QuiesceEscalator | None:
    """The app's escalator, or None off PostgreSQL (gates are PostgreSQL-only)."""
    engine = getattr(store, "engine", None)
    if getattr(getattr(engine, "dialect", None), "name", None) != "postgresql":
        return None
    return QuiesceEscalator(store)


__all__ = [
    "ESCALATED", "ESCALATE_AFTER_SECONDS", "EscalationAsk", "EscalationPass", "QuiesceEscalator",
    "REFUSED", "UNCERTAIN", "build_escalator",
]
