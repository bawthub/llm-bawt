"""Transactional terminal-state helpers shared by abort and normal finalization.

An unresolved tool ending with its parent has no known execution result. We
record the observation's end time and result_complete=False, not synthetic
success/error output. A later canonical result can still replace that unknown.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import update
from sqlmodel import Session, select

from .tool_call_store import ToolCallRecord


def interrupt_unresolved_tools(session: Session, turn_id: str, ended_at: datetime) -> None:
    session.execute(
        update(ToolCallRecord)
        .where(ToolCallRecord.turn_id == turn_id)
        .where(ToolCallRecord.ended_at.is_(None))
        .where(ToolCallRecord.result_text.is_(None))
        .values(ended_at=ended_at.timestamp(), result_complete=False)
    )


def request_payload_with_audit(previous: str | None, payload: dict) -> str:
    previous_payload = json.loads(previous) if previous else {}
    merged = dict(payload)
    if isinstance(previous_payload, dict) and "abort_request" in previous_payload:
        merged["abort_request"] = previous_payload["abort_request"]
    return json.dumps(merged, ensure_ascii=False, default=str)


def reap_unowned_turns(store, bot_id: str, current_turn_id: str) -> list[dict]:
    """Reap abandoned ordinary turns, never live owners or cancellation intent."""
    from .turn_execution import turn_executions
    from .turn_logs import TurnLog

    with Session(store.engine) as session:
        current = session.get(TurnLog, current_turn_id)
        if current is None:
            return []
        rows = session.exec(select(TurnLog).where(
            TurnLog.bot_id == bot_id,
            TurnLog.ended_at.is_(None),
            TurnLog.status != "cancelling",
            TurnLog.id != current_turn_id,
            TurnLog.created_at < current.created_at,
            TurnLog.id.not_in(turn_executions.active_ids()),
        ).with_for_update()).all()
        now = datetime.now(timezone.utc)
        result = []
        for row in rows:
            row.status, row.end_reason, row.ended_at = "timeout", "timeout", now
            interrupt_unresolved_tools(session, row.id, now)
            session.add(row)
            result.append({"id": row.id, "user_id": row.user_id})
        session.commit()
        return result


class TurnAbortCoordinator:
    """Claim cancellation before bridge teardown; keep provenance in turn audit."""

    def __init__(self, store):
        self.store = store

    def request(self, turn_id: str, *, source: str, peer: str | None) -> bool:
        from .turn_logs import TurnLog

        with Session(self.store.engine) as session:
            row = session.exec(select(TurnLog).where(TurnLog.id == turn_id).with_for_update()).first()
            if row is None or row.ended_at is not None or row.status not in ("pending", "streaming", "cancelling"):
                return False
            if row.status == "cancelling":
                return True  # exact request-scoped retries are idempotent
            now = datetime.now(timezone.utc)
            payload = json.loads(row.request_json) if row.request_json else {}
            payload["abort_request"] = {
                "source": source, "peer": peer,
                # Namespace is known; the authenticated human actor is not.
                "actor": None, "user_id": row.user_id,
                "turn_id": row.id, "request_id": row.agent_request_id,
                "requested_at": now.isoformat(), "outcome": "requested",
            }
            row.request_json = json.dumps(payload, ensure_ascii=False, default=str)
            # Intent is durable, but execution has not acknowledged termination.
            # The finalizer or successful abort RPC will stamp the terminal state.
            row.status = "cancelling"
            session.add(row)
            session.commit()
            return True

    def record_outcome(self, turn_id: str, outcome: str) -> None:
        from .turn_logs import TurnLog

        with Session(self.store.engine) as session:
            row = session.exec(select(TurnLog).where(TurnLog.id == turn_id).with_for_update()).first()
            if row is None:
                return
            payload = json.loads(row.request_json) if row.request_json else {}
            audit = payload.get("abort_request")
            if not isinstance(audit, dict):
                return
            audit.update(outcome=outcome, acknowledged_at=datetime.now(timezone.utc).isoformat())
            row.request_json = json.dumps(payload, ensure_ascii=False, default=str)
            session.add(row)
            session.commit()
