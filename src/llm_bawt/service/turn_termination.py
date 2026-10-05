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


# TASK-1015: an unacknowledged Stop is bounded. Fresh cancellation intent is
# never reaped (the worker may still acknowledge), but once no live owner
# exists in this process and the request is older than the longest bridge
# budget, nothing can ever acknowledge it — the row would read "cancelling"
# forever. It settles as aborted/abort_unconfirmed: honest about the unknown.
UNCONFIRMED_ABORT_GRACE_SECONDS = 1800.0


def _abort_requested_at(row) -> datetime | None:
    try:
        audit = json.loads(row.request_json or "{}").get("abort_request") or {}
        stamp = datetime.fromisoformat(str(audit.get("requested_at")))
    except (TypeError, ValueError, AttributeError):
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def reap_unowned_turns(store, bot_id: str, current_turn_id: str) -> list[dict]:
    """Reap abandoned ordinary turns, never live owners or fresh cancellation intent."""
    from .turn_execution import turn_executions
    from .turn_logs import TurnLog

    with Session(store.engine) as session:
        current = session.get(TurnLog, current_turn_id)
        if current is None:
            return []
        rows = session.exec(select(TurnLog).where(
            TurnLog.bot_id == bot_id,
            TurnLog.ended_at.is_(None),
            TurnLog.id != current_turn_id,
            TurnLog.created_at < current.created_at,
            TurnLog.id.not_in(turn_executions.active_ids()),
        ).with_for_update()).all()
        now = datetime.now(timezone.utc)
        result = []
        for row in rows:
            if row.status == "cancelling":
                requested = _abort_requested_at(row)
                if requested is None or (now - requested).total_seconds() < UNCONFIRMED_ABORT_GRACE_SECONDS:
                    continue
                row.status, row.end_reason = "aborted", "abort_unconfirmed"
            else:
                row.status, row.end_reason = "timeout", "timeout"
            row.ended_at = now
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
