"""Transactional terminal-state helpers shared by abort and normal finalization.

An unresolved tool ending with its parent has no known execution result. We
record the observation's end time and result_complete=False, not synthetic
success/error output. A later canonical result can still replace that unknown.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from sqlalchemy import update
from sqlmodel import Session, select

from .tool_call_store import ToolCallRecord
from .turn_gates import EXECUTION_EXIT_KEY


def interrupt_unresolved_tools(session: Session, turn_id: str, ended_at: datetime) -> None:
    session.execute(
        update(ToolCallRecord)
        .where(ToolCallRecord.turn_id == turn_id)
        .where(ToolCallRecord.ended_at.is_(None))
        .where(ToolCallRecord.result_text.is_(None))
        .values(ended_at=ended_at.timestamp(), result_complete=False)
    )


def reconcile_finished_tool(engine, *, turn_id: str, record_id: int | None = None) -> None:
    """Finish a start that raced its parent without locking the parent row.

    A start commits before reading the terminal state; a terminal writer sweeps
    again after its own commit. Whichever commits last observes the other. A
    real result always wins over this provisional interrupted marker.
    """
    from .turn_logs import TurnLog

    with Session(engine) as session:
        ended_at = session.exec(select(TurnLog.ended_at).where(TurnLog.id == turn_id)).first()
        if ended_at is None:
            return
        statement = (
            update(ToolCallRecord)
            .where(ToolCallRecord.turn_id == turn_id)
            .where(ToolCallRecord.ended_at.is_(None))
            .where(ToolCallRecord.result_text.is_(None))
            .values(ended_at=ended_at.timestamp(), result_complete=False)
        )
        if record_id is not None:
            statement = statement.where(ToolCallRecord.id == record_id)
        session.execute(statement)
        session.commit()


def reconcile_finished_tool_safely(engine, *, turn_id: str) -> None:
    """Reconcile after the terminal commit; never relabel that commit as failed."""
    try:
        reconcile_finished_tool(engine, turn_id=turn_id)
    except Exception:
        logging.getLogger(__name__).exception("post-commit tool reconciliation failed for turn=%s", turn_id)


# TASK-1232 r2-M1: the ONE list of request_json keys a rewrite must preserve.
# AUDIT_KEYS survive every rewrite of a turn row's request_json, including the
# delivery re-arm writes that start a new attempt (write_reservation,
# save_turn's reserved UPDATE). ATTEMPT_KEYS describe a single attempt: they
# survive late writes within it (update_turn) but never carry into the next.
# TASK-1318: "system_prompt_pin" is written only by SystemPromptPinStore; on
# the pin turn it carries the pinned prompt bytes other turns point at.
AUDIT_KEYS = (
    "abort_request", "ordinary_request_digest", "user_stop", "recovery", "manual_settle", "attempts",
    "system_prompt_pin",
)
ATTEMPT_KEYS = ("abort_cause", EXECUTION_EXIT_KEY)
# TASK-1234 Fix 2: how many earlier attempts ``attempts`` keeps (newest last).
ATTEMPT_HISTORY_LIMIT = 10


def turn_payload(request_json: str | None) -> dict:
    """A turn row's ``request_json`` as a dict; unreadable or non-object → {}."""
    try:
        payload = json.loads(request_json) if request_json else {}
    except (TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def dump_payload(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _preserved_payload(previous: str | None, payload: dict, *, rearm: bool) -> dict:
    # A preserved key from ``previous`` wins over the same key in ``payload``:
    # audits are written only by their own writers, never by a request body.
    previous_payload = turn_payload(previous)
    merged = dict(payload)
    for key in AUDIT_KEYS if rearm else AUDIT_KEYS + ATTEMPT_KEYS:
        if key in previous_payload:
            merged[key] = previous_payload[key]
    return merged


def request_payload_with_audit(previous: str | None, payload: dict | None, *, rearm: bool = False) -> str | None:
    """``payload`` with the preserved keys of ``previous`` carried over.

    ``rearm`` marks a write that starts a new attempt: per-attempt keys drop.
    A ``None`` payload keeps only the preserved keys (``None`` if there are none).
    """
    merged = _preserved_payload(previous, payload or {}, rearm=rearm)
    if payload is None and not merged:
        return None
    return dump_payload(merged)


def _iso(value) -> str | None:
    return value.isoformat() if hasattr(value, "isoformat") else (str(value) if value else None)


def rearm_request_payload(prior, payload: dict) -> str:
    """request_json for a write that starts a new attempt on an existing row.

    TASK-1234 Fix 2. ``prior`` is the row before the write (``request_json``,
    ``status``, ``end_reason``, ``error_text``, ``created_at``, ``ended_at``) or
    None. Audits carry over; a prior attempt that actually started is appended
    to ``attempts``, since the row's own columns now describe the new attempt
    (``created_at`` is when the current attempt was admitted). A row still
    ``reserved`` never started and leaves nothing to record; a reservation a
    Stop settled is recorded as its ``aborted`` attempt.
    """
    merged = _preserved_payload(prior["request_json"] if prior else None, payload, rearm=True)
    if prior is not None and prior["status"] != "reserved":
        history = [entry for entry in merged.get("attempts") or [] if isinstance(entry, dict)]
        error = prior["error_text"]
        history.append({
            "n": (history[-1].get("n") or len(history)) + 1 if history else 1,
            "started": _iso(prior["created_at"]), "ended": _iso(prior["ended_at"]),
            "status": prior["status"], "end_reason": prior["end_reason"],
            "error": str(error)[:500] if error else None,
        })
        merged["attempts"] = history[-ATTEMPT_HISTORY_LIMIT:]
    return dump_payload(merged)


def merge_audit(request_json: str | None, key: str, values: dict) -> str:
    """``request_json`` with ``values`` merged into the ``key`` audit dict."""
    payload = turn_payload(request_json)
    existing = payload.get(key)
    payload[key] = {**(existing if isinstance(existing, dict) else {}), **values}
    return dump_payload(payload)


def stamp_abort_cause(row, cause: str | None) -> None:
    """TASK-1231B: record why the bridge itself cancelled this attempt.

    ``end_reason`` stays ``aborted`` (clients switch on it); the cause is the
    per-attempt ``abort_cause`` key (ATTEMPT_KEYS). A user Stop owns rows
    already ``cancelling``, and an ended row's outcome is never rewritten.
    """
    if not cause or row.ended_at is not None or row.status == "cancelling":
        return
    payload = turn_payload(row.request_json)
    payload["abort_cause"] = {"cause": cause, "at": datetime.now(timezone.utc).isoformat()}
    row.request_json = dump_payload(payload)


def settle_open_turn(session: Session, row, *, audit_key: str, audit: dict, now: datetime) -> bool:
    """The one fenced settle for an open turn held FOR NO KEY UPDATE.

    First writer wins: a row that already ended is left untouched (False).
    Every abort-class settle ends ``aborted``/``aborted``, the only pair the
    clients treat as a stopped reply (A1). The cause lives in the audit under
    ``audit_key`` (``user_stop`` or ``recovery``), never in ``end_reason``.
    """
    if row.ended_at is not None:
        return False
    payload = turn_payload(row.request_json)
    payload[audit_key] = {**audit, "settled_at": now.isoformat()}
    row.request_json = dump_payload(payload)
    row.status, row.end_reason, row.ended_at = "aborted", "aborted", now
    session.add(row)
    from .turn_terminal_effects import record_terminal_effects
    record_terminal_effects(session, row)
    return True


# TASK-1015: an unacknowledged Stop is bounded. Fresh cancellation intent is
# never reaped (the worker may still acknowledge), but once no live owner
# exists in this process and the request is older than the longest bridge
# budget, nothing can ever acknowledge it — the row would read "cancelling"
# forever. It settles aborted/aborted with recovery.cause=abort_unconfirmed:
# honest about the unknown, and still rendered as a stopped reply (A1).
UNCONFIRMED_ABORT_GRACE_SECONDS = 1800.0


def _abort_requested_at(row) -> datetime | None:
    try:
        audit = turn_payload(row.request_json).get("abort_request") or {}
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
        ).with_for_update(key_share=True)).all()
        now = datetime.now(timezone.utc)
        result = []
        for row in rows:
            # An ordinary agent START has no durable execution owner yet. Its
            # bridge command may have been accepted even when this process no
            # longer has a TurnExecution handle. A newer turn's first output
            # cannot prove that command stopped, so never clear this admission
            # fence by guessing a timeout (including after a Stop request).
            if turn_payload(row.request_json).get("ordinary_request_digest"):
                continue
            if row.status == "cancelling":
                requested = _abort_requested_at(row)
                if requested is None or (now - requested).total_seconds() < UNCONFIRMED_ABORT_GRACE_SECONDS:
                    continue
                settle_open_turn(session, row, audit_key="recovery", now=now, audit={
                    "cause": "abort_unconfirmed", "by": "reap_unowned_turns",
                    "current_turn_id": current_turn_id,
                })
            else:
                # Not an abort: the stale-reap timeout keeps update_turn's
                # timeout→completed repair path (turn_logs.update_turn).
                row.status, row.end_reason, row.ended_at = "timeout", "timeout", now
                session.add(row)
                from .turn_terminal_effects import record_terminal_effects
                record_terminal_effects(session, row)
            result.append({"id": row.id, "user_id": row.user_id, "usage": row.token_usage_json})
        session.commit()
    for reaped in result:
        try:
            reconcile_finished_tool(store.engine, turn_id=reaped["id"])
        except Exception:
            logging.getLogger(__name__).exception(
                "post-commit tool reconciliation failed for turn=%s", reaped["id"],
            )
    # turn_complete for each reaped turn pushes its committed balance (TASK-1461).
    from .usage.balance import balance_update_for_turn
    return [{"id": r["id"], "user_id": r["user_id"],
             "balance_update": balance_update_for_turn(store.engine, r["usage"])} for r in result]


class TurnAbortCoordinator:
    """Claim cancellation before bridge teardown; keep provenance in turn audit."""

    def __init__(self, store):
        self.store = store

    def request(self, turn_id: str, *, source: str, peer: str | None) -> bool:
        from .turn_logs import TurnLog

        with Session(self.store.engine) as session:
            row = session.exec(select(TurnLog).where(TurnLog.id == turn_id).with_for_update(key_share=True)).first()
            if row is None or row.ended_at is not None or row.status not in ("pending", "streaming", "cancelling"):
                return False
            if row.status == "cancelling":
                return True  # exact request-scoped retries are idempotent
            now = datetime.now(timezone.utc)
            payload = turn_payload(row.request_json)
            payload["abort_request"] = {
                "source": source, "peer": peer,
                # Namespace is known; the authenticated human actor is not.
                "actor": None, "user_id": row.user_id,
                "turn_id": row.id, "request_id": row.agent_request_id,
                "requested_at": now.isoformat(), "outcome": "requested",
            }
            row.request_json = dump_payload(payload)
            # Intent is durable, but execution has not acknowledged termination.
            # The finalizer or successful abort RPC will stamp the terminal state.
            row.status = "cancelling"
            session.add(row)
            session.commit()
            return True

    def settle(self, turn_id: str, audit: dict) -> bool:
        """User Stop always stops (TASK-1232 P4): settle the open row now.

        False when another writer (the worker's own finalize) ended it first.
        """
        from .turn_logs import TurnLog

        with Session(self.store.engine) as session:
            row = session.exec(select(TurnLog).where(TurnLog.id == turn_id).with_for_update(key_share=True)).first()
            if row is None:
                return False
            now = datetime.now(timezone.utc)
            settled = settle_open_turn(session, row, audit_key="user_stop", audit=audit, now=now)
            if not settled:
                # The worker's terminal stands; still record what the bridge said.
                payload = turn_payload(row.request_json)
                payload["user_stop"] = {**audit, "settled": False, "recorded_at": now.isoformat()}
                row.request_json = dump_payload(payload)
                session.add(row)
            session.commit()
        if settled:
            try:
                reconcile_finished_tool(self.store.engine, turn_id=turn_id)
            except Exception:
                logging.getLogger(__name__).exception(
                    "post-commit tool reconciliation failed for turn=%s", turn_id,
                )
        return settled

    def record_outcome(self, turn_id: str, outcome: str) -> None:
        from .turn_logs import TurnLog

        with Session(self.store.engine) as session:
            row = session.exec(select(TurnLog).where(TurnLog.id == turn_id).with_for_update(key_share=True)).first()
            if row is None:
                return
            payload = turn_payload(row.request_json)
            audit = payload.get("abort_request")
            if not isinstance(audit, dict):
                return
            audit.update(outcome=outcome, acknowledged_at=datetime.now(timezone.utc).isoformat())
            row.request_json = dump_payload(payload)
            session.add(row)
            session.commit()
