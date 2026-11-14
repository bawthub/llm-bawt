"""Fenced, read-only durable evidence and recovery candidate construction."""
from __future__ import annotations

from dataclasses import dataclass
import json

from sqlalchemy import text
from sqlmodel import Session, select

from agent_bridge.events import AgentEventKind
from ..approval_models import ToolApprovalRequest
from .chat_pending_questions import PendingQuestion
from .tool_call_store import ToolCallRecord
from .turn_log_schema import TurnLog
from .turn_terminal_fold import ReplayContent
from .turn_terminal_protocol import FollowerFenceLost, OBSERVATION_STATES, _db_now


@dataclass(frozen=True)
class RecoveryEvidence:
    row: TurnLog
    tools: tuple[ToolCallRecord, ...]
    questions: tuple[PendingQuestion, ...]
    approvals: tuple[ToolApprovalRequest, ...]
    boot_id: str | None
    limited: bool = False


def load_recovery_evidence(protocol, lease) -> RecoveryEvidence:
    """Snapshot under the current lease; release locks before any Redis/RPC wait."""
    with Session(protocol.engine) as session:
        row = protocol._row(session, lease.turn_id)
        protocol._fence(row, lease, _db_now(session))
        if row.terminal_protocol_state not in OBSERVATION_STATES or row.ended_at is not None:
            raise FollowerFenceLost("observation already has a terminal winner")
        tools = session.exec(select(ToolCallRecord).where(
            ToolCallRecord.turn_id == row.id, ToolCallRecord.bot_id == row.bot_id,
            ToolCallRecord.user_id == row.user_id,
        ).order_by(ToolCallRecord.id).limit(10_001)).all()
        questions = session.exec(select(PendingQuestion).where(
            PendingQuestion.turn_id == row.id, PendingQuestion.bot_id == row.bot_id,
            PendingQuestion.user_id == row.user_id,
        ).order_by(PendingQuestion.created_at, PendingQuestion.tool_use_id).limit(1_001)).all()
        approvals = session.exec(select(ToolApprovalRequest).where(
            ToolApprovalRequest.turn_id == row.id, ToolApprovalRequest.bot_id == row.bot_id,
            ToolApprovalRequest.user_id == row.user_id,
        ).order_by(ToolApprovalRequest.created_at, ToolApprovalRequest.id).limit(1_001)).all()
        boot = None
        if protocol.engine.dialect.name == "postgresql" and row.agent_request_id and row.agent_backend:
            boot = session.execute(text(
                "SELECT boot_id FROM bridge_run_accepts WHERE backend=:backend AND request_id=:request"
            ), {"backend": row.agent_backend, "request": row.agent_request_id}).scalar()
        # Detached scalar models cannot lazy-read after the fenced snapshot.
        session.expunge_all()
        return RecoveryEvidence(row, tuple(tools[:10_000]), tuple(questions[:1_000]),
                                tuple(approvals[:1_000]), boot,
                                len(tools) > 10_000 or len(questions) > 1_000 or len(approvals) > 1_000)


def recovery_candidate(evidence: RecoveryEvidence, content: ReplayContent | None, *,
                       incomplete_reason: str | None = None) -> dict:
    """Use durable partials on incomplete replay, never append a replay suffix.

    Interaction rows are read, not reconstructed: replay must never recreate an
    approval/question/reset side effect. Missing required receipts force error.
    Tools merge by stable SDK/call identity; unknown results remain explicit.
    """
    row = evidence.row
    if evidence.limited:
        incomplete_reason = incomplete_reason or "recovery_evidence_limit"
    if content is not None and content.reset_ack:
        # Reset receipt owns app-authored text and must be integrated before
        # protocol rollout to reset turns. Never replay that receipt writer.
        incomplete_reason = incomplete_reason or "recovery_reset_receipt_unresolved"
    questions = {item.tool_use_id: item for item in evidence.questions}
    approvals = {item.id: item for item in evidence.approvals}
    if content is not None:
        for event in content.interactions:
            records = questions if event.kind is AgentEventKind.AWAIT_TOOL_RESULT else approvals
            if not event.tool_use_id or event.tool_use_id not in records:
                incomplete_reason = incomplete_reason or "recovery_interaction_receipt_missing"
    calls = {}
    for tool in evidence.tools:
        key = tool.tool_use_id or tool.call_id or f"record:{tool.id}"
        calls[key] = {"name": tool.tool_name, "result": tool.result_text,
                      "known": tool.result_text is not None, "complete": tool.result_complete is True}
    if content is not None and incomplete_reason is None:
        for tool in content.tools:
            key = tool.get("tool_use_id") or tool.get("event_id")
            prior = calls.get(key)
            if prior is not None and prior["complete"]:
                continue
            value = tool.get("result")
            if value is not None and not isinstance(value, str):
                value = json.dumps(value, ensure_ascii=False, default=str)
            if prior is None or tool.get("result_observed"):
                calls[key] = {"name": tool.get("name") or (prior or {}).get("name", "unknown"),
                              "result": value, "known": value is not None,
                              "complete": bool(tool.get("result_observed"))}
        if not content.cancelled and any(not call["known"] for call in calls.values()):
            incomplete_reason = "recovery_tool_result_missing"
    tool_context = "\n\n".join(
        f"[{call['name']}]\n{call['result'] if call['known'] else '[Result unavailable in retained evidence.]'}"
        for call in calls.values()
    )
    question = next((item.tool_use_id for item in reversed(evidence.questions)
                     if not (item.live_deadline and item.status in ("answered", "skipped"))), None)
    approval = next((item.id for item in reversed(evidence.approvals)), None)
    if content is None or incomplete_reason is not None:
        status, reason = "error", "error"
        response, reasoning = row.response_text or "", row.reasoning
        usage = json.loads(row.token_usage_json) if row.token_usage_json else None
        error = f"Incomplete turn recovery: {incomplete_reason or 'recovery_content_unavailable'}."
        attachments = None
    else:
        status, reason = ("aborted", "aborted") if content.cancelled else (
            ("error", "upstream_error") if content.error_text else
            ("ok", "question" if question else "approval" if approval else "stop")
        )
        response, reasoning = content.response_text, content.reasoning or row.reasoning
        usage, error = content.token_usage, content.error_text
        attachments = list(content.attachments) or None
    if row.terminal_protocol_state == "binding":
        response, reasoning, usage, attachments, tool_context = "", None, None, None, ""
        status, reason = "error", "error"
        error = "Incomplete turn recovery: owner lost before history binding."
        question = approval = None
    return dict(status=status, end_reason=reason, response_text=response,
                reasoning=reasoning, token_usage=usage, attachments=attachments,
                tool_context=tool_context, error_text=error,
                abort_cause=content.abort_cause if content and status == "aborted" else None,
                completion_fields={"question_id": question, "approval_id": approval,
                                   "animation": row.animation},
                audit_key="recovery", audit={"by": "terminal_protocol", "cause": incomplete_reason,
                                              "abort_cause": content.abort_cause if content else None})
