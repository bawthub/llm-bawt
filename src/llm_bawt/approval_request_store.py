"""Durable approval request lifecycle and harness audit persistence."""

from __future__ import annotations

import json
import logging
from hashlib import sha256
from typing import Any

from sqlalchemy import func, update
from sqlmodel import Session, select

from .approval_mcp_store import McpApprovalStoreMixin
from .approval_orchestration_store import OrchestrationApprovalStoreMixin
from .approval_models import (
    ApprovalPersistError,
    ApprovalStoreUnavailable,
    CONT_NOT_NEEDED,
    CONT_PENDING,
    KIND_HARNESS,
    REQ_APPROVED,
    REQ_CANCELLED,
    REQ_DENIED,
    REQ_EXPIRED,
    REQ_PENDING,
    REQ_RESPONDED,
    REQ_SUPERSEDED,
    ToolApprovalRequest,
    _utcnow,
)


logger = logging.getLogger(__name__)


def _continuation_id(request_id):
    return "approval-cont-" + sha256(request_id.encode()).hexdigest()[:32]


class ApprovalRequestStoreMixin(OrchestrationApprovalStoreMixin, McpApprovalStoreMixin):
    """Generic request audit plus legacy harness continuation persistence."""

    def record_request(
        self,
        *,
        request_id: str,
        bot_id: str,
        user_id: str,
        turn_id: str,
        backend: str,
        tool_name: str,
        tool_arguments: dict[str, Any],
        subject: str,
        grant_key: str,
        policy_id: str | None,
        severity: str,
        prompt: str,
        trigger_message_id: str | None = None,
        session_key: str | None = None,
        session_id: str | None = None,
    ) -> ToolApprovalRequest:
        """Persist a new pending approval. Idempotent on request_id.

        Returns the committed (or pre-existing) row. Raises
        ``ApprovalPersistError`` if the request cannot be durably committed —
        no DB engine, or the insert/commit failed. TASK-306 Section A: the
        single caller treats a raise as a hard, agent-visible failure and must
        NOT swallow it.
        """
        if self.engine is None:
            raise ApprovalPersistError(
                f"approval store has no DB engine; cannot persist request {request_id}"
            )
        try:
            with Session(self.engine) as session:
                existing = session.get(ToolApprovalRequest, request_id)
                if existing is not None:
                    return existing
                row = ToolApprovalRequest(
                    id=request_id,
                    bot_id=(bot_id or "unknown").strip() or "unknown",
                    user_id=(user_id or "unknown").strip() or "unknown",
                    turn_id=(turn_id or "unknown").strip() or "unknown",
                    trigger_message_id=trigger_message_id or None,
                    session_key=session_key or None,
                    caller_context_json=json.dumps({"session_id": session_id})
                    if session_id
                    else None,
                    backend=backend or "claude-code",
                    tool_name=tool_name or "",
                    tool_arguments_json=json.dumps(
                        tool_arguments
                        if isinstance(tool_arguments, dict)
                        else {"value": tool_arguments},
                        ensure_ascii=False,
                        default=str,
                    ),
                    subject=subject or "",
                    grant_key=grant_key or "",
                    policy_id=policy_id,
                    severity=severity or "medium",
                    prompt=prompt or "",
                    status=REQ_PENDING,
                )
                session.add(row)
                session.commit()
                session.refresh(row)
                return row
        except ApprovalPersistError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to record approval request id=%s", request_id)
            raise ApprovalPersistError(
                f"insert failed for approval request {request_id}: {exc}"
            ) from exc

    def get_request(self, request_id: str) -> ToolApprovalRequest | None:
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval database unavailable")
        with Session(self.engine) as session:
            return session.get(ToolApprovalRequest, request_id)

    def resolve_request(
        self,
        request_id: str,
        *,
        status: str,
        resolved_by: str | None = None,
        resolved_turn_id: str | None = None,
        message: str = "",
        continuation_owner: str | None = None,
    ) -> ToolApprovalRequest | None:
        """First terminal decision wins, including message and dispatch ownership."""
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval database unavailable")
        if status not in (
            REQ_APPROVED,
            REQ_DENIED,
            REQ_CANCELLED,
            REQ_RESPONDED,
            REQ_EXPIRED,
            REQ_SUPERSEDED,
        ):
            raise ValueError("Invalid approval resolution status")
        with Session(self.engine) as session:
            values = dict(
                status=status,
                resolved_at=_utcnow(),
                resolved_by=resolved_by,
                resolved_turn_id=resolved_turn_id,
                resolution_message=message,
            )
            if continuation_owner is not None:
                if continuation_owner not in ("server", "client", "none"):
                    raise ValueError("Invalid continuation owner")
                values["continuation_owner"] = continuation_owner
            session.exec(
                update(ToolApprovalRequest)
                .where(
                    ToolApprovalRequest.id == request_id,
                    ToolApprovalRequest.status == REQ_PENDING,
                )
                .values(**values)
            )
            session.commit()
            return session.get(ToolApprovalRequest, request_id)

    def list_requests(
        self,
        *,
        status: str | None = None,
        bot_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ToolApprovalRequest]:
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval database unavailable")
        with Session(self.engine) as session:
            stmt = select(ToolApprovalRequest)
            if status:
                stmt = stmt.where(ToolApprovalRequest.status == status)
            if bot_id:
                stmt = stmt.where(ToolApprovalRequest.bot_id == bot_id)
            stmt = (
                stmt.order_by(
                    ToolApprovalRequest.created_at.desc(), ToolApprovalRequest.id
                )
                .offset(max(0, offset))
                .limit(min(max(1, limit), 200))
            )
            return list(session.exec(stmt).all())

    def count_requests(self, *, status=None, bot_id=None):
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval database unavailable")
        with Session(self.engine) as session:
            stmt = select(func.count()).select_from(ToolApprovalRequest)
            if status:
                stmt = stmt.where(ToolApprovalRequest.status == status)
            if bot_id:
                stmt = stmt.where(ToolApprovalRequest.bot_id == bot_id)
            return session.exec(stmt).one()

    def prepare_harness_continuation(self, request_id):
        """Durably enqueue server-owned continuation without a fire-and-forget gap."""
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval database unavailable")
        with Session(self.engine) as session:
            session.exec(
                update(ToolApprovalRequest)
                .where(
                    ToolApprovalRequest.id == request_id,
                    ToolApprovalRequest.request_kind == KIND_HARNESS,
                    ToolApprovalRequest.status.in_(
                        [REQ_APPROVED, REQ_DENIED, REQ_RESPONDED]
                    ),
                    ToolApprovalRequest.continuation_owner == "server",
                    ToolApprovalRequest.continuation_state == CONT_NOT_NEEDED,
                )
                .values(
                    continuation_state=CONT_PENDING,
                    continuation_next_attempt_at=_utcnow(),
                    continuation_id=_continuation_id(request_id),
                )
            )
            session.commit()
            return session.get(ToolApprovalRequest, request_id)

    def set_grant_state(self, request_id, state, error=None):
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval database unavailable")
        with Session(self.engine) as session:
            allowed_prior = (
                ["pending", "failed"] if state == "failed" else ["dispatching"]
            )
            session.exec(
                update(ToolApprovalRequest)
                .where(
                    ToolApprovalRequest.id == request_id,
                    ToolApprovalRequest.grant_state.in_(allowed_prior),
                )
                .values(grant_state=state, grant_error=error)
            )
            session.commit()

    def claim_client_grant(self, request_id):
        """One publisher; an interrupted send is uncertain, never silently re-granted."""
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval database unavailable")
        with Session(self.engine) as session:
            result = session.exec(
                update(ToolApprovalRequest)
                .where(
                    ToolApprovalRequest.id == request_id,
                    ToolApprovalRequest.grant_state.in_(["pending", "failed"]),
                )
                .values(grant_state="dispatching", grant_error=None)
            )
            session.commit()
            return int(result.rowcount or 0) == 1

    def find_stranded_harness_continuations(self, *, limit=20):
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval database unavailable")
        with Session(self.engine) as session:
            return list(
                session.exec(
                    select(ToolApprovalRequest)
                    .where(
                        ToolApprovalRequest.request_kind == KIND_HARNESS,
                        ToolApprovalRequest.continuation_owner == "server",
                        ToolApprovalRequest.continuation_state == CONT_NOT_NEEDED,
                        ToolApprovalRequest.status.in_(
                            [REQ_APPROVED, REQ_DENIED, REQ_RESPONDED]
                        ),
                    )
                    .limit(limit)
                ).all()
            )
