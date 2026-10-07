"""Server-owned orchestration approval requests (TASK-1030).

An orchestration request is a human decision gate owned by a durable
server-side coordinator (the deploy step of a BawtHub release). The row shares
``tool_approval_requests`` with harness/MCP approvals so it appears in the same
Approvals queue and resolves through the same first-decision-wins
``resolve_request`` CAS. Unlike those kinds it grants nothing and dispatches no
agent continuation: ``grant_state`` is ``not_applicable`` (never claimable by
``claim_client_grant``) and ``continuation_owner`` is ``none``. The
coordinator observes the decision and acts on it.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlmodel import Session

from .approval_models import (
    ApprovalPersistError,
    KIND_ORCHESTRATION,
    REQ_PENDING,
    ToolApprovalRequest,
)

logger = logging.getLogger(__name__)

ORCHESTRATION_GRANT_STATE = "not_applicable"


class OrchestrationApprovalStoreMixin:
    """Insert-if-absent persistence for coordinator-owned approval gates."""

    def record_orchestration_request(
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
        severity: str,
        prompt: str,
        operations_snapshot: dict[str, Any] | None = None,
        caller_context: dict[str, Any] | None = None,
        session_key: str | None = None,
    ) -> tuple[ToolApprovalRequest, bool]:
        """Return ``(row, created)``; an existing id is returned unchanged.

        The stored row is authoritative: a coordinator that crashed after the
        insert and recomputed its arguments must act on what the operator saw,
        not on the recomputation. Raises ``ApprovalPersistError`` when the
        request cannot be durably committed.
        """
        if self.engine is None:
            raise ApprovalPersistError(
                f"approval store has no DB engine; cannot persist request {request_id}"
            )
        try:
            with Session(self.engine) as session:
                existing = session.get(ToolApprovalRequest, request_id)
                if existing is not None:
                    return existing, False
                row = ToolApprovalRequest(
                    id=request_id,
                    bot_id=bot_id,
                    user_id=user_id,
                    turn_id=turn_id,
                    session_key=session_key or None,
                    backend=backend,
                    tool_name=tool_name,
                    tool_arguments_json=json.dumps(
                        tool_arguments, ensure_ascii=False, sort_keys=True, default=str
                    ),
                    subject=subject,
                    grant_key=grant_key,
                    policy_id=None,
                    severity=severity,
                    prompt=prompt,
                    status=REQ_PENDING,
                    request_kind=KIND_ORCHESTRATION,
                    continuation_owner="none",
                    grant_state=ORCHESTRATION_GRANT_STATE,
                    operations_snapshot_json=json.dumps(
                        operations_snapshot, ensure_ascii=False, sort_keys=True, default=str
                    )
                    if operations_snapshot is not None
                    else None,
                    caller_context_json=json.dumps(caller_context, sort_keys=True, default=str)
                    if caller_context
                    else None,
                )
                session.add(row)
                session.commit()
                session.refresh(row)
                return row, True
        except IntegrityError:
            # A concurrent recorder won the insert; theirs is authoritative.
            with Session(self.engine) as session:
                existing = session.get(ToolApprovalRequest, request_id)
            if existing is None:
                raise ApprovalPersistError(
                    f"orchestration request {request_id} conflicted but is missing"
                )
            return existing, False
        except ApprovalPersistError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to record orchestration request id=%s", request_id)
            raise ApprovalPersistError(
                f"insert failed for orchestration request {request_id}: {exc}"
            ) from exc


__all__ = ["ORCHESTRATION_GRANT_STATE", "OrchestrationApprovalStoreMixin"]
