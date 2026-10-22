"""MCP approval execution claims and continuation outbox persistence."""

from __future__ import annotations

import json
import logging
import re
from datetime import timedelta
from hashlib import sha256
from typing import Any

from sqlalchemy import and_, case, func, not_, or_, update
from sqlalchemy import text as sa_text
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from .approval_models import (
    ApprovalDuplicatePending,
    ApprovalPersistError,
    ApprovalStoreUnavailable,
    CONT_DELIVERED,
    CONT_DISPATCHING,
    CONT_FAILED,
    CONT_NOT_NEEDED,
    CONT_PENDING,
    CONT_SUPPRESSED,
    CONTINUATION_LEASE_SECONDS,
    EXEC_FAILED,
    EXEC_PENDING,
    EXEC_RUNNING,
    EXEC_SKIPPED,
    EXEC_SUCCEEDED,
    ExecutionDeadlinePassed,
    KIND_MCP,
    OPS_RUN_TOOL,
    REQ_APPROVED,
    REQ_CANCELLED,
    REQ_PENDING,
    ToolApprovalRequest,
    _new_id,
    _utcnow,
)


logger = logging.getLogger(__name__)


def _unfinished_execution(request_id: str, *, state: str, claim_token: str | None):
    """This row, still without a result, in ``state`` (and under this claim)."""
    conditions = [
        ToolApprovalRequest.id == request_id,
        ToolApprovalRequest.result_json.is_(None),
        ToolApprovalRequest.execution_state == state,
    ]
    if claim_token is not None:
        conditions.append(ToolApprovalRequest.execution_claim_token == claim_token)
    return conditions


def requeued_continuation_state():
    """The state for every dispatching → pending writer (TASK-1232 S1).

    Evaluated in the UPDATE against the current row, so a purge that committed
    after the writer's read still wins: a cancelled request is suppressed,
    never handed back to the outbox.
    """
    return case((ToolApprovalRequest.status == REQ_CANCELLED, CONT_SUPPRESSED), else_=CONT_PENDING)


def _awaiting_execution(*, lease_seconds: int):
    """No result yet, and unclaimed or its claim lease has expired.

    The one predicate shared by the execution claim, app recovery and ops
    admission, so the three can never disagree about what is claimable.
    """
    stale_before = _utcnow() - timedelta(seconds=max(0, lease_seconds))
    return and_(
        ToolApprovalRequest.result_json.is_(None),
        or_(
            ToolApprovalRequest.execution_state == EXEC_PENDING,
            and_(
                ToolApprovalRequest.execution_state == EXEC_RUNNING,
                or_(
                    ToolApprovalRequest.execution_started_at.is_(None),
                    ToolApprovalRequest.execution_started_at <= stale_before,
                ),
            ),
        ),
    )


def _approved_ops_run():
    """SQL form of ``is_approved_ops_run``: the ops reconciler admits these (TASK-1220)."""
    return and_(
        ToolApprovalRequest.request_kind == KIND_MCP,
        ToolApprovalRequest.status == REQ_APPROVED,
        ToolApprovalRequest.tool_name == OPS_RUN_TOOL,
    )


_SENTINEL_IDENTITIES = {"unknown", "none", "null", "undefined"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _required_mcp_identity(value: Any, field: str, *, max_length: int = 256) -> str:
    normalized = str(value or "").strip()
    if (
        not normalized
        or len(normalized) > max_length
        or normalized.lower() in _SENTINEL_IDENTITIES
        or any(ord(char) < 32 or ord(char) == 127 for char in normalized)
    ):
        raise ApprovalPersistError(f"MCP approval {field} is missing or invalid")
    return normalized


def _validate_mcp_caller_context(
    *,
    caller_context_json: str | None,
    bot_id: str,
    user_id: str,
    turn_id: str,
    trigger_message_id: str | None,
    session_key: str | None,
    backend: str,
    tool_use_id: str | None,
) -> str:
    try:
        context = json.loads(
            _required_mcp_identity(
                caller_context_json, "caller_context_json", max_length=16_384
            )
        )
    except (json.JSONDecodeError, TypeError) as error:
        raise ApprovalPersistError(
            "MCP approval caller context is malformed"
        ) from error
    if not isinstance(context, dict):
        raise ApprovalPersistError("MCP approval caller context must be an object")
    expected = {
        "bot_id": bot_id,
        "user_id": user_id,
        "turn_id": turn_id,
        "trigger_message_id": trigger_message_id,
        "session_key": session_key,
        "backend": backend,
        "tool_use_id": tool_use_id,
    }
    for field, value in expected.items():
        canonical = _required_mcp_identity(value, field)
        if (
            _required_mcp_identity(context.get(field), f"caller_context.{field}")
            != canonical
        ):
            raise ApprovalPersistError(
                f"MCP approval caller context {field} does not match the row"
            )
    for field in ("session_id", "agent_request_id", "issued_at"):
        _required_mcp_identity(context.get(field), f"caller_context.{field}")
    return json.dumps(context, ensure_ascii=False, sort_keys=True)


def _continuation_id(request_id):
    return "approval-cont-" + sha256(request_id.encode()).hexdigest()[:32]


def _mcp_record_result(row: ToolApprovalRequest, created: bool, with_created: bool):
    return (row, created) if with_created else row


def _same_mcp_invocation(
    row: ToolApprovalRequest,
    *,
    tool_use_id: str,
    invocation_hash: str,
    bot_id: str,
    user_id: str,
    turn_id: str,
    backend: str,
) -> bool:
    return (
        row.request_kind == KIND_MCP
        and row.tool_use_id == tool_use_id
        and row.invocation_hash == invocation_hash
        and row.bot_id == bot_id
        and row.user_id == user_id
        and row.turn_id == turn_id
        and row.backend == backend
    )


def _lock_mcp_invocation(
    session: Session, bot_id: str, user_id: str, invocation_hash: str
) -> None:
    """Transaction-scoped lock on one caller's identical invocation (Postgres).

    SQLite (tests) serializes writers on its own, so there is nothing to take.
    """
    if session.get_bind().dialect.name != "postgresql":
        return
    session.execute(
        sa_text("SELECT pg_advisory_xact_lock(hashtext(:key))"),
        {"key": f"mcp-approval\x1f{bot_id}\x1f{user_id}\x1f{invocation_hash}"},
    )


def _pending_duplicate_id(
    session: Session, bot_id: str, user_id: str, invocation_hash: str
) -> str | None:
    """Id of an identical MCP call still awaiting approval, if any.

    ``invocation_hash`` is the canonical hash of tool name + arguments, so it
    already includes a caller-supplied ``idempotency_key``.
    """
    return session.exec(
        select(ToolApprovalRequest.id)
        .where(
            ToolApprovalRequest.request_kind == KIND_MCP,
            ToolApprovalRequest.status == REQ_PENDING,
            ToolApprovalRequest.bot_id == bot_id,
            ToolApprovalRequest.user_id == user_id,
            ToolApprovalRequest.invocation_hash == invocation_hash,
        )
        .order_by(ToolApprovalRequest.created_at)
        .limit(1)
    ).first()


class McpApprovalStoreMixin:
    """Persist fenced MCP execution and server-owned continuation state."""

    def record_mcp_request(
        self,
        *,
        request_id: str,
        tool_use_id: str | None,
        mcp_server: str,
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
        invocation_hash: str,
        caller_context_json: str | None = None,
        operations_snapshot: dict[str, Any] | None = None,
        continuation_capable: bool = False,
        trigger_message_id: str | None = None,
        session_key: str | None = None,
        with_created: bool = False,
    ) -> ToolApprovalRequest | tuple[ToolApprovalRequest, bool]:
        """Persist a new MCP-kind gated request. Idempotent on request_id.

        Same durability contract as :meth:`record_request` — raises
        ``ApprovalPersistError`` on missing engine / insert failure so the MCP
        interceptor never returns ``approval_required`` to the caller unless
        the row is committed. Starts in status=REQ_PENDING (approval flow) +
        execution_state=EXEC_PENDING (MCP execution has not run yet).
        """
        request_id = _required_mcp_identity(request_id, "request_id", max_length=128)
        tool_use_id = _required_mcp_identity(tool_use_id, "tool_use_id", max_length=128)
        mcp_server = _required_mcp_identity(mcp_server, "mcp_server", max_length=128)
        bot_id = _required_mcp_identity(bot_id, "bot_id", max_length=128)
        user_id = _required_mcp_identity(user_id, "user_id", max_length=128)
        turn_id = _required_mcp_identity(turn_id, "turn_id", max_length=128)
        trigger_message_id = _required_mcp_identity(
            trigger_message_id, "trigger_message_id", max_length=128
        )
        session_key = _required_mcp_identity(session_key, "session_key", max_length=128)
        backend = _required_mcp_identity(backend, "backend", max_length=64)
        tool_name = _required_mcp_identity(tool_name, "tool_name", max_length=128)
        invocation_hash = _required_mcp_identity(
            invocation_hash, "invocation_hash", max_length=64
        )
        if not _SHA256_RE.fullmatch(invocation_hash):
            raise ApprovalPersistError("MCP approval invocation_hash is invalid")
        if not continuation_capable:
            raise ApprovalPersistError(
                "MCP approval must have a routable continuation context"
            )
        caller_context_json = _validate_mcp_caller_context(
            caller_context_json=caller_context_json,
            bot_id=bot_id,
            user_id=user_id,
            turn_id=turn_id,
            trigger_message_id=trigger_message_id,
            session_key=session_key,
            backend=backend,
            tool_use_id=tool_use_id,
        )
        if self.engine is None:
            raise ApprovalPersistError(
                f"approval store has no DB engine; cannot persist MCP request {request_id}"
            )
        try:
            with Session(self.engine) as session:
                # TASK-959: one pending approval per identical call. The lock
                # serializes concurrent identical calls through the pending
                # check + insert below; it is released at commit/rollback.
                _lock_mcp_invocation(session, bot_id, user_id, invocation_hash)
                existing = session.get(ToolApprovalRequest, request_id)
                if existing is None:
                    pending = _pending_duplicate_id(
                        session, bot_id, user_id, invocation_hash
                    )
                    if pending is not None:
                        raise ApprovalDuplicatePending(pending)
                if existing is not None:
                    if not _same_mcp_invocation(
                        existing,
                        tool_use_id=tool_use_id,
                        invocation_hash=invocation_hash,
                        bot_id=bot_id,
                        user_id=user_id,
                        turn_id=turn_id,
                        backend=backend,
                    ):
                        raise ApprovalPersistError(
                            "approval request id is already bound to a different invocation"
                        )
                    return _mcp_record_result(existing, False, with_created)
                row = ToolApprovalRequest(
                    id=request_id,
                    bot_id=bot_id,
                    user_id=user_id,
                    turn_id=turn_id,
                    trigger_message_id=trigger_message_id,
                    session_key=session_key,
                    backend=backend,
                    tool_name=tool_name,
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
                    request_kind=KIND_MCP,
                    tool_use_id=tool_use_id,
                    mcp_server=mcp_server,
                    invocation_hash=invocation_hash,
                    caller_context_json=caller_context_json,
                    operations_snapshot_json=json.dumps(
                        operations_snapshot, sort_keys=True
                    )
                    if operations_snapshot is not None
                    else None,
                    continuation_owner="server",
                    continuation_id=_continuation_id(request_id),
                    continuation_capable=bool(continuation_capable),
                    execution_state=EXEC_PENDING,
                    continuation_state=CONT_NOT_NEEDED,
                )
                session.add(row)
                session.commit()
                session.refresh(row)
                return _mcp_record_result(row, True, with_created)
        except ApprovalPersistError:
            raise
        except IntegrityError as exc:
            # A concurrent transport retry can race the optimistic lookup.
            # The primary key is the final arbiter; return the winner only
            # when it represents the exact same trusted invocation.
            with Session(self.engine) as session:
                existing = session.get(ToolApprovalRequest, request_id)
                if existing is not None and _same_mcp_invocation(
                    existing,
                    tool_use_id=tool_use_id,
                    invocation_hash=invocation_hash,
                    bot_id=bot_id,
                    user_id=user_id,
                    turn_id=turn_id,
                    backend=backend,
                ):
                    return _mcp_record_result(existing, False, with_created)
            raise ApprovalPersistError(
                f"conflicting insert for MCP approval request {request_id}"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to record MCP approval request id=%s", request_id)
            raise ApprovalPersistError(
                f"insert failed for MCP approval request {request_id}: {exc}"
            ) from exc

    def claim_mcp_execution(
        self,
        request_id: str,
        *,
        lease_seconds: int = 300,
    ) -> ToolApprovalRequest | None:
        """Atomically claim an approved MCP request for execution.

        Returns the claimed row (execution_state=EXEC_RUNNING) or None if:
          * request does not exist
          * status is not REQ_APPROVED
          * execution has already completed (EXEC_SUCCEEDED / EXEC_FAILED / EXEC_SKIPPED)
          * another worker holds a fresh RUNNING lease (started_at within lease_seconds)

        Expired generic claims become uncertain failures without replay. Only
        ops_run with an immutable snapshot may re-enter execution with the same
        durable approval idempotency key. Every claimant receives a fencing token.
        """
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval database unavailable")
        now = _utcnow()
        with Session(self.engine) as session:
            claimable = and_(
                ToolApprovalRequest.id == request_id,
                ToolApprovalRequest.status == REQ_APPROVED,
                ToolApprovalRequest.request_kind == KIND_MCP,
                _awaiting_execution(lease_seconds=lease_seconds),
            )
            stmt = (
                update(ToolApprovalRequest)
                .where(claimable)
                .values(
                    execution_state=EXEC_RUNNING,
                    execution_claim_token=_new_id(),
                    execution_started_at=now,
                    execution_attempts=ToolApprovalRequest.execution_attempts + 1,
                )
            )
            result = session.exec(stmt)
            session.commit()
            if int(result.rowcount or 0) != 1:
                return None
            row = session.get(ToolApprovalRequest, request_id)
        if row.execution_attempts > 1 and (
            row.tool_name != "ops_run" or not row.operations_snapshot_json
        ):
            self.complete_mcp_execution(
                request_id,
                result_json=json.dumps(
                    {
                        "status": "failed",
                        "uncertain": True,
                        "error": "Execution lease expired. Side effects may have occurred; manual reconciliation required. Do not replay.",
                    }
                ),
                is_error=True,
                error="Uncertain execution; manual reconciliation required",
                claim_token=row.execution_claim_token,
            )
            return None
        return row

    def execution_fence(
        self,
        request_id: str,
        claim_token: str,
        *,
        approved_within: timedelta | None = None,
        clock=_utcnow,
    ):
        """Guard for side effects that may start only while this claim is held.

        Returns ``fence(session)``. Inside the caller's transaction it renews the
        claim's lease. With ``approved_within`` it also requires the approval
        (``resolved_at``, else ``created_at``) to be no older than that window,
        measured when the fence runs.
        - Claim superseded or completed: raises ``ApprovalPersistError``.
        - Claim held but window closed: raises ``ExecutionDeadlinePassed``.
        A newer claimant cannot re-claim until the transaction commits, and it
        sees what was committed. So neither a stale claimant nor a stalled lone
        one can start work after its claim or deadline is gone (TASK-1220).
        """

        def fence(session: Session) -> None:
            # A purged request (TASK-1232 B1) is cancelled: no work may start.
            held = [*_unfinished_execution(request_id, state=EXEC_RUNNING, claim_token=claim_token),
                    ToolApprovalRequest.status != REQ_CANCELLED]
            in_window = (
                [func.coalesce(ToolApprovalRequest.resolved_at, ToolApprovalRequest.created_at)
                 >= clock() - approved_within]
                if approved_within is not None else []
            )
            renewed = session.exec(
                update(ToolApprovalRequest)
                .where(*held, *in_window)
                .values(execution_started_at=_utcnow())
            )
            if int(renewed.rowcount or 0) == 1:
                return
            still_held = session.exec(
                select(ToolApprovalRequest.id).where(*held)
            ).first()
            if still_held is not None:
                raise ExecutionDeadlinePassed("Approval execution window closed; work not started")
            raise ApprovalPersistError("Execution claim lost; work not started")

        return fence

    def complete_mcp_execution(
        self,
        request_id: str,
        *,
        result_json: str,
        is_error: bool,
        error: str | None = None,
        skipped: bool = False,
        claim_token: str | None = None,
    ) -> ToolApprovalRequest | None:
        """CAS result and outbox enqueue in ONE transaction; fence stale workers."""
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval database unavailable")
        with Session(self.engine) as session:
            row = session.get(ToolApprovalRequest, request_id)
            if row is None:
                return None
            if row.result_json is not None:
                return row
            if skipped and row.status == REQ_APPROVED:
                return row
            now = _utcnow()
            conditions = _unfinished_execution(
                request_id,
                state=EXEC_PENDING if skipped else EXEC_RUNNING,
                claim_token=claim_token,
            )
            # TASK-1232 B1: decided in the UPDATE from the current row, never
            # from the snapshot above, so a purge between the two is not undone.
            keep = or_(
                ToolApprovalRequest.status == REQ_CANCELLED,
                ToolApprovalRequest.continuation_state.in_([CONT_SUPPRESSED, CONT_DELIVERED, CONT_FAILED]),
            )
            enqueue = ToolApprovalRequest.continuation_capable.is_(True)
            result = session.exec(
                update(ToolApprovalRequest)
                .where(*conditions)
                .values(
                    execution_state=EXEC_SKIPPED
                    if skipped
                    else (EXEC_FAILED if is_error else EXEC_SUCCEEDED),
                    execution_finished_at=now,
                    result_json=result_json,
                    result_is_error=bool(is_error),
                    execution_error=error,
                    execution_claim_token=None,
                    continuation_state=case(
                        (keep, ToolApprovalRequest.continuation_state),
                        (enqueue, CONT_PENDING),
                        else_=CONT_NOT_NEEDED,
                    ),
                    continuation_next_attempt_at=case(
                        (keep, ToolApprovalRequest.continuation_next_attempt_at),
                        (enqueue, now),
                        else_=None,
                    ),
                    continuation_id=row.continuation_id or _continuation_id(request_id),
                )
            )
            session.commit()
            session.expire_all()
            if int(result.rowcount or 0) != 1:
                raise ApprovalPersistError(
                    "Execution claim lost; result not overwritten"
                )
            return session.get(ToolApprovalRequest, request_id)

    def enqueue_continuation(
        self,
        request_id: str,
    ) -> ToolApprovalRequest | None:
        """Mark a completed MCP execution as CONT_PENDING for the outbox worker.

        No-op if the row is not continuation_capable, has no terminal execution
        result, or is already in a continuation lifecycle. Idempotent.
        """
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval database unavailable")
        with Session(self.engine) as session:
            # One CAS on the current row (TASK-1232 B1): a read-then-write
            # here could re-enqueue a request purged between the two.
            session.exec(
                update(ToolApprovalRequest)
                .where(
                    ToolApprovalRequest.id == request_id,
                    ToolApprovalRequest.continuation_state == CONT_NOT_NEEDED,
                    ToolApprovalRequest.status != REQ_CANCELLED,
                    ToolApprovalRequest.continuation_capable.is_(True),
                    ToolApprovalRequest.execution_state.in_([EXEC_SUCCEEDED, EXEC_FAILED, EXEC_SKIPPED]),
                )
                .values(
                    continuation_id=func.coalesce(
                        ToolApprovalRequest.continuation_id, _continuation_id(request_id)),
                    continuation_state=CONT_PENDING,
                    continuation_next_attempt_at=_utcnow(),
                )
            )
            session.commit()
            return session.get(ToolApprovalRequest, request_id)

    def claim_continuation(
        self,
        request_id: str,
        *,
        lease_seconds: int = CONTINUATION_LEASE_SECONDS,
    ) -> ToolApprovalRequest | None:
        """Atomically transition CONT_PENDING → CONT_DISPATCHING for the worker.

        Same lease semantics as ``claim_mcp_execution``: a DISPATCHING row with
        an expired lease and no continuation_delivered_at may be reclaimed.
        """
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval database unavailable")
        now = _utcnow()
        stale_before = now - timedelta(seconds=max(0, lease_seconds))
        with Session(self.engine) as session:
            claimable = and_(
                ToolApprovalRequest.id == request_id,
                ToolApprovalRequest.continuation_delivered_at.is_(None),
                # Purged (TASK-1232 B1). ``suppressed`` is already excluded:
                # only pending or dispatching rows are claimable.
                ToolApprovalRequest.status != REQ_CANCELLED,
                or_(
                    and_(
                        ToolApprovalRequest.continuation_state == CONT_PENDING,
                        or_(
                            ToolApprovalRequest.continuation_next_attempt_at.is_(None),
                            ToolApprovalRequest.continuation_next_attempt_at <= now,
                        ),
                    ),
                    and_(
                        ToolApprovalRequest.continuation_state == CONT_DISPATCHING,
                        or_(
                            ToolApprovalRequest.continuation_next_attempt_at.is_(None),
                            ToolApprovalRequest.continuation_next_attempt_at
                            <= stale_before,
                        ),
                    ),
                ),
            )
            stmt = (
                update(ToolApprovalRequest)
                .where(claimable)
                .values(
                    continuation_state=CONT_DISPATCHING,
                    continuation_claim_token=_new_id(),
                    continuation_id=_continuation_id(request_id),
                    continuation_next_attempt_at=now,
                    continuation_attempts=ToolApprovalRequest.continuation_attempts + 1,
                )
            )
            result = session.exec(stmt)
            session.commit()
            if int(result.rowcount or 0) != 1:
                return None
            return session.get(ToolApprovalRequest, request_id)

    def mark_continuation_delivered(
        self, request_id: str, *, claim_token: str | None = None
    ):
        return self._finish_continuation(request_id, claim_token=claim_token)

    def mark_continuation_failed(
        self,
        request_id: str,
        *,
        error: str,
        max_attempts: int = 5,
        backoff_seconds: int = 30,
        claim_token: str | None = None,
    ):
        return self._finish_continuation(
            request_id,
            claim_token=claim_token,
            error=error,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
        )

    def _finish_continuation(
        self,
        request_id,
        *,
        claim_token=None,
        error=None,
        max_attempts=5,
        backoff_seconds=30,
    ):
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval database unavailable")
        with Session(self.engine) as session:
            row = session.get(ToolApprovalRequest, request_id)
            if row is None or row.continuation_state != CONT_DISPATCHING:
                return row
            expected = (
                claim_token if claim_token is not None else row.continuation_claim_token
            )
            now = _utcnow()
            failed = error is not None
            terminal_failure = failed and row.continuation_attempts >= max_attempts
            values = dict(
                continuation_state=(CONT_FAILED if terminal_failure else requeued_continuation_state())
                if failed
                else CONT_DELIVERED,
                continuation_last_error=str(error)[:2000] if failed else None,
                continuation_delivered_at=None if failed else now,
                continuation_claim_token=None,
                continuation_next_attempt_at=(
                    now
                    + timedelta(
                        seconds=min(
                            backoff_seconds
                            * 2 ** max(0, row.continuation_attempts - 1),
                            600,
                        )
                    )
                )
                if failed and not terminal_failure
                else None,
            )
            session.exec(
                update(ToolApprovalRequest)
                .where(
                    ToolApprovalRequest.id == request_id,
                    ToolApprovalRequest.continuation_state == CONT_DISPATCHING,
                    ToolApprovalRequest.continuation_claim_token == expected,
                )
                .values(**values)
            )
            session.commit()
            session.expire_all()
            return session.get(ToolApprovalRequest, request_id)

    def defer_continuation_claim(
        self,
        request_id: str,
        *,
        claim_token: str,
        reason: str,
        delay_seconds: float,
    ) -> bool:
        """Return a claim lost to target admission (TASK-1082) without failing it.

        Waiting for the target's running turn is not a delivery attempt: the
        attempt is refunded, the claim is fenced by its token, and the reason
        stays visible on the row. Mirrors delivery ``requeue(refund_attempt)``.
        """
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval database unavailable")
        with Session(self.engine) as session:
            result = session.exec(
                update(ToolApprovalRequest)
                .where(
                    ToolApprovalRequest.id == request_id,
                    ToolApprovalRequest.continuation_state == CONT_DISPATCHING,
                    ToolApprovalRequest.continuation_claim_token == claim_token,
                )
                .values(
                    continuation_state=requeued_continuation_state(),
                    continuation_claim_token=None,
                    continuation_last_error=str(reason)[:2000],
                    continuation_next_attempt_at=_utcnow()
                    + timedelta(seconds=max(0.0, delay_seconds)),
                    continuation_attempts=case(
                        (ToolApprovalRequest.continuation_attempts > 0,
                         ToolApprovalRequest.continuation_attempts - 1),
                        else_=0,
                    ),
                )
            )
            session.commit()
            return int(result.rowcount or 0) == 1

    def continuation_claim_fence(self, request_id: str, claim_token: str):
        """Fence proving ``claim_token`` still owns the continuation (TASK-1082).

        The returned callable runs inside the caller's transaction (turn
        admission shares this database via the shared engine). On PostgreSQL
        it locks the approval row FOR UPDATE, so a reclaim cannot interleave
        between the ownership check and the caller's write before commit.
        """
        params = {"id": request_id, "state": CONT_DISPATCHING, "token": claim_token}

        def holds(conn) -> bool:
            lock = " FOR UPDATE" if conn.dialect.name == "postgresql" else ""
            return conn.execute(sa_text(
                "SELECT 1 FROM tool_approval_requests WHERE id=:id"
                " AND continuation_state=:state AND continuation_claim_token=:token"
                + lock
            ), params).first() is not None

        return holds

    def renew_continuation_claim(self, request_id, claim_token):
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval database unavailable")
        with Session(self.engine) as session:
            result = session.exec(
                update(ToolApprovalRequest)
                .where(
                    ToolApprovalRequest.id == request_id,
                    ToolApprovalRequest.continuation_state == CONT_DISPATCHING,
                    ToolApprovalRequest.continuation_claim_token == claim_token,
                )
                .values(continuation_next_attempt_at=_utcnow())
            )
            session.commit()
            return int(result.rowcount or 0) == 1

    def find_pending_continuations(
        self,
        *,
        limit: int = 50,
        include_dispatching_older_than_s: int = 120,
    ) -> list[ToolApprovalRequest]:
        """Outbox worker query: continuations ready for dispatch.

        Includes CONT_PENDING rows whose next_attempt_at is due, plus any
        CONT_DISPATCHING rows whose lease expired without delivery (crash /
        app restart mid-dispatch).
        """
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval database unavailable")
        now = _utcnow()
        stale_before = now - timedelta(seconds=include_dispatching_older_than_s)
        with Session(self.engine) as session:
            stmt = (
                select(ToolApprovalRequest)
                .where(
                    (
                        (ToolApprovalRequest.continuation_state == CONT_PENDING)
                        & (
                            (ToolApprovalRequest.continuation_next_attempt_at.is_(None))
                            | (ToolApprovalRequest.continuation_next_attempt_at <= now)
                        )
                    )
                    | (
                        (ToolApprovalRequest.continuation_state == CONT_DISPATCHING)
                        & (ToolApprovalRequest.continuation_delivered_at.is_(None))
                        & (
                            ToolApprovalRequest.continuation_next_attempt_at
                            <= stale_before
                        )
                    )
                )
                .order_by(ToolApprovalRequest.continuation_next_attempt_at)
                .limit(limit)
            )
            return list(session.exec(stmt).all())

    def find_recoverable_mcp_requests(self, *, limit=20, lease_seconds=300):
        """App-side recovery: resolved MCP rows with no committed result.

        Excludes APPROVED ops_run only — the ops reconciler admits those. Denied,
        cancelled and responded ops_run rows still need the app's skip-completion
        (their receipt would otherwise strand after a resolve/complete crash).
        """
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval database unavailable")
        with Session(self.engine) as session:
            return list(
                session.exec(
                    select(ToolApprovalRequest)
                    .where(
                        ToolApprovalRequest.request_kind == KIND_MCP,
                        ToolApprovalRequest.status != REQ_PENDING,
                        _awaiting_execution(lease_seconds=lease_seconds),
                        not_(_approved_ops_run()),
                    )
                    .order_by(ToolApprovalRequest.created_at)
                    .limit(limit)
                ).all()
            )

    def find_admittable_ops_requests(self, *, limit=10, lease_seconds=300):
        """Ops admission: approved ops_run rows awaiting (re-)admission, oldest first."""
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval database unavailable")
        with Session(self.engine) as session:
            return list(
                session.exec(
                    select(ToolApprovalRequest)
                    .where(_approved_ops_run(), _awaiting_execution(lease_seconds=lease_seconds))
                    .order_by(ToolApprovalRequest.resolved_at, ToolApprovalRequest.created_at)
                    .limit(limit)
                ).all()
            )

    def oldest_unadmitted_ops_resolved_at(self):
        """When the oldest approved ops_run without a result was approved (health)."""
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval database unavailable")
        with Session(self.engine) as session:
            return session.exec(
                select(ToolApprovalRequest.resolved_at)
                .where(_approved_ops_run(), ToolApprovalRequest.result_json.is_(None))
                .order_by(ToolApprovalRequest.resolved_at)
                .limit(1)
            ).first()

    def renew_mcp_execution_claim(self, request_id, claim_token):
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval database unavailable")
        with Session(self.engine) as session:
            result = session.exec(
                update(ToolApprovalRequest)
                .where(
                    ToolApprovalRequest.id == request_id,
                    ToolApprovalRequest.execution_state == EXEC_RUNNING,
                    ToolApprovalRequest.execution_claim_token == claim_token,
                )
                .values(execution_started_at=_utcnow())
            )
            session.commit()
            return int(result.rowcount or 0) == 1

    def find_stranded_mcp_results(self, *, limit=20):
        if self.engine is None:
            raise ApprovalStoreUnavailable("Approval database unavailable")
        with Session(self.engine) as session:
            return list(
                session.exec(
                    select(ToolApprovalRequest)
                    .where(
                        ToolApprovalRequest.request_kind == KIND_MCP,
                        ToolApprovalRequest.result_json.is_not(None),
                        ToolApprovalRequest.continuation_capable == True,  # noqa: E712
                        ToolApprovalRequest.continuation_state == CONT_NOT_NEEDED,
                        ToolApprovalRequest.status != REQ_CANCELLED,
                    )
                    .limit(limit)
                ).all()
            )
