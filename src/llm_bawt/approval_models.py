"""Storage + audit for approval-gated tool policies (TASK-289, TASK-290).

Source of truth for the feature lives here, in llm-bawt. Two tables:

* ``tool_approval_policies`` — the operator-configured rules. Compiled into the
  pure :class:`agent_bridge.approval.ApprovalPolicy` bundle that bridges fetch
  over HTTP and evaluate in their per-tool permission hook.
* ``tool_approval_requests`` — one row per gated tool call: full audit trail of
  what was asked, which policy matched, and how it resolved. Mirrors the
  ``chat_pending_questions`` design (TASK-269) so the deny→resolve→continuation
  lifecycle is durable and idempotent.

Evaluation semantics are NOT here — they're in the pure engine. This module is
storage, compilation, and the request state machine only.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text
from sqlmodel import Field, SQLModel

from agent_bridge.approval import (
    ApprovalPolicy,
    MatcherType,
    PolicyAction,
    Severity,
    humanize_subject,
)


logger = logging.getLogger(__name__)


class ApprovalStoreUnavailable(RuntimeError):
    """Policy source unavailable; never equivalent to an empty policy bundle."""


class ApprovalPersistError(RuntimeError):
    """A tool approval request could not be durably persisted.

    TASK-306 Section A: callers that gate a tool on approval require a
    *confirmed* commit. This is raised when the row cannot be written (no DB
    engine, or the insert/commit failed). It must be surfaced honestly to the
    agent/user — never swallowed — because a swallowed failure means the user
    never sees the approval and the agent proceeds on a false premise.
    """


class ApprovalDuplicatePending(ApprovalPersistError):
    """An identical MCP call already has an approval awaiting a decision.

    TASK-959: the caller (same bot + user, same tool + canonical args) re-sent
    a gated call while the first is still pending. No new approval is created;
    the caller must wait for ``existing_request_id``. Subclasses
    ``ApprovalPersistError`` so any caller that does not handle it explicitly
    still fails closed (nothing executes, nothing new is persisted).
    """

    def __init__(self, existing_request_id: str):
        super().__init__(
            f"identical MCP call already awaiting approval: {existing_request_id}"
        )
        self.existing_request_id = existing_request_id


def _new_id() -> str:
    return uuid.uuid4().hex


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_aware_utc(dt: datetime | None) -> datetime | None:
    """Coerce a possibly-naive datetime to tz-aware UTC.

    Postgres round-trips ``TIMESTAMP WITH TIME ZONE`` as aware, but SQLite
    (test env) strips tzinfo on read. Comparisons of ``_utcnow()`` (aware)
    against a stored value would then raise ``TypeError``. Coerce here so
    the lease/backoff math works uniformly.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


# Request lifecycle states.
REQ_PENDING = "pending"
REQ_APPROVED = "approved"
REQ_DENIED = "denied"
# Cancelled = user dismissed the request WITHOUT warning the agent. Unlike
# `denied` (which dispatches a "you were refused" continuation that costs the
# agent tokens to acknowledge), cancel is silent: no grant, no continuation.
REQ_CANCELLED = "cancelled"
# Responded = user declined to run the tool but sent the agent their own
# guidance (e.g. correcting a false-positive gate) instead of the canned deny.
# Like deny: no grant, tool not run — but the continuation is user-authored.
REQ_RESPONDED = "responded"
REQ_EXPIRED = "expired"
REQ_SUPERSEDED = "superseded"

# TASK-639: request kind. "harness" = classic bridge-hook approvals (Bash and
# other bridge-layer gated tools; the client dispatches a one-shot grant +
# continuation). "mcp" = server-side approval interception at the BawtHub
# MCP server; the approved tool is executed under a fenced server-side claim and the
# result is delivered back through a durable continuation outbox — the agent
# never re-issues the call. Default stays "harness" so existing rows are
# unambiguous.
KIND_HARNESS = "harness"
KIND_MCP = "mcp"
# TASK-1030: a server-owned orchestration decision (e.g. the deploy step of a
# durable BawtHub release). The coordinator that recorded it observes the
# decision and acts; resolving it grants nothing and dispatches no agent
# continuation (continuation_owner="none").
KIND_ORCHESTRATION = "orchestration"

# TASK-639: execution state machine for MCP-kind approvals. Harness-kind rows
# leave this at EXEC_NOT_APPLICABLE — execution happens on the CLIENT after
# the one-shot grant lands.
EXEC_NOT_APPLICABLE = "not_applicable"
EXEC_PENDING = "pending"
EXEC_RUNNING = "running"
EXEC_SUCCEEDED = "succeeded"
EXEC_FAILED = "failed"
EXEC_SKIPPED = "skipped"

# TASK-639: continuation outbox state. Harness-kind = CONT_NOT_NEEDED (client
# dispatches its own continuation). MCP-kind = CONT_PENDING once the tool has
# executed; a lifespan worker moves it through DISPATCHING → DELIVERED (or
# FAILED with backoff).
CONT_NOT_NEEDED = "not_needed"
CONT_PENDING = "pending"
CONT_DISPATCHING = "dispatching"
CONT_DELIVERED = "delivered"
CONT_FAILED = "failed"


class ToolApprovalPolicy(SQLModel, table=True):
    """One operator-configured approval rule. Compiles to an ApprovalPolicy."""

    __tablename__ = "tool_approval_policies"

    id: str = Field(
        default_factory=_new_id,
        sa_column=Column(String(64), primary_key=True),
    )
    enabled: bool = Field(
        default=True, sa_column=Column(Boolean, nullable=False, index=True)
    )
    # "*" = any bridge, else a backend name ("claude-code", "codex", "openclaw").
    backend_scope: str = Field(
        default="*", sa_column=Column(String(64), nullable=False, index=True)
    )
    # "*" = any tool, else a tool name ("Bash", "Write", …). MCP-tail aware.
    tool_name: str = Field(
        default="*", sa_column=Column(String(128), nullable=False, index=True)
    )
    matcher_type: str = Field(default="always", sa_column=Column(String(16), nullable=False))
    pattern: str = Field(default="", sa_column=Column(Text, nullable=False))
    # Which tool-input field to derive the subject from ("" = per-tool default,
    # "*" = whole input JSON).
    field: str = Field(default="", sa_column=Column(String(128), nullable=False))
    action: str = Field(
        default="require_approval", sa_column=Column(String(24), nullable=False)
    )
    severity: str = Field(default="medium", sa_column=Column(String(16), nullable=False))
    category: str | None = Field(default=None, sa_column=Column(String(64), nullable=True))
    approval_prompt: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    # Lower = evaluated first (first match wins). Named order_index — ``order`` is
    # a SQL reserved word.
    order_index: int = Field(default=100, sa_column=Column(Integer, nullable=False, index=True))
    # Bumped on every update — gives the bundle a per-row revision for debugging
    # and lets the UI show "version N" (TASK-289 versioning semantics).
    version: int = Field(default=1, sa_column=Column(Integer, nullable=False))
    created_at: datetime = Field(
        default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )
    updated_at: datetime = Field(
        default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )
    created_by: str | None = Field(default=None, sa_column=Column(String(128), nullable=True))
    updated_by: str | None = Field(default=None, sa_column=Column(String(128), nullable=True))

    def to_policy(self) -> ApprovalPolicy:
        """Compile this row into the pure evaluation dataclass."""
        return ApprovalPolicy(
            id=self.id,
            backend_scope=self.backend_scope or "*",
            tool_name=self.tool_name or "*",
            matcher_type=MatcherType.coerce(self.matcher_type),
            pattern=self.pattern or "",
            field=self.field or "",
            action=PolicyAction.coerce(self.action),
            severity=Severity.coerce(self.severity),
            category=self.category,
            approval_prompt=self.approval_prompt,
            order=self.order_index,
            enabled=self.enabled,
            version=self.version,
        )

    def to_api(self) -> dict[str, Any]:
        """Full row as a JSON-able dict for the admin API."""
        return {
            "id": self.id,
            "enabled": self.enabled,
            "backend_scope": self.backend_scope,
            "tool_name": self.tool_name,
            "matcher_type": self.matcher_type,
            "pattern": self.pattern,
            "field": self.field,
            "action": self.action,
            "severity": self.severity,
            "category": self.category,
            "approval_prompt": self.approval_prompt,
            "order": self.order_index,
            "version": self.version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "created_by": self.created_by,
            "updated_by": self.updated_by,
        }


class ToolApprovalRequest(SQLModel, table=True):
    """Durable audit + state for one gated tool call (mirrors PendingQuestion)."""

    __tablename__ = "tool_approval_requests"

    # SDK tool_use id doubles as PK so a duplicate APPROVAL_REQUIRED event
    # (Redis replay / multi-tab race) is idempotent.
    id: str = Field(sa_column=Column(String(128), primary_key=True))
    created_at: datetime = Field(
        default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )
    bot_id: str = Field(sa_column=Column(String(128), nullable=False, index=True))
    user_id: str = Field(sa_column=Column(String(128), nullable=False, index=True))
    turn_id: str = Field(sa_column=Column(String(128), nullable=False, index=True))
    trigger_message_id: str | None = Field(
        default=None, sa_column=Column(String(128), nullable=True, index=True)
    )
    session_key: str | None = Field(default=None, sa_column=Column(String(128), nullable=True))
    backend: str = Field(default="claude-code", sa_column=Column(String(64), nullable=False))
    tool_name: str = Field(sa_column=Column(String(128), nullable=False))
    tool_arguments_json: str = Field(sa_column=Column(Text, nullable=False))
    subject: str = Field(sa_column=Column(Text, nullable=False))
    grant_key: str = Field(sa_column=Column(String(64), nullable=False, index=True))
    policy_id: str | None = Field(default=None, sa_column=Column(String(64), nullable=True, index=True))
    severity: str = Field(default="medium", sa_column=Column(String(16), nullable=False))
    prompt: str = Field(default="", sa_column=Column(Text, nullable=False))
    # pending → approved | denied | expired | superseded
    status: str = Field(default=REQ_PENDING, sa_column=Column(String(24), nullable=False, index=True))
    resolved_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    resolved_by: str | None = Field(default=None, sa_column=Column(String(128), nullable=True))
    resolved_turn_id: str | None = Field(default=None, sa_column=Column(String(128), nullable=True))

    resolution_message: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    continuation_owner: str = Field(default="client", sa_column=Column(String(16), nullable=False, server_default="client"))
    operations_snapshot_json: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    execution_claim_token: str | None = Field(default=None, sa_column=Column(String(64), nullable=True))
    continuation_claim_token: str | None = Field(default=None, sa_column=Column(String(64), nullable=True))
    continuation_id: str | None = Field(default=None, sa_column=Column(String(128), nullable=True))
    grant_state: str = Field(default="pending", sa_column=Column(String(24), nullable=False, server_default="pending"))
    grant_error: str | None = Field(default=None, sa_column=Column(Text, nullable=True))

    # TASK-639 --- MCP-kind extensions --------------------------------------
    # Two lifecycles share this table: the classic "harness" flow (Bash and
    # other bridge-hook gated tools; client re-issues on approve) and the new
    # "mcp" flow (BawtHub MCP server intercepts; server claims the stored invocation
    # on approve; result delivered via the durable continuation outbox).
    request_kind: str = Field(
        default=KIND_HARNESS,
        sa_column=Column(String(16), nullable=False, index=True, server_default=KIND_HARNESS),
    )
    # For MCP-kind the row `id` is an app-generated approval-request id; the
    # SDK's tool_use_id (used by the client harness to correlate a resumed
    # ToolResultBlock) is captured separately when the caller-context header
    # provided it. For harness-kind this stays None (their `id` IS the SDK
    # tool_use_id, per pre-TASK-639 invariant).
    tool_use_id: str | None = Field(
        default=None, sa_column=Column(String(128), nullable=True, index=True)
    )
    # The BawtHub MCP server that received the intercepted call (e.g. "bawthub").
    # None for harness-kind.
    mcp_server: str | None = Field(default=None, sa_column=Column(String(128), nullable=True))
    # sha256(tool_name + canonical(args_json)) — used to bind the internal
    # signed approval-bypass to the exact stored invocation so a model-forged
    # bypass header can't swap args at execute time.
    invocation_hash: str | None = Field(
        default=None, sa_column=Column(String(64), nullable=True)
    )
    # Verified server-owned caller context (bot/user/turn/trigger/session,
    # backend, tool_use_id). Claude supplies an exact per-call stamp; Codex
    # supplies a signed request header bound to the MCP protocol request id.
    caller_context_json: str | None = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    # Pending MCP approvals are only valid for routable interactive turns.
    # Raw MCP clients may call non-gated tools but fail closed when a policy
    # requires approval.
    continuation_capable: bool = Field(
        default=False,
        sa_column=Column(Boolean, nullable=False, server_default="false"),
    )

    # ---- MCP execution state machine (only meaningful for KIND_MCP rows) ---
    execution_state: str = Field(
        default=EXEC_NOT_APPLICABLE,
        sa_column=Column(String(16), nullable=False, server_default=EXEC_NOT_APPLICABLE, index=True),
    )
    execution_attempts: int = Field(
        default=0, sa_column=Column(Integer, nullable=False, server_default="0"),
    )
    execution_started_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    execution_finished_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    execution_error: str | None = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    # Normalized MCP result payload (JSON-serialized). For an ordinary success
    # this is the tool's return value; for a refusal/deny/respond it is a
    # structured refusal envelope. Never contains an inflight/pending marker.
    result_json: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    result_is_error: bool | None = Field(
        default=None, sa_column=Column(Boolean, nullable=True),
    )

    # ---- continuation outbox state (only meaningful for KIND_MCP rows) -----
    continuation_state: str = Field(
        default=CONT_NOT_NEEDED,
        sa_column=Column(String(16), nullable=False, server_default=CONT_NOT_NEEDED, index=True),
    )
    continuation_attempts: int = Field(
        default=0, sa_column=Column(Integer, nullable=False, server_default="0"),
    )
    continuation_last_error: str | None = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    continuation_next_attempt_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True, index=True)
    )
    continuation_delivered_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )

    def to_api(self) -> dict[str, Any]:
        try:
            args = json.loads(self.tool_arguments_json) if self.tool_arguments_json else {}
        except (json.JSONDecodeError, TypeError):
            args = {}
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "bot_id": self.bot_id,
            "user_id": self.user_id,
            "turn_id": self.turn_id,
            "trigger_message_id": self.trigger_message_id,
            "session_key": self.session_key,
            "backend": self.backend,
            "tool_name": self.tool_name,
            "tool_arguments": args,
            "subject": self.subject,
            "label": humanize_subject(self.subject or ""),
            "grant_key": self.grant_key,
            "policy_id": self.policy_id,
            "severity": self.severity,
            "prompt": self.prompt,
            "status": self.status,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolved_by": self.resolved_by,
            "resolution_message": self.resolution_message,
            "continuation_owner": self.continuation_owner,
            "continuation_id": self.continuation_id,
            "continuation_last_error": self.continuation_last_error,
            "grant_state": self.grant_state,
            "grant_error": self.grant_error,
            "operations_snapshot": json.loads(self.operations_snapshot_json) if self.operations_snapshot_json else None,
            "resolved_turn_id": self.resolved_turn_id,
            # TASK-639 — MCP-kind fields. Present on every row (server_default
            # backfills), so ApprovalCard hydration stays field-compatible with
            # pre-TASK-639 harness rows.
            "request_kind": self.request_kind,
            "tool_use_id": self.tool_use_id,
            "mcp_server": self.mcp_server,
            "invocation_hash": self.invocation_hash,
            "continuation_capable": bool(self.continuation_capable),
            "execution_state": self.execution_state,
            "execution_attempts": int(self.execution_attempts or 0),
            "execution_started_at": (
                self.execution_started_at.isoformat() if self.execution_started_at else None
            ),
            "execution_finished_at": (
                self.execution_finished_at.isoformat() if self.execution_finished_at else None
            ),
            "execution_error": self.execution_error,
            "result_is_error": self.result_is_error,
            # result_json is intentionally excluded from the default API dict —
            # it can be large; callers that need it fetch via the dedicated
            # /v1/tool-approval-requests/{id}/result endpoint.
            "continuation_state": self.continuation_state,
            "continuation_attempts": int(self.continuation_attempts or 0),
            "continuation_delivered_at": (
                self.continuation_delivered_at.isoformat()
                if self.continuation_delivered_at else None
            ),
        }



class ToolApprovalPolicyRevision(SQLModel, table=True):
    """Immutable full row revisions, including deletion tombstones."""
    __tablename__ = "tool_approval_policy_revisions"
    id: str = Field(sa_column=Column(String(128), primary_key=True))
    policy_id: str = Field(index=True)
    version: int
    change: str
    actor: str | None = None
    created_at: datetime = Field(default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False))
    snapshot_json: str = Field(sa_column=Column(Text, nullable=False))

    def to_api(self):
        return {"id": self.id, "policy_id": self.policy_id, "version": self.version,
                "change": self.change, "actor": self.actor, "created_at": self.created_at.isoformat(),
                "policy": json.loads(self.snapshot_json)}


class ToolApprovalDecision(SQLModel, table=True):
    """Evaluation audit separate from gated requests; also captures allow/deny."""
    __tablename__ = "tool_approval_decisions"
    id: str = Field(default_factory=_new_id, primary_key=True)
    created_at: datetime = Field(default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False))
    backend: str
    tool_name: str
    action: str
    severity: str
    subject: str = Field(sa_column=Column(Text, nullable=False))
    policy_id: str | None = Field(default=None, index=True)
    policy_version: int | None = None
    policy_snapshot_json: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    bundle_etag: str
    invocation_hash: str
    bot_id: str | None = None
    user_id: str | None = None
    turn_id: str | None = None
    source: str = "mcp"
    session_key: str | None = None
    session_id: str | None = None
    request_id: str | None = None
    tool_use_id: str | None = None
    outcome: str | None = None  # policy action may require approval but a grant allowed it

    def to_api(self):
        value = self.model_dump(mode="json")
        value["policy"] = json.loads(value.pop("policy_snapshot_json")) if self.policy_snapshot_json else None
        value.pop("policy_snapshot_json", None)
        return value
