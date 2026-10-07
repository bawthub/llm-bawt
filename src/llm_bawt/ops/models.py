"""Operation catalog + job ledger SQLModel tables (TASK-639).

Two tables:

* ``ops_operations`` — operator-configured operations. ``slug`` is the
  stable identifier the agent references via ``ops_run(operation=slug, ...)``.
  ``command_script`` is an operator-authored Docker JSON spec; agents never
  supply it. Arguments are declared via ``args_schema_json``; unknown
  properties are rejected and dynamic selectors use validated arguments.

* ``ops_jobs`` — one row per invocation. Snapshots the operation version +
  script hash so subsequent catalog edits never mutate an in-flight job.
  Carries caller context (bot / user / turn / approval request) for audit
  and for the continuation dispatcher.

Both tables live in the same ``askllm`` DB served by the shared
:func:`~llm_bawt.utils.db.get_shared_engine` pool.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlmodel import Field, SQLModel


# ---------------------------------------------------------------------------
# Operation risk / category enums (open strings; UI provides suggestions)
# ---------------------------------------------------------------------------

RISK_LOW = "low"
RISK_MEDIUM = "medium"
RISK_HIGH = "high"
RISK_CRITICAL = "critical"

# ---------------------------------------------------------------------------
# Executor kinds — docker (:mod:`.executor`) and the durable BawtHub release
# coordinator (TASK-1030, :mod:`.release_executor`); each accepts only its spec.
# ---------------------------------------------------------------------------

EXECUTOR_DOCKER = "docker"
EXECUTOR_RELEASE = "release"

# ---------------------------------------------------------------------------
# Job state machine
# ---------------------------------------------------------------------------

# The catalog validation + row commit landed but the executor hasn't been
# invoked yet. Only transient — held for a few ms.
JOB_QUEUED = "queued"
# Executor invoked; host unit has not yet been confirmed accepted.
JOB_DISPATCHING = "dispatching"
# Docker worker wrote a running receipt immediately before execution.
JOB_RUNNING = "running"
# Docker daemon owns the worker; no action result yet.
JOB_ACCEPTED = "accepted"
# Terminal: runner wrote exit_code=0 status.
JOB_SUCCEEDED = "succeeded"
# Terminal: runner wrote non-zero exit status.
JOB_FAILED = "failed"
# Terminal: worker's execution deadline elapsed; Docker side effect may be unknown.
JOB_TIMED_OUT = "timed_out"
# Terminal: operator/caller cancelled prior to run OR mid-run.
JOB_CANCELLED = "cancelled"
# Terminal: reconciler could not find the host unit AND no status file
# arrived within the stale window. Distinct from failed because we don't
# know the exit code.
JOB_LOST = "lost"

JOB_TERMINAL_STATES = {
    JOB_SUCCEEDED, JOB_FAILED, JOB_TIMED_OUT, JOB_CANCELLED, JOB_LOST,
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(value: datetime | None) -> str | None:
    """Stable UTC ISO string. Pooled DB connections may carry different session
    TimeZones, so the same instant can load as +00:00 or -04:00; serializing
    un-normalized values makes ops snapshot hashes flap between reads."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# OpsOperation
# ---------------------------------------------------------------------------

class OpsOperation(SQLModel, table=True):
    """One operator-configured operation. Editable in BawtHub.

    Creating / enabling a row here makes it appear in ``ops_list_operations``
    and become invocable via ``ops_run(operation=slug, ...)`` without any
    code change or service reload.
    """

    __tablename__ = "ops_operations"

    id: str = Field(sa_column=Column(String(64), primary_key=True))
    slug: str = Field(
        sa_column=Column(String(128), nullable=False, unique=True, index=True)
    )
    title: str = Field(default="", sa_column=Column(String(256), nullable=False))
    description: str = Field(default="", sa_column=Column(Text, nullable=False))
    enabled: bool = Field(
        default=False,
        sa_column=Column(Boolean, nullable=False, server_default="false", index=True),
    )
    # Only docker is implemented. Extra executor kinds land alongside
    # their implementation modules.
    executor_kind: str = Field(
        default=EXECUTOR_DOCKER,
        sa_column=Column(String(32), nullable=False),
    )
    # Legacy columns retained for compatibility. Docker operations require
    # these empty; TASK-639 explicitly rejects SSH/systemd execution.
    target_host: str = Field(default="", sa_column=Column(String(256), nullable=False))
    run_as_user: str | None = Field(
        default=None, sa_column=Column(String(64), nullable=True)
    )
    working_directory: str | None = Field(
        default=None, sa_column=Column(String(1024), nullable=True)
    )
    # Operator-authored Docker JSON action spec; never arbitrary shell.
    # Dynamic selectors reference validated arguments by name.
    command_script: str = Field(default="", sa_column=Column(Text, nullable=False))
    # JSON Schema for the args dict. Empty {} = no args.
    # ``additionalProperties: false`` is enforced at validation time regardless
    # of the schema (unknown args are rejected).
    args_schema_json: str = Field(
        default="{}", sa_column=Column(Text, nullable=False)
    )
    args_defaults_json: str = Field(
        default="{}", sa_column=Column(Text, nullable=False)
    )
    # Executor guardrails.
    timeout_seconds: int = Field(
        default=300,
        sa_column=Column(Integer, nullable=False, server_default="300"),
    )
    # Delay before the host unit starts running. Non-zero for self-affecting
    # operations (bridge/redis/app restart) so the caller's response has time
    # to stream back before the restart lands.
    start_delay_seconds: int = Field(
        default=0, sa_column=Column(Integer, nullable=False, server_default="0"),
    )
    max_output_bytes: int = Field(
        default=65536,
        sa_column=Column(Integer, nullable=False, server_default="65536"),
    )
    max_concurrent: int | None = Field(
        default=None, sa_column=Column(Integer, nullable=True)
    )
    risk_level: str = Field(
        default=RISK_MEDIUM,
        sa_column=Column(String(16), nullable=False, server_default=RISK_MEDIUM),
    )
    category: str | None = Field(
        default=None, sa_column=Column(String(64), nullable=True, index=True)
    )
    # Informational — a prefix the approval-policy UI can suggest / preview.
    # Actual gating enforcement stays in ``tool_approval_policies``.
    approval_prompt_prefix: str | None = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    # Monotonic per-slug. Bumped on every ``update()``. Jobs snapshot this.
    version: int = Field(
        default=1, sa_column=Column(Integer, nullable=False, server_default="1"),
    )
    # sha256(command_script). Snapshotted on jobs so a later edit can't
    # rewrite an in-flight job's script.
    script_hash: str = Field(
        default="", sa_column=Column(String(64), nullable=False)
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    created_by: str | None = Field(
        default=None, sa_column=Column(String(128), nullable=True)
    )
    updated_by: str | None = Field(
        default=None, sa_column=Column(String(128), nullable=True)
    )
    # Soft-delete marker. Disabled + soft-deleted rows are hidden from
    # ``ops_list_operations`` but preserved for audit.
    soft_deleted_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )

    def to_api(self, *, include_script: bool = True) -> dict[str, Any]:
        return {
            "id": self.id,
            "slug": self.slug,
            "title": self.title,
            "description": self.description,
            "enabled": bool(self.enabled),
            "executor_kind": self.executor_kind,
            "target_host": self.target_host,
            "run_as_user": self.run_as_user,
            "working_directory": self.working_directory,
            "command_script": self.command_script if include_script else None,
            "args_schema_json": self.args_schema_json,
            "args_defaults_json": self.args_defaults_json,
            "timeout_seconds": int(self.timeout_seconds or 0),
            "start_delay_seconds": int(self.start_delay_seconds or 0),
            "max_output_bytes": int(self.max_output_bytes or 0),
            "max_concurrent": self.max_concurrent,
            "risk_level": self.risk_level,
            "category": self.category,
            "approval_prompt_prefix": self.approval_prompt_prefix,
            "version": int(self.version or 1),
            "script_hash": self.script_hash,
            "created_at": _iso_utc(self.created_at),
            "updated_at": _iso_utc(self.updated_at),
            "created_by": self.created_by,
            "updated_by": self.updated_by,
            "soft_deleted_at": (
                _iso_utc(self.soft_deleted_at)
            ),
        }

    def to_agent_summary(self) -> dict[str, Any]:
        """The subset returned by ``ops_list_operations`` — enough for the
        agent to form a valid ``ops_run`` call, without leaking the script."""
        import json
        try:
            args_schema = json.loads(self.args_schema_json or "{}")
        except (json.JSONDecodeError, TypeError):
            args_schema = {}
        try:
            args_defaults = json.loads(self.args_defaults_json or "{}")
        except (json.JSONDecodeError, TypeError):
            args_defaults = {}
        return {
            "slug": self.slug,
            "enabled": self.enabled,
            "title": self.title,
            "description": self.description,
            "risk_level": self.risk_level,
            "category": self.category,
            "target_host": self.target_host,
            "timeout_seconds": int(self.timeout_seconds or 0),
            "args_schema": args_schema,
            "args_defaults": args_defaults,
        }


# ---------------------------------------------------------------------------
# OpsJob
# ---------------------------------------------------------------------------

class OpsJob(SQLModel, table=True):
    """One invocation. Rows survive app / bridge / redis restarts so the
    reconciler + approval outbox can recover their state.

    The DB row is canonical metadata; the isolated worker's receipt on the
    dedicated Docker volume is the source of truth for completion. The
    independent reconciler imports receipt timestamps, exit code and output.
    """

    __tablename__ = "ops_jobs"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_ops_jobs_idempotency_key"),
    )

    id: str = Field(sa_column=Column(String(64), primary_key=True))

    # Operation snapshot — do NOT resolve the operation by slug at read time
    # for a running/terminal job; the row already carries the exact version.
    operation_slug: str = Field(
        sa_column=Column(String(128), nullable=False, index=True)
    )
    operation_version: int = Field(
        sa_column=Column(Integer, nullable=False, server_default="1")
    )
    operation_script_hash: str = Field(
        default="", sa_column=Column(String(64), nullable=False)
    )

    # Validated caller args (post-JSON-Schema check) + a redacted variant for
    # UI / logs (fields marked ``x-sensitive`` in the schema are masked here).
    args_json: str = Field(default="{}", sa_column=Column(Text, nullable=False))
    display_args_json: str = Field(
        default="{}", sa_column=Column(Text, nullable=False)
    )

    # Complete immutable invocation and original input for conflict detection.
    invocation_snapshot_json: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    request_payload_json: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    caller_actor: str | None = Field(default=None, sa_column=Column(String(128), nullable=True))

    # Caller context (nullable — direct MCP callers may not have all of these).
    caller_bot_id: str | None = Field(
        default=None, sa_column=Column(String(128), nullable=True, index=True)
    )
    caller_user_id: str | None = Field(
        default=None, sa_column=Column(String(128), nullable=True)
    )
    caller_turn_id: str | None = Field(
        default=None, sa_column=Column(String(128), nullable=True, index=True)
    )
    caller_session_key: str | None = Field(
        default=None, sa_column=Column(String(128), nullable=True)
    )
    caller_backend: str | None = Field(
        default=None, sa_column=Column(String(64), nullable=True)
    )
    # If this job was dispatched via an approved MCP-kind approval request,
    # record its id for audit + result linking.
    approval_request_id: str | None = Field(
        default=None, sa_column=Column(String(128), nullable=True, index=True)
    )

    # Lifecycle.
    state: str = Field(
        default=JOB_QUEUED,
        sa_column=Column(String(16), nullable=False, server_default=JOB_QUEUED, index=True),
    )
    host_unit_name: str | None = Field(
        default=None, sa_column=Column(String(128), nullable=True)
    )
    submitted_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False, index=True),
    )
    dispatched_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    started_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    finished_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    exit_code: int | None = Field(
        default=None, sa_column=Column(Integer, nullable=True)
    )
    output_tail: str | None = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    error_text: str | None = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    status_file_path: str | None = Field(
        default=None, sa_column=Column(String(1024), nullable=True)
    )
    log_file_path: str | None = Field(
        default=None, sa_column=Column(String(1024), nullable=True)
    )
    # Deduplication key — the MCP approval request id for approved calls, or
    # an explicit invocation id passed by a direct caller. UNIQUE index.
    idempotency_key: str = Field(
        sa_column=Column(String(128), nullable=False)
    )
    retry_count: int = Field(
        default=0, sa_column=Column(Integer, nullable=False, server_default="0"),
    )
    last_reconcile_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )

    def to_api(self, *, include_output: bool = False) -> dict[str, Any]:
        import json
        try:
            args = json.loads(self.display_args_json or "{}")
        except (json.JSONDecodeError, TypeError):
            args = {}
        row: dict[str, Any] = {
            "id": self.id,
            "job_id": self.id,
            "operation": self.operation_slug,
            "operation_version": int(self.operation_version or 1),
            "operation_script_hash": self.operation_script_hash,
            "args": args,
            "caller": {
                "actor": self.caller_actor,
                "bot_id": self.caller_bot_id,
                "user_id": self.caller_user_id,
                "turn_id": self.caller_turn_id,
                "session_key": self.caller_session_key,
                "backend": self.caller_backend,
            },
            "approval_request_id": self.approval_request_id,
            "state": self.state,
            "host_unit_name": self.host_unit_name,
            "submitted_at": _iso_utc(self.submitted_at),
            "dispatched_at": _iso_utc(self.dispatched_at),
            "started_at": _iso_utc(self.started_at),
            "finished_at": _iso_utc(self.finished_at),
            "exit_code": self.exit_code,
            "error_text": self.error_text,
            "idempotency_key": self.idempotency_key,
            "retry_count": int(self.retry_count or 0),
            "terminal": self.state in JOB_TERMINAL_STATES,
            "last_reconcile_at": (
                _iso_utc(self.last_reconcile_at)
            ),
        }
        # TASK-997: structured deploy/rollback record (None for ordinary ops),
        # so the UI never has to parse raw output for release state.
        from .image_deploy import parse_deployment
        row["deployment"] = parse_deployment(self.output_tail)
        if include_output:
            row["output_tail"] = self.output_tail
            row["status_file_path"] = self.status_file_path
            row["log_file_path"] = self.log_file_path
        return row


class OpsOperationRevision(SQLModel, table=True):
    """Append-only full catalog history, including enable/delete and attribution."""
    __tablename__ = "ops_operation_revisions"
    __table_args__ = (UniqueConstraint("operation_id", "version", name="uq_ops_revision"),)
    id: str = Field(default_factory=_new_id, primary_key=True)
    operation_id: str = Field(index=True)
    operation_slug: str = Field(index=True)
    version: int
    snapshot_json: str = Field(sa_column=Column(Text, nullable=False))
    actor: str | None = None
    recorded_at: datetime = Field(default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False))

    def to_api(self):
        import json
        return {"operation_id": self.operation_id, "operation_slug": self.operation_slug,
                "version": self.version, "actor": self.actor,
                "recorded_at": _iso_utc(self.recorded_at),
                "operation": json.loads(self.snapshot_json)}


__all__ = [
    "OpsOperationRevision",
    "JOB_ACCEPTED",
    "OpsOperation",
    "OpsJob",
    "RISK_LOW",
    "RISK_MEDIUM",
    "RISK_HIGH",
    "RISK_CRITICAL",
    "EXECUTOR_DOCKER",
    "EXECUTOR_RELEASE",
    "JOB_QUEUED",
    "JOB_DISPATCHING",
    "JOB_RUNNING",
    "JOB_SUCCEEDED",
    "JOB_FAILED",
    "JOB_TIMED_OUT",
    "JOB_CANCELLED",
    "JOB_LOST",
    "JOB_TERMINAL_STATES",
]
