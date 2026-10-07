"""Durable release orchestration records for the BawtHub production pipeline."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text, UniqueConstraint
from sqlmodel import Field, SQLModel

RELEASE_PREFLIGHT = "preflight"
RELEASE_DISPATCHING_BUILD = "dispatching_build"
RELEASE_BUILDING = "building"
RELEASE_BUILD_FAILED_SAFE = "build_failed_safe"
RELEASE_BUILD_PARTIAL = "build_partial"
RELEASE_BUILD_COMPLETE = "build_complete"
RELEASE_AWAITING_DEPLOY_APPROVAL = "awaiting_deploy_approval"
RELEASE_DEPLOYING = "deploying"
RELEASE_DEPLOYED = "deployed"
RELEASE_DEPLOY_FAILED_RESTORED = "deploy_failed_restored"
# Build complete and deployable, but the deploy was denied, expired, or
# interlocked (deploy operation disabled). Nothing was deployed.
RELEASE_DEPLOY_DECLINED = "deploy_declined"
RELEASE_LOST = "lost_requires_inspection"

RELEASE_TERMINAL_STATES = {
    RELEASE_BUILD_FAILED_SAFE,
    RELEASE_DEPLOYED,
    RELEASE_DEPLOY_FAILED_RESTORED,
    RELEASE_DEPLOY_DECLINED,
    RELEASE_LOST,
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _loads(raw: str | None, fallback: Any = None) -> Any:
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return fallback


class ReleaseRun(SQLModel, table=True):
    """One logical release from remote preflight through verified deployment."""

    __tablename__ = "ops_release_runs"
    __table_args__ = (
        UniqueConstraint("parent_job_id", name="uq_ops_release_parent_job"),
        UniqueConstraint("release_request_id", name="uq_ops_release_request"),
    )

    id: str = Field(sa_column=Column(String(64), primary_key=True))
    parent_job_id: str = Field(sa_column=Column(String(64), nullable=False, index=True))
    release_request_id: str = Field(sa_column=Column(String(128), nullable=False, index=True))
    state: str = Field(
        default=RELEASE_PREFLIGHT,
        sa_column=Column(String(40), nullable=False, index=True, server_default=RELEASE_PREFLIGHT),
    )
    release_task: str = Field(sa_column=Column(String(64), nullable=False))
    bump: str = Field(default="patch", sa_column=Column(String(16), nullable=False))
    llm_bawt_mode: str = Field(default="auto", sa_column=Column(String(16), nullable=False))

    github_repository: str = Field(sa_column=Column(String(256), nullable=False))
    workflow_path: str = Field(sa_column=Column(String(512), nullable=False))
    canonical_branch: str = Field(sa_column=Column(String(256), nullable=False))
    llm_bawt_repository: str | None = Field(default=None, sa_column=Column(String(256), nullable=True))
    llm_bawt_branch: str | None = Field(default=None, sa_column=Column(String(256), nullable=True))
    source_sha: str | None = Field(default=None, sa_column=Column(String(40), nullable=True))
    llm_bawt_sha: str | None = Field(default=None, sa_column=Column(String(40), nullable=True))
    release_plan_json: str | None = Field(default=None, sa_column=Column(Text, nullable=True))

    dispatch_started_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    github_run_id: str | None = Field(default=None, sa_column=Column(String(32), nullable=True, index=True))
    github_run_attempt: int | None = Field(default=None, sa_column=Column(Integer, nullable=True))
    github_run_url: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    rerun_count: int = Field(default=0, sa_column=Column(Integer, nullable=False, server_default="0"))

    receipt_json: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    receipt_verified_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    deployable: bool = Field(
        default=False, sa_column=Column(Boolean, nullable=False, server_default="false")
    )
    warning_text: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    version: str | None = Field(default=None, sa_column=Column(String(32), nullable=True))
    tag: str | None = Field(default=None, sa_column=Column(String(128), nullable=True))
    digest: str | None = Field(default=None, sa_column=Column(String(80), nullable=True))
    image_repository: str | None = Field(default=None, sa_column=Column(String(256), nullable=True))

    deploy_approval_request_id: str | None = Field(
        default=None, sa_column=Column(String(128), nullable=True, index=True)
    )
    deploy_job_id: str | None = Field(default=None, sa_column=Column(String(64), nullable=True, index=True))
    # Pump throttle (many status readers, one cheap GitHub poll per interval)
    # and the deadline of the current phase (correlation, rerun registration,
    # deploy-approval expiry, deploy-preparation retry window).
    next_check_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True, index=True)
    )
    phase_deadline_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    error_code: str | None = Field(default=None, sa_column=Column(String(64), nullable=True))
    error_text: str | None = Field(default=None, sa_column=Column(Text, nullable=True))

    claim_token: str | None = Field(default=None, sa_column=Column(String(64), nullable=True, index=True))
    claim_expires_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True, index=True)
    )
    reconcile_attempts: int = Field(
        default=0, sa_column=Column(Integer, nullable=False, server_default="0")
    )
    event_seq: int = Field(default=0, sa_column=Column(Integer, nullable=False, server_default="0"))
    created_at: datetime = Field(
        default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False, index=True)
    )
    updated_at: datetime = Field(
        default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False, index=True)
    )
    finished_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )

    def to_api(self, *, events: list["ReleaseEvent"] | None = None) -> dict[str, Any]:
        value = {
            "id": self.id,
            "parent_job_id": self.parent_job_id,
            "release_request_id": self.release_request_id,
            "state": self.state,
            "terminal": self.state in RELEASE_TERMINAL_STATES,
            "release_task": self.release_task,
            "bump": self.bump,
            "llm_bawt_mode": self.llm_bawt_mode,
            "github_repository": self.github_repository,
            "workflow_path": self.workflow_path,
            "canonical_branch": self.canonical_branch,
            "llm_bawt_repository": self.llm_bawt_repository,
            "llm_bawt_branch": self.llm_bawt_branch,
            "source_sha": self.source_sha,
            "llm_bawt_sha": self.llm_bawt_sha,
            "release_plan": _loads(self.release_plan_json, {}),
            "dispatch_started_at": _iso(self.dispatch_started_at),
            "github_run_id": self.github_run_id,
            "github_run_attempt": self.github_run_attempt,
            "github_run_url": self.github_run_url,
            "rerun_count": int(self.rerun_count or 0),
            "receipt": _loads(self.receipt_json),
            "receipt_verified_at": _iso(self.receipt_verified_at),
            "deployable": bool(self.deployable),
            "warning_text": self.warning_text,
            "version": self.version,
            "tag": self.tag,
            "digest": self.digest,
            "image_repository": self.image_repository,
            "deploy_approval_request_id": self.deploy_approval_request_id,
            "deploy_job_id": self.deploy_job_id,
            "next_check_at": _iso(self.next_check_at),
            "phase_deadline_at": _iso(self.phase_deadline_at),
            "error_code": self.error_code,
            "error_text": self.error_text,
            "reconcile_attempts": int(self.reconcile_attempts or 0),
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
            "finished_at": _iso(self.finished_at),
        }
        if events is not None:
            value["events"] = [event.to_api() for event in events]
        return value


class ReleaseEvent(SQLModel, table=True):
    """Append-only release transition/audit event."""

    __tablename__ = "ops_release_events"
    __table_args__ = (
        UniqueConstraint("release_run_id", "sequence", name="uq_ops_release_event_sequence"),
    )

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    release_run_id: str = Field(sa_column=Column(String(64), nullable=False, index=True))
    sequence: int = Field(sa_column=Column(Integer, nullable=False))
    event_type: str = Field(sa_column=Column(String(64), nullable=False, index=True))
    state: str = Field(sa_column=Column(String(40), nullable=False, index=True))
    detail_json: str = Field(default="{}", sa_column=Column(Text, nullable=False))
    created_at: datetime = Field(
        default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False, index=True)
    )

    def to_api(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "release_run_id": self.release_run_id,
            "sequence": self.sequence,
            "event_type": self.event_type,
            "state": self.state,
            "detail": _loads(self.detail_json, {}),
            "created_at": _iso(self.created_at),
        }


__all__ = [
    "ReleaseRun",
    "ReleaseEvent",
    "RELEASE_PREFLIGHT",
    "RELEASE_DISPATCHING_BUILD",
    "RELEASE_BUILDING",
    "RELEASE_BUILD_FAILED_SAFE",
    "RELEASE_BUILD_PARTIAL",
    "RELEASE_BUILD_COMPLETE",
    "RELEASE_AWAITING_DEPLOY_APPROVAL",
    "RELEASE_DEPLOYING",
    "RELEASE_DEPLOYED",
    "RELEASE_DEPLOY_FAILED_RESTORED",
    "RELEASE_DEPLOY_DECLINED",
    "RELEASE_LOST",
    "RELEASE_TERMINAL_STATES",
]
