"""Prompt-only companions to the existing scheduled_jobs/job_runs identities."""
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, CheckConstraint, Column, DateTime, Index, Text, UniqueConstraint, text
from sqlmodel import Field, SQLModel

from .prompt_timing import PromptTiming


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class PromptSchedule(SQLModel, table=True):
    __tablename__ = "prompt_schedules"
    __table_args__ = (
        CheckConstraint("lifecycle IN ('active','paused','completed','cancelled')"),
        CheckConstraint("missed_policy IN ('skip','run_latest')"),
        CheckConstraint("misfire_grace_seconds BETWEEN 60 AND 604800"),
        CheckConstraint("revision > 0"),
        CheckConstraint("length(trim(name)) BETWEEN 1 AND 120"),
        CheckConstraint("length(trim(prompt)) BETWEEN 1 AND 100000"),
        Index("ix_prompt_schedules_owner_lifecycle", "owner_user_id", "lifecycle"),
    )
    job_id: str = Field(primary_key=True, foreign_key="scheduled_jobs.id")
    owner_user_id: str = Field(index=True)
    name: str = Field(max_length=120)
    description: str = Field(default="", sa_column=Column(Text, nullable=False))
    prompt: str = Field(sa_column=Column(Text, nullable=False))
    requested_model: str | None = None
    target_mode: str = "specific"
    candidate_bot_ids: list[str] | None = Field(default=None, sa_column=Column(JSON))
    create_request_hash: str | None = None
    # Versioned, validated PromptTiming schema, kept together to avoid two
    # divergent timing validators/representations in API and persistence.
    timing_json: dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))
    clear_context: bool = True
    augment_memory: bool = True
    extract_memory: bool = True
    dedicated_session_id: str | None = None
    missed_policy: str = "run_latest"
    misfire_grace_seconds: int = 3600
    lifecycle: str = "active"
    revision: int = 1
    created_at: datetime = Field(default_factory=now_utc, sa_column=Column(DateTime(timezone=True), nullable=False))
    updated_at: datetime = Field(default_factory=now_utc, sa_column=Column(DateTime(timezone=True), nullable=False))
    cancelled_at: datetime | None = Field(default=None, sa_column=Column(DateTime(timezone=True)))

    def timing(self) -> PromptTiming:
        return PromptTiming.model_validate(self.timing_json)


NONTERMINAL = ("pending", "queued", "running")


class PromptOccurrence(SQLModel, table=True):
    __tablename__ = "prompt_occurrences"
    __table_args__ = (
        UniqueConstraint("job_id", "occurrence_key", name="uq_prompt_occurrence_key"),
        CheckConstraint("kind IN ('scheduled','manual')"),
        CheckConstraint("state IN ('pending','queued','running','succeeded','failed','skipped','cancelled','unknown')"),
        Index("ix_prompt_occurrences_job_created", "job_id", "created_at"),
        # DB-level final guard, including manual-vs-automatic races. Both
        # dialects support partial unique indexes; SQLite tests are NOT proof
        # of PostgreSQL row-lock concurrency behavior.
        Index("uq_prompt_occurrences_active", "job_id", unique=True,
              postgresql_where=text("state IN ('pending','queued','running')"),
              sqlite_where=text("state IN ('pending','queued','running')")),
    )
    run_id: str = Field(primary_key=True, foreign_key="job_runs.id")
    job_id: str = Field(foreign_key="scheduled_jobs.id", index=True)
    occurrence_key: str
    scheduled_for: datetime = Field(sa_column=Column(DateTime(timezone=True), nullable=False))
    kind: str = "scheduled"
    schedule_revision: int
    snapshot_json: dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))
    session_id: str | None = None
    delivery_id: str | None = Field(default=None, unique=True)
    state: str = Field(default="pending", index=True)
    enqueue_attempts: int = 0
    last_error: str | None = Field(default=None, sa_column=Column(Text))
    created_at: datetime = Field(default_factory=now_utc, sa_column=Column(DateTime(timezone=True), nullable=False))
    updated_at: datetime = Field(default_factory=now_utc, sa_column=Column(DateTime(timezone=True), nullable=False))
