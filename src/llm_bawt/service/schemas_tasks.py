"""Background task + scheduled-job request/response schemas.

Split out of ``service/schemas.py`` (TASK-557). ``schemas.py`` re-imports every
name here so ``from ..schemas import X`` across the service is unchanged.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class TaskSubmitRequest(BaseModel):
    """Request to submit a background task."""
    task_type: str
    payload: dict[str, Any]
    bot_id: str | None = None  # Will use config DEFAULT_BOT if not specified
    user_id: str  # Required - must be passed explicitly
    priority: int = 0

class TaskSubmitResponse(BaseModel):
    """Response after submitting a task."""
    task_id: str
    status: str = "pending"

class TaskStatusResponse(BaseModel):
    """Response for task status."""
    task_id: str
    status: str
    result: Any | None = None
    error: str | None = None
    processing_time_ms: float | None = None

class TaskListItem(BaseModel):
    """A task entry for task listing."""
    task_id: str
    status: str
    task_type: str | None = None
    bot_id: str | None = None
    user_id: str | None = None
    priority: int | None = None
    created_at: str | None = None
    completed_at: str | None = None
    processing_time_ms: float | None = None
    error: str | None = None

class TaskListResponse(BaseModel):
    """Response for listing tasks."""
    tasks: list[TaskListItem]
    total_count: int
    filters: dict[str, Any] = Field(default_factory=dict)

class ScheduledJobInfo(BaseModel):
    """A scheduled job with latest run summary."""
    id: str
    job_type: str
    bot_id: str
    enabled: bool
    interval_minutes: int
    last_run_at: datetime | None = None
    next_run_at: datetime | None = None
    created_at: datetime | None = None
    last_status: str | None = None
    last_duration_ms: int | None = None
    last_error: str | None = None
    parameter_schema: dict[str, Any] | None = None
    parameters: dict[str, Any] | None = None
    configuration_error: str | None = None
    stored_parameters: dict[str, Any] | None = None

class ScheduledJobsResponse(BaseModel):
    """Response for listing scheduled jobs."""
    jobs: list[ScheduledJobInfo]
    total_count: int
    filters: dict[str, Any] = Field(default_factory=dict)

class JobRunInfo(BaseModel):
    """A single scheduler job run record."""
    id: str
    job_id: str
    job_type: str | None = None
    bot_id: str
    status: str
    started_at: datetime
    finished_at: datetime | None = None
    duration_ms: int | None = None
    error_message: str | None = None
    result: Any | None = None

class JobRunsResponse(BaseModel):
    """Response for listing scheduler job run history."""
    runs: list[JobRunInfo]
    total_count: int
    filters: dict[str, Any] = Field(default_factory=dict)
