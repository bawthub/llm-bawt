"""TASK-1396/1402: relay BawtHub's committed task and project writes to open UIs.

BawtHub owns task and project state and calls these after its outermost
transaction commits. The events go out on the existing global unified stream,
the same one ``bot_profile_changed`` uses, so every open window invalidates
its task views. They are deliberately narrow: two closed event types, row ids
only, no content, one route each. There is no generic publish endpoint. The
LAN/compose network boundary is the auth for ``/v1/*``
(``reference/security-model.md``).
"""

from __future__ import annotations

import logging
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from ..dependencies import get_service

logger = logging.getLogger(__name__)

router = APIRouter()

# A larger or unknown set is sent as null: invalidate every task view.
MAX_IDS = 100

# A BawtHub row id (cuid). Shared so the two events can't drift apart.
RowId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$")]

_CLOSED = ConfigDict(extra="forbid", strict=True, populate_by_name=False)


def _unique(value: list[str] | None, field: str) -> list[str] | None:
    if value is not None and len(set(value)) != len(value):
        raise ValueError(f"{field} must be unique")
    return value


class TaskChangedEvent(BaseModel):
    """``{"_type": "task_changed", "task_ids": [AgentTask.id, ...] | null}``."""

    model_config = _CLOSED

    type: Literal["task_changed"] = Field(alias="_type")
    # Required, but null is a valid value ("everything may have changed").
    task_ids: list[RowId] | None = Field(min_length=1, max_length=MAX_IDS)

    @field_validator("task_ids")
    @classmethod
    def _unique_ids(cls, value: list[str] | None) -> list[str] | None:
        return _unique(value, "task_ids")

    def to_event(self) -> dict:
        return {"_type": "task_changed", "task_ids": self.task_ids}


class ProjectChangedEvent(BaseModel):
    """``{"_type": "project_changed", "project_ids": [AgentProject.id, ...] | null}``."""

    model_config = _CLOSED

    type: Literal["project_changed"] = Field(alias="_type")
    # Required, but null is a valid value ("everything may have changed").
    project_ids: list[RowId] | None = Field(min_length=1, max_length=MAX_IDS)

    @field_validator("project_ids")
    @classmethod
    def _unique_ids(cls, value: list[str] | None) -> list[str] | None:
        return _unique(value, "project_ids")

    def to_event(self) -> dict:
        return {"_type": "project_changed", "project_ids": self.project_ids}


async def _publish(event: dict, what: str) -> dict:
    """Publish on the global unified stream; 503 if it is down."""
    subscriber = getattr(get_service(), "_redis_subscriber", None)
    if subscriber is None:
        logger.warning("%s change not published: the event stream is unavailable", what.capitalize())
        raise HTTPException(status_code=503, detail=f"{what} event stream unavailable")
    stream_id = await subscriber.publish_global_config_event(event)
    if stream_id is None:
        # publish_tool_event already logged the Redis error.
        raise HTTPException(status_code=503, detail=f"{what} event stream unavailable")
    return {"stream_id": stream_id}


@router.post("/v1/events/task-changed", status_code=status.HTTP_202_ACCEPTED, tags=["Events"])
async def publish_task_changed(event: TaskChangedEvent):
    """Tell open UIs these tasks changed. 503 if the event stream is down.

    Callers treat this as best-effort: the task write already committed, and
    the UI reconciles on SSE reconnect.
    """
    return await _publish(event.to_event(), "task")


@router.post("/v1/events/project-changed", status_code=status.HTTP_202_ACCEPTED, tags=["Events"])
async def publish_project_changed(event: ProjectChangedEvent):
    """Tell open UIs these projects were created, renamed or deleted.

    Best-effort, like task-changed. UIs treat it as a full task refresh,
    since task rows show their project's name and color.
    """
    return await _publish(event.to_event(), "project")
