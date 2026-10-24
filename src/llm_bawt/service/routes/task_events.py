"""TASK-1396: relay BawtHub's committed task writes to open UIs.

BawtHub owns task state and calls this after its outermost task transaction
commits. The event goes out on the existing global unified stream, the same
one ``bot_profile_changed`` uses, so every open window invalidates its task
views. It is deliberately narrow: one event type, task ids only, no content.
There is no generic publish endpoint. The LAN/compose network boundary is the
auth for ``/v1/*`` (``reference/security-model.md``).
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
MAX_TASK_IDS = 100

TaskId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$")]


class TaskChangedEvent(BaseModel):
    """``{"_type": "task_changed", "task_ids": [AgentTask.id, ...] | null}``."""

    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=False)

    type: Literal["task_changed"] = Field(alias="_type")
    # Required, but null is a valid value ("everything may have changed").
    task_ids: list[TaskId] | None = Field(min_length=1, max_length=MAX_TASK_IDS)

    @field_validator("task_ids")
    @classmethod
    def _unique(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and len(set(value)) != len(value):
            raise ValueError("task_ids must be unique")
        return value

    def to_event(self) -> dict:
        return {"_type": "task_changed", "task_ids": self.task_ids}


@router.post("/v1/events/task-changed", status_code=status.HTTP_202_ACCEPTED, tags=["Events"])
async def publish_task_changed(event: TaskChangedEvent):
    """Tell open UIs these tasks changed. 503 if the event stream is down.

    Callers treat this as best-effort: the task write already committed, and
    the UI reconciles on SSE reconnect.
    """
    subscriber = getattr(get_service(), "_redis_subscriber", None)
    if subscriber is None:
        logger.warning("Task change not published: the event stream is unavailable")
        raise HTTPException(status_code=503, detail="task event stream unavailable")
    stream_id = await subscriber.publish_global_config_event(event.to_event())
    if stream_id is None:
        # publish_tool_event already logged the Redis error.
        raise HTTPException(status_code=503, detail="task event stream unavailable")
    return {"stream_id": stream_id}
