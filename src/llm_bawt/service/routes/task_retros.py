"""Usage-only read contract for BawtHub task retros (TASK-2019)."""
from datetime import timedelta

from fastapi import APIRouter, HTTPException, Response
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.exc import SQLAlchemyError

from ...bot_tags import BehaviorTag, BotTags
from ...utils.db import get_shared_engine
from ..dependencies import get_service
from ..task_retro_read import (
    MAX_RETRO_LINKS, cached_claude_quota, read_retro_messages, read_retro_turns, window_datetime,
)

router = APIRouter()


class RetroLink(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bot_id: str = Field(min_length=1, max_length=128)
    trigger_message_id: str = Field(min_length=1, max_length=128)
    task_ref: str = Field(min_length=1, max_length=128)


class RetroRead(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str = Field(min_length=1, max_length=128)
    links: list[RetroLink] = Field(default_factory=list, max_length=MAX_RETRO_LINKS)
    task_refs: list[str] | None = Field(default=None, min_length=1, max_length=2)
    start: AwareDatetime | None = None
    end: AwareDatetime | None = None

    @model_validator(mode="after")
    def bounded_window(self):
        if (self.start is None) != (self.end is None):
            raise ValueError("start and end must be supplied together")
        if self.start is not None and (self.end <= self.start or self.end - self.start > timedelta(days=7)):
            raise ValueError("window must be positive and at most seven days")
        if self.start is None and self.task_refs is None:
            raise ValueError("a task or bounded week is required")
        if not self.user_id.strip():
            raise ValueError("user_id must not be blank")
        return self


@router.post("/v1/task-retros/read", tags=["System"])
def task_retro_read(body: RetroRead, response: Response):
    response.headers["Cache-Control"] = "no-store"
    tags = BotTags().snapshot()
    if not tags.available:
        raise HTTPException(503, "Bot visibility unavailable")
    bots = sorted(bot for bot in tags.known_bots if not tags.has(bot, BehaviorTag.PRIVATE))
    links = [link.model_dump() for link in body.links if link.bot_id in bots]
    try:
        with get_shared_engine(get_service().config).connect() as conn:
            # Bound query runtime as well as returned rows; no full-turn detail
            # or provider call is hidden behind this aggregate read.
            from sqlalchemy import text
            conn.execute(text("SET LOCAL statement_timeout = '4000ms'"))
            result = read_retro_turns(
                conn, user=body.user_id, bots=bots, links=links, refs=body.task_refs,
                start=body.start, end=body.end,
            )
            result.update(read_retro_messages(
                conn, user=body.user_id, bots=bots, refs=body.task_refs,
                start=body.start, end=body.end,
            ))
            snapshot = cached_claude_quota()
            result["claude_quota"] = None
            if snapshot is not None:
                quota = read_retro_turns(
                    conn, user=body.user_id, bots=bots, links=links,
                    start=window_datetime(snapshot["start"]),
                    end=window_datetime(snapshot["fetched_at"]),
                )
                result["claude_quota"] = {**snapshot, **quota, "basis": "recorded_visible_user_fleet"}
    except SQLAlchemyError as exc:
        raise HTTPException(503, "Task usage temporarily unavailable") from exc
    return result
