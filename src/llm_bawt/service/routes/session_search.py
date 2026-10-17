"""Bounded read-only session search, registered before the session-id route."""
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, field_validator

from ...mcp_server.storage import get_storage
from ...memory.session_search import SessionSearch
from ..logging import get_service_logger

router = APIRouter()
log = get_service_logger(__name__)


class SessionSearchItem(BaseModel):
    id: str
    bot_id: str
    user_id: str
    title: str
    status: Literal["active", "archived"]
    started_at: datetime | None
    ended_at: datetime | None
    models: str = ""
    provider: str = ""
    has_predecessor: bool = False
    rank: int = 0

    @field_validator("started_at", "ended_at")
    @classmethod
    def utc_timestamps(cls, value: datetime | None) -> datetime | None:
        # The registry stores naive UTC; never let the browser interpret it
        # as its own timezone (which can change the displayed calendar date).
        return value.replace(tzinfo=timezone.utc) if value and value.tzinfo is None else value


class SessionSearchResponse(BaseModel):
    sessions: list[SessionSearchItem]
    total_count: int


@router.get("/v1/sessions/search", response_model=SessionSearchResponse, tags=["Sessions"])
def search_sessions(
    user: str = Query(..., min_length=1, max_length=200),
    query: str = Query("", max_length=300),
    bot_id: str | None = Query(None, max_length=200),
    status: Literal["all", "active", "archived"] = Query("all"),
    since: float | None = Query(None, ge=0, le=253402300799, allow_inf_nan=False),
    sort_by: Literal["relevance", "recent", "oldest", "title"] = Query("relevance"),
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0, le=100000),
):
    """Search titles, bot slugs and model/provider labels, never SDK IDs or message bodies.

    A user is mandatory (no default-user fallback on cross-bot discovery).
    Archived sessions are included; deleted and unowned legacy rows are not.
    Date filtering and ordering refer to when the session started.
    """
    if not user.strip():
        raise HTTPException(status_code=400, detail="user is required")
    try:
        engine = get_storage().engine
        if engine is None:
            raise RuntimeError("Session storage unavailable")
        return SessionSearch(engine).search(
            user_id=user.strip(), query=query, bot_id=bot_id, status=status,
            since=since, sort_by=sort_by, limit=limit, offset=offset,
        )
    except Exception:
        log.exception("Session search unavailable")
        raise HTTPException(status_code=503, detail="Session search unavailable; retry shortly") from None
