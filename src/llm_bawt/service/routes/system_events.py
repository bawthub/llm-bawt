"""Read-only, user-scoped durable Events feed (TASK-1745)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query, Response

from ...system_events import MAX_EVENTS
from ..dependencies import get_service

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/v1/system/events", tags=["Event Stream"])
def list_system_events(
    response: Response,
    user_id: str = Query(..., min_length=1, max_length=128),
    after: int | None = Query(None, ge=0),
    before: int | None = Query(None, ge=1),
    limit: int = Query(100, ge=1, le=MAX_EVENTS),
):
    """Only one user per read. BawtHub's proxy binds user_id to requestUser.

    llm-bawt /v1/* is internal and intentionally unauthenticated; never link
    this endpoint directly from a browser without the BawtHub identity guard.
    """
    response.headers["Cache-Control"] = "private, no-store"
    if after is not None and before is not None:
        raise HTTPException(status_code=400, detail="Specify after or before, not both")
    emitter = getattr(get_service(), "_system_events", None)
    if emitter is None:
        raise HTTPException(status_code=503, detail="System event history unavailable")
    try:
        events, has_more = emitter.list_page(user_id, after=after, before=before, limit=limit)
    except Exception as exc:
        logger.exception("System event history unavailable")
        raise HTTPException(status_code=503, detail="System event history unavailable") from exc
    return {"events": events, "has_more": has_more, "capability": "ready"}
