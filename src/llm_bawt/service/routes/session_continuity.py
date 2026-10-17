"""Read-only session viewer and internal bridge restoration receipts (TASK-1117)."""
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import text

from ...mcp_server.storage import get_storage
from ...memory.session_continuity import SessionContinuity
from ..dependencies import get_effective_bot_id
from .sessions import _owned_session_or_404, _resolve_user, _to_info

router = APIRouter()


async def session_details(session_id: str, bot_id: str, user_id: str, offset: int, limit: int):
    row = await _owned_session_or_404(session_id, bot_id, user_id)
    info = _to_info(row)
    # Registry timestamps are naive UTC; never let browsers interpret them as local time.
    for field in ("started_at", "ended_at", "archived_at"):
        value = getattr(info, field)
        if value:
            parsed = datetime.fromisoformat(value)
            setattr(info, field, (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).isoformat())
    metadata = info.session_metadata or {}
    continuity = metadata.get("continuity")
    continuity = continuity if isinstance(continuity, dict) else None
    page = SessionContinuity(get_storage().engine).resolve(
        continuity or {}, bot_id=bot_id, user_id=user_id, offset=offset, limit=limit,
    )
    return {"session": info, "continuity": continuity, **page}


@router.get("/v1/sessions/{session_id}/details", tags=["Sessions"])
async def get_session_details(
    session_id: str, bot_id: str = Query(...), user: str | None = Query(None),
    offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=100),
):
    return await session_details(session_id, get_effective_bot_id(bot_id), _resolve_user(user), offset, limit)


@router.get("/v1/sessions/by-message/{message_id}/details", tags=["Sessions"])
async def get_message_session_details(
    message_id: str, bot_id: str = Query(...), user: str | None = Query(None),
    offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=100),
):
    bot, owner = get_effective_bot_id(bot_id), _resolve_user(user)
    with get_storage().engine.connect() as conn:
        session_id = conn.execute(text("""
            SELECT s.id FROM sessions s JOIN messages m ON m.session_id=s.id AND m.bot_id=s.bot_id
            WHERE m.id=:message AND m.bot_id=:bot AND s.user_id=:user AND s.status <> 'deleted'
        """), {"message": message_id, "bot": bot, "user": owner}).scalar()
    if not session_id:
        raise HTTPException(404, "Session not found for this message")
    return await session_details(session_id, bot, owner, offset, limit)


class ContinuityReceipt(BaseModel):
    payload_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    restored: bool


@router.post("/v1/sessions/{session_id}/continuity-receipt", tags=["Sessions"])
async def acknowledge_continuity(session_id: str, body: ContinuityReceipt, bot_id: str = Query(...)):
    # Internal bridge callback, not exposed by the dedicated public viewer proxy.
    # The digest must match the payload already recorded by the app for this bot/session.
    updated = SessionContinuity(get_storage().engine).acknowledge(
        session_id, get_effective_bot_id(bot_id), body.payload_digest, restored=body.restored,
    )
    return {"updated": updated}
