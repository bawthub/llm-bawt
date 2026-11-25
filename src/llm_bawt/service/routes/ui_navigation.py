"""A tab reports what it did with a ``ui_navigate`` request (TASK-2626)."""
from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, Body, HTTPException, Query

from ... import ui_navigation as nav
from ..dependencies import get_service
from .sessions import _resolve_user

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/ui/navigate", tags=["UI"])


@router.post("/{request_id}/ack")
async def ack_navigation(
    request_id: str,
    user: str | None = Query(None),
    bot_id: str = Body(..., embed=True, max_length=64),
    outcome: Literal["claimed", "navigated", "released", "offered"] = Body(..., embed=True),
):
    """Accept one tab's outcome. Unknown, expired or another user's request: 404."""
    import redis.asyncio as aioredis

    user_id = _resolve_user(user)
    if not nav.valid_request_id(request_id):
        raise HTTPException(status_code=404, detail="navigation request not found")
    url = get_service().config.REDIS_URL
    if not url:
        raise HTTPException(status_code=503, detail="live events unavailable")
    client = aioredis.from_url(url, decode_responses=True, socket_timeout=2.0,
                               socket_connect_timeout=2.0)
    try:
        accepted = await nav.record_ack(client, request_id=request_id,
                                        bot_id=bot_id.strip().lower(), user_id=user_id,
                                        outcome=outcome)
    except Exception:
        log.warning("ui_navigate ack: redis unavailable", exc_info=True)
        raise HTTPException(status_code=503, detail="live events unavailable") from None
    finally:
        await client.aclose()
    if not accepted:
        raise HTTPException(status_code=404, detail="navigation request not found")
    return {"ok": True}
