"""The user-Stop queue hold API (TASK-1327): inspect, Resume, Cancel all."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from ..dependencies import get_service
from ..user_stop_hold import cancel_all, resume, snapshot

router = APIRouter()


def _engine():
    engine = get_service()._turn_log_store.engine
    if engine is None:
        raise HTTPException(status_code=503, detail="Queue hold unavailable")
    return engine


def _bot(bot_id: str) -> str:
    return bot_id.strip().lower()


def _wake(bot_id: str) -> None:
    dispatcher = getattr(get_service(), "_inter_bot_dispatcher", None)
    if dispatcher is not None:
        dispatcher.wake(bot_id)


@router.get("/v1/bots/{bot_id}/queue-hold", tags=["Inter-Bot Deliveries"])
async def get_queue_hold(bot_id: str):
    """The bot's live user-Stop hold and the rows it holds; ``hold`` is null when none."""
    hold = await asyncio.to_thread(snapshot, _engine(), _bot(bot_id))
    return {"bot_id": _bot(bot_id), "hold": hold.to_api() if hold is not None else None}


@router.post("/v1/bots/{bot_id}/queue-hold/resume", tags=["Inter-Bot Deliveries"])
async def resume_queue_hold(bot_id: str):
    """Release the hold; the held rows dispatch in FIFO order."""
    released = await asyncio.to_thread(resume, _engine(), _bot(bot_id))
    _wake(_bot(bot_id))
    return {"bot_id": _bot(bot_id), "released_gate_ids": released}


@router.post("/v1/bots/{bot_id}/queue-hold/cancel-all", tags=["Inter-Bot Deliveries"])
async def cancel_all_queue_hold(bot_id: str):
    """Cancel every held row, then release the hold (409 when no hold is live)."""
    cancelled = await asyncio.to_thread(cancel_all, _engine(), _bot(bot_id))
    if cancelled is None:
        raise HTTPException(status_code=409, detail="No live queue hold")
    dispatcher = getattr(get_service(), "_inter_bot_dispatcher", None)
    if dispatcher is not None:
        for record in cancelled:
            await dispatcher._emit(record)
    _wake(_bot(bot_id))
    return {"bot_id": _bot(bot_id), "cancelled": [record.id for record in cancelled]}
