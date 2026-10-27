"""UI-only drill-down into what one context_recall read (TASK-1519).

BawtHub's recall card calls this through its chat proxy when the user opens a
coverage chip. Bots never see it, so the recall packet stays the same size.
Owner mismatch, an expired or unknown handle, and an owner bot that is no
longer searchable all return the same 404.
"""
from __future__ import annotations

from functools import lru_cache
from typing import get_args

from fastapi import APIRouter, HTTPException, Query

from ...recall.contracts import RecallError, Source
from ...recall.drilldown import MAX_PAGE, EvidenceDrilldown
from ...recall.ledger import MAX_LEDGER_ENTRIES, PostgresLedger
from ...recall.postgres import RecallDatabase
from ...recall.task_source import TaskSource
from ...utils.db import get_shared_engine
from ..dependencies import get_service

router = APIRouter()
SOURCES = frozenset(get_args(Source))
UNAVAILABLE = "Record list unavailable (older recall or expired)"


@lru_cache(maxsize=1)
def _drilldown() -> EvidenceDrilldown:
    engine = get_shared_engine(get_service().config)
    return EvidenceDrilldown(RecallDatabase(engine), PostgresLedger(engine), TaskSource())


@router.get("/v1/recall/evidence", tags=["Recall"])
async def recall_evidence(
    handle: str = Query(..., min_length=16, max_length=64, description="Result handle from the packet"),
    bot_id: str = Query(..., min_length=1, max_length=128, description="Bot that ran the recall"),
    user_id: str = Query(..., min_length=1, max_length=128, description="User the recall ran for"),
    source: str | None = Query(None, description="List this source's records; omit for the summary"),
    offset: int = Query(0, ge=0, le=MAX_LEDGER_ENTRIES),
    limit: int = Query(20, ge=1, le=MAX_PAGE),
) -> dict:
    if source is not None and source not in SOURCES:
        raise HTTPException(status_code=422, detail="Unknown source")
    drilldown = _drilldown()
    try:
        if source is None:
            result = await drilldown.summary(handle, bot_id, user_id)
        else:
            result = await drilldown.page(handle, bot_id, user_id, source, offset, limit)
    except RecallError as error:
        # Fixed codes only; source contents never ride an error.
        raise HTTPException(status_code=503, detail=f"Recall evidence unavailable: {error.code}") from None
    if result is None:
        raise HTTPException(status_code=404, detail=UNAVAILABLE)
    return result
