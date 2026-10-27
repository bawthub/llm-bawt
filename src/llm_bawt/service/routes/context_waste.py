"""``GET /v1/turn-logs/waste``: the recurring context-waste report (TASK-1585).

The BawtHub operator UI passes its session user server-side. Every known bot
except private ones is in scope, for that user only. Registered ahead of the
turn-logs router so ``/v1/turn-logs/{turn_id}`` cannot capture ``waste``.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from llm_bawt.bot_access import BotAccessError, BotPolicyResolver

from ..context_waste import SETTING_KEY, build_report, operator_scope, resolve_thresholds
from ..context_waste.registry_keys import declared_mcp_keys
from ..context_waste.scope import MAX_DAYS
from ..dependencies import get_runtime_settings_store, get_service, get_turn_log_store

router = APIRouter()

_STATUS = {"invalid_request": 422, "access_denied": 404}


def load_thresholds() -> dict:
    """The operator's ``context_waste_thresholds`` over the generic seed."""
    try:
        settings = get_runtime_settings_store(get_service().config).get_scope_settings("global", "*")
    except Exception:
        settings = {}
    return resolve_thresholds(settings.get(SETTING_KEY))


def _report(engine, user_id: str, bot: str | None, days: int, mcp_keys) -> dict:
    access = BotPolicyResolver(engine).resolve_sync(None)
    scope = operator_scope(access, user_id=user_id, bot=bot, days=days)
    return build_report(engine, scope, load_thresholds(), mcp_keys)


@router.get("/v1/turn-logs/waste", tags=["Debug"])
async def context_waste(
    user_id: str = Query(..., min_length=1, max_length=128),
    bot: str | None = Query(None, min_length=1, max_length=64),
    days: int = Query(7, ge=1, le=MAX_DAYS),
):
    engine = get_turn_log_store().engine
    if engine is None:
        raise HTTPException(status_code=503, detail="Turn logs DB unavailable")
    # The registry listing is async; the report's blocking SQL runs off the loop.
    mcp_keys = await declared_mcp_keys()
    try:
        return await run_in_threadpool(_report, engine, user_id, bot, days, mcp_keys)
    except BotAccessError as error:
        raise HTTPException(status_code=_STATUS.get(error.code, 503), detail=error.code) from None
