"""Provider usage by exact model access path, never model-name heuristics.

GET /v1/usage             all registered reports
GET /v1/usage?provider=…  one explicit report
GET /v1/usage?access_path=…  report for a frozen turn's route
GET /v1/usage?bot_id=…  report for the bot's bound catalog endpoint
PUT/DELETE /v1/usage/{provider}/balance  set/clear a user-entered API balance

Unknown/unbound paths report unavailable; they never borrow a different
account's quota. A usage snapshot is cached by report provider, not bot; a
path/bot query echoes the resolved ``access_path_key``.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal, InvalidOperation
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..dependencies import get_service
from ..usage import (AllUsage, ProviderUsage, clear_balance, get_adapter, get_all, get_usage,
                     has_provider, list_providers, normalize_access_path, set_balance,
                     usage_adapter_for_access_path)
from ..usage.balance import BALANCE_PROVIDER
from ..usage.canonical import STATUS_NOT_IMPLEMENTED
from ..usage.routing import bot_access_path

router = APIRouter()
# NUMERIC(18, 8) holds under 10**10; a larger "balance" is a typo, not credit.
MAX_BALANCE_USD = Decimal(10) ** 9


def _provider_for_bot(service, bot_id: str) -> str | None:
    """Resolve the one usage report for the bot's bound endpoint."""
    adapter = usage_adapter_for_access_path(bot_access_path(service.config, bot_id))
    return adapter.provider if adapter else None


def _unavailable(path: str | None) -> ProviderUsage:
    return ProviderUsage(
        provider="unavailable",
        display_name="Usage unavailable",
        available=False,
        status=STATUS_NOT_IMPLEMENTED,
        error=f"No usage report for access path {path or 'unknown'!r}.",
    )


@router.get("/v1/usage", tags=["Usage"])
async def get_usage_endpoint(
    provider: str | None = Query(None, description="Canonical usage provider id."),
    bot_id: str | None = Query(None, description="Resolve from this bot's bound endpoint."),
    access_path: str | None = Query(None, description="Access path frozen on a turn."),
    force: bool = Query(False, description="Bypass the per-provider cache."),
) -> ProviderUsage | AllUsage:
    if provider is not None and (bot_id or access_path):
        raise HTTPException(status_code=400, detail="Select one of provider, bot_id or access_path")
    if bot_id and access_path:
        raise HTTPException(status_code=400, detail="Select either bot_id or access_path")

    path = access_path
    if provider is None and bot_id:
        path = bot_access_path(get_service().config, bot_id)
    if provider is None and (bot_id or access_path):
        adapter = usage_adapter_for_access_path(path)
        if adapter is None:
            return _unavailable(path)
        snap = await _provider_report(adapter.provider, force=force)
        # The report is cached per provider; the resolved route is per request.
        return snap.model_copy(update={"access_path_key": normalize_access_path(path)})

    if provider is not None:
        return await _provider_report(provider, force=force)

    return await get_all(force=force, catalog=get_service().config.ensure_model_catalog())


async def _provider_report(provider: str, *, force: bool = False) -> ProviderUsage:
    if not has_provider(provider):
        raise HTTPException(status_code=404, detail=f"Unknown provider '{provider}'. Registered: {list_providers()}")
    snap = await get_usage(provider, force=force)
    if snap is None:
        raise HTTPException(status_code=404, detail=f"Unknown provider '{provider}'.")
    return snap


class BalanceAnchorRequest(BaseModel):
    amount_usd: Any


def _anchored_provider(provider: str):
    adapter = get_adapter(provider)
    if adapter is None:
        raise HTTPException(status_code=404, detail=f"Unknown provider '{provider}'.")
    if not adapter.capabilities()["manual_balance_supported"]:  # the flag reports advertise
        detail = ("This provider reports its own balance" if adapter.balance_mode == BALANCE_PROVIDER
                  else "This provider has no prepaid API balance")
        raise HTTPException(status_code=409, detail=detail)
    return adapter


@router.put("/v1/usage/{provider}/balance", tags=["Usage"])
async def set_balance_endpoint(provider: str, request: BalanceAnchorRequest) -> ProviderUsage:
    """Set "$X left as of now" (server clock); later turns on this provider count down."""
    _anchored_provider(provider)
    raw = request.amount_usd
    try:
        amount = Decimal(str(raw))
    except (InvalidOperation, TypeError, ValueError):
        amount = Decimal("NaN")
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)) or not amount.is_finite() \
            or amount <= 0 or amount >= MAX_BALANCE_USD:
        raise HTTPException(status_code=400, detail="API balance must be a positive USD amount")
    await asyncio.to_thread(set_balance, provider, amount)
    return await _provider_report(provider)


@router.delete("/v1/usage/{provider}/balance", tags=["Usage"])
async def clear_balance_endpoint(provider: str) -> ProviderUsage:
    _anchored_provider(provider)
    await asyncio.to_thread(clear_balance, provider)
    return await _provider_report(provider)
