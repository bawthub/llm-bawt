"""Provider usage by exact model access path, never model-name heuristics.

GET /v1/usage             all registered reports
GET /v1/usage?provider=…  one explicit report
GET /v1/usage?access_path=…  report for a frozen turn's route
GET /v1/usage?bot_id=…  report for the bot's bound catalog endpoint

Unknown/unbound paths report unavailable; they never borrow a different
account's quota. A usage snapshot is cached by report provider, not bot.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ..dependencies import get_service
from ..usage import (AllUsage, ProviderUsage, get_all, get_usage, has_provider,
                     list_providers, usage_adapter_for_access_path)
from ..usage.canonical import STATUS_NOT_IMPLEMENTED
from ..usage.routing import bot_access_path

router = APIRouter()


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
        provider = adapter.provider

    if provider is not None:
        if not has_provider(provider):
            raise HTTPException(status_code=404, detail=f"Unknown provider '{provider}'. Registered: {list_providers()}")
        snap = await get_usage(provider, force=force)
        if snap is None:
            raise HTTPException(status_code=404, detail=f"Unknown provider '{provider}'.")
        return snap

    return await get_all(force=force)
