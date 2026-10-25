"""Anthropic API prepaid balance, plus spend from organization reports or the ledger.

Anthropic has no balance endpoint, so the balance is a user-set anchor counted
down per turn (``usage.balance``; the registry overlays it on every read). The
spend detail comes from the Admin cost report when an Admin key is connected
(account-wide), otherwise from this app's per-turn ledger. Neither is a Claude
subscription window. Admin key retrieval is deliberately isolated from the
inference provider/token broker.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx

from ..canonical import STATUS_ERROR, STATUS_OK, STATUS_RATE_LIMITED, STATUS_UNAUTHORIZED, UsageSpend
from .api_key_balance import ApiKeyBalanceAdapter, utc_datetime

logger = logging.getLogger(__name__)
COST_URL = "https://api.anthropic.com/v1/organizations/cost_report"
ANTHROPIC_VERSION = "2023-06-01"
ADMIN_CONNECTION = "anthropic-admin"
INFERENCE_CONNECTION = "anthropic-api"


def _decimal(value: Any) -> Decimal:
    try:
        n = Decimal(str(value))
        return n if n.is_finite() else Decimal(0)
    except (InvalidOperation, TypeError, ValueError):
        return Decimal(0)


def _windows(now: datetime) -> tuple[datetime, datetime, datetime]:
    today = now.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return today, today - timedelta(days=6), today.replace(day=1)


def cost_totals(buckets: list[dict], *, now: datetime) -> tuple[Decimal, Decimal, Decimal]:
    """Aggregate USD cents as Decimal over UTC calendar day, 7 days, and month."""
    today_start, seven_start, month_start = _windows(now)
    totals = [Decimal(0), Decimal(0), Decimal(0)]
    for bucket in buckets:
        started = utc_datetime(bucket.get("starting_at"))
        if started is None or started > now:
            continue
        for result in bucket.get("results") or []:
            if result.get("currency") != "USD":
                continue
            amount = _decimal(result.get("amount")) / 100  # API reports cents, not dollars
            if started >= today_start:
                totals[0] += amount
            if started >= seven_start:
                totals[1] += amount
            if started >= month_start:
                totals[2] += amount
    return tuple(totals)


async def _pages(client: httpx.AsyncClient, url: str, key: str, params: dict[str, str]) -> list[dict]:
    buckets: list[dict] = []
    seen: set[str] = set()
    page: str | None = None
    for _ in range(64):
        query = {**params, **({"page": page} if page else {})}
        response = await client.get(
            url, params=query,
            headers={"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION,
                     "User-Agent": "llm-bawt/usage-report"},
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise ValueError("Malformed Anthropic report")
        buckets.extend(payload["data"])
        if not payload.get("has_more"):
            return buckets
        page = payload.get("next_page")
        if not isinstance(page, str) or not page or page in seen:
            raise ValueError("Malformed Anthropic report pagination")
        seen.add(page)
    raise ValueError("Anthropic report exceeded pagination limit")


class AnthropicApiUsageAdapter(ApiKeyBalanceAdapter):
    """The generic anchored API key, plus Anthropic spend detail."""

    provider = "anthropic_api"
    display_name = "Anthropic API"
    backend = "claude-code"
    access_path_keys = ("anthropic-api",)
    connection_id = INFERENCE_CONNECTION
    legacy_balance_setting = "anthropic_api_balance_anchor"

    def _admin_key(self) -> str | None:
        from ...dependencies import get_service
        from ...providers.base import CredentialStore, STATUS_CONNECTED

        record = CredentialStore(get_service().config).load(ADMIN_CONNECTION)
        if record is None or record.status != STATUS_CONNECTED:
            return None
        key = record.secret.get("admin_key")
        return key if isinstance(key, str) and key.startswith("sk-ant-admin") else None

    def _ledger_spend(self, now: datetime, since: datetime) -> tuple[Decimal, ...]:
        from .. import _balance_store

        return _balance_store().spend(self.provider, since=since, starts=_windows(now))

    async def fetch(self):
        now = datetime.now(timezone.utc)
        key = None
        try:
            key = await asyncio.to_thread(self._admin_key)
            if not key:
                # Before the key was connected, an anthropic-api label was only
                # catalog intent, never proof this key paid: floor the ledger.
                connected_at = await asyncio.to_thread(self.credential_connected_at)
                if connected_at is None:
                    raise ValueError("Anthropic API connection time is unavailable")
                totals = await asyncio.to_thread(self._ledger_spend, now, connected_at)
                return self._spend(totals, "bawt_recorded_estimate")
        except Exception as exc:  # noqa: BLE001 — no fabricated zero-spend on failure
            logger.warning("Anthropic API spend unavailable: %s", type(exc).__name__)
            return self._base(available=False, status=STATUS_ERROR,
                              error="Anthropic API spend or connection time unavailable")
        try:
            _, _, month_start = _windows(now)
            async with httpx.AsyncClient(timeout=15.0) as client:
                cost = await _pages(client, COST_URL, key, {
                    "starting_at": month_start.isoformat(), "group_by[]": "description",
                    "bucket_width": "1d", "limit": "31",
                })
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            status = STATUS_UNAUTHORIZED if code in (401, 403) else STATUS_RATE_LIMITED if code == 429 else STATUS_ERROR
            return self._base(available=False, status=status,
                              error=f"Anthropic usage report returned HTTP {code}")
        except Exception as exc:  # noqa: BLE001 — the balance overlay is independent
            logger.warning("Anthropic API cost report unavailable: %s", type(exc).__name__)
            return self._base(available=False, status=STATUS_ERROR, error="Anthropic cost report unavailable")
        # Cost reports are daily and lag; they never debit the real-time balance.
        return self._spend(cost_totals(cost, now=now), "anthropic_reported")

    def _spend(self, totals: tuple[Decimal, ...], source: str):
        today, seven, month = totals
        spend = UsageSpend(today_usd=float(today), seven_day_usd=float(seven),
                           month_usd=float(month), source=source)
        return self._base(available=True, status=STATUS_OK, spend=spend, limits=[])
