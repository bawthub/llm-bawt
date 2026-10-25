"""Anthropic API spend from organization reports, or local turn-log estimate.

The admin report is account-wide, while the no-admin-key fallback includes only
turns recorded by this app. Neither is a Claude subscription window. API key
retrieval is deliberately isolated from the inference provider/token broker.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx
from sqlalchemy import text

from ..base import UsageAdapter
from ..canonical import STATUS_ERROR, STATUS_OK, STATUS_RATE_LIMITED, STATUS_UNAUTHORIZED, UsageSpend

logger = logging.getLogger(__name__)
COST_URL = "https://api.anthropic.com/v1/organizations/cost_report"
ANTHROPIC_VERSION = "2023-06-01"
ADMIN_CONNECTION = "anthropic-admin"
BUDGET_SETTING = "anthropic_api_monthly_budget_usd"


def _decimal(value: Any) -> Decimal:
    try:
        n = Decimal(str(value))
        return n if n.is_finite() else Decimal(0)
    except (InvalidOperation, TypeError, ValueError):
        return Decimal(0)


def _utc_date(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else None
    except (ValueError, TypeError):
        return None


def _windows(now: datetime) -> tuple[datetime, datetime, datetime]:
    today = now.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return today, today - timedelta(days=6), today.replace(day=1)


def cost_totals(buckets: list[dict], *, now: datetime) -> tuple[Decimal, Decimal, Decimal]:
    """Aggregate USD cents as Decimal over UTC calendar day, 7 days, and month."""
    today_start, seven_start, month_start = _windows(now)
    totals = [Decimal(0), Decimal(0), Decimal(0)]
    for bucket in buckets:
        started = _utc_date(bucket.get("starting_at"))
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


def read_local_spend(engine: Any, now: datetime) -> tuple[Decimal, Decimal, Decimal]:
    """Count only recorded turns on the Anthropic API access path.

    Parse JSON in Python so a malformed legacy row cannot poison the entire
    aggregate. The SQL prefilter avoids loading unrelated providers' histories.
    """
    if engine is None:
        raise RuntimeError("turn-log database unavailable")
    today_start, seven_start, month_start = _windows(now)
    start = min(seven_start, month_start)
    total = [Decimal(0), Decimal(0), Decimal(0)]
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT created_at, model, token_usage_json FROM turn_logs
            WHERE created_at >= :start AND token_usage_json IS NOT NULL
              AND (model LIKE :model_suffix OR token_usage_json LIKE :path_hint)
        """), {"start": start, "model_suffix": "%@anthropic-api", "path_hint": '%"anthropic-api"%'})
        for created_at, model, raw in rows:
            try:
                usage = json.loads(raw)
            except (TypeError, ValueError):
                continue
            if not isinstance(usage, dict):
                continue
            frozen = usage.get("access_path")
            path = frozen.get("key") if isinstance(frozen, dict) else None
            # An explicit frozen path beats the model alias if they disagree.
            if path is None:
                path = str(model or "").rsplit("@", 1)[-1] if "@" in str(model or "") else None
            if path != "anthropic-api":
                continue
            cost = _decimal(usage.get("total_cost_usd"))
            if cost <= 0:
                continue
            started = created_at.astimezone(timezone.utc)
            if started >= today_start:
                total[0] += cost
            if started >= seven_start:
                total[1] += cost
            if started >= month_start:
                total[2] += cost
    return tuple(total)


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


class AnthropicApiUsageAdapter(UsageAdapter):
    provider = "anthropic_api"
    display_name = "Anthropic API"
    backend = "claude-code"
    access_path_keys = ("anthropic-api",)
    kind = "spend"

    def _admin_key(self) -> str | None:
        from ...dependencies import get_service
        from ...providers.base import CredentialStore, STATUS_CONNECTED

        record = CredentialStore(get_service().config).load(ADMIN_CONNECTION)
        if record is None or record.status != STATUS_CONNECTED:
            return None
        key = record.secret.get("admin_key")
        return key if isinstance(key, str) and key.startswith("sk-ant-admin") else None

    def _budget(self) -> Decimal | None:
        from ...dependencies import get_service
        from ....runtime_settings import RuntimeSettingsStore

        settings = RuntimeSettingsStore(get_service().config).get_scope_settings("global", "*")
        amount = _decimal(settings.get(BUDGET_SETTING))
        return amount if amount > 0 else None

    def _local_spend(self, now: datetime) -> tuple[Decimal, Decimal, Decimal]:
        from ...dependencies import get_service
        return read_local_spend(get_service()._turn_log_store.engine, now)

    async def fetch(self):
        now = datetime.now(timezone.utc)
        try:
            key = await asyncio.to_thread(self._admin_key)
            budget = await asyncio.to_thread(self._budget)
            if not key:
                totals = await asyncio.to_thread(self._local_spend, now)
                source = "bawt_recorded_estimate"
            else:
                _, _, month_start = _windows(now)
                async with httpx.AsyncClient(timeout=15.0) as client:
                    # The cost endpoint includes non-token charges (e.g. web search).
                    # Fetch only the report consumed by this spend view: an unused
                    # usage request must not mask a successful cost report.
                    cost = await _pages(client, COST_URL, key, {
                        "starting_at": month_start.isoformat(), "group_by[]": "description",
                        "bucket_width": "1d", "limit": "31",
                    })
                totals = cost_totals(cost, now=now)
                source = "anthropic_reported"
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            status = STATUS_UNAUTHORIZED if code in (401, 403) else STATUS_RATE_LIMITED if code == 429 else STATUS_ERROR
            return self._base(kind=self.kind, available=False, status=status,
                              error=f"Anthropic usage report returned HTTP {code}")
        except Exception as exc:  # noqa: BLE001 — no fabricated zero-spend on failure
            logger.warning("Anthropic API spend unavailable: %s", type(exc).__name__)
            return self._base(kind=self.kind, available=False, status=STATUS_ERROR,
                              error="Anthropic API spend unavailable")

        today, seven, month = totals
        spend = UsageSpend(
            today_usd=float(today), seven_day_usd=float(seven), month_usd=float(month),
            monthly_budget_usd=float(budget) if budget else None,
            monthly_used_pct=float(month / budget * 100) if budget else None,
            source=source,
        )
        return self._base(available=True, status=STATUS_OK, spend=spend, limits=[])
