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
from ..canonical import STATUS_ERROR, STATUS_OK, STATUS_RATE_LIMITED, STATUS_UNAUTHORIZED, UsageBalance, UsageSpend

logger = logging.getLogger(__name__)
COST_URL = "https://api.anthropic.com/v1/organizations/cost_report"
ANTHROPIC_VERSION = "2023-06-01"
ADMIN_CONNECTION = "anthropic-admin"
INFERENCE_CONNECTION = "anthropic-api"
BALANCE_SETTING = "anthropic_api_balance_anchor"


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


def read_local_spend(engine: Any, now: datetime, *, since: datetime) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    """Count verified API-path turns strictly after the connection/anchor time.

    A model alias before the key was connected was only catalog intent; it was
    not proof that the bridge used that key. A frozen path is required as well.
    The fourth result is the full post-anchor debit, even across month changes.
    """
    if engine is None:
        raise RuntimeError("turn-log database unavailable")
    today_start, seven_start, month_start = _windows(now)
    total = [Decimal(0), Decimal(0), Decimal(0), Decimal(0)]
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT created_at, token_usage_json FROM turn_logs
            WHERE created_at > :start AND created_at <= :end AND token_usage_json IS NOT NULL
              AND token_usage_json LIKE :path_hint
        """), {"start": since, "end": now, "path_hint": '%"anthropic-api"%'})
        for created_at, raw in rows:
            try:
                usage = json.loads(raw)
            except (TypeError, ValueError):
                continue
            if not isinstance(usage, dict):
                continue
            frozen = usage.get("access_path")
            if not isinstance(frozen, dict) or frozen.get("key") != "anthropic-api":
                continue
            cost = _decimal(usage.get("total_cost_usd"))
            if cost <= 0:
                continue
            started = created_at.astimezone(timezone.utc)
            if started <= since or started > now:
                continue
            total[3] += cost
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

    def _base(self, **overrides):
        return super()._base(balance_supported=True, **overrides)

    def _admin_key(self) -> str | None:
        from ...dependencies import get_service
        from ...providers.base import CredentialStore, STATUS_CONNECTED

        record = CredentialStore(get_service().config).load(ADMIN_CONNECTION)
        if record is None or record.status != STATUS_CONNECTED:
            return None
        key = record.secret.get("admin_key")
        return key if isinstance(key, str) and key.startswith("sk-ant-admin") else None

    def _anchor_and_connection(self) -> tuple[Decimal | None, datetime | None, datetime | None]:
        from ...dependencies import get_service
        from ...providers.base import CredentialStore
        from ....runtime_settings import RuntimeSettingsStore

        config = get_service().config
        settings = RuntimeSettingsStore(config).get_scope_settings("global", "*")
        raw = settings.get(BALANCE_SETTING)
        amount = _decimal(raw.get("amount_usd")) if isinstance(raw, dict) else Decimal(0)
        anchor = _utc_date(raw.get("as_of")) if isinstance(raw, dict) else None
        connection = CredentialStore(config).load(INFERENCE_CONNECTION)
        connected_at = _utc_date(connection.connected_at) if connection else None
        if amount <= 0 or (anchor and anchor > datetime.now(timezone.utc)):
            anchor = None  # invalid DB data is never a trustworthy zero-debit anchor
        return (amount if anchor else None, anchor, connected_at)

    def _local_spend(self, now: datetime, since: datetime) -> tuple[Decimal, Decimal, Decimal, Decimal]:
        from ...dependencies import get_service
        return read_local_spend(get_service()._turn_log_store.engine, now, since=since)

    async def fetch(self):
        now = datetime.now(timezone.utc)
        try:
            key = await asyncio.to_thread(self._admin_key)
            amount, anchor, connected_at = await asyncio.to_thread(self._anchor_and_connection)
            if connected_at is None:
                raise ValueError("Anthropic API connection time is unavailable")
            since = max((time for time in (anchor, connected_at) if time is not None))
            local = await asyncio.to_thread(self._local_spend, now, since)
        except Exception as exc:  # noqa: BLE001 — no fabricated zero-spend on failure
            logger.warning("Anthropic API balance unavailable: %s", type(exc).__name__)
            return self._base(available=False, status=STATUS_ERROR,
                              error="Anthropic API balance or connection time unavailable")

        balance = None
        if amount is not None and anchor is not None:
            used = local[3]
            balance = UsageBalance(amount_usd=float(amount), remaining_usd=float(amount - used),
                                   used_usd=float(used), as_of=int(anchor.timestamp()))

        totals = local[:3]
        source = "bawt_recorded_estimate"
        if key:
            try:
                _, _, month_start = _windows(now)
                async with httpx.AsyncClient(timeout=15.0) as client:
                    # Cost reports are daily and lag; their first bucket cannot
                    # isolate an arbitrary anchor time. Never use them to debit
                    # a real-time prepaid balance (local frozen-path turns do).
                    cost = await _pages(client, COST_URL, key, {
                        "starting_at": month_start.isoformat(), "group_by[]": "description",
                        "bucket_width": "1d", "limit": "31",
                    })
                totals = cost_totals(cost, now=now)
                source = "anthropic_reported"
            except httpx.HTTPStatusError as exc:
                code = exc.response.status_code
                status = STATUS_UNAUTHORIZED if code in (401, 403) else STATUS_RATE_LIMITED if code == 429 else STATUS_ERROR
                return self._base(available=False, status=status, balance=balance,
                                  error=f"Anthropic usage report returned HTTP {code}; local balance is still available")
            except Exception as exc:  # noqa: BLE001 — preserve a valid local balance
                logger.warning("Anthropic API cost report unavailable: %s", type(exc).__name__)
                return self._base(available=False, status=STATUS_ERROR, balance=balance,
                                  error="Anthropic cost report unavailable; local balance is still available")

        today, seven, month = totals
        spend = UsageSpend(today_usd=float(today), seven_day_usd=float(seven),
                           month_usd=float(month), source=source)
        return self._base(available=True, status=STATUS_OK, spend=spend, balance=balance, limits=[])
