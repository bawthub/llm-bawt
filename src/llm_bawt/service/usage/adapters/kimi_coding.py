"""Kimi For Coding subscription usage adapter.

The coding-plan API exposes quota at ``GET /coding/v1/usages``.  The same API
key used for inference authorizes the read.  Kimi's payload has drifted between
``used`` and ``remaining`` fields, so parsing accepts either and preserves the
native response in ``raw`` for future refinement.
"""

from __future__ import annotations

import logging
from datetime import datetime

import httpx

from ...providers.bridge_keys import KimiCodingKeyAdapter
from ..base import UsageAdapter
from .stored_key import stored_api_key
from ..canonical import (
    STATUS_ERROR,
    STATUS_OK,
    STATUS_RATE_LIMITED,
    STATUS_UNAUTHORIZED,
    ProviderUsage,
    UsageLimit,
)

logger = logging.getLogger(__name__)

USAGE_URL = "https://api.kimi.com/coding/v1/usages"


def _number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _reset_seconds(value: object) -> int | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


def _used_and_limit(detail: dict) -> tuple[float | None, float | None]:
    limit = _number(detail.get("limit"))
    used = _number(detail.get("used"))
    if used is None and limit is not None:
        remaining = _number(detail.get("remaining"))
        if remaining is not None:
            used = max(0.0, limit - remaining)
    return used, limit


def _percentage(used: float | None, limit: float | None) -> float | None:
    if used is None or limit is None or limit <= 0:
        return None
    return round(max(0.0, used / limit * 100), 1)


def _window_label(window: dict) -> tuple[str, str, str | None]:
    duration = _number(window.get("duration"))
    unit = str(window.get("timeUnit") or "").upper()
    if duration is not None and "MINUTE" in unit:
        minutes = int(duration)
        if minutes >= 60 and minutes % 60 == 0:
            hours = minutes // 60
            if hours == 5:
                return "session_5h", "5-hour request limit", "5h"
            return f"window_{hours}h", f"{hours}-hour request limit", f"{hours}h"
        return f"window_{minutes}m", f"{minutes}-minute request limit", f"{minutes}m"
    if duration is not None and "HOUR" in unit:
        hours = int(duration)
        return f"window_{hours}h", f"{hours}-hour request limit", f"{hours}h"
    return "requests", "Request limit", None


def _to_limit(
    detail: dict,
    *,
    limit_id: str,
    label: str,
    window: str | None,
) -> UsageLimit | None:
    used, ceiling = _used_and_limit(detail)
    if used is None and ceiling is None:
        return None
    reset = detail.get("resetTime", detail.get("resetAt", detail.get("reset_at")))
    return UsageLimit(
        id=limit_id,
        label=label,
        used_pct=_percentage(used, ceiling),
        used=used,
        limit=ceiling,
        unit="requests",
        resets_at=_reset_seconds(reset),
        window=window,
        active=True,
    )


def _parse_limits(payload: dict) -> list[UsageLimit]:
    limits: list[UsageLimit] = []
    raw_limits = payload.get("limits")
    if isinstance(raw_limits, list):
        for item in raw_limits:
            if not isinstance(item, dict):
                continue
            detail = item.get("detail")
            if not isinstance(detail, dict):
                detail = item
            raw_window = item.get("window")
            window_data = raw_window if isinstance(raw_window, dict) else {}
            limit_id, label, window = _window_label(window_data)
            parsed = _to_limit(
                detail,
                limit_id=limit_id,
                label=label,
                window=window,
            )
            if parsed is not None:
                limits.append(parsed)

    weekly = payload.get("usage")
    if isinstance(weekly, dict):
        parsed = _to_limit(
            weekly,
            limit_id="weekly_all",
            label="Weekly request limit",
            window="7d",
        )
        if parsed is not None:
            limits.append(parsed)

    order = {"session_5h": 0, "weekly_all": 1}
    limits.sort(key=lambda item: order.get(item.id, 99))
    return limits


def _safe_json(response: httpx.Response) -> dict | None:
    try:
        value = response.json()
        return value if isinstance(value, dict) else {"value": value}
    except ValueError:
        return {"text": (response.text or "")[:500]}


class KimiCodingUsageAdapter(UsageAdapter):
    provider = "kimi_coding"
    display_name = "Kimi For Coding"
    backend = "claude-code"
    access_path_keys = ("kimi-coding-chat",)

    async def fetch(self) -> ProviderUsage:
        stored = await stored_api_key("kimi-coding", "Kimi For Coding", KimiCodingKeyAdapter.env_vars)
        if stored.key is None:
            return self._base(available=False, status=stored.status, error=stored.error, limits=[])
        key = stored.key

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.get(
                    USAGE_URL,
                    headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
                )
        except httpx.HTTPError as exc:
            return self._base(
                available=False,
                status=STATUS_ERROR,
                error=f"network error reaching Kimi usage: {exc}",
            )

        if response.status_code == 401:
            return self._base(
                available=False,
                status=STATUS_UNAUTHORIZED,
                error="Kimi rejected the API key on the usage endpoint.",
                raw=_safe_json(response),
            )
        if response.status_code == 429:
            return self._base(
                available=False,
                status=STATUS_RATE_LIMITED,
                error="Kimi usage endpoint is rate-limited; retry shortly.",
                raw=_safe_json(response),
            )
        if response.status_code >= 400:
            return self._base(
                available=False,
                status=STATUS_ERROR,
                error=f"HTTP {response.status_code} from Kimi usage endpoint.",
                raw=_safe_json(response),
            )

        payload = _safe_json(response)
        if payload is None or "text" in payload:
            return self._base(
                available=False,
                status=STATUS_ERROR,
                error="Non-JSON response from Kimi usage endpoint.",
                raw=payload,
            )

        if payload == {}:
            # Live answer for a valid key with no Kimi coding subscription:
            # HTTP 200 with an empty object, so there is no shape to parse.
            return self._base(
                available=False,
                status=STATUS_OK,
                error="No Kimi coding plan limits on this key.",
                raw=payload,
            )

        limits = _parse_limits(payload)
        if not limits:
            return self._base(
                available=False,
                status=STATUS_ERROR,
                error="Kimi authenticated but returned no recognizable usage limits.",
                raw=payload,
            )

        return self._base(
            available=True,
            status=STATUS_OK,
            limits=limits,
            raw=payload,
        )
