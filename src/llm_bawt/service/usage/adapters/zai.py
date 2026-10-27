"""z.ai (Zhipu GLM Coding Plan) usage adapter.

Queries the GLM Coding Plan quota via z.ai's monitor API and maps it into
canonical form. Unlike Claude, the same API key that authorizes inference
(``ZAI_API_KEY``) authorizes the usage read — no separate OAuth login.

Endpoint: ``GET https://api.z.ai/api/monitor/usage/quota/limit``

The response's ``data.limits[]`` uses an opaque ``(type, unit, number)``
scheme, decoded against z.ai's own ``glm-plan-usage`` plugin plus live
reset-cadence confirmation:

  - ``TIME_LIMIT``   (unit=5, month)  -> monthly MCP pool (search / web-reader / vision)
  - ``TOKENS_LIMIT`` (unit=3, hour)   -> 5-hour GLM token window
  - ``TOKENS_LIMIT`` (unit=6, week)   -> weekly GLM token window

``percentage`` is percent-used (0-100); ``nextResetTime`` is epoch-ms. The
native payload is always preserved as ``raw`` so unmapped or changed fields
are never silently lost — the first live call after any upstream change is
the moment to tighten this mapper.
"""

from __future__ import annotations

import logging
import time

import httpx

from ...providers.bridge_keys import ZaiKeyAdapter
from ..base import UsageAdapter
from .stored_key import stored_api_key
from ..canonical import (
    ProviderUsage,
    UsageLimit,
    STATUS_OK,
    STATUS_UNAUTHORIZED,
    STATUS_RATE_LIMITED,
    STATUS_ERROR,
)

logger = logging.getLogger(__name__)

QUOTA_URL = "https://api.z.ai/api/monitor/usage/quota/limit"
_PLAN_LABEL = {"lite": "Lite", "pro": "Pro", "max": "Max"}


def _reset_ms(value) -> int | None:
    """z.ai ``nextResetTime`` is epoch-ms -> unix seconds."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return int(v / 1000) if v > 1e12 else int(v)


def _pct(value) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 1)
    except (TypeError, ValueError):
        return None


def _token_window(item: dict) -> tuple[str, str, str]:
    """``(canonical id, label, window)`` for a TOKENS_LIMIT entry.

    Decoded by z.ai's unit enum (3=hour -> 5h, 6=week -> weekly), with a
    reset-cadence fallback: a 5h rolling window always resets within hours,
    a weekly window within days.
    """
    unit = item.get("unit")
    if unit == 3:
        return ("session_5h", "5-hour token limit", "5h")
    if unit == 6:
        return ("weekly_all", "Weekly token limit", "7d")
    resets = _reset_ms(item.get("nextResetTime"))
    if resets is not None:
        horizon = resets - int(time.time())
        if horizon <= 86_400:  # resets within a day -> 5h rolling window
            return ("session_5h", "5-hour token limit", "5h")
        return ("weekly_all", "Weekly token limit", "7d")
    return ("tokens", "Token limit", None)


def _from_limit_item(item: dict, idx: int) -> UsageLimit | None:
    typ = item.get("type")
    pct = _pct(item.get("percentage"))
    resets = _reset_ms(item.get("nextResetTime"))
    if typ == "TIME_LIMIT":
        # Monthly MCP pool (search / web-reader / vision). The per-tool
        # breakdown (usageDetails) is preserved on the snapshot via ``raw``.
        used = item.get("currentValue")
        total = item.get("usage")
        return UsageLimit(
            id="monthly_mcp",
            label="MCP usage (monthly)",
            used_pct=pct,
            resets_at=resets,
            window="30d",
            used=float(used) if isinstance(used, (int, float)) else None,
            limit=float(total) if isinstance(total, (int, float)) else None,
            unit="calls",
        )
    if typ == "TOKENS_LIMIT":
        cid, label, window = _token_window(item)
        return UsageLimit(id=cid, label=label, used_pct=pct, resets_at=resets, window=window)
    # Unknown type — surface generically so it isn't silently dropped.
    cid = str(typ or f"window_{idx}").lower()
    label = str(typ or f"window {idx}").replace("_", " ").title()
    return UsageLimit(id=cid, label=label, used_pct=pct, resets_at=resets)


def _safe_json(resp: httpx.Response) -> dict | None:
    try:
        body = resp.json()
        return body if isinstance(body, dict) else {"value": body}
    except Exception:  # noqa: BLE001
        return {"text": (resp.text or "")[:500]}


class ZaiUsageAdapter(UsageAdapter):
    provider = "zai"
    display_name = "z.ai · GLM"
    backend = "claude-code"
    access_path_keys = ("zai-anthropic",)

    async def fetch(self) -> ProviderUsage:
        stored = await stored_api_key("zai", "Z.AI", ZaiKeyAdapter.env_vars)
        if stored.key is None:
            return self._base(available=False, status=stored.status, error=stored.error)
        key = stored.key
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(QUOTA_URL, headers=headers)
        except httpx.HTTPError as e:
            return self._base(
                available=False, status=STATUS_ERROR, error=f"network error reaching z.ai: {e}"
            )

        if resp.status_code == 401:
            return self._base(
                available=False,
                status=STATUS_UNAUTHORIZED,
                error="z.ai rejected the API key on the usage endpoint.",
                raw=_safe_json(resp),
            )
        if resp.status_code == 429:
            return self._base(
                available=False,
                status=STATUS_RATE_LIMITED,
                error="z.ai usage endpoint is rate-limited; retry shortly.",
            )
        if resp.status_code >= 400:
            return self._base(
                available=False,
                status=STATUS_ERROR,
                error=f"HTTP {resp.status_code} from z.ai usage endpoint.",
                raw=_safe_json(resp),
            )

        try:
            payload = resp.json()
        except ValueError:
            return self._base(
                available=False,
                status=STATUS_ERROR,
                error="Non-JSON response from z.ai usage endpoint.",
                raw={"text": (resp.text or "")[:500]},
            )

        data = payload.get("data") if isinstance(payload, dict) else None
        level = None
        limits: list[UsageLimit] = []
        if isinstance(data, dict):
            level = data.get("level")
            raw_limits = data.get("limits")
            if isinstance(raw_limits, list):
                for i, item in enumerate(raw_limits):
                    if isinstance(item, dict):
                        lim = _from_limit_item(item, i)
                        if lim is not None:
                            limits.append(lim)

        # 5h first, then weekly, then monthly MCP, then anything else.
        order = {"session_5h": 0, "weekly_all": 1, "monthly_mcp": 2}
        limits.sort(key=lambda l: order.get(l.id, 99))

        raw = payload if isinstance(payload, dict) else {"value": payload}

        if not limits:
            # Key authenticated but reported no Coding Plan limits — a
            # pay-as-you-go key rather than a subscription. Surfaced (not
            # hidden) with a clear note; cached as OK since re-querying won't
            # change until the key is put on a plan.
            return self._base(
                available=False,
                status=STATUS_OK,
                error="No GLM Coding Plan limits on this key (pay-as-you-go).",
                raw=raw,
            )

        display_name = self.display_name
        if level:
            tier = _PLAN_LABEL.get(str(level).lower(), str(level).title())
            display_name = f"{self.display_name} · {tier}"

        return self._base(
            available=True,
            status=STATUS_OK,
            display_name=display_name,
            limits=limits,
            raw=raw,
        )
