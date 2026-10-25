"""OpenAI ChatGPT (codex subscription) usage adapter.

The official Codex client reads subscription limits from
``GET https://chatgpt.com/backend-api/wham/usage``. This adapter uses that
non-inference endpoint as the authoritative refresh path. The claude-code
bridge also passively harvests equivalent ``x-codex-*`` headers from real
``/responses`` turns and publishes a canonical Redis snapshot; that snapshot
is a fallback only when the dedicated endpoint is temporarily unavailable.

Mapping — window identity comes from its reported duration, never its slot.
``primary``/``secondary`` are positions, not meanings: Plus sends a 5-hour
primary plus a weekly secondary, while Pro-tier plans (e.g. Prolite) send a
single weekly window in the primary slot.
- 300-minute window   → ``session_5h``  (the rolling 5-hour limit)
- 10080-minute window → ``weekly_all``  (the weekly all-models limit)
- any other duration  → ``window_<dur>`` with a duration label
- unknown duration    → slot fallback (primary = 5h, secondary = weekly)
- ``plan_type`` → tiered display name, e.g. ``ChatGPT · codex · Plus``
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time

from ..base import UsageAdapter
from ..canonical import (
    ProviderUsage,
    UsageLimit,
    STATUS_ERROR,
    STATUS_OK,
    STATUS_USAGE_STALE,
)

logger = logging.getLogger(__name__)

# Shared contract with the bridge-side writer. Keep both in lockstep.
REDIS_KEY = "llm_bawt:usage:openai_chatgpt:snapshot"
_TTL_SECONDS = 7 * 24 * 3600
_SNAPSHOT_MAX_AGE_SECONDS = 60
_USAGE_URL = "https://chatgpt.com/backend-api/wham/usage"
_USAGE_URL_ENV = "OPENAI_CHATGPT_USAGE_URL"

_PLAN_LABEL = {
    "free": "Free",
    "plus": "Plus",
    "pro": "Pro",
    "team": "Team",
    "business": "Business",
    "enterprise": "Enterprise",
}


def _display_name(plan_type) -> str:
    base = OpenAIChatGPTUsageAdapter.display_name
    if not plan_type:
        return base
    raw = str(plan_type).strip()
    if not raw:
        return base
    label = _PLAN_LABEL.get(raw.lower(), raw.replace("_", " ").title())
    return f"{base} · {label}"


def _window_label(minutes) -> str | None:
    try:
        m = int(minutes)
    except (TypeError, ValueError):
        return None
    if m <= 0:
        return None
    if m % 1440 == 0:
        return f"{m // 1440}d"
    if m % 60 == 0:
        return f"{m // 60}h"
    return f"{m}m"


_SESSION = ("session_5h", "5-hour limit", "5h")
_WEEKLY = ("weekly_all", "Weekly · all models", "7d")
_SLOT_FALLBACK = {"primary": _SESSION, "secondary": _WEEKLY}


def _window_identity(window: dict | None, slot: str) -> tuple[str, str, str]:
    """(id, label, window) from the reported duration; slot only if unknown."""
    dur = _window_label(window.get("window_minutes")) if isinstance(window, dict) else None
    if dur is None:
        return _SLOT_FALLBACK[slot]
    for identity in (_SESSION, _WEEKLY):
        if identity[2] == dur:
            return identity
    return f"window_{dur}", f"{dur} limit", dur



def _snapshot_age(snap: dict | None) -> float | None:
    if not isinstance(snap, dict):
        return None
    try:
        return time.time() - float(snap.get("captured_at"))
    except (TypeError, ValueError):
        return None


def _snapshot_is_stale(snap: dict | None) -> bool:
    age = _snapshot_age(snap)
    if age is None or age > _SNAPSHOT_MAX_AGE_SECONDS:
        return True
    now = time.time()
    for key in ("primary", "secondary"):
        window = snap.get(key) if isinstance(snap, dict) else None
        if not isinstance(window, dict):
            continue
        try:
            reset_at = window.get("reset_at")
            if reset_at is not None and float(reset_at) <= now:
                return True
        except (TypeError, ValueError):
            pass
    return False


def _advance_reset(reset_at, window_minutes, now: int):
    """Roll an elapsed reset boundary forward by whole windows past ``now``.

    Codex snapshots are frozen between turns, so a stored ``reset_at`` can sit
    well in the past. For a rolling window that just means N whole windows have
    elapsed; project the boundary forward so the displayed reset is the next
    real one instead of a stale past timestamp.
    """
    if reset_at is None:
        return None
    try:
        period = int(window_minutes) * 60
    except (TypeError, ValueError):
        period = 0
    if period <= 0 or reset_at > now:
        return reset_at
    missed = (now - reset_at) // period + 1
    return int(reset_at + missed * period)


def _limit_from_window(window: dict | None, slot: str, now: int) -> UsageLimit | None:
    if not isinstance(window, dict):
        return None
    cid, label, fallback_window = _window_identity(window, slot)
    used = window.get("used_percent")
    if used is None:
        return None
    try:
        used_pct = float(used)
    except (TypeError, ValueError):
        return None
    reset_at = window.get("reset_at")
    window_minutes = window.get("window_minutes")
    # A rolling window whose reset boundary has already passed has, by
    # definition, rolled over: usage is back to zero until the next codex turn
    # writes a fresh snapshot. The frozen ``used_percent`` is stale — don't
    # replay it as a scary full bar. Zero it and project the reset forward.
    if reset_at is not None and reset_at <= now:
        used_pct = 0.0
        reset_at = _advance_reset(reset_at, window_minutes, now)
    return UsageLimit(
        id=cid,
        label=label,
        used_pct=used_pct,
        resets_at=reset_at,
        window=_window_label(window_minutes) or fallback_window,
        active=True,
    )


class OpenAIChatGPTUsageAdapter(UsageAdapter):
    provider = "openai_chatgpt"
    display_name = "ChatGPT · codex"
    backend = "claude-code"
    access_path_keys = ("openai-oauth",)

    async def _read_snapshot(self) -> dict | None:
        url = (
            os.getenv("LLM_BAWT_REDIS_URL")
            or os.getenv("REDIS_URL")
            or "redis://redis:6379/0"
        )
        try:
            import redis.asyncio as redis

            client = redis.from_url(
                url,
                decode_responses=True,
                socket_timeout=2,
                socket_connect_timeout=2,
            )
            try:
                raw = await client.get(REDIS_KEY)
            finally:
                await client.aclose()
        except Exception as e:  # noqa: BLE001
            logger.warning("codex usage snapshot read failed: %s", e)
            return None
        if not raw:
            return None
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else None
        except (ValueError, TypeError):
            return None

    async def _write_snapshot(self, snap: dict) -> None:
        url = (
            os.getenv("LLM_BAWT_REDIS_URL")
            or os.getenv("REDIS_URL")
            or "redis://redis:6379/0"
        )
        try:
            import redis.asyncio as redis

            client = redis.from_url(
                url,
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=2,
                socket_connect_timeout=2,
            )
            try:
                await client.set(REDIS_KEY, json.dumps(snap), ex=_TTL_SECONDS)
            finally:
                await client.aclose()
        except Exception as e:  # noqa: BLE001
            logger.warning("codex usage snapshot write failed: %s", e)

    @staticmethod
    def _window_from_payload(window: dict | None) -> dict:
        if not isinstance(window, dict):
            return {}
        duration = window.get("limit_window_seconds")
        try:
            minutes = int(duration) // 60 if duration is not None else None
        except (TypeError, ValueError):
            minutes = None
        return {
            "used_percent": window.get("used_percent"),
            "window_minutes": minutes,
            "reset_at": window.get("reset_at"),
            "reset_after_seconds": window.get("reset_after_seconds"),
        }

    @classmethod
    def _snapshot_from_payload(cls, payload: dict) -> dict | None:
        rate_limit = payload.get("rate_limit")
        if not isinstance(rate_limit, dict):
            return None
        primary = cls._window_from_payload(rate_limit.get("primary_window"))
        secondary = cls._window_from_payload(rate_limit.get("secondary_window"))
        if primary.get("used_percent") is None and secondary.get("used_percent") is None:
            return None
        return {
            "captured_at": int(time.time()),
            "plan_type": payload.get("plan_type"),
            "active_limit": payload.get("rate_limit_reached_type"),
            "primary": primary,
            "secondary": secondary,
            "credits": payload.get("credits") if isinstance(payload.get("credits"), dict) else {},
        }

    async def _fetch_usage_once(self, result) -> tuple[dict | None, int | None]:
        """Fetch the official non-inference Codex usage endpoint once."""
        import httpx

        headers = {
            "Authorization": f"Bearer {result.token}",
            "Accept": "application/json",
            "User-Agent": "llm-bawt-usage/1",
        }
        if result.account_id:
            headers["ChatGPT-Account-Id"] = result.account_id
        url = os.getenv(_USAGE_URL_ENV) or _USAGE_URL
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            logger.warning("codex usage endpoint failed: HTTP %s", resp.status_code)
            return None, resp.status_code
        try:
            payload = resp.json()
        except ValueError:
            logger.warning("codex usage endpoint returned non-JSON data")
            return None, resp.status_code
        snap = self._snapshot_from_payload(payload) if isinstance(payload, dict) else None
        if snap is None:
            logger.warning("codex usage endpoint returned no usable limit windows")
            return None, resp.status_code
        await self._write_snapshot(snap)
        return snap, resp.status_code

    async def _fetch_live_snapshot(self) -> dict | None:
        """Fetch usage directly, force-rotating the app-owned token once on 401."""
        try:
            from ..codex_oauth import get_access_token

            result = await asyncio.to_thread(get_access_token)
            if not result.token:
                logger.warning("codex usage endpoint: no ChatGPT token available")
                return None
            snap, status = await self._fetch_usage_once(result)
            if snap is not None or status != 401:
                return snap
            logger.info("codex usage endpoint got 401 — force-refreshing token")
            result = await asyncio.to_thread(get_access_token, force_refresh=True)
            if not result.token:
                return None
            snap, _ = await self._fetch_usage_once(result)
            return snap
        except Exception as e:  # noqa: BLE001
            logger.warning("codex usage endpoint failed: %s", e)
            return None

    async def fetch(self) -> ProviderUsage:
        passive_snap = await self._read_snapshot()
        snap = await self._fetch_live_snapshot()
        live = snap is not None
        if snap is None:
            snap = passive_snap
        if snap is None:
            return self._base(
                available=False,
                status=STATUS_ERROR,
                error=(
                    "Could not fetch Codex usage, and no passive snapshot from a "
                    "recent model response is available."
                ),
            )
        stale = not live and _snapshot_is_stale(snap)

        now = int(time.time())
        limits = [
            limit
            for slot in ("primary", "secondary")
            if (limit := _limit_from_window(snap.get(slot), slot, now)) is not None
        ]

        display = _display_name(snap.get("plan_type"))
        if not limits:
            return self._base(
                available=False,
                status=STATUS_OK,
                display_name=display,
                error="Codex snapshot present but carried no usable limit windows.",
                raw=snap,
            )

        return self._base(
            available=True,
            status=STATUS_USAGE_STALE if stale else STATUS_OK,
            display_name=display,
            error=(
                "Showing cached Codex usage; the dedicated usage endpoint is "
                "temporarily unavailable. Credential health is checked separately."
                if stale
                else None
            ),
            fetched_at=snap.get("captured_at"),
            cached=not live,
            limits=limits,
            raw=snap,
        )
