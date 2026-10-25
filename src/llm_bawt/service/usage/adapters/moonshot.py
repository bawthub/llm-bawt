"""Moonshot AI (Kimi) usage adapter.

Moonshot is **prepaid pay-per-token**, not a subscription: there are no
rolling plan windows to poll (unlike Claude Max's 5h/weekly limits or the
z.ai GLM Coding Plan). What it *does* expose — and what xAI has no equivalent
of — is an account balance:

    GET https://api.moonshot.ai/v1/users/me/balance
    Authorization: Bearer $MOONSHOT_API_KEY

so instead of xAI's empty-limits placeholder we can surface real remaining
credit, which is the number that actually predicts "will my next turn work".

**Mapping note (deliberate, read before "fixing" it).** A prepaid balance has
no denominator: we know what's left, never what was topped up. The canonical
``balance`` (``source="provider_reported"``, TASK-1461) carries only
``remaining_usd``; the deposit, debits and anchor time stay ``None`` rather
than being invented. The legacy ``credit_balance`` limit row (balance in
``limit``, no ``used_pct``) remains for older clients only.

Response shape is handled defensively — it was not confirmable against a live
200 at implementation time (the vault key was revoked; see TASK-654). The
documented shape is::

    {"code": 0, "status": true, "scode": "0x0",
     "data": {"available_balance": 49.53, "voucher_balance": 0,
              "cash_balance": 49.53}}
"""

from __future__ import annotations

import logging
import os
import time

import httpx

from ..balance import BALANCE_PROVIDER
from ..base import UsageAdapter
from ..canonical import (
    STATUS_ERROR,
    STATUS_OK,
    STATUS_RATE_LIMITED,
    STATUS_UNAUTHORIZED,
    UsageBalance,
    UsageLimit,
)

logger = logging.getLogger(__name__)

API_KEY_ENVS = ("MOONSHOT_API_KEY", "LLM_BAWT_MOONSHOT_API_KEY", "KIMI_API_KEY")
DEFAULT_BASE_URL = "https://api.moonshot.ai/v1"
BASE_URL_ENV = "MOONSHOT_USAGE_BASE_URL"

# Below this many dollars the account is close to refusing requests, so the UI
# should shout. Override with MOONSHOT_LOW_BALANCE_USD.
_DEFAULT_LOW_BALANCE = 5.0

_TIMEOUT = httpx.Timeout(connect=10.0, read=15.0, write=10.0, pool=10.0)


class MoonshotUsageAdapter(UsageAdapter):
    """Prepaid Kimi credit, surfaced as remaining account balance."""

    provider = "moonshot"
    display_name = "Moonshot · Kimi"
    backend = "claude-code"  # reached via the claude-code bridge proxy
    access_path_keys = ("moonshot-anthropic",)
    kind = "balance"
    balance_mode = BALANCE_PROVIDER
    # Not a timer: readers refetch after a Moonshot turn completes or a page
    # opens; this only collapses a burst of those into one balance request.
    report_ttl = 10.0

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _api_key() -> str | None:
        for env in API_KEY_ENVS:
            key = os.getenv(env)
            if key:
                return key
        return None

    @staticmethod
    def _base_url() -> str:
        return (os.getenv(BASE_URL_ENV) or DEFAULT_BASE_URL).rstrip("/")

    @staticmethod
    def _low_balance() -> float:
        try:
            return float(
                os.getenv("MOONSHOT_LOW_BALANCE_USD", str(_DEFAULT_LOW_BALANCE))
            )
        except (TypeError, ValueError):
            return _DEFAULT_LOW_BALANCE

    @staticmethod
    def _extract_balance(payload: dict) -> float | None:
        """Pull available balance out of the documented (or a near) shape."""
        data = payload.get("data")
        candidates = [data, payload] if isinstance(data, dict) else [payload]
        for src in candidates:
            if not isinstance(src, dict):
                continue
            for key in ("available_balance", "cash_balance", "balance"):
                val = src.get(key)
                if isinstance(val, (int, float)):
                    return float(val)
        return None

    # -- fetch -----------------------------------------------------------

    async def fetch(self):
        key = self._api_key()
        if not key:
            return self._base(
                available=False,
                status=STATUS_UNAUTHORIZED,
                error=(
                    "No Moonshot API key configured "
                    "(set MOONSHOT_API_KEY / LLM_BAWT_MOONSHOT_API_KEY)."
                ),
                limits=[],
            )

        url = f"{self._base_url()}/users/me/balance"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(url, headers=headers)
        except httpx.HTTPError as e:
            # Network-level failure is an expected mode, not a crash.
            return self._base(
                available=False, status=STATUS_ERROR,
                error=f"Moonshot balance request failed: {e}", limits=[],
            )

        if resp.status_code in (401, 403):
            return self._base(
                available=False,
                status=STATUS_UNAUTHORIZED,
                error=(
                    "Moonshot rejected the API key "
                    f"({resp.status_code}) — it may be revoked or expired."
                ),
                limits=[],
            )
        if resp.status_code == 429:
            return self._base(
                available=False, status=STATUS_RATE_LIMITED,
                error="Moonshot balance endpoint rate-limited (429).", limits=[],
            )
        if resp.status_code >= 400:
            return self._base(
                available=False, status=STATUS_ERROR,
                error=f"Moonshot balance HTTP {resp.status_code}.", limits=[],
            )

        try:
            payload = resp.json()
        except ValueError:
            return self._base(
                available=False, status=STATUS_ERROR,
                error="Moonshot balance returned non-JSON.", limits=[],
            )

        balance = self._extract_balance(payload if isinstance(payload, dict) else {})
        if balance is None:
            # Authenticated fine, but we couldn't find the number — report ok-ish
            # with no limits rather than inventing one.
            logger.warning(
                "Moonshot balance: unrecognized payload keys=%s",
                list(payload)[:8] if isinstance(payload, dict) else type(payload),
            )
            return self._base(
                available=False,
                status=STATUS_ERROR,
                error="Could not parse available_balance from Moonshot response.",
                limits=[],
                raw=payload if isinstance(payload, dict) else None,
            )

        severity = "critical" if balance <= self._low_balance() else "normal"
        limit = UsageLimit(
            id="credit_balance",
            label="Available credit",
            # No denominator exists for a prepaid pool — see module docstring.
            used_pct=None,
            used=None,
            limit=balance,
            unit="USD",
            window=None,
            severity=severity,
            active=True,
        )
        return self._base(
            available=True,
            status=STATUS_OK,
            # The limit row serves clients from before ``balance``; remove it
            # once every deployed BawtHub reads kind="balance" (TASK-1461).
            limits=[limit],
            balance=UsageBalance(amount_usd=None, remaining_usd=balance, used_usd=None,
                                 as_of=None, source="provider_reported", updated_at=time.time()),
            raw=payload if isinstance(payload, dict) else None,
        )
