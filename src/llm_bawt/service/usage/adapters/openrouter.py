"""OpenRouter usage adapter: provider-reported credit (TASK-1733).

``GET /api/v1/credits`` answers with the connected inference key (verified
live 2026-10-29; no management key needed)::

    {"data": {"total_credits": 10, "total_usage": 0.97}}

Both are account-wide USD, so unlike Moonshot the deposit and debits are
known: ``remaining = total_credits - total_usage``. This app's own per-day
chat/maintenance split still comes from the local ledger (``spend.daily``).
"""

from __future__ import annotations

import logging
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
)
from .api_key_balance import utc_datetime
from .stored_key import stored_api_key

logger = logging.getLogger(__name__)

#: OpenRouter's own API origin (first-party, not deployment data).
CREDITS_URL = "https://openrouter.ai/api/v1/credits"
_TIMEOUT = httpx.Timeout(connect=10.0, read=15.0, write=10.0, pool=10.0)


class OpenRouterUsageAdapter(UsageAdapter):
    """OpenRouter API key: account credit read from the provider."""

    provider = "openrouter"
    display_name = "OpenRouter"
    backend = "claude-code"  # reached via the claude-code bridge proxy
    access_path_keys = ("openrouter-anthropic",)
    kind = "balance"
    balance_mode = BALANCE_PROVIDER
    connection_id = "openrouter"
    # Not a timer: collapses a burst of post-turn refetches into one request.
    report_ttl = 10.0

    def credential_connected_at(self):
        from ...dependencies import get_service
        from ...providers.base import CredentialStore

        record = CredentialStore(get_service().config).load(self.connection_id)
        return utc_datetime(record.connected_at) if record else None

    def _fail(self, status: str, error: str):
        return self._base(available=False, status=status, error=error, limits=[])

    async def fetch(self):
        # The bridge reads this key only through the app broker, never env.
        stored = await stored_api_key(self.connection_id, self.display_name, ())
        if stored.key is None:
            return self._fail(stored.status, stored.error)
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(CREDITS_URL, headers={"Authorization": f"Bearer {stored.key}"})
        except httpx.HTTPError as exc:
            return self._fail(STATUS_ERROR, f"OpenRouter credit request failed: {type(exc).__name__}")
        if resp.status_code in (401, 403):
            return self._fail(STATUS_UNAUTHORIZED,
                              f"OpenRouter rejected the key for credits ({resp.status_code}).")
        if resp.status_code == 429:
            return self._fail(STATUS_RATE_LIMITED, "OpenRouter credits endpoint rate-limited (429).")
        if resp.status_code >= 400:
            return self._fail(STATUS_ERROR, f"OpenRouter credits HTTP {resp.status_code}.")
        try:
            data = resp.json().get("data") or {}
            total, used = float(data["total_credits"]), float(data["total_usage"])
        except (ValueError, KeyError, TypeError, AttributeError):
            logger.warning("OpenRouter credits: unrecognized payload")
            return self._fail(STATUS_ERROR, "Could not parse OpenRouter credits.")
        return self._base(
            available=True, status=STATUS_OK, limits=[],
            balance=UsageBalance(amount_usd=total, remaining_usd=total - used, used_usd=used,
                                 as_of=None, source="provider_reported", updated_at=time.time()),
        )
