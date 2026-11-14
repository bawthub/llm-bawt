"""xAI prepaid balance: management-reported when configured, manual otherwise.

The reporting credential has no relationship to inference-key resolution. A
configured but unavailable report never silently falls back to the manual
anchor. Ledger spend may still be shown separately, never subtracted here.
"""

from __future__ import annotations

import asyncio
import hashlib
import time

import httpx

from ...providers.base import STATUS_CONNECTED
from ...providers.xai_management import CONNECTION_ID, balance_url, prepaid_usd, reporting_headers
from ..balance import BALANCE_ANCHOR, BALANCE_PROVIDER
from ..canonical import STATUS_ERROR, STATUS_OK, STATUS_RATE_LIMITED, STATUS_UNAUTHORIZED, UsageBalance
from .api_key_balance import ApiKeyBalanceAdapter


class XaiUsageAdapter(ApiKeyBalanceAdapter):
    provider = "xai"
    display_name = "xAI · Grok"
    backend = "claude-code"
    access_path_keys = ("xai-chat", "xai-responses")
    connection_id = "xai"
    report_ttl = 10.0  # debounce on-demand refreshes, including turn completions

    def _management_connection(self):
        from ...dependencies import get_service
        from ...providers.base import CredentialStore

        store = CredentialStore(get_service().config)
        if not store.available:
            raise RuntimeError("Reporting connection store unavailable")
        return store.load(CONNECTION_ID)

    @property
    def balance_mode(self):
        try:
            return BALANCE_ANCHOR if self._management_connection() is None else BALANCE_PROVIDER
        except Exception:  # unknown configuration must not enable manual writes
            return BALANCE_PROVIDER

    def report_identity(self):
        record = self._management_connection()
        if record is None:
            return ("manual",)
        # No plaintext key in the identity/cache/public payload. Include key and
        # team as well as timestamps so same-time rotations cannot reuse credit.
        key = str(record.secret.get("management_key") or "")
        return ("management", record.status, record.meta.get("team_id"), record.updated_at,
                hashlib.sha256(key.encode()).hexdigest())

    def anchored_balance_row(self, store):
        # Provider-mode ledger charges did not advance the manual counter.
        # Reconcile before every anchor read, including cached report overlays;
        # an unavailable recount must never resurrect overstated credit.
        return store.recount(self.provider)

    def _unavailable(self, status: str, error: str):
        if status != STATUS_RATE_LIMITED:
            from .. import invalidate_cache

            # Revocation/parse/network failures must not let the next ordinary
            # read resurrect a previously good balance as if it still worked.
            invalidate_cache(self.provider)
        return self._base(available=False, status=status, manual_balance_supported=False, error=error)

    async def fetch(self):
        try:
            record = await asyncio.to_thread(self._management_connection)
        except Exception:
            return self._unavailable(STATUS_ERROR, "xAI reporting connection unavailable")
        if record is None:
            return await super().fetch()
        native = {"manual_balance_supported": False}
        key = record.secret.get("management_key")
        if record.status != STATUS_CONNECTED or not isinstance(key, str) or not key:
            return self._unavailable(STATUS_UNAUTHORIZED,
                                     "xAI management reporting is not connected; reconnect the management key")
        try:
            url = balance_url(record.meta.get("team_id"))
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=False) as client:
                response = await client.get(url, headers=reporting_headers(key))
            if response.status_code != 200:
                code = response.status_code
                status = STATUS_UNAUTHORIZED if code in (401, 403) else STATUS_RATE_LIMITED if code == 429 else STATUS_ERROR
                return self._unavailable(status, f"xAI prepaid-balance read returned HTTP {code}")
            remaining = prepaid_usd(response.json())
        except httpx.HTTPError:
            return self._unavailable(STATUS_ERROR, "xAI prepaid balance is unreachable")
        except ValueError:
            return self._unavailable(STATUS_ERROR, "xAI prepaid balance or team configuration is invalid")
        return self._base(available=True, status=STATUS_OK, **native, limits=[],
                          balance=UsageBalance(amount_usd=None, remaining_usd=float(remaining), used_usd=None,
                                               as_of=None, source="provider_reported", updated_at=time.time()))
