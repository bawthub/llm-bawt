"""Read-only Anthropic organization-usage credential, isolated from inference.

This is deliberately not an ApiKeyAdapter: the generic /providers/{id}/token
broker exposes inference keys, whereas admin credentials must never be handed
to an inference client or bridge. Only the usage adapter reads this slot.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx

from .base import AUTH_API_KEY, STATUS_CONNECTED, ConnectionRecord, ProviderAdapter, ValidateResult

# Validation and reporting both call the same official Admin endpoint; keep the
# credential adapter independent of the usage registry to avoid import cycles.
_COST_URL = "https://api.anthropic.com/v1/organizations/cost_report"
_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicAdminAdapter(ProviderAdapter):
    id = "anthropic-admin"
    label = "Usage reporting key (Admin)"
    auth_methods = (AUTH_API_KEY,)
    access_path_keys = ()  # NEVER an inference credential

    def descriptor(self) -> dict:
        result = super().descriptor()
        result.update({
            "category": "usage",
            "description": "Read-only organization spending reports; does not run models or replace the Anthropic API inference key.",
            "credential_label": "Usage reporting key (Admin)",
            "credential_help": "Use an Anthropic Admin API key (sk-ant-admin…). This key reads organization usage only.",
            "setup_url": "https://platform.claude.com/settings/admin-keys",
        })
        return result

    @staticmethod
    def _probe(key: str) -> ValidateResult:
        if not key.startswith("sk-ant-admin"):
            return ValidateResult(ok=False, detail="Usage reporting requires an Admin key (sk-ant-admin…), not an inference key.")
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        try:
            response = httpx.get(
                _COST_URL,
                params={"starting_at": (today - timedelta(days=1)).isoformat(),
                        "ending_at": today.isoformat(), "limit": 1},
                headers={"x-api-key": key, "anthropic-version": _ANTHROPIC_VERSION,
                         "User-Agent": "llm-bawt/usage-report"},
                timeout=15.0,
            )
        except httpx.HTTPError:
            return ValidateResult(ok=False, detail="Anthropic usage report is unreachable. Try again.")
        if response.is_error:
            return ValidateResult(ok=False, detail=f"Anthropic Admin usage read returned HTTP {response.status_code}.")
        try:
            if not isinstance(response.json().get("data"), list):
                raise ValueError("missing data")
        except (ValueError, AttributeError):
            return ValidateResult(ok=False, detail="Anthropic Admin usage read returned an invalid report.")
        return ValidateResult(ok=True, account="organization usage")

    def set_api_key(self, api_key: str) -> ValidateResult:
        key = (api_key or "").strip()
        result = self._probe(key)
        if result.ok:
            self.store.save(ConnectionRecord(
                provider=self.id, status=STATUS_CONNECTED, auth_method=AUTH_API_KEY,
                account=result.account, secret={"admin_key": key},
            ))
        return result

    def validate(self, record: ConnectionRecord) -> ValidateResult:
        return self._probe(str(record.secret.get("admin_key") or ""))
