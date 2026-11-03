"""Read-only xAI management credential; deliberately NOT an inference adapter.

Public contract: docs.x.ai/developers/rest-api-reference/management/{auth,billing}.
Only validation and invoice-preview GETs are used. No API-key/billing mutations,
inference access paths or token-broker method are exposed by this adapter.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any

import httpx

from .base import AUTH_API_KEY, STATUS_CONNECTED, ConnectionRecord, ProviderAdapter, ValidateResult

MANAGEMENT_ORIGIN = "https://management-api.x.ai"
VALIDATION_URL = f"{MANAGEMENT_ORIGIN}/auth/management-keys/validation"
CONNECTION_ID = "xai-management"
_TEAM_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_INT64 = re.compile(r"-?[0-9]{1,19}\Z")


def team_id_from_validation(payload: Any) -> str:
    """Resolve only a team-scoped key, never an organization's ID as a team.

    Current keys use scope/scopeId; teamId is the deprecated pre-scope shape.
    Do not fall back to teamId when an explicit non-team scope is present.
    """
    if not isinstance(payload, dict):
        raise ValueError("Invalid management-key validation response")
    scope = payload.get("scope")
    if scope == "SCOPE_TEAM":
        team_id = payload.get("scopeId")
    elif scope in (None, "SCOPE_UNSPECIFIED"):
        team_id = payload.get("teamId")
    else:
        raise ValueError("Use a team-scoped xAI management key for balance reporting")
    if not isinstance(team_id, str) or not _TEAM_ID.fullmatch(team_id):
        raise ValueError("Management key did not identify a valid xAI team")
    return team_id


def balance_url(team_id: str) -> str:
    """Current-cycle postpaid invoice preview; carries the live prepaid position.

    The prepaid/balance ledger total omits spend until xAI posts each month's
    SPEND entry (around the 11th of the next month), so it overstates credit.
    """
    if not isinstance(team_id, str) or not _TEAM_ID.fullmatch(team_id):
        raise ValueError("Invalid xAI team ID")
    return f"{MANAGEMENT_ORIGIN}/v1/billing/teams/{team_id}/postpaid/invoice/preview"


def _cents(container: Any, field: str) -> int:
    node = container.get(field) if isinstance(container, dict) else None
    value = node.get("val") if isinstance(node, dict) else None
    if not isinstance(value, str) or not _INT64.fullmatch(value):
        raise ValueError("Invalid xAI prepaid credit field")
    cents = int(value)
    if not -(2**63) <= cents < 2**63:
        raise ValueError("Invalid xAI prepaid credit field")
    return cents


def prepaid_usd(payload: Any) -> Decimal:
    """Remaining prepaid credit, matching the xAI console balance.

    From the invoice preview's coreInvoice: prepaidCredits is the credit
    available to this cycle and prepaidCreditsUsed is what this cycle has
    consumed, both as negative-signed int64 USD cents. Remaining is
    -(prepaidCredits) - -(prepaidCreditsUsed). Missing fields raise; no
    local spend reconstruction or fabricated zero.
    """
    core = payload.get("coreInvoice") if isinstance(payload, dict) else None
    return Decimal(-_cents(core, "prepaidCredits") + _cents(core, "prepaidCreditsUsed")) / 100


def reporting_headers(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}", "User-Agent": "llm-bawt/balance-report"}


class XaiManagementAdapter(ProviderAdapter):
    id = CONNECTION_ID
    label = "xAI management (balance reporting)"
    auth_methods = (AUTH_API_KEY,)
    access_path_keys = ()

    def descriptor(self) -> dict:
        result = super().descriptor()
        result.update({
            "category": "usage",
            "description": "Read-only xAI team prepaid balance; separate from the xAI inference key. Does not run models or change billing.",
            "credential_label": "Management key (read-only)",
            "credential_help": "Use a team-scoped xAI Management key with prepaid-balance read permission. The team ID is discovered from the key; validation also checks balance access. No write permissions are needed.",
            "setup_url": "https://console.x.ai",
        })
        return result

    @staticmethod
    def _probe(key: str) -> tuple[ValidateResult, str | None]:
        if not key:
            return ValidateResult(ok=False, detail="Enter an xAI Management key, not an inference key."), None
        try:
            # Never follow a redirect with this credential, nor accept a key
            # based only on validation: validation requires no ACL permissions.
            response = httpx.get(VALIDATION_URL, headers=reporting_headers(key), timeout=15.0, follow_redirects=False)
            if response.status_code != 200:
                return ValidateResult(ok=False, detail=f"xAI management validation returned HTTP {response.status_code}."), None
            team_id = team_id_from_validation(response.json())
            response = httpx.get(balance_url(team_id), headers=reporting_headers(key), timeout=15.0, follow_redirects=False)
            if response.status_code != 200:
                return ValidateResult(ok=False, detail=f"xAI prepaid-balance read returned HTTP {response.status_code}; check the management key's read permissions."), None
            prepaid_usd(response.json())
        except httpx.HTTPError:
            return ValidateResult(ok=False, detail="xAI management reporting is unreachable. Try again."), None
        except ValueError:
            return ValidateResult(ok=False, detail="xAI management reporting requires a team-scoped key and a valid prepaid-balance response."), None
        return ValidateResult(ok=True, account=f"xAI team {team_id}"), team_id

    def set_api_key(self, api_key: str) -> ValidateResult:
        key = (api_key or "").strip()
        result, team_id = self._probe(key)
        if result.ok:
            self.store.save(ConnectionRecord(
                provider=self.id, status=STATUS_CONNECTED, auth_method=AUTH_API_KEY,
                account=result.account, meta={"team_id": team_id}, secret={"management_key": key},
            ))
        return result

    def validate(self, record: ConnectionRecord) -> ValidateResult:
        result, _team_id = self._probe(str(record.secret.get("management_key") or ""))
        return result
