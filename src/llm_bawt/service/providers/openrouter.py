"""OpenRouter provider adapter — API-key auth (TASK-822).

OpenRouter is a straight API-key provider: the user pastes an ``sk-or-…`` key,
we validate it against ``GET /api/v1/key`` (returns the key's label/limits),
and persist it encrypted in the :class:`CredentialStore`. No OAuth, no refresh
chain — the key is static until rotated.

Consumers:
- The claude-code bridge proxy's OpenRouter adapter resolves the key via the
  app broker endpoint ``GET /v1/providers/openrouter/token`` (never env vars).
- Model discovery (``/v1/models/upstream?provider=openrouter``) lists the
  public OpenRouter catalog; no key required for that endpoint.
"""

from __future__ import annotations

import logging

import httpx

from .base import (
    AUTH_API_KEY,
    STATUS_CONNECTED,
    ConnectionRecord,
    ProviderAdapter,
    ValidateResult,
)

logger = logging.getLogger(__name__)

_KEY_CHECK_URL = "https://openrouter.ai/api/v1/key"
_SECRET_FIELD = "api_key"


class OpenRouterAdapter(ProviderAdapter):
    """API-key connection for openrouter.ai."""

    id = "openrouter"
    label = "OpenRouter"
    auth_methods = (AUTH_API_KEY,)
    access_path_keys = ("openrouter-anthropic",)

    # --- api key -------------------------------------------------------------
    def set_api_key(self, api_key: str) -> ValidateResult:
        key = (api_key or "").strip()
        if not key:
            return ValidateResult(ok=False, detail="API key is empty")
        account, detail = self._probe(key)
        if account is None:
            return ValidateResult(ok=False, detail=detail)
        record = ConnectionRecord(
            provider=self.id,
            status=STATUS_CONNECTED,
            auth_method=AUTH_API_KEY,
            account=account,
            secret={_SECRET_FIELD: key},
        )
        self.store.save(record)
        logger.info("OpenRouter API key connected (label=%s)", account)
        return ValidateResult(ok=True, account=account)

    def validate(self, record: ConnectionRecord) -> ValidateResult:
        key = (record.secret or {}).get(_SECRET_FIELD)
        if not key:
            return ValidateResult(ok=False, detail="no API key stored")
        account, detail = self._probe(key)
        if account is None:
            return ValidateResult(ok=False, detail=detail)
        return ValidateResult(ok=True, account=account)

    # --- broker access -------------------------------------------------------
    def api_key(self) -> str | None:
        """Decrypted key for internal consumers (token broker); None if absent."""
        record = self.store.load(self.id)
        if not record or record.status != STATUS_CONNECTED:
            return None
        return (record.secret or {}).get(_SECRET_FIELD) or None

    # --- internals -----------------------------------------------------------
    @staticmethod
    def _probe(key: str) -> tuple[str | None, str | None]:
        """Validate the key upstream. Returns ``(account_label, error_detail)``."""
        try:
            resp = httpx.get(
                _KEY_CHECK_URL,
                headers={"Authorization": f"Bearer {key}"},
                timeout=15.0,
            )
        except httpx.HTTPError as exc:
            return None, f"OpenRouter unreachable: {exc}"
        if resp.status_code == 401:
            return None, "OpenRouter rejected the API key"
        if resp.is_error:
            return None, f"OpenRouter key check returned HTTP {resp.status_code}"
        try:
            data = resp.json().get("data") or {}
        except ValueError:
            data = {}
        label = str(data.get("label") or "").strip() or "openrouter"
        return label, None
