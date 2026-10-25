"""Anthropic API provider adapter — API-key auth (TASK-825).

Raw ``sk-ant-…`` API key for users without a Claude subscription — distinct
from the ``claude`` adapter (subscription OAuth, TASK-635). Validated against
``GET /v1/models`` with the ``x-api-key`` + ``anthropic-version`` headers
(401 = bad key). Persisted encrypted in the CredentialStore and consumed
DB-first via :func:`.api_key.resolve_api_key`.
"""

from __future__ import annotations

import logging

import httpx

from .api_key import ApiKeyAdapter

logger = logging.getLogger(__name__)

_KEY_CHECK_URL = "https://api.anthropic.com/v1/models"
_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicKeyAdapter(ApiKeyAdapter):
    """API-key connection for api.anthropic.com (raw key, not Claude sub)."""

    id = "anthropic-api"
    label = "Anthropic API"
    access_path_keys = ("anthropic-api",)

    @staticmethod
    def _probe(key: str) -> tuple[str | None, str | None]:
        if key.startswith("sk-ant-admin"):
            return None, "This is an Admin key; it can't run models. Add it under Usage reporting key (Admin)."
        try:
            resp = httpx.get(
                _KEY_CHECK_URL,
                headers={"x-api-key": key, "anthropic-version": _ANTHROPIC_VERSION},
                timeout=15.0,
            )
        except httpx.HTTPError as exc:
            return None, f"Anthropic unreachable: {exc}"
        if resp.status_code == 401:
            return None, "Anthropic rejected the API key"
        if resp.is_error:
            return None, f"Anthropic key check returned HTTP {resp.status_code}"
        org = str(resp.headers.get("anthropic-organization-id") or "").strip()
        return org or "anthropic", None
