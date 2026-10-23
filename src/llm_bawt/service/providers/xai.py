"""xAI provider adapter — API-key auth (TASK-825).

Straight API-key provider: the user pastes an ``xai-…`` key, we validate it
against ``GET /v1/api-key`` (returns the key's name and ACLs) and persist it
encrypted in the CredentialStore. Consumed DB-first by ``GrokClient`` /
``ResponsesClient`` via :func:`.api_key.resolve_api_key` — env ``XAI_API_KEY``
remains only as a deprecated fallback.
"""

from __future__ import annotations

import logging

import httpx

from .api_key import ApiKeyAdapter

logger = logging.getLogger(__name__)

_KEY_CHECK_URL = "https://api.x.ai/v1/api-key"


class XaiAdapter(ApiKeyAdapter):
    """API-key connection for api.x.ai (Grok models)."""

    id = "xai"
    label = "xAI"
    access_path_keys = ("xai-chat", "xai-responses")

    @staticmethod
    def _probe(key: str) -> tuple[str | None, str | None]:
        try:
            resp = httpx.get(
                _KEY_CHECK_URL,
                headers={"Authorization": f"Bearer {key}"},
                timeout=15.0,
            )
        except httpx.HTTPError as exc:
            return None, f"xAI unreachable: {exc}"
        if resp.status_code in (400, 401, 403):
            return None, "xAI rejected the API key"
        if resp.is_error:
            return None, f"xAI key check returned HTTP {resp.status_code}"
        try:
            data = resp.json()
        except ValueError:
            data = {}
        label = str(data.get("name") or "").strip() or "xai"
        return label, None
