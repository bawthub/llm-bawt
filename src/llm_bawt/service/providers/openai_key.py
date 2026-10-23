"""OpenAI platform provider adapter — API-key auth (TASK-825).

Raw ``sk-…`` platform API key, validated against ``GET /v1/models``. Distinct
from the ``codex`` adapter (ChatGPT-subscription OAuth): this one powers direct
OpenAI API usage in ``OpenAIClient`` / ``ResponsesClient``, resolved DB-first
via :func:`.api_key.resolve_api_key` — env ``OPENAI_API_KEY`` remains only as a
deprecated fallback.
"""

from __future__ import annotations

import logging

import httpx

from .api_key import ApiKeyAdapter

logger = logging.getLogger(__name__)

_KEY_CHECK_URL = "https://api.openai.com/v1/models"


class OpenAIKeyAdapter(ApiKeyAdapter):
    """API-key connection for api.openai.com (platform key, not ChatGPT sub)."""

    id = "openai-api"
    label = "OpenAI API"
    access_path_keys = ("openai-api", "openai-responses")

    @staticmethod
    def _probe(key: str) -> tuple[str | None, str | None]:
        try:
            resp = httpx.get(
                _KEY_CHECK_URL,
                headers={"Authorization": f"Bearer {key}"},
                timeout=15.0,
            )
        except httpx.HTTPError as exc:
            return None, f"OpenAI unreachable: {exc}"
        if resp.status_code == 401:
            return None, "OpenAI rejected the API key"
        if resp.is_error:
            return None, f"OpenAI key check returned HTTP {resp.status_code}"
        org = str(resp.headers.get("openai-organization") or "").strip()
        return org or "openai", None
