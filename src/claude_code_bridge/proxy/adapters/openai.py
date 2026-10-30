"""OpenAI platform adapter — standard Responses API with brokered API-key auth.

This is intentionally distinct from :mod:`openai_chatgpt`: ``openai`` uses a
paid OpenAI platform API key against ``api.openai.com/v1/responses``, while
``openai_chatgpt`` uses subscription OAuth against the ChatGPT Codex backend.

The key remains encrypted in llm-bawt's CredentialStore.  Each inference asks
the internal provider broker for the current key, so rotation and disconnects
are visible immediately rather than hidden by a bridge-side credential cache.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import ClassVar

import httpx

from agent_bridge import bridge_token, preview_credentials

from .base import ProviderAdapter

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.openai.com/v1"
BASE_URL_ENV = "OPENAI_API_BASE_URL"


class OpenAIAdapter(ProviderAdapter):
    """OpenAI's standard Responses API using the connected platform key."""

    name: ClassVar[str] = "openai"

    @staticmethod
    def _fetch_broker_key() -> str:
        if preview_credentials.enabled():
            raise RuntimeError("OpenAI API keys are not served in preview")
        api_url = (os.environ.get("LLM_BAWT_API_URL") or "").rstrip("/")
        if not api_url:
            raise RuntimeError(
                "LLM_BAWT_API_URL not set — cannot reach the OpenAI API-key broker"
            )
        headers = bridge_token.headers()
        try:
            response = httpx.get(
                f"{api_url}/v1/providers/openai-api/token",
                headers=headers,
                timeout=15.0,
            )
        except httpx.HTTPError as exc:
            raise RuntimeError(f"OpenAI API-key broker unreachable: {exc}") from exc
        if response.is_error:
            # Match llm-bawt's DB-first credential resolver: an explicitly
            # unconfigured DB connection may use the deployment's legacy env
            # key. Other broker failures remain visible instead of being masked.
            env_key = os.getenv("OPENAI_API_KEY") if response.status_code == 503 else None
            if env_key:
                logger.warning(
                    "Using OPENAI_API_KEY from the bridge environment because "
                    "no DB-connected OpenAI key is installed"
                )
                return env_key
            raise RuntimeError(
                f"OpenAI API-key broker returned {response.status_code}: "
                f"{(response.text or '')[:200]} — connect an OpenAI API key "
                "in Settings → Providers."
            )
        key = (response.json() or {}).get("access_token")
        if not key:
            raise RuntimeError("OpenAI API-key broker returned no key")
        logger.info("Fetched OpenAI platform API key from app broker")
        return str(key)

    @staticmethod
    def _base_url() -> str:
        return (os.getenv(BASE_URL_ENV) or DEFAULT_BASE_URL).rstrip("/")

    async def authorize(self) -> tuple[str, str]:
        key = await asyncio.to_thread(self._fetch_broker_key)
        return key, self._base_url()
