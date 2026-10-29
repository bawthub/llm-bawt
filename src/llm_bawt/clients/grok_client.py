"""Grok (xAI) API client.

Uses the Responses API (/v1/responses) via the OpenAI SDK with xAI's base URL.
API documentation: https://docs.x.ai/api
"""

from __future__ import annotations

import logging

from .responses_client import ResponsesClient
from ..utils.config import Config

logger = logging.getLogger(__name__)


class GrokClient(ResponsesClient):
    """Client for Grok (xAI) API via the Responses API.

    The xAI API is OpenAI-compatible, so we extend ResponsesClient
    with the xAI base URL and API key handling.
    """

    SUPPORTS_STREAMING = True
    XAI_BASE_URL = "https://api.x.ai/v1"

    def __init__(
        self,
        model: str,
        config: Config,
        api_key: str | None = None,
        model_definition: dict | None = None,
    ):
        self._provided_api_key = api_key
        effective_key = self._resolve_api_key(config, api_key)

        if not effective_key:
            raise ValueError(
                "xAI API key not found. Connect the xAI provider in the UI "
                "(Settings → Providers), or set XAI_API_KEY / "
                "LLM_BAWT_XAI_API_KEY as a legacy fallback."
            )

        super().__init__(
            model=model,
            config=config,
            base_url=self.XAI_BASE_URL,
            api_key=effective_key,
            model_definition=model_definition,
        )

    def _cache_routing_fields(self, key: str) -> dict:
        """xAI routes its prompt cache by the ``x-grok-conv-id`` header.

        Measured TASK-1729: header 6/6 cache hits vs ``prompt_cache_key`` 4/6.
        """
        return {"extra_headers": {"x-grok-conv-id": key}}

    @staticmethod
    def _resolve_api_key(config: Config, explicit: str | None) -> str | None:
        """DB-first (CredentialStore via the `xai` adapter), env as legacy fallback."""
        from ..service.providers.api_key import resolve_api_key

        return resolve_api_key(
            config,
            "xai",
            env_vars=("XAI_API_KEY", "LLM_BAWT_XAI_API_KEY"),
            explicit=explicit,
            config_attr="XAI_API_KEY",
        )

    def get_styling(self) -> tuple[str | None, str]:
        return None, "bright_magenta"
