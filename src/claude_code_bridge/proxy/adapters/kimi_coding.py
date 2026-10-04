"""Kimi For Coding subscription adapter (Chat Completions dialect).

This is deliberately separate from :mod:`moonshot`. Moonshot OpenPlatform uses
``api.moonshot.ai`` and exposes an Anthropic Messages endpoint plus prepaid
balance. The coding subscription uses ``api.kimi.com/coding/v1`` and exposes
Chat Completions for inference; its subscription windows are available from the
separate ``/usages`` endpoint used by the app's usage adapter.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from typing import ClassVar

import httpx

from ..stream_cc import chat_completions_to_anthropic_sse
from ..translate_cc import anthropic_to_chat_completions
from ..request_context import ProxyRequestContext
from .base import ProviderAdapter

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.kimi.com/coding/v1"
_TIMEOUT = httpx.Timeout(connect=15.0, read=600.0, write=60.0, pool=15.0)
_KEY_ENVS = ("KIMI_CODING_API_KEY", "KIMI_API_KEY")


class KimiCodingAdapter(ProviderAdapter):
    """Kimi coding-plan subscription via Chat Completions translation."""

    name: ClassVar[str] = "kimi_coding"

    @staticmethod
    def _api_key() -> str:
        for env in _KEY_ENVS:
            key = os.getenv(env)
            if key:
                return key
        raise RuntimeError(
            "Kimi For Coding API key required: set KIMI_CODING_API_KEY "
            "(KIMI_API_KEY also accepted) on the claude-code-bridge container."
        )

    @staticmethod
    def _base_url() -> str:
        return (os.getenv("KIMI_CODING_BASE_URL") or _DEFAULT_BASE_URL).rstrip("/")

    async def authorize(self) -> tuple[str, str]:
        return self._api_key(), self._base_url()

    async def call(
        self,
        anthropic_body: dict,
        upstream_model: str,
        context: ProxyRequestContext | None = None,
    ) -> AsyncIterator[bytes]:
        key, base_url = await self.authorize()
        body = anthropic_to_chat_completions(
            anthropic_body,
            upstream_model,
            allowed_skill_names=context.skill_names if context else (),
        )
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        logger.debug(
            "Kimi For Coding → %s model=%s tools=%d messages=%d",
            url,
            upstream_model,
            len(body.get("tools") or []),
            len(body.get("messages") or []),
        )

        client = await self.http_client()
        async with client.stream(
            "POST", url, json=body, headers=headers, timeout=_TIMEOUT
        ) as resp:
            if resp.status_code >= 400:
                detail = (await resp.aread()).decode("utf-8", "replace")
                raise RuntimeError(
                    f"Kimi For Coding upstream {resp.status_code}: {detail[:500]}"
                )
            async for frame in chat_completions_to_anthropic_sse(
                resp.aiter_lines(),
                anthropic_model=anthropic_body.get("model", upstream_model),
                tool_schemas=anthropic_body.get("tools"),
                allowed_skill_names=context.skill_names if context else (),
            ):
                yield frame
