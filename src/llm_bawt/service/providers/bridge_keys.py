"""API-key providers the claude-code bridge proxy calls (TASK-1476).

Z.AI (GLM Coding Plan), Kimi For Coding and Moonshot OpenPlatform each take
one static API key. The key is stored encrypted as ``provider_connection:<id>``.
The bridge reads it from the app's broker on every call, and the app-side
usage and model-discovery readers use :func:`.api_key.resolve_stored_api_key`.
Their legacy env vars apply only while no record exists, and a disconnect
leaves a ``disconnected`` record so the env key can't come back.
"""

from __future__ import annotations

import httpx

from .api_key import ApiKeyAdapter

_TIMEOUT_S = 15.0


def _probe(label: str, url: str, headers: dict[str, str]) -> tuple[str | None, str | None]:
    """Validate a key with one authenticated GET; never echoes the key."""
    try:
        resp = httpx.get(url, headers=headers, timeout=_TIMEOUT_S)
    except httpx.HTTPError as exc:
        return None, f"{label} unreachable: {type(exc).__name__}"
    if resp.status_code in (400, 401, 403):
        return None, f"{label} rejected the API key"
    if resp.is_error:
        return None, f"{label} key check returned HTTP {resp.status_code}"
    return label, None


class ZaiKeyAdapter(ApiKeyAdapter):
    """Z.AI API key (GLM models through the Anthropic-compatible endpoint)."""

    id = "zai"
    label = "Z.AI"
    access_path_keys = ("zai-anthropic",)
    keeps_disconnected_record = True
    #: Legacy env names, read only while no record exists.
    env_vars = ("ZAI_API_KEY", "Z_AI_API_KEY")

    @staticmethod
    def _probe(key: str) -> tuple[str | None, str | None]:
        # The usage endpoint answers 401 for a bad key and is free to call.
        return _probe("Z.AI", "https://api.z.ai/api/monitor/usage/quota/limit",
                      {"x-api-key": key, "anthropic-version": "2023-06-01"})


class KimiCodingKeyAdapter(ApiKeyAdapter):
    """Kimi For Coding subscription key."""

    id = "kimi-coding"
    label = "Kimi For Coding"
    access_path_keys = ("kimi-coding-chat",)
    keeps_disconnected_record = True
    env_vars = ("KIMI_CODING_API_KEY", "KIMI_API_KEY")

    @staticmethod
    def _probe(key: str) -> tuple[str | None, str | None]:
        return _probe("Kimi For Coding", "https://api.kimi.com/coding/v1/models",
                      {"Authorization": f"Bearer {key}", "Accept": "application/json"})


class MoonshotKeyAdapter(ApiKeyAdapter):
    """Moonshot OpenPlatform key (pay-per-token Kimi)."""

    id = "moonshot"
    label = "Moonshot"
    access_path_keys = ("moonshot-anthropic",)
    keeps_disconnected_record = True
    env_vars = ("MOONSHOT_API_KEY", "LLM_BAWT_MOONSHOT_API_KEY", "KIMI_API_KEY")

    @staticmethod
    def _probe(key: str) -> tuple[str | None, str | None]:
        return _probe("Moonshot", "https://api.moonshot.ai/v1/models",
                      {"Authorization": f"Bearer {key}"})


__all__ = ["KimiCodingKeyAdapter", "MoonshotKeyAdapter", "ZaiKeyAdapter"]
