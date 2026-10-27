"""API-key providers the claude-code bridge proxy calls (TASK-1476).

Z.AI (GLM Coding Plan), Kimi For Coding and Moonshot OpenPlatform each take
one static API key. The key is stored encrypted as ``provider_connection:<id>``.
The bridge reads it from the app's broker on every call, and the app-side
usage and model-discovery readers use :func:`.api_key.resolve_stored_api_key`.
Their legacy env vars apply only while no record exists, and a disconnect
leaves a ``disconnected`` record so the env key can't come back.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx

from .api_key import ApiKeyAdapter

_TIMEOUT_S = 15.0
_AUTH_CODES = (401, 403)


class ZaiEnvelopeError:
    """A z.ai failure carried inside an HTTP 200 body.

    z.ai's monitor API answers ``{"code": 401, "success": false, ...}`` with
    HTTP 200 for a bad or expired key, so the HTTP status alone can't tell a
    rejected key from a working one.
    """

    __slots__ = ("code", "unauthorized")

    def __init__(self, code: Any, unauthorized: bool):
        self.code = code
        self.unauthorized = unauthorized

    @classmethod
    def from_payload(cls, payload: Any) -> ZaiEnvelopeError | None:
        """The failure ``payload`` reports, or None when it reports success."""
        if not isinstance(payload, dict):
            return None
        code = payload.get("code")
        failed = payload.get("success") is False or (
            isinstance(code, int) and not isinstance(code, bool) and code not in (0, 200))
        if not failed:
            return None
        return cls(code, code in _AUTH_CODES)


def _zai_body_error(resp: httpx.Response) -> str | None:
    try:
        payload = resp.json()
    except ValueError:
        return "Z.AI key check returned a non-JSON body"
    failure = ZaiEnvelopeError.from_payload(payload)
    if failure is None:
        return None
    if failure.unauthorized:
        return "Z.AI rejected the API key"
    return f"Z.AI key check failed with code {failure.code}"


def _probe(label: str, url: str, headers: dict[str, str],
           body_error: Callable[[httpx.Response], str | None] | None = None,
           ) -> tuple[str | None, str | None]:
    """Validate a key with one authenticated GET; never echoes the key.

    ``body_error`` inspects a successful response for a failure the provider
    reports in its body rather than its HTTP status.
    """
    try:
        resp = httpx.get(url, headers=headers, timeout=_TIMEOUT_S)
    except httpx.HTTPError as exc:
        return None, f"{label} unreachable: {type(exc).__name__}"
    if resp.status_code in (400, 401, 403):
        return None, f"{label} rejected the API key"
    if resp.is_error:
        return None, f"{label} key check returned HTTP {resp.status_code}"
    if body_error is not None:
        error = body_error(resp)
        if error is not None:
            return None, error
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
        # The usage endpoint is free to call. It rejects a bad key in the body
        # (HTTP 200, ``code: 401``), so the body is checked too.
        return _probe("Z.AI", "https://api.z.ai/api/monitor/usage/quota/limit",
                      {"x-api-key": key, "anthropic-version": "2023-06-01"},
                      body_error=_zai_body_error)


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


__all__ = ["KimiCodingKeyAdapter", "MoonshotKeyAdapter", "ZaiEnvelopeError", "ZaiKeyAdapter"]
