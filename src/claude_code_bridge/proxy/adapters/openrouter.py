"""OpenRouter adapter — Anthropic Messages API passthrough (TASK-822).

OpenRouter exposes a native Anthropic-compatible Messages surface (probed live
2026-08-28; ``POST /api/v1/messages`` answers Anthropic-shaped
``authentication_error`` JSON while bogus routes 404 with HTML — the route
exists and speaks the wire format). So this rides the shared
:class:`~.anthropic_passthrough.AnthropicPassthroughAdapter`: no translation,
raw SSE relay.

Model namespacing: catalog rows carry ``openrouter/<upstream-id>`` where the
upstream id itself contains slashes (``qwen/qwen3-coder``). ``routes.py``
splits on the FIRST ``/`` only, so ``openrouter/qwen/qwen3-coder`` →
provider ``openrouter``, upstream ``qwen/qwen3-coder``. OpenRouter's own
model ids are exactly what its Anthropic endpoint expects.

Auth — DB-driven, NOT env vars (the point of TASK-822): the key lives
encrypted in the app's CredentialStore and is resolved via the broker
endpoint ``GET /v1/providers/openrouter/token`` (same pattern as the
ChatGPT adapter, TASK-636). A short in-memory TTL cache keeps the broker
off the hot path while still picking up key rotation within minutes.
OpenRouter authenticates with ``Authorization: Bearer``; the ``x-api-key``
mirror is a zero-cost fallback matching the Moonshot precedent.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import ClassVar

import httpx

from agent_bridge import bridge_token

from .anthropic_passthrough import AnthropicPassthroughAdapter

logger = logging.getLogger(__name__)

# Static API key — cache briefly so a rotated key propagates without a bridge
# restart, but the broker isn't hit on every request.
_KEY_CACHE_TTL_S = 300.0


class OpenRouterAdapter(AnthropicPassthroughAdapter):
    """OpenRouter via its native Anthropic Messages endpoint."""

    name: ClassVar[str] = "openrouter"
    LABEL: ClassVar[str] = "OpenRouter"
    DEFAULT_BASE_URL: ClassVar[str] = "https://openrouter.ai/api"
    BASE_URL_ENV: ClassVar[str] = "OPENROUTER_BASE_URL"
    # Deliberately empty: the key comes from the app broker, never env.
    API_KEY_ENVS: ClassVar[tuple[str, ...]] = ()

    def __init__(self) -> None:
        super().__init__()
        self._cached_key: str | None = None
        self._cached_at: float = 0.0
        self._key_lock = threading.Lock()

    # -- credentials (broker-backed, overrides env lookup) ----------------
    def _api_key(self) -> str:
        with self._key_lock:
            if (
                self._cached_key
                and (time.time() - self._cached_at) < _KEY_CACHE_TTL_S
            ):
                return self._cached_key
        key = self._fetch_broker_key()
        with self._key_lock:
            self._cached_key = key
            self._cached_at = time.time()
        return key

    @staticmethod
    def _fetch_broker_key() -> str:
        api_url = (os.environ.get("LLM_BAWT_API_URL") or "").rstrip("/")
        if not api_url:
            raise RuntimeError(
                "LLM_BAWT_API_URL not set — cannot reach the OpenRouter key broker"
            )
        headers = bridge_token.headers()
        try:
            resp = httpx.get(
                f"{api_url}/v1/providers/openrouter/token",
                headers=headers,
                timeout=15.0,
            )
        except httpx.HTTPError as e:
            raise RuntimeError(f"OpenRouter key broker unreachable: {e}") from e
        if resp.is_error:
            raise RuntimeError(
                f"OpenRouter key broker returned {resp.status_code}: "
                f"{(resp.text or '')[:200]} — connect an OpenRouter API key "
                "in Settings → Providers."
            )
        key = (resp.json() or {}).get("access_token")
        if not key:
            raise RuntimeError("OpenRouter key broker returned no key")
        logger.info("Fetched OpenRouter API key from app broker")
        return key

    def _supports_deferral(self, upstream_model: str) -> bool:
        """OpenRouter only honors the tool-search/deferred-tools beta when the
        routed model is Anthropic's ("Deferred custom tools are only supported
        on Anthropic models…" — live 400, 2026-08-28). For everything else
        (moonshot, qwen, …) the deferral markers must be stripped or the
        request is rejected before it ever reaches the model."""
        return upstream_model.startswith("anthropic/")

    def _auth_headers(self, api_key: str) -> dict[str, str]:
        """Bearer is OpenRouter's documented contract; x-api-key mirror is a
        harmless compatibility fallback (Moonshot precedent)."""
        return {
            "Authorization": f"Bearer {api_key}",
            "x-api-key": api_key,
        }
