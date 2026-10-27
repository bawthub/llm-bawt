"""Moonshot AI (Kimi) adapter — Anthropic Messages API passthrough.

Moonshot exposes a **native Anthropic Messages API** surface — the same one
their documented Claude Code integration targets:

    https://api.moonshot.ai/anthropic/v1/messages

so this rides the shared
:class:`~.anthropic_passthrough.AnthropicPassthroughAdapter` (no translation,
raw SSE relay) and only supplies the endpoint + credentials.

**Why passthrough and not the default Responses-API path** (probed live
2026-07-27, unauthenticated — the routing signal is independent of the key):
Moonshot has **no** ``/v1/responses`` endpoint. Unknown routes there answer
with a distinctive ``{"error":"url.not_found"}`` shape, real routes reach the
auth layer and answer ``{"error":{"type":"invalid_authentication_error"}}``:

    POST /v1/responses              → 404 url.not_found      (no such route)
    POST /anthropic/v1/bogus_route  → 404 url.not_found      (control)
    POST /anthropic/v1/messages     → 401 invalid_auth        (route EXISTS)
    POST /v1/chat/completions       → 401 invalid_auth        (route exists)

So the ``ProviderAdapter`` default ``call`` (which posts to ``/responses``)
would 404 here. Chat-Completions would need a whole translation layer we'd
have to write; the Anthropic surface needs none.

Auth: a Moonshot API key (``sk-…``). Moonshot's Claude Code guide sets it via
``ANTHROPIC_AUTH_TOKEN``, i.e. an ``Authorization: Bearer`` header — so this
overrides the base class's ``x-api-key`` default. We send **both** headers
with the same value: Bearer is the documented contract, and the ``x-api-key``
mirror is a zero-cost fallback in case the gateway prefers the
api.anthropic.com style. (Not yet confirmed against a live 200 — the vault key
was revoked at implementation time; see the task notes.)

Models are namespaced ``moonshot/<model>``; the proxy strips the ``moonshot/``
prefix before handing the bare upstream model to ``call``. Note Moonshot's
context variants carry a **bracket suffix** in the model id itself —
e.g. ``moonshot/kimi-k3[1m]`` for the 1M-context K3. ``routes.py`` splits on
the first ``/`` only, so brackets pass through untouched.

Quirk worth knowing (server-side, nothing to do here): Moonshot rescales
sampling temperature as ``real = requested * 0.6``. We deliberately do NOT
pre-compensate — dividing by 0.6 would push the SDK's default ``temperature:1``
to 1.67 and out of the accepted range.
"""

from __future__ import annotations

from typing import ClassVar

from .anthropic_passthrough import AnthropicPassthroughAdapter


class MoonshotAdapter(AnthropicPassthroughAdapter):
    """Kimi via Moonshot's native Anthropic Messages endpoint."""

    name: ClassVar[str] = "moonshot"
    LABEL: ClassVar[str] = "Moonshot"
    DEFAULT_BASE_URL: ClassVar[str] = "https://api.moonshot.ai/anthropic"
    BASE_URL_ENV: ClassVar[str] = "MOONSHOT_BASE_URL"
    API_KEY_ENVS: ClassVar[tuple[str, ...]] = (
        "MOONSHOT_API_KEY",
        "LLM_BAWT_MOONSHOT_API_KEY",
        "KIMI_API_KEY",
    )
    BROKER_PROVIDER_ID: ClassVar[str] = "moonshot"

    def _auth_headers(self, api_key: str) -> dict[str, str]:
        """Bearer (Moonshot's documented ``ANTHROPIC_AUTH_TOKEN`` style),
        plus an ``x-api-key`` mirror as a harmless compatibility fallback."""
        return {
            "Authorization": f"Bearer {api_key}",
            "x-api-key": api_key,
        }
