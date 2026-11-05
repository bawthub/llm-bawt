"""Local inference adapter — Anthropic Messages API passthrough to a LAN box.

Ollama (>= 0.14) and llama.cpp's ``llama-server`` both expose a native
Anthropic-compatible ``/v1/messages`` surface, so a local model rides the
Claude Agent SDK exactly like Z.AI or Moonshot do: nothing to translate, the
shared passthrough in :class:`~.anthropic_passthrough.AnthropicPassthroughAdapter`
does the work. This module only supplies the endpoint and (non-)credentials.

Default upstream is Ollama on localhost; override with
``LOCAL_ANTHROPIC_BASE_URL`` to point at a different host or at llama-server.

Auth: local servers ignore the key, but the Anthropic wire contract still
expects an ``x-api-key`` header, so we always send one. ``LOCAL_ANTHROPIC_API_KEY``
is honoured if set; otherwise a fixed placeholder is used — a missing key is
NOT an error here, unlike the cloud adapters.

Tool deferral: neither Ollama nor llama.cpp implements Anthropic's
tool-search/deferred-tools beta, so every deferred tool is inlined and the
server-side search tool dropped (lossless — see the base class).
"""

from __future__ import annotations

import os
from typing import ClassVar

from .anthropic_passthrough import AnthropicPassthroughAdapter

#: Sent when no real key is configured. Local servers don't validate it.
PLACEHOLDER_KEY = "local"


class LocalAdapter(AnthropicPassthroughAdapter):
    """Anthropic→Anthropic passthrough to a local Ollama / llama-server."""

    name: ClassVar[str] = "local"
    LABEL: ClassVar[str] = "Local"
    DEFAULT_BASE_URL: ClassVar[str] = "http://localhost:11434"
    BASE_URL_ENV: ClassVar[str] = "LOCAL_ANTHROPIC_BASE_URL"
    API_KEY_ENVS: ClassVar[tuple[str, ...]] = ("LOCAL_ANTHROPIC_API_KEY",)
    SUPPORTS_TOOL_DEFERRAL: ClassVar[bool] = False

    def _api_key(self) -> str:
        """Key is optional for local servers — never raise on a missing one."""
        for env in self.API_KEY_ENVS:
            key = os.getenv(env)
            if key:
                return key
        return PLACEHOLDER_KEY
