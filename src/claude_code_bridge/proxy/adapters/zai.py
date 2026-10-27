"""Z.AI (Zhipu GLM) adapter — Anthropic Messages API passthrough.

Unlike the OpenAI ChatGPT adapter, Z.AI exposes a **native Anthropic
Messages API** surface (the same one their Claude Code integration targets):

    https://api.z.ai/api/anthropic/v1/messages

So there's nothing to translate — the Claude Agent SDK already speaks this
wire format. The streaming passthrough (plus the namespaced-model rewrite and
the prompt-cache usage tap) lives in
:class:`~.anthropic_passthrough.AnthropicPassthroughAdapter`; this module only
supplies Z.AI's endpoint and credentials.

Auth: a Z.AI API key (the value you'd normally set as ``ANTHROPIC_AUTH_TOKEN``
for Claude Code). Read from ``ZAI_API_KEY`` (``Z_AI_API_KEY`` accepted as an
alias). Z.AI's Anthropic endpoint authenticates with an ``x-api-key`` header,
same as api.anthropic.com — i.e. the base class default.
"""

from __future__ import annotations

from typing import ClassVar

from .anthropic_passthrough import AnthropicPassthroughAdapter


class ZaiAdapter(AnthropicPassthroughAdapter):
    """Pure Anthropic→Anthropic passthrough to Z.AI's GLM models."""

    name: ClassVar[str] = "zai"
    LABEL: ClassVar[str] = "Z.AI"
    DEFAULT_BASE_URL: ClassVar[str] = "https://api.z.ai/api/anthropic"
    BASE_URL_ENV: ClassVar[str] = "ZAI_BASE_URL"
    API_KEY_ENVS: ClassVar[tuple[str, ...]] = ("ZAI_API_KEY", "Z_AI_API_KEY")
    BROKER_PROVIDER_ID: ClassVar[str] = "zai"
