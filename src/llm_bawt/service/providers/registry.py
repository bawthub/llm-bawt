"""Provider adapter registry.

Maps a provider id -> adapter instance (constructed with the app Config). Phase 2
adds the LLM api-key adapters (openai/anthropic/grok); GitHub ships in Phase 1.
"""

from __future__ import annotations

from ...utils.config import Config
from .anthropic_key import AnthropicKeyAdapter
from .base import ProviderAdapter
from .claude import ClaudeAdapter
from .codex import CodexAdapter
from .github import GitHubAdapter
from .github_deploy import (
    GhcrPullAdapter,
    GitHubReleaseAdapter,
    GitHubReleaseDispatchAdapter,
)
from .openai_key import OpenAIKeyAdapter
from .openrouter import OpenRouterAdapter
from .xai import XaiAdapter
from .x import XAdapter

# Adapter classes keyed by provider id. Extend here as providers are added.
# TASK-635: the single `claude` adapter supersedes the old claude-sub +
# claude-usage dual flow (ClaudeSubAdapter remains as its unregistered base).
# TASK-637: `codex` is status-only (health/expiry) until TASK-636 phase 2.
# TASK-825: xai / openai-api / anthropic-api are the raw API-key adapters —
# clients resolve keys DB-first via providers.api_key.resolve_api_key.
_ADAPTER_CLASSES: dict[str, type[ProviderAdapter]] = {
    GitHubAdapter.id: GitHubAdapter,
    ClaudeAdapter.id: ClaudeAdapter,
    CodexAdapter.id: CodexAdapter,
    OpenRouterAdapter.id: OpenRouterAdapter,
    XaiAdapter.id: XaiAdapter,
    OpenAIKeyAdapter.id: OpenAIKeyAdapter,
    AnthropicKeyAdapter.id: AnthropicKeyAdapter,
    XAdapter.id: XAdapter,
    # TASK-997: release-deploy credentials (DB-stored, never env).
    GitHubReleaseAdapter.id: GitHubReleaseAdapter,
    GitHubReleaseDispatchAdapter.id: GitHubReleaseDispatchAdapter,
    GhcrPullAdapter.id: GhcrPullAdapter,
}


def get_adapter(config: Config, provider_id: str) -> ProviderAdapter | None:
    cls = _ADAPTER_CLASSES.get(provider_id)
    if cls is None:
        return None
    return cls(config)


def all_adapters(config: Config) -> list[ProviderAdapter]:
    return [cls(config) for cls in _ADAPTER_CLASSES.values()]
