"""Provider adapter registry.

Maps a provider id -> adapter instance (constructed with the app Config). Phase 2
adds the LLM api-key adapters (openai/anthropic/grok); GitHub ships in Phase 1.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from ...utils.config import Config
from .anthropic_admin import AnthropicAdminAdapter
from .anthropic_key import AnthropicKeyAdapter
from .base import (
    CONNECTED_HEALTH_STATES,
    HEALTH_BROKEN,
    HEALTH_UNCONFIGURED,
    HEALTH_UNKNOWN,
    ProviderAdapter,
)
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

logger = logging.getLogger(__name__)

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
    AnthropicAdminAdapter.id: AnthropicAdminAdapter,
    XAdapter.id: XAdapter,
    # TASK-997: release-deploy credentials (DB-stored, never env).
    GitHubReleaseAdapter.id: GitHubReleaseAdapter,
    GitHubReleaseDispatchAdapter.id: GitHubReleaseDispatchAdapter,
    GhcrPullAdapter.id: GhcrPullAdapter,
}


def _access_path_owners() -> dict[str, type[ProviderAdapter]]:
    owners: dict[str, type[ProviderAdapter]] = {}
    for cls in _ADAPTER_CLASSES.values():
        for key in cls.access_path_keys:
            if key in owners:
                raise RuntimeError(
                    f"access path {key!r} is claimed by both {owners[key].id!r} and {cls.id!r}"
                )
            owners[key] = cls
    return owners


#: access_paths.key -> the adapter whose credential authorizes it (TASK-1297).
_ACCESS_PATH_OWNERS = _access_path_owners()


def get_adapter(config: Config, provider_id: str) -> ProviderAdapter | None:
    cls = _ADAPTER_CLASSES.get(provider_id)
    if cls is None:
        return None
    return cls(config)


def all_adapters(config: Config) -> list[ProviderAdapter]:
    return [cls(config) for cls in _ADAPTER_CLASSES.values()]


def access_path_connections(config: Config, access_path_keys: Iterable[str]) -> dict[str, dict[str, Any] | None]:
    """Connection state for each access path, keyed by ``access_paths.key``.

    ``None`` means no provider connection authorizes that path (local engines,
    or a vendor with no adapter yet). Otherwise::

        {"provider_id", "label", "state", "detail", "connected"}

    ``connected`` is true when the credential's health state is usable now.
    Each owning adapter's health is probed once, however many paths it owns.
    """
    health_by_provider: dict[str, dict[str, Any]] = {}
    result: dict[str, dict[str, Any] | None] = {}
    for key in access_path_keys:
        cls = _ACCESS_PATH_OWNERS.get(key)
        if cls is None:
            result[key] = None
            continue
        if cls.id not in health_by_provider:
            try:
                health_by_provider[cls.id] = cls(config).health()
            except Exception as exc:  # noqa: BLE001 — one bad adapter must not hide the rest
                logger.warning("health probe failed for %s: %s", cls.id, exc)
                health_by_provider[cls.id] = {"state": HEALTH_UNKNOWN, "detail": str(exc)[:200]}
        health = health_by_provider[cls.id]
        state = str(health.get("state") or HEALTH_UNKNOWN)
        result[key] = {
            "provider_id": cls.id,
            "label": cls.label,
            "state": state,
            "detail": health.get("detail"),
            "connected": state in CONNECTED_HEALTH_STATES,
        }
    return result


#: Synthesized route states beyond the provider health states (TASK-1302 A0).
ROUTE_LOCAL = "local"
ROUTE_UNTRACKED = "untracked"


def route_eligibility(
    access_path_key: str,
    auth_mechanism: str | None,
    connection: dict[str, Any] | None,
) -> dict[str, Any]:
    """Whether a bot may be newly assigned to a route, and why not (TASK-1302 A0).

    The one rule behind the bot-save gate and every route list. A route is
    eligible only when it needs no credential (``auth_mechanism == "none"``) or
    its owning provider is connected now (health ok or warning). An owned path
    in any other health state, and a remote path no provider owns (BawtHub
    doesn't manage its credential), is not. ``connection`` is the
    :func:`access_path_connections` entry for the path.
    """
    if (auth_mechanism or "").strip().lower() == "none":
        return {"eligible": True, "state": ROUTE_LOCAL, "reason": None}
    if connection is None:
        return {
            "eligible": False,
            "state": ROUTE_UNTRACKED,
            "reason": (
                f"BawtHub doesn't manage the credential for {access_path_key}; "
                "this route can't be assigned yet."
            ),
        }
    state = str(connection.get("state") or HEALTH_UNKNOWN)
    if state in CONNECTED_HEALTH_STATES:
        return {"eligible": True, "state": state, "reason": None}
    label = connection.get("label") or access_path_key
    detail = connection.get("detail")
    suffix = f" ({detail})" if detail else ""
    if state == HEALTH_UNCONFIGURED:
        reason = f"{label} is not connected. Connect it in Integrations or pick a connected route."
    elif state == HEALTH_BROKEN:
        reason = f"{label} needs reconnecting{suffix}."
    else:
        reason = f"{label} status couldn't be checked{suffix}."
    return {"eligible": False, "state": state, "reason": reason}
