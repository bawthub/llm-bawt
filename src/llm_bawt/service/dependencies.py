"""Shared FastAPI dependencies and service helpers.

Process-wide singleton stores
-----------------------------
The ``get_*_store`` / ``get_profile_manager`` helpers below return cached
instances of Stores so that route handlers don't pay the construction
cost (table-existence checks, schema migrations) on every request.

After the TASK-202 refactor, **all Stores share a single SQLAlchemy
engine** (see :func:`llm_bawt.utils.db.get_shared_engine`), so caching
the Store is no longer necessary to avoid connection-pool leaks — the
underlying pool is the same regardless of how many Store instances
exist. The cache survives mainly to avoid repeating
``CREATE TABLE IF NOT EXISTS`` on every call.

The cache is keyed on ``id(config)`` so a fresh config (e.g. after
``/v1/admin/reload``) yields a fresh Store. **Do NOT** call
``engine.dispose()`` on evicted Stores — the engine is shared with every
other Store and disposing it would break the whole process.
"""

import threading
from typing import TYPE_CHECKING, Any

try:
    from fastapi import Depends, HTTPException
except ImportError:  # pragma: no cover - FastAPI is optional at import time
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)

    def Depends(dep):  # type: ignore[misc]
        return dep

from .logging import generate_request_id as _generate_request_id
from .schemas import UserProfileAttribute

if TYPE_CHECKING:
    from ..media.assets import MediaAssetStore
    from ..media.db import MediaGenerationStore
    from .background_service import BackgroundService
    from .turn_logs import TurnLogStore

_service: "BackgroundService | None" = None

# --------------------------------------------------------------------------
# Process-wide store singletons (TASK-202: prevent connection-pool leaks).
# --------------------------------------------------------------------------
# Re-entrant by requirement, not convenience: ``factory()`` runs while the
# lock is held, and a factory may resolve a sibling singleton through this
# same helper (``get_ops_service`` builds on ``get_ops_store``). With a plain
# Lock that nesting self-deadlocks the calling thread forever — and because
# the MCP tools await these accessors on the event loop, one such call wedges
# the whole MCP server. A cache hit on the inner store hides it, so the
# deadlock only fires on the paths where the inner store isn't warm yet.
_store_lock = threading.RLock()
_store_cache: dict[tuple[str, int], Any] = {}


def _get_or_build_store(name: str, config: Any, factory):
    """Return a cached store instance, building it on first request.

    Keyed on ``(name, id(config))`` so the cache invalidates when the
    config object is replaced (e.g. settings reload).

    NOTE (TASK-202): every Store now shares one process-wide SQLAlchemy
    engine. We do **not** call ``engine.dispose()`` on evicted Stores
    because that would tear down the shared pool used by every other
    Store. Eviction simply drops the Python reference; the shared engine
    keeps running.
    """
    key = (name, id(config))
    cached = _store_cache.get(key)
    if cached is not None:
        return cached
    with _store_lock:
        cached = _store_cache.get(key)
        if cached is not None:
            return cached
        # Drop stale instances under the same name with a different config.
        # Do NOT dispose their engine — it's the shared pool.
        for existing_key in [k for k in _store_cache if k[0] == name]:
            _store_cache.pop(existing_key, None)
        instance = factory()
        _store_cache[key] = instance
        return instance


def reset_store_cache() -> None:
    """Drop all cached stores. Used at shutdown / reload.

    Does NOT dispose the underlying shared engine — that's owned by
    :mod:`llm_bawt.utils.db` and disposing it would break every other
    Store. Call ``llm_bawt.utils.db.reset_shared_engines`` separately if
    you really need to recycle the pool (tests only).
    """
    with _store_lock:
        _store_cache.clear()


def get_profile_manager(config: Any):
    """Process-wide ``ProfileManager`` singleton. See module docstring."""
    from ..profiles import ProfileManager

    return _get_or_build_store("profile_manager", config, lambda: ProfileManager(config))


def get_bot_profile_store(config: Any):
    """Process-wide ``BotProfileStore`` singleton."""
    from ..runtime_settings import BotProfileStore

    return _get_or_build_store("bot_profile_store", config, lambda: BotProfileStore(config))


def get_runtime_settings_store(config: Any):
    """Process-wide ``RuntimeSettingsStore`` singleton."""
    from ..runtime_settings import RuntimeSettingsStore

    return _get_or_build_store(
        "runtime_settings_store", config, lambda: RuntimeSettingsStore(config)
    )


def get_model_catalog_engine(config: Any):
    """Return the shared PostgreSQL engine for normalized model catalog access."""
    from ..utils.db import get_shared_engine

    return get_shared_engine(config)


# TASK-214: get_avatar_animation_store() was removed. The avatar animation
# catalog now lives in bawthub Prisma and travels on each chat request as
# `animations: list[ChatRequestAnimation]`. llm-bawt is stateless w.r.t.
# the catalog.


def get_prompt_template_store(config: Any):
    """Process-wide ``PromptTemplateStore`` singleton."""
    from ..prompt_registry import PromptTemplateStore

    return _get_or_build_store(
        "prompt_template_store", config, lambda: PromptTemplateStore(config)
    )


def get_media_asset_store(config: Any) -> "MediaAssetStore":
    """Process-wide ``MediaAssetStore`` singleton for a config generation."""
    from ..media.assets import MediaAssetStore

    return _get_or_build_store(
        "media_asset_store", config, lambda: MediaAssetStore(config)
    )


def get_media_generation_store(config: Any) -> "MediaGenerationStore":
    """Process-wide ``MediaGenerationStore`` singleton for a config generation."""
    from ..media.db import MediaGenerationStore

    return _get_or_build_store(
        "media_generation_store", config, lambda: MediaGenerationStore(config)
    )


def get_tool_approval_policy_store(config: Any):
    """Process-wide ``ToolApprovalPolicyStore`` singleton (TASK-290)."""
    from ..approval_policies import ToolApprovalPolicyStore

    return _get_or_build_store(
        "tool_approval_policy_store", config, lambda: ToolApprovalPolicyStore(config)
    )


def get_ops_store(config: Any):
    """Process-wide ``OpsStore`` singleton (TASK-639)."""
    from ..ops import OpsStore

    return _get_or_build_store("ops_store", config, lambda: OpsStore(config))


def get_ops_service(config: Any):
    """Process-wide ``OpsService`` singleton (TASK-639).

    Registers both the Docker worker and durable release executors. Callers
    swapping executors for tests should construct their own OpsService.
    """
    from ..ops.release_executor import build_ops_service

    return _get_or_build_store(
        "ops_service", config, lambda: build_ops_service(config, get_ops_store(config))
    )


def set_service(service: "BackgroundService | None") -> None:
    """Set the global background service instance."""
    global _service
    _service = service


def get_service() -> "BackgroundService":
    """Get the initialized background service instance."""
    if _service is None:
        raise RuntimeError("Service not initialized")
    return _service


def get_turn_log_store() -> "TurnLogStore":
    """Return the service-owned store shared by streaming writers and readers."""
    return get_service()._turn_log_store


def get_effective_bot_id(bot_id: str | None = None) -> str:
    """Resolve optional bot_id to the configured service default."""
    return bot_id or get_service()._default_bot


def require_memory_client(bot_id: str = Depends(get_effective_bot_id)):
    """Dependency that requires a live memory client for the effective bot."""
    client = get_service().get_memory_client(bot_id)
    if not client:
        raise HTTPException(status_code=503, detail="Memory service unavailable")
    return client


def attribute_to_response(attr: Any) -> UserProfileAttribute:
    """Convert a profile attribute ORM/domain object to API schema."""
    return UserProfileAttribute(
        id=attr.id,
        category=attr.category.value if hasattr(attr.category, "value") else str(attr.category),
        key=attr.key,
        value=attr.value,
        confidence=attr.confidence,
        source=attr.source,
        created_at=attr.created_at.isoformat() if attr.created_at else None,
        updated_at=attr.updated_at.isoformat() if attr.updated_at else None,
    )


def generate_request_id() -> str:
    """Expose request ID generation as a shared dependency helper."""
    return _generate_request_id()
