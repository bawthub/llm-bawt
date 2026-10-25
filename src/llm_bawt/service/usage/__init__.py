"""Provider-pluggable usage registry.

Exposes a canonical, multi-provider view of subscription usage. Adapters are
registered by their canonical provider id; the HTTP route (``routes/usage.py``)
calls :func:`get_usage` for a single provider or :func:`get_all` for the
all-backends view.

Caching: the upstream endpoints (notably Claude's ``/api/oauth/usage``) are
aggressively rate-limited, so successful snapshots are cached per-provider for
``LLM_BAWT_USAGE_CACHE_TTL`` seconds (default 120). On a fetch error or a 429
we serve the last-good snapshot marked ``cached=True`` rather than failing the
UI. A per-provider lock collapses concurrent requests into a single upstream
call.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

from .base import UsageAdapter
from .canonical import (
    AllUsage,
    ProviderUsage,
    STATUS_OK,
    STATUS_RATE_LIMITED,
    STATUS_ERROR,
)
from .adapters import (
    AnthropicApiUsageAdapter,
    ClaudeUsageAdapter,
    KimiCodingUsageAdapter,
    MoonshotUsageAdapter,
    OpenAIChatGPTUsageAdapter,
    XaiUsageAdapter,
    ZaiUsageAdapter,
)

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, UsageAdapter] = {}
_ACCESS_PATH_ADAPTERS: dict[str, UsageAdapter] = {}


def register(adapter: UsageAdapter) -> None:
    """Register one usage source and reject overlapping access-path ownership."""
    for key in adapter.access_path_keys:
        previous = _ACCESS_PATH_ADAPTERS.get(key)
        if previous is not None and previous.provider != adapter.provider:
            raise ValueError(f"usage access path {key!r} claimed by {previous.provider!r} and {adapter.provider!r}")
    old = _REGISTRY.get(adapter.provider)
    if old is not None:
        for key in old.access_path_keys:
            _ACCESS_PATH_ADAPTERS.pop(key, None)
    _REGISTRY[adapter.provider] = adapter
    for key in adapter.access_path_keys:
        _ACCESS_PATH_ADAPTERS[key] = adapter


def usage_adapter_for_access_path(key: str | None) -> UsageAdapter | None:
    """Exact report owner; unknown/local/API paths without reporting return None."""
    return _ACCESS_PATH_ADAPTERS.get(str(key or "").strip().lower())


def get_cached(provider: str) -> ProviderUsage | None:
    """Read last authoritative snapshot without fetching or refreshing upstream."""
    hit = _cache.get(provider)
    return _cached_copy(hit[1]) if hit else None


def invalidate_cache(provider: str) -> None:
    """Discard a report after its runtime settings change."""
    _cache.pop(provider, None)


# Default registrations. Add a provider = create the adapter, import it here,
# register an instance.
register(ClaudeUsageAdapter())
register(AnthropicApiUsageAdapter())
register(ZaiUsageAdapter())
register(OpenAIChatGPTUsageAdapter())
register(XaiUsageAdapter())
# Moonshot OpenPlatform and Kimi For Coding are separate products. Do not
# surface an unauthorized Moonshot card when only the Kimi subscription is
# configured; register OpenPlatform usage once its own key exists.
if os.getenv("MOONSHOT_API_KEY") or os.getenv("LLM_BAWT_MOONSHOT_API_KEY"):
    register(MoonshotUsageAdapter())
register(KimiCodingUsageAdapter())


def list_providers() -> list[str]:
    return list(_REGISTRY.keys())


def has_provider(provider: str) -> bool:
    return provider in _REGISTRY


def _cache_ttl() -> float:
    try:
        return float(os.getenv("LLM_BAWT_USAGE_CACHE_TTL", "120"))
    except (TypeError, ValueError):
        return 120.0


# provider -> (fetched_monotonic, ProviderUsage)
_cache: dict[str, tuple[float, ProviderUsage]] = {}
_locks: dict[str, asyncio.Lock] = {}


def _cached_copy(snap: ProviderUsage) -> ProviderUsage:
    c = snap.model_copy(deep=True)
    c.cached = True
    return c


async def get_usage(provider: str, *, force: bool = False) -> ProviderUsage | None:
    """Return a canonical snapshot for one provider, or None if unregistered."""
    adapter = _REGISTRY.get(provider)
    if adapter is None:
        return None

    ttl = _cache_ttl()
    now = time.monotonic()
    hit = _cache.get(provider)
    if not force and hit and (now - hit[0]) < ttl:
        return _cached_copy(hit[1])

    lock = _locks.setdefault(provider, asyncio.Lock())
    async with lock:
        # Re-check: another waiter may have just refreshed it.
        hit = _cache.get(provider)
        if not force and hit and (time.monotonic() - hit[0]) < ttl:
            return _cached_copy(hit[1])
        try:
            snap = await adapter.fetch()
        except Exception as e:  # noqa: BLE001 — never fail the UI on a fetch error
            logger.warning("Usage fetch failed for %s: %s", provider, e)
            if hit is not None:
                stale = _cached_copy(hit[1])
                stale.error = f"refresh failed: {e}"
                return stale
            return ProviderUsage(
                provider=provider,
                display_name=adapter.display_name,
                backend=adapter.backend,
                kind=adapter.kind,
                available=False,
                status=STATUS_ERROR,
                error=str(e),
                fetched_at=int(time.time()),
            )

        # Cache only authoritative snapshots; on a transient 429, prefer the
        # last-good snapshot if we have one.
        if snap.status == STATUS_OK:
            _cache[provider] = (time.monotonic(), snap)
        elif snap.status == STATUS_RATE_LIMITED and hit is not None and snap.balance is None:
            # Keep a freshly computed local balance even if its separate Admin
            # cost report hit a rate limit; a cached balance would be stale.
            stale = _cached_copy(hit[1])
            stale.error = "rate-limited; showing last cached snapshot"
            return stale
        return snap


async def get_all(*, force: bool = False) -> AllUsage:
    """Fetch every registered provider concurrently (the all-backends view)."""
    providers = list_providers()
    results = await asyncio.gather(
        *(get_usage(p, force=force) for p in providers),
        return_exceptions=False,
    )
    snaps = [r for r in results if r is not None]
    return AllUsage(fetched_at=int(time.time()), providers=snaps)


__all__ = [
    "UsageAdapter",
    "ProviderUsage",
    "UsageLimit",
    "AllUsage",
    "register",
    "usage_adapter_for_access_path",
    "get_cached",
    "invalidate_cache",
    "list_providers",
    "has_provider",
    "get_usage",
    "get_all",
]

from .canonical import UsageLimit  # noqa: E402  (re-export for convenience)
