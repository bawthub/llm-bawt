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
call. An adapter may shorten its TTL (``report_ttl``): a provider balance API
uses it as a short debounce, refreshed on demand after that provider's turns.

Balances are never cached: an anchored balance (``balance_mode == "anchor"``)
is read from its counter row on every request and overlaid on the report.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from .balance import BALANCE_ANCHOR
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
    OpenAIApiUsageAdapter,
    OpenAIChatGPTUsageAdapter,
    OpenRouterUsageAdapter,
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


def normalize_access_path(key: str | None) -> str:
    return str(key or "").strip().lower()


def usage_adapter_for_access_path(key: str | None) -> UsageAdapter | None:
    """Exact report owner; unknown/local/API paths without reporting return None."""
    return _ACCESS_PATH_ADAPTERS.get(normalize_access_path(key))


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
# Pay-per-token API keys counted down from a manual anchor (TASK-1461).
register(OpenAIApiUsageAdapter())
register(OpenRouterUsageAdapter())
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


def _cache_ttl(adapter: UsageAdapter) -> float:
    if adapter.report_ttl is not None:
        return adapter.report_ttl
    try:
        return float(os.getenv("LLM_BAWT_USAGE_CACHE_TTL", "120"))
    except (TypeError, ValueError):
        return 120.0


def _balance_store():
    from ..dependencies import get_service
    from .balance import ApiBalanceStore

    return ApiBalanceStore(get_service()._turn_log_store.engine)


def _with_anchored_balance(adapter: UsageAdapter, snap: ProviderUsage) -> ProviderUsage:
    """Overlay the live counter row on a (possibly cached) report: never cached itself."""
    from .balance import anchored_balance

    out = snap.model_copy(deep=True)
    try:
        balance, reason = anchored_balance(_balance_store().read(adapter.provider),
                                           adapter.credential_connected_at())
    except Exception as exc:  # noqa: BLE001 — unknown credit is shown as unknown, never as zero
        logger.warning("Balance read failed for %s: %s", adapter.provider, type(exc).__name__)
        balance, reason = None, "API balance unavailable"
    out.balance = balance
    if reason:
        out.error = reason
    return out


def _with_ledger_spend(adapter: UsageAdapter, snap: ProviderUsage) -> ProviderUsage:
    """Live ledger spend: per-day chat vs maintenance, plus totals when no report.

    Floored at the credential's connection time like the Anthropic ledger path:
    before it, a path label is not proof this key paid. Never cached; a failed
    read leaves the report as it was (no fabricated zero).
    """
    from datetime import datetime, timedelta, timezone

    from .canonical import UsageSpend

    out = snap.model_copy(deep=True)
    try:
        connected_at = adapter.credential_connected_at()
        if connected_at is None:
            return out
        now = datetime.now(timezone.utc)
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        store = _balance_store()
        daily = store.daily_spend(adapter.provider, since=connected_at, start=today - timedelta(days=6))
        if out.spend is None:
            totals = store.spend(adapter.provider, since=connected_at,
                                 starts=(today, today - timedelta(days=6), today.replace(day=1)))
            out.spend = UsageSpend(today_usd=float(totals[0]), seven_day_usd=float(totals[1]),
                                   month_usd=float(totals[2]), source="bawt_recorded_estimate")
        out.spend.daily = daily
    except Exception as exc:  # noqa: BLE001 — spend detail is optional
        logger.warning("Ledger spend read failed for %s: %s", adapter.provider, type(exc).__name__)
    return out


def set_balance(provider: str, amount) -> None:
    """Anchor a user-entered balance as of now (``balance_mode == "anchor"`` only)."""
    _balance_store().set_anchor(provider, amount)


def clear_balance(provider: str) -> bool:
    return _balance_store().clear(provider)


def get_adapter(provider: str) -> UsageAdapter | None:
    return _REGISTRY.get(provider)


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
    snap = await _report(adapter, force=force)
    if adapter.balance_mode == BALANCE_ANCHOR:
        snap = await asyncio.to_thread(_with_anchored_balance, adapter, snap)
    if adapter.balance_mode:
        snap = await asyncio.to_thread(_with_ledger_spend, adapter, snap)
    return snap


async def _report(adapter: UsageAdapter, *, force: bool) -> ProviderUsage:
    provider = adapter.provider
    ttl = _cache_ttl(adapter)
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
            return adapter._base(available=False, status=STATUS_ERROR, error=str(e))

        # Cache only authoritative snapshots; on a transient 429, prefer the
        # last-good snapshot if we have one.
        if snap.status == STATUS_OK:
            _cache[provider] = (time.monotonic(), snap)
        elif snap.status == STATUS_RATE_LIMITED and hit is not None:
            # An anchored balance is overlaid fresh after this; a provider-
            # reported one is only as old as the last good fetch.
            stale = _cached_copy(hit[1])
            stale.error = "rate-limited; showing last cached snapshot"
            return stale
        return snap


def providers_with_models(catalog: Any) -> list[str]:
    """Registered providers whose access paths serve at least one catalog model.

    The catalog is the live one the service installs at startup and after
    every model change, so adding a model shows its provider's card and
    removing the last one hides it. No catalog means no models.
    """
    if catalog is None:
        return []
    served = {normalize_access_path(e.access_path.key) for e in catalog.endpoints()}
    return [
        provider
        for provider, adapter in _REGISTRY.items()
        if served.intersection(normalize_access_path(k) for k in adapter.access_path_keys)
    ]


async def get_all(*, force: bool = False, catalog: Any = None) -> AllUsage:
    """Fetch every provider with a catalog model concurrently (the all-backends view)."""
    providers = providers_with_models(catalog)
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
    "normalize_access_path",
    "get_cached",
    "invalidate_cache",
    "get_adapter",
    "set_balance",
    "clear_balance",
    "list_providers",
    "has_provider",
    "get_usage",
    "get_all",
    "providers_with_models",
]

from .canonical import UsageLimit  # noqa: E402  (re-export for convenience)
