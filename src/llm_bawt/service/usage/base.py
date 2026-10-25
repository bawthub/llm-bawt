"""Provider usage-adapter interface.

A ``UsageAdapter`` knows how to fetch ONE provider's subscription usage and
map it into the canonical :class:`ProviderUsage`. Each adapter is free to
source its data however it needs — a direct OAuth call (Claude), a static API
key (z.ai), or an RPC to the bridge that owns the credential (codex) — the
registry and the HTTP route don't care.

Adapters must NOT raise for *expected* failure modes (missing credential,
upstream 403/429); they return a ``ProviderUsage`` with the appropriate
``status`` so the UI can render it. Raising is reserved for genuinely
unexpected errors, which the registry catches and turns into a stale/error
snapshot.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import datetime

from .canonical import ProviderUsage, STATUS_NOT_IMPLEMENTED


class UsageAdapter(ABC):
    provider: str = ""             # canonical provider id
    display_name: str = ""         # human label
    backend: str | None = None     # owning agent backend, if any
    access_path_keys: tuple[str, ...] = ()  # exact catalog paths this report covers
    kind: str = "subscription_windows"
    # Prepaid credit (see usage.balance): "anchor" = user-entered amount counted
    # down per turn; "provider" = the provider's balance API; None = no balance.
    balance_mode: str | None = None
    # Seconds a successful report is reused; None = LLM_BAWT_USAGE_CACHE_TTL.
    report_ttl: float | None = None
    # Runtime-setting key of a pre-counter anchor to migrate once, if any.
    legacy_balance_setting: str | None = None

    @abstractmethod
    async def fetch(self) -> ProviderUsage:
        """Return a fresh canonical snapshot. May be slow (external call)."""
        raise NotImplementedError

    def credential_connected_at(self) -> datetime | None:
        """When the billed credential was last connected; an older anchor is void."""
        return None

    # -- helpers for subclasses ------------------------------------------

    def _base(self, **overrides) -> ProviderUsage:
        """Build a ProviderUsage pre-filled with this adapter's identity."""
        fields = dict(
            provider=self.provider,
            display_name=self.display_name,
            backend=self.backend,
            kind=self.kind,
            balance_supported=bool(self.balance_mode),
            fetched_at=int(time.time()),
        )
        fields.update(overrides)
        return ProviderUsage(**fields)


class NotImplementedAdapter(UsageAdapter):
    """Registered-but-unimplemented provider.

    Surfaces the backend in the all-providers view as a clearly-labeled
    placeholder instead of silently dropping it. Subclass and set the three
    identity attributes; the canonical model + UI wiring already handle it.
    """

    async def fetch(self) -> ProviderUsage:
        return self._base(
            available=False,
            status=STATUS_NOT_IMPLEMENTED,
            error="Usage reporting for this provider is not implemented yet.",
            limits=[],
        )
