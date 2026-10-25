"""Canonical, provider-agnostic usage model.

Every provider (Claude, z.ai, OpenAI/codex, ...) maps its native
subscription/limit payload into these shapes so the UI renders a single
consistent structure regardless of backend.

A ``ProviderUsage`` is one subscription/account's state. Its ``limits`` are
the individual rolling windows the provider enforces (e.g. Claude's 5-hour
session limit, weekly all-models, weekly Sonnet-only). A provider that isn't
implemented yet returns ``available=False, status="not_implemented"`` with an
empty ``limits`` list — the registry/endpoint surface it so the UI can show a
placeholder rather than omitting the backend entirely.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# Canonical status values. Keep this list small and stable — the UI switches
# on it. ``ok`` is the only state that carries authoritative live limits.
STATUS_OK = "ok"
STATUS_NOT_IMPLEMENTED = "not_implemented"
STATUS_UNAUTHORIZED = "unauthorized"
STATUS_RATE_LIMITED = "rate_limited"
STATUS_STALE = "stale"  # credential present but access token expired (shared mode)
STATUS_USAGE_STALE = "usage_stale"  # cached limits shown; live usage refresh failed
STATUS_ERROR = "error"


class UsageLimit(BaseModel):
    """One rolling-window limit within a subscription."""

    id: str                       # stable key, e.g. "session_5h", "weekly_all"
    label: str                    # human label, e.g. "5-hour limit"
    used_pct: float | None = None  # 0-100, the primary number; None if unknown
    used: float | None = None      # optional absolute amount consumed
    limit: float | None = None     # optional absolute ceiling
    unit: str | None = None        # e.g. "tokens", "requests"; None if pct-only
    resets_at: int | None = None   # unix seconds when this window resets
    window: str | None = None      # descriptive window, e.g. "5h", "7d"
    severity: str | None = None    # provider-native severity, e.g. "normal"/"critical"
    active: bool | None = None     # whether this window is currently the binding one
    # Models this window counts against: catalog model keys or model-family
    # names (``fable`` covers ``claude-fable-5-1``). None = every model on the
    # subscription. See ``usage.quota.limit_applies``.
    model_scope: list[str] | None = None


class UsageSpend(BaseModel):
    """USD spend, never confused with subscription quota percentages."""

    today_usd: float
    seven_day_usd: float
    month_usd: float
    source: str  # anthropic_reported | bawt_recorded_estimate


class UsageBalance(BaseModel):
    """Manual prepaid balance minus app-recorded charges after its anchor."""

    amount_usd: float
    remaining_usd: float
    used_usd: float
    as_of: int  # unix seconds; the anchor is stamped on the server
    source: str = "bawt_recorded_estimate"


class ProviderUsage(BaseModel):
    """One provider/subscription's usage snapshot in canonical form."""

    provider: str                  # canonical provider id, e.g. "claude"
    display_name: str              # human label, e.g. "Claude"
    backend: str | None = None     # owning agent backend, e.g. "claude-code"
    kind: str = "subscription_windows"  # subscription_windows | spend
    available: bool = False        # True only when live limits were fetched
    status: str = STATUS_ERROR     # one of STATUS_* above
    error: str | None = None       # human-readable detail when not ok
    fetched_at: int | None = None  # unix seconds the snapshot was produced
    cached: bool = False           # True if served from cache / stale
    limits: list[UsageLimit] = Field(default_factory=list)
    spend: UsageSpend | None = None
    balance: UsageBalance | None = None
    balance_supported: bool = False  # new-server capability; older spend servers must not accept an anchor
    raw: dict | None = None        # provider-native payload, for debug/refine


class AllUsage(BaseModel):
    """Aggregate across every registered provider (the all-backends view)."""

    fetched_at: int
    providers: list[ProviderUsage] = Field(default_factory=list)
