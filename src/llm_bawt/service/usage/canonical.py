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


class UsageSpendDay(BaseModel):
    """One UTC day of this app's ledgered spend, split by what paid for it (TASK-1733)."""

    date: str  # UTC calendar day, YYYY-MM-DD
    chat_usd: float
    maintenance_usd: float
    unpriced_calls: int = 0  # billed calls with no known cost (not in the USD figures)


class UsageSpend(BaseModel):
    """USD spend, never confused with subscription quota percentages."""

    today_usd: float
    seven_day_usd: float
    month_usd: float
    source: str  # anthropic_reported | bawt_recorded_estimate
    # Always from this app's ledger (bawt_recorded_estimate), whatever ``source``
    # is: the last 7 UTC days, oldest first, chat vs background maintenance.
    daily: list[UsageSpendDay] | None = None


class UsageBalance(BaseModel):
    """Prepaid API credit (see ``usage.balance``).

    ``bawt_recorded_estimate``: a user-set anchor minus recorded turn costs
    since; every field is set, and ``unpriced_turns`` counts turns since the
    anchor with no recorded cost (the remainder excludes them, so it is an
    upper bound). ``provider_reported``: only ``remaining_usd`` is known; the
    deposit, debits and anchor time stay None, never invented.
    """

    amount_usd: float | None
    remaining_usd: float
    used_usd: float | None
    as_of: int | None  # unix seconds; the anchor is stamped on the server
    source: str = "bawt_recorded_estimate"  # bawt_recorded_estimate | provider_reported
    updated_at: float | None = None  # fractional unix seconds; orders turn_complete updates
    unpriced_turns: int = 0  # bawt_recorded_estimate only: counted turns with no recorded cost


class ProviderUsage(BaseModel):
    """One provider/subscription's usage snapshot in canonical form."""

    provider: str                  # canonical provider id, e.g. "claude"
    display_name: str              # human label, e.g. "Claude"
    backend: str | None = None     # owning agent backend, e.g. "claude-code"
    kind: str = "subscription_windows"  # subscription_windows | balance (spend: pre-counter servers)
    access_path_key: str | None = None  # echoed for a path/bot query: the route it resolved
    available: bool = False        # True only when live limits were fetched
    status: str = STATUS_ERROR     # one of STATUS_* above
    error: str | None = None       # human-readable detail when not ok
    fetched_at: int | None = None  # unix seconds the snapshot was produced
    cached: bool = False           # True if served from cache / stale
    limits: list[UsageLimit] = Field(default_factory=list)
    spend: UsageSpend | None = None
    balance: UsageBalance | None = None
    balance_supported: bool = False  # new-server capability; older spend servers must not accept an anchor
    # True only when PUT/DELETE /v1/usage/{provider}/balance accept a manual anchor
    # (balance_mode "anchor"); false when the provider's own API is authoritative.
    manual_balance_supported: bool = False
    raw: dict | None = None        # provider-native payload, for debug/refine


class AllUsage(BaseModel):
    """Aggregate across every registered provider (the all-backends view)."""

    fetched_at: int
    providers: list[ProviderUsage] = Field(default_factory=list)
