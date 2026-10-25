"""Claude (Anthropic subscription) usage adapter.

Calls the internal ``/api/oauth/usage`` endpoint with a ``user:profile``-scoped
OAuth token (see ``claude_oauth.py``) and maps the response to canonical form.

NOTE ON THE RESPONSE SHAPE: this endpoint is undocumented. ``_to_canonical``
is written defensively — it maps the limit windows we expect from the
interactive ``/usage`` display (5-hour session, weekly all-models, weekly
Sonnet/Opus) but also generically picks up any ``{utilization, resets_at}``
entry, and ALWAYS attaches the provider-native payload as ``raw``. The first
successful live call is the moment to confirm field names and tighten this
mapper; until then the raw payload is preserved so nothing is lost.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from ..base import UsageAdapter
from ..canonical import (
    ProviderUsage,
    UsageLimit,
    STATUS_OK,
    STATUS_UNAUTHORIZED,
    STATUS_RATE_LIMITED,
    STATUS_STALE,
    STATUS_ERROR,
)
from ..claude_oauth import credentials_mode, load_usage_token, usage_credentials_path

logger = logging.getLogger(__name__)

USAGE_URL = "https://api.anthropic.com/api/oauth/usage"
# Header the Claude CLI sends on OAuth-authenticated calls.
_ANTHROPIC_BETA = "oauth-2025-04-20"

# The response's authoritative source is the `limits[]` array; map its `kind`
# -> (canonical id, label, window). `weekly_scoped` is handled specially (label
# derived from scope.model.display_name, e.g. "Sonnet").
_KIND_MAP: dict[str, tuple[str, str, str]] = {
    "session": ("session_5h", "5-hour limit", "5h"),
    "weekly_all": ("weekly_all", "Weekly · all models", "7d"),
}

# Fallback only — top-level window keys, used if `limits[]` is ever absent.
_TOPLEVEL_WINDOWS: list[tuple[str, str, str, str]] = [
    # (response key, canonical id, label, window)
    ("five_hour", "session_5h", "5-hour limit", "5h"),
    ("seven_day", "weekly_all", "Weekly · all models", "7d"),
    ("seven_day_opus", "weekly_opus", "Weekly · Opus", "7d"),
    ("seven_day_sonnet", "weekly_sonnet", "Weekly · Sonnet", "7d"),
]


def _parse_reset(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # Heuristic: ms vs s.
        v = float(value)
        return int(v / 1000) if v > 1e12 else int(v)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if s.isdigit():
            return _parse_reset(int(s))
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            return None
    return None


def _parse_pct(value, *, allow_fraction: bool = True) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    # Anthropic's `limits[].percent` is ALREADY a 0-100 number — never rescale
    # it. Only the legacy top-level `utilization` fallback can be a 0-1 ratio,
    # and even there the literal value 1 means 1%, not 100%: a weekly window
    # that just reset reports `percent: 1`, which the old `<= 1.0` rule turned
    # into 100%. Rescale strictly inside the open interval (0, 1) only.
    if allow_fraction and 0.0 < v < 1.0:
        return round(v * 100, 1)
    return round(v, 1)


def _from_limit_item(item: dict, idx: int) -> UsageLimit | None:
    """Map one entry of the response's authoritative `limits[]` array.

    Item shape: ``{kind, group, percent, severity, resets_at, scope, is_active}``.
    ``kind`` is ``session`` / ``weekly_all`` / ``weekly_scoped`` (the last
    carries ``scope.model.display_name``, e.g. "Sonnet").
    """
    kind = (item.get("kind") or "").lower()
    # `percent` is provider-native 0-100; never treat it as a fraction.
    pct = _parse_pct(item.get("percent"), allow_fraction=False)
    resets = _parse_reset(item.get("resets_at"))
    if kind == "weekly_scoped":
        model = (((item.get("scope") or {}).get("model") or {}).get("display_name")) or "scoped"
        cid, label, window = f"weekly_{model.lower()}", f"Weekly · {model}", "7d"
    elif kind in _KIND_MAP:
        cid, label, window = _KIND_MAP[kind]
    else:
        group = (item.get("group") or "").lower()
        cid = kind or f"window_{idx}"
        label = (kind or f"window {idx}").replace("_", " ").title()
        window = "7d" if group == "weekly" else "5h" if group == "session" else None
    return UsageLimit(
        id=cid, label=label, used_pct=pct, resets_at=resets, window=window,
        severity=item.get("severity"), active=item.get("is_active"),
    )


def _from_toplevel(data: dict) -> list[UsageLimit]:
    """Fallback mapper over top-level window keys (`{utilization, resets_at}`)."""
    out: list[UsageLimit] = []
    for key, cid, label, window in _TOPLEVEL_WINDOWS:
        val = data.get(key)
        if isinstance(val, dict):
            pct = _parse_pct(val.get("utilization"))
            resets = _parse_reset(val.get("resets_at"))
            if pct is not None or resets is not None:
                out.append(UsageLimit(
                    id=cid, label=label, used_pct=pct, resets_at=resets, window=window,
                ))
    return out


def _to_canonical(adapter: "ClaudeUsageAdapter", data: dict) -> ProviderUsage:
    """Map the /api/oauth/usage payload to canonical limits.

    The response's ``limits[]`` array is the authoritative rate-limit list (what
    the official client renders): session + weekly windows with percent,
    severity and reset times. We map that and deliberately ignore the parallel
    top-level window keys (duplicates) and the dollar ``spend`` / ``extra_usage``
    blocks (preserved in ``raw``). If ``limits[]`` is ever absent we fall back to
    the top-level keys.
    """
    limits: list[UsageLimit] = []
    if isinstance(data, dict) and isinstance(data.get("limits"), list):
        for i, item in enumerate(data["limits"]):
            if isinstance(item, dict):
                lim = _from_limit_item(item, i)
                if lim is not None:
                    limits.append(lim)
    elif isinstance(data, dict):
        limits = _from_toplevel(data)
    order = {"session_5h": 0, "weekly_all": 1, "weekly_sonnet": 2, "weekly_opus": 3}
    limits.sort(key=lambda l: order.get(l.id, 99))
    return adapter._base(
        available=True,
        status=STATUS_OK,
        limits=limits,
        raw=data if isinstance(data, dict) else {"value": data},
    )


class ClaudeUsageAdapter(UsageAdapter):
    provider = "claude"
    display_name = "Claude"
    backend = "claude-code"
    access_path_keys = ("anthropic-oauth",)

    async def fetch(self) -> ProviderUsage:
        cred = load_usage_token()
        if cred.state == "missing":
            return self._base(
                available=False,
                status=STATUS_UNAUTHORIZED,
                error=(
                    "No Claude usage credential found at "
                    f"{usage_credentials_path()}. Point CLAUDE_USAGE_CREDENTIALS_PATH "
                    "at a user:profile-scoped claudeAiOauth bundle (e.g. your "
                    "interactive Claude Code login). See docs/usage-endpoint.md."
                ),
            )
        if cred.state == "stale":
            if credentials_mode() == "owned":
                error = (
                    "Owned usage credential could not be refreshed (refresh "
                    "token rejected upstream). Re-mint the dedicated login: "
                    "isolated `claude login` + copy the bundle to "
                    f"{usage_credentials_path()}. Using Claude Code does NOT "
                    "fix this — the owned bundle is a separate login."
                )
            else:
                error = (
                    "Usage credential's access token has expired. It refreshes "
                    "automatically the next time you use Claude Code (the shared "
                    "login owner); this app does not refresh it to avoid rotating "
                    "the token out from under your session."
                )
            return self._base(
                available=False,
                status=STATUS_STALE,
                error=error,
            )
        token = cred.token
        headers = {
            "Authorization": f"Bearer {token}",
            "anthropic-beta": _ANTHROPIC_BETA,
            "User-Agent": "llm-bawt-usage/1",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(USAGE_URL, headers=headers)

        if resp.status_code == 403:
            return self._base(
                available=False,
                status=STATUS_UNAUTHORIZED,
                error="Token lacks user:profile scope for /api/oauth/usage.",
                raw=_safe_json(resp),
            )
        if resp.status_code == 429:
            return self._base(
                available=False,
                status=STATUS_RATE_LIMITED,
                error="Upstream rate-limited the usage endpoint.",
            )
        if resp.status_code >= 400:
            return self._base(
                available=False,
                status=STATUS_ERROR,
                error=f"HTTP {resp.status_code} from usage endpoint.",
                raw=_safe_json(resp),
            )
        return _to_canonical(self, resp.json())


def _safe_json(resp: httpx.Response) -> dict | None:
    try:
        body = resp.json()
        return body if isinstance(body, dict) else {"value": body}
    except Exception:  # noqa: BLE001
        return {"text": (resp.text or "")[:500]}
