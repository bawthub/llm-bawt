"""Shared claude-code-bridge constants + helpers (TASK-555).

Relocated from ``bridge.py`` so the split-out mixin modules can import them
without a cycle back through ``bridge``. ``bridge.py`` re-imports every name
here (preserving ``from .bridge import _get_fresh_oauth_token`` in __main__).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path

import httpx

logger = logging.getLogger("claude_code_bridge.bridge")

SESSION_PREFIX = "claude-code:"

# TASK-615/501 Phase 2: the bridge seed-policy constants (SEED_SETTING_KEY /
# CONTINUITY_SETTING_KEY) are gone. llm-bawt (maybe_build_session_seed) is the
# sole seed authority and pushes pre-assembled messages via inject_messages; the
# bridge no longer reads continuity or self-fetches a seed.

# TASK-490: MCP tool context body. Canonical default lives in the app registry
# (agents.mcp_tool_context_template); the bridge cannot import llm_bawt, so it
# fetches the (overridable) body via GET /v1/prompts/{key} and falls back to
# this BYTE-IDENTICAL copy. The leading "\n\n" separator is added at append time.
MCP_TOOL_CONTEXT_KEY = "agents.mcp_tool_context_template"
_MCP_TOOL_CONTEXT_FALLBACK = (
    "## MCP Tool Context\n"
    "Your bot_id is \"{bot_slug}\". When using bawthub MCP tools:\n"
    "- Memory/message tools: always pass bot_id=\"{bot_slug}\"\n"
    "- Profile tool with entity_type=\"user\": use entity_id=\"nick\" (the user)\n"
    "- Profile tool with entity_type=\"bot\": use entity_id=\"{bot_slug}\" (yourself)\n"
    "- When you actually begin/resume an existing task in ordinary chat, call "
    "tasks_associate_current(task_id=...). Do not link from TASK-N text alone.\n"
    "- tasks_create(..., associate_current_turn=true) creates and links new work; "
    "tasks_update(..., associate_current_turn=true) claims/links existing work."
)
# Stamped on synthetic seed transcript entries. Not load-bearing for resume
# (the CLI reads message content, not this), just keeps entries well-formed.
_SEED_CLI_VERSION = "2.1.191"
# Mirrors the SDK's _sanitize_path: every non-alphanumeric char -> '-'.
_SEED_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9]")


def _bot_slug_from_session_key(session_key: str) -> str:
    sk = (session_key or "").strip()
    if not sk:
        return ""
    return sk.split(":", 1)[0]


def _fmt_tokens(n: int | None) -> str:
    """Human-format a token count: 67288 -> '67.3k', 1486 -> '1.5k'."""
    if n is None:
        return "?"
    n = int(n)
    if n < 1000:
        return str(n)
    return f"{n / 1000:.1f}k"


def _usage_input_total(usage: dict | None) -> int:
    """Total prompt tokens in an Anthropic-shaped usage dict."""
    if not isinstance(usage, dict):
        return 0
    return (
        int(usage.get("input_tokens", 0) or 0)
        + int(usage.get("cache_read_input_tokens", 0) or 0)
        + int(usage.get("cache_creation_input_tokens", 0) or 0)
    )


# Per-1M-token USD rates for proxy providers the CLI prices as "unknown"
# (which falls through to Opus-tier $5/$25 and wildly overstates Grok cost).
_XAI_RATES = {
    # model substring -> (input, output, cache_read, cache_write)
    "grok-4.5": (2.0, 6.0, 0.5, 2.0),
    "grok-4.3": (1.25, 2.50, 0.20, 1.25),
    "grok-4.20": (1.25, 2.50, 0.20, 1.25),
    "grok-build": (1.0, 2.0, 0.20, 1.0),
}
_XAI_DEFAULT_RATES = (2.0, 6.0, 0.5, 2.0)


def _estimate_proxy_cost_usd(model: str | None, usage: dict | None) -> float | None:
    """Estimate turn cost for proxy-routed models from token counts.

    Returns None when we can't price the model (leave SDK total_cost_usd).
    """
    if not model or not isinstance(usage, dict):
        return None
    m = str(model).strip().lower()
    # Kimi For Coding is a flat subscription surface. The Claude CLI does not
    # know that model and otherwise prices it as Opus-tier pay-as-you-go, which
    # is pure fiction. There is no marginal per-token charge to estimate.
    if m.startswith("kimi_coding/"):
        return 0.0
    if not m.startswith("xai/"):
        return None
    rates = _XAI_DEFAULT_RATES
    for key, r in _XAI_RATES.items():
        if key in m:
            rates = r
            break
    in_rate, out_rate, cr_rate, cw_rate = rates
    inp = int(usage.get("input_tokens", 0) or 0)
    cr = int(usage.get("cache_read_input_tokens", 0) or usage.get("cache_read_tokens", 0) or 0)
    cw = int(
        usage.get("cache_creation_input_tokens", 0)
        or usage.get("cache_creation_tokens", 0)
        or 0
    )
    out = int(usage.get("output_tokens", 0) or 0)
    cost = (inp * in_rate + cr * cr_rate + cw * cw_rate + out * out_rate) / 1_000_000
    return round(cost, 6) if cost > 0 else 0.0


def _pick_iteration_usage(
    latest_assistant_usage: dict | None,
    latest_stream_usage: dict | None,
    cumulative_usage: dict | None,
    *,
    proxy_model: bool,
) -> dict:
    """Choose the usage snapshot that best represents final context size.

    Preference:
      1. For proxy streams (xAI / ChatGPT / z.ai): prefer the last
         ``message_delta`` when it reports a higher total input than the
         AssistantMessage snapshot. The synthetic Anthropic stream only
         finalizes real usage on message_delta; AssistantMessage can keep
         message_start zeros or a partial merge that under-counts.
      2. Otherwise last AssistantMessage usage (true final API view).
      3. Stream usage, then cumulative ResultMessage.usage.
    """
    am = latest_assistant_usage if isinstance(latest_assistant_usage, dict) else None
    sm = latest_stream_usage if isinstance(latest_stream_usage, dict) else None
    cu = cumulative_usage if isinstance(cumulative_usage, dict) else None

    if proxy_model and sm and _usage_input_total(sm) > 0:
        if not am or _usage_input_total(sm) >= _usage_input_total(am):
            return sm
    if am and _usage_input_total(am) > 0:
        return am
    if sm and _usage_input_total(sm) > 0:
        return sm
    if am:
        return am
    if sm:
        return sm
    return cu or {}



def _read_latest_compact_metadata(session_id: str | None) -> dict | None:
    """Return the most recent ``compact_boundary`` metadata for a session.

    The Claude Agent SDK does NOT emit ``compact_boundary`` on the wire — a
    ``/compact`` turn surfaces only as ``SystemMessage(subtype="status")`` with
    a ``compact_result`` string ("success"/"failed"); the actual pre/post token
    counts are written solely to the on-disk transcript. So to report the new
    resident context size we read it back from the session's ``.jsonl`` (located
    by ``<session_id>.jsonl`` under ``~/.claude/projects/``) and return the last
    ``compactMetadata`` dict ({trigger, preTokens, postTokens, durationMs, ...}).
    Best-effort: returns ``None`` if the file or marker can't be found.
    """
    if not session_id:
        return None
    try:
        base = Path.home() / ".claude" / "projects"
        matches = list(base.glob(f"*/{session_id}.jsonl")) or list(
            base.glob(f"**/{session_id}.jsonl")
        )
        if not matches:
            return None
        meta: dict | None = None
        with open(matches[0], "r") as fh:
            for line in fh:
                if "compact_boundary" not in line:
                    continue
                try:
                    entry = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if entry.get("subtype") == "compact_boundary":
                    cm = entry.get("compactMetadata")
                    if isinstance(cm, dict):
                        meta = cm  # keep the LAST boundary in file order
        return meta
    except Exception:
        logger.debug("compact metadata read failed for %s", session_id, exc_info=True)
        return None


# TASK-635: the bridge is a READ-ONLY consumer of the app-owned Claude
# credential. The app (llm-bawt) is the sole refresher of the rotate-on-use
# refresh chain — the bridge must NEVER refresh or write a bundle (two
# independent refreshers racing on one bundle is the exact invalid_grant
# failure this design removes).
#
# TASK-636: the bind mount is GONE. The DB row (not a host file) is the source
# of truth; this bridge reaches it via the broker endpoint.
#
# TASK-671: resolve the broker token on every direct-Anthropic turn. App-owned
# proactive refresh rotates and immediately revokes the prior access token, so
# a bridge cache cannot safely infer validity from the old token's expiry.
#
# Resolution order:
#   1. broker endpoint GET /v1/providers/claude/token (?force=true post-401)
#   2. the legacy self-owned bundle at ~/.claude/.credentials.json (read-only
#      fallback — the ~/.claude dir mount is harness session state, not a
#      credential mount; explicitly out of scope for TASK-636)
#   4. env CLAUDE_CODE_OAUTH_TOKEN (long-lived setup-token)

_LEGACY_CREDENTIALS_PATH = Path.home() / ".claude" / ".credentials.json"
_REFRESH_BUFFER_MS = 5 * 60 * 1000

def _read_oauth_bundle(path: Path) -> dict | None:
    """Read a claudeAiOauth bundle (wrapper or bare form). Never writes."""
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text())
    except Exception as e:
        logger.warning("Failed to read Claude credential %s: %s", path, e)
        return None
    if not isinstance(data, dict):
        return None
    bundle = data.get("claudeAiOauth")
    if bundle is None and data.get("accessToken"):
        bundle = data
    return bundle if isinstance(bundle, dict) else None


def _token_expired_or_stale(expires_at: int | None) -> bool:
    if not expires_at:
        return False
    now_ms = int(time.time() * 1000)
    return (now_ms + _REFRESH_BUFFER_MS) >= int(expires_at)


def _fetch_broker_token(
    *, force: bool = False, rejected_token_sha256: str | None = None,
) -> tuple[str | None, int | None]:
    """Read the app-owned token; optionally recover a specific rejected token.

    The digest lets the sole refresher reuse a newer generation under its lock.
    No bearer/refresh token is placed in a query string or log.
    """
    api_url = (os.environ.get("LLM_BAWT_API_URL") or "").rstrip("/")
    if not api_url:
        return None, None
    headers = {}
    secret = os.environ.get("BRIDGE_CLAUDE_TOKEN_SECRET")
    if secret:
        headers["X-Bridge-Token"] = secret
    params = {"force": "true"} if force else {}
    if force and rejected_token_sha256:
        params["rejected_token_sha256"] = rejected_token_sha256
    try:
        resp = httpx.get(
            f"{api_url}/v1/providers/claude/token",
            params=params or None,
            headers=headers,
            timeout=25.0,  # a broker-side upstream refresh can take ~15s
        )
        if resp.is_error:
            logger.warning(
                "Claude token broker returned %s: %s",
                resp.status_code,
                (resp.text or "")[:200],
            )
            return None, None
        payload = resp.json()
        token = payload.get("access_token")
        expires_at = payload.get("expires_at")
        if token:
            logger.info("Fetched Claude access token from app broker (state=%s)", payload.get("state"))
        return token or None, expires_at if isinstance(expires_at, int) else None
    except Exception as e:
        logger.warning("Claude token broker unreachable: %s", e)
        return None, None


def _get_fresh_oauth_token(*, force_refresh: bool = False) -> str | None:
    """Return the app-owned Claude token without ever refreshing locally.

    Every call consults the broker because the app's proactive OAuth rotation
    revokes the previous access token before its advertised expiry. ``force``
    asks the app to refresh upstream after a confirmed 401; the bridge remains
    read-only in both cases.
    """
    stale_candidate: str | None = None

    # 1. Broker endpoint — the app owns the refresh chain (DB row is truth).
    token, _expires_at = _fetch_broker_token(force=force_refresh)
    if token:
        return token

    # 2. Legacy self-owned bundle at ~/.claude/.credentials.json, READ-ONLY
    #    (interactive TUI login fallback — the ~/.claude dir mount is harness
    #    session state, not a credential bind mount).
    bundle = _read_oauth_bundle(_LEGACY_CREDENTIALS_PATH)
    if bundle:
        token = bundle.get("accessToken")
        if token and not _token_expired_or_stale(bundle.get("expiresAt")):
            return token
        stale_candidate = stale_candidate or token

    # 3. Long-lived setup-token from env.
    env_token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    if env_token:
        return env_token

    # Last resort: a stale token beats none (clock skew may save it).
    if stale_candidate:
        logger.warning("Only a stale Claude access token is available — using it anyway")
    return stale_candidate


def _is_cli_crash(exc: Exception) -> bool:
    """Return True if the exception looks like a CLI subprocess crash (exit code 1)."""
    msg = str(exc).lower()
    return "exit code: 1" in msg or "exit code 1" in msg


def _is_auth_failure(exc: Exception, stderr_lines: list[str]) -> bool:
    haystack = "\n".join([str(exc), *stderr_lines]).lower()
    auth_markers = (
        "oauth token has expired",
        "failed to authenticate",
        "authentication_error",
        "401",
        "unauthorized",
        "invalid api key",
        "invalid token",
        "token has expired",
        "invalid_grant",
        "refresh token not found or invalid",
        "claude oauth refresh failed",
    )
    return any(marker in haystack for marker in auth_markers)
