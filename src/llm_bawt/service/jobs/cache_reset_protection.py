"""One-way hold for an observed native write, not an expiry decision."""
from __future__ import annotations

import json

from sqlalchemy import text

from agent_bridge.prompt_cache_clock import prompt_cache_clock, sdk_session_ref

CACHE_WRITE_HOLD = "observed native cache write protection window"
_WINDOW_MS = 3_600_000

# Ended turns of exactly this accepted thread and owner. Return only the newest
# qualifying same-domain observation inside the window, never entire histories
# or raw identities. Do not range-filter on wall time: NTP can step that clock.
_RECENT_WRITE = text("""
    WITH scoped AS MATERIALIZED (
        SELECT tl.created_at, tl.id,
               CASE WHEN pg_input_is_valid(tl.token_usage_json, 'jsonb')
               THEN CAST(tl.token_usage_json AS jsonb)->'prompt_cache' END AS cache
        FROM messages m
        JOIN turn_logs tl ON tl.trigger_message_id=m.id AND tl.bot_id=:bot AND tl.user_id=:owner
        WHERE m.bot_id=regexp_replace(lower(:bot), '[^a-z0-9_]', '', 'g')
          AND m.session_id=:thread AND m.role='user' AND tl.ended_at IS NOT NULL
          AND tl.token_usage_json LIKE '%sdk_session_ref%'
    ), latest_binding AS (
        SELECT cache->>'sdk_session_ref' AS ref
        FROM scoped
        WHERE cache->>'sdk_session_ref' ~ '^[0-9a-f]{64}$'
        ORDER BY created_at DESC, id DESC LIMIT 1
    )
    SELECT max(p.observed_ms)
    FROM scoped u JOIN latest_binding b ON b.ref=:session_ref
    CROSS JOIN LATERAL (SELECT u.cache->'protection_write' AS proof) w
    CROSS JOIN LATERAL (
        SELECT CASE WHEN jsonb_typeof(w.proof->'observed_ms')='number'
                         AND w.proof->>'observed_ms' ~ '^[0-9]{1,15}$'
               THEN CAST(w.proof->>'observed_ms' AS bigint) END AS observed_ms
    ) p
    WHERE u.cache->>'route'='native' AND u.cache->>'access_path'='anthropic-oauth'
      AND u.cache->>'sdk_ttl'='1h' AND u.cache->>'sdk_session_ref'=:session_ref
      AND w.proof->>'source'='native_oauth_relay_1h_write' AND w.proof->>'clock_domain'=:domain
      AND jsonb_typeof(w.proof->'model')='string' AND w.proof->>'model' <> ''
      AND jsonb_typeof(w.proof->'write_tokens')='number'
      AND w.proof->>'write_tokens' ~ '^[1-9][0-9]*$'
      AND p.observed_ms > :cutoff AND p.observed_ms <= :now
""")


def _object(value: object) -> dict:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def _accepted_sdk_ref(candidate: dict) -> str | None:
    metadata = _object(candidate.get("session_metadata"))
    key = _object(metadata.get("agent_session_keys")).get("claude_code")
    if not isinstance(key, str) or not key.strip() or ":" in key:
        return None
    return sdk_session_ref(key.strip())


def cache_write_hold_reason(conn, candidate: dict) -> str | None:
    """Only called for otherwise eligible scheduler candidates, not evictions."""
    if not candidate["session_id"] or not candidate["user_id"]:
        return None
    session_ref = _accepted_sdk_ref(candidate)
    clock = prompt_cache_clock() if session_ref else None
    if clock is None:
        return None
    domain, now = clock
    observed = conn.execute(_RECENT_WRITE, {
        "bot": candidate["bot_id"], "owner": candidate["user_id"], "thread": candidate["session_id"],
        "session_ref": session_ref, "domain": domain, "now": now, "cutoff": now - _WINDOW_MS,
    }).scalar_one()
    return CACHE_WRITE_HOLD if observed is not None else None
