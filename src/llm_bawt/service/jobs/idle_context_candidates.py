"""Read-only, bot-pointer-based candidate selection for idle maintenance.

This is a permissive prefilter, not admission authority. The shared lifecycle
rechecks the session/usage fences and busy state before accepting a reset.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

from sqlalchemy import text

MIN_RESIDENT_TOKENS = 100_000
MIN_IDLE = timedelta(minutes=15)
# Privacy boundary: exclude before reading any session or turn for this bot.
PROTECTED_BOT_IDS = frozenset({"mira"})


class IdleContextCandidates:
    def __init__(self, engine, *, protected_bot_ids=()):
        self.engine = engine
        self.protected_bot_ids = PROTECTED_BOT_IDS | frozenset(
            slug.strip().lower() for slug in protected_bot_ids if isinstance(slug, str)
        )

    def read(self, target_bot: str) -> list[dict]:
        """Select one pointed-to session per bot; never guess a missing pointer.

        Archived sessions are valid: an explicit turn can resume one. Latest
        usage is read from its canonical turn, not from an arbitrary user's
        active session. Later bot activity invalidates an older measurement.
        """
        with self.engine.connect() as conn:
            return self._read(conn, target_bot)

    def read_one(self, conn, bot_id: str) -> dict | None:
        """Recheck using the lifecycle's existing target-lock transaction."""
        rows = self._read(conn, bot_id)
        return rows[0] if rows else None

    def _read(self, conn, target_bot: str) -> list[dict]:
        rows = conn.execute(text("""
                WITH targets AS MATERIALIZED (
                    SELECT slug, last_session_id FROM bot_profiles
                    WHERE (:target_bot='*' OR slug=:target_bot)
                      AND lower(slug) != ALL(:protected)
                )
                SELECT b.slug AS bot_id, s.id AS session_id, s.user_id,
                       t.id AS usage_turn_id, t.ended_at, t.token_usage_json,
                       latest.id AS latest_turn_id,
                       latest.ended_at AS last_completed_at
                FROM targets b
                LEFT JOIN sessions s ON s.id=b.last_session_id
                  AND s.bot_id=b.slug AND s.status <> 'deleted'
                LEFT JOIN LATERAL (
                    SELECT tl.id, tl.ended_at, tl.token_usage_json
                    FROM turn_logs tl
                    JOIN messages m ON m.bot_id=regexp_replace(lower(b.slug), '[^a-z0-9_]', '', 'g')
                      AND m.id=tl.trigger_message_id
                    WHERE s.id IS NOT NULL AND tl.bot_id=b.slug
                      AND m.session_id=s.id
                    ORDER BY tl.created_at DESC, tl.id DESC LIMIT 1
                ) t ON true
                LEFT JOIN LATERAL (
                    SELECT tl.id, tl.ended_at FROM turn_logs tl
                    WHERE s.id IS NOT NULL AND tl.bot_id=b.slug
                    ORDER BY tl.created_at DESC, tl.id DESC LIMIT 1
                ) latest ON true
                ORDER BY b.slug
        """), {"target_bot": target_bot, "protected": list(self.protected_bot_ids)})
        return [dict(row) for row in rows.mappings()]

    @staticmethod
    def skip_reason(candidate: dict, *, now: datetime) -> str | None:
        if not candidate["session_id"] or not candidate["user_id"]:
            return "no valid accepted session"
        if not candidate["usage_turn_id"] or not candidate["ended_at"]:
            return "missing completed context measurement"
        if candidate["latest_turn_id"] != candidate["usage_turn_id"]:
            return "newer bot activity invalidated measurement"
        completed = candidate["last_completed_at"]
        if completed is None or now - completed <= MIN_IDLE:
            return "idle less than threshold"
        try:
            usage = json.loads(candidate["token_usage_json"] or "null")
            if not isinstance(usage, dict):
                return "resident context unknown"
            resident = int(usage.get("resident_tokens") or 0) or int(usage.get("input_tokens") or 0)
        except (TypeError, ValueError, OverflowError):
            return "resident context unknown"
        if resident <= MIN_RESIDENT_TOKENS:
            return "resident context within threshold"
        return None
