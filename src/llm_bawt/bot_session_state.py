"""The bot's last accepted session, independent of message authorship.

Callers own admission/rotation and their transaction. Reads never select a
session by user or by timestamp, and never repair or move the pointer. A null
pointer means no accepted session has been recorded yet, not permission to
choose a historical session heuristically.
"""
from __future__ import annotations

from sqlalchemy import bindparam, text

_POINTERS = text("""
    SELECT b.slug, b.last_session_id, s.id, s.bot_id, s.user_id,
           s.started_at, s.ended_at, s.archived_at, s.status,
           s.session_metadata
    FROM bot_profiles AS b
    LEFT JOIN sessions AS s ON s.id = b.last_session_id
    WHERE b.slug IN :bots
""").bindparams(bindparam("bots", expanding=True))


class BotSessionState:
    """Transaction-bound authority for ``bot_profiles.last_session_id``."""

    @staticmethod
    def ensure_schema(conn) -> None:
        """Bootstrap in either bot-profile/session creation order (PostgreSQL).

        The nullable column needs no historical data rewrite. Install the FK
        once both tables exist; NOT VALID avoids scanning existing rows during
        startup while still enforcing every subsequent write. Validation of
        pre-existing values belongs to an explicit schema upgrade, not reads.
        """
        if conn.dialect.name != "postgresql":
            return
        tables = conn.execute(text("""
            SELECT to_regclass('bot_profiles') AS bots,
                   to_regclass('sessions') AS sessions
        """)).mappings().one()
        if tables["bots"] is None:
            return
        conn.execute(text("""
            ALTER TABLE bot_profiles
            ADD COLUMN IF NOT EXISTS last_session_id VARCHAR(36)
        """))
        if tables["sessions"] is None:
            return
        # Both bootstrappers use this lock, so concurrent first construction
        # cannot race the pg_constraint check and add the same FK twice.
        conn.execute(text("SELECT pg_advisory_xact_lock(hashtext('bot-session-state-schema'))"))
        present = conn.execute(text("""
            SELECT 1 FROM pg_constraint
            WHERE conrelid = 'bot_profiles'::regclass
              AND conname = 'fk_bot_profiles_last_session'
        """)).scalar_one_or_none()
        if present is None:
            conn.execute(text("""
                ALTER TABLE bot_profiles
                ADD CONSTRAINT fk_bot_profiles_last_session
                FOREIGN KEY (last_session_id) REFERENCES sessions(id)
                ON DELETE SET NULL NOT VALID
            """))

    @staticmethod
    def resolve(conn, *, bot_id: str) -> dict | None:
        """Read the bot-wide last session, including explicitly resumed archives.

        Missing profiles and dangling/cross-bot/deleted pointers are errors;
        only a registered bot with a null pointer has no recorded session.
        """
        row = conn.execute(_POINTERS, {"bots": [bot_id]}).mappings().one_or_none()
        if row is None:
            raise ValueError(f"Unknown bot: {bot_id}")
        return BotSessionState._validated(row, bot_id)

    @staticmethod
    def resolve_many(conn, *, bot_ids) -> dict[str, dict | None]:
        """``resolve`` for many bots in one query (TASK-1400 readout).

        Same rules per bot: ``None`` means no recorded session. Unknown bots
        and invalid pointers are absent from the result, never guessed.
        """
        bots = list(dict.fromkeys(bot_ids))
        if not bots:
            return {}
        resolved: dict[str, dict | None] = {}
        for row in conn.execute(_POINTERS, {"bots": bots}).mappings().all():
            bot_id = row["slug"]
            try:
                resolved[bot_id] = BotSessionState._validated(row, bot_id)
            except ValueError:
                continue
        return resolved

    @staticmethod
    def _validated(row, bot_id: str) -> dict | None:
        if row["last_session_id"] is None:
            return None
        if row["id"] is None or row["bot_id"] != bot_id or row["status"] == "deleted":
            raise ValueError(f"Invalid last session for bot: {bot_id}")
        return {key: value for key, value in row.items() if key not in ("last_session_id", "slug")}

    @staticmethod
    def accept(conn, *, bot_id: str, session_id: str) -> None:
        """Stamp an accepted binding/first rotation under the admission lock.

        Never call on completion, history reads, queued reservations, or keyed
        rotation replay. User identity is deliberately not a selection fence:
        users and bots author messages in the bot's conversation.
        """
        result = conn.execute(text("""
            UPDATE bot_profiles SET last_session_id = :session
            WHERE slug = :bot AND EXISTS (
                SELECT 1 FROM sessions
                WHERE id = :session AND bot_id = :bot AND status <> 'deleted'
            )
        """), {"bot": bot_id, "session": session_id})
        if result.rowcount != 1:
            raise ValueError(f"Cannot accept session {session_id!r} for bot {bot_id!r}")
