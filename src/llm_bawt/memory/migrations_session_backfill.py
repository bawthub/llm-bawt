"""TASK-284: backfill NULL ``messages.session_id`` onto a legacy thread per bot.

After the session-lifecycle cutover (Increment B) every NEW message is stamped
with an active ``session_id``, but ~24k historical rows written before that
predate session tracking and carry ``session_id IS NULL``. They cannot be
split by user — the partitioned ``messages`` table has **no ``user_id``
column** (only ``bot_id`` + ``timestamp``) — so they bucket per bot.

Strategy: create ONE ``status='archived'`` legacy session per bot
(``id = 'legacy-<bot>'``, owned by ``DEFAULT_USER``, metadata-flagged) and
point every NULL-session message for that bot at it.

- **Completed**, so it can never collide with the one-active-per-(bot,user)
  partial unique index (``idx_sessions_one_active_per_bot_user``).
- **Historical**, so it can't affect the live read path (chat reads via
  ``conversation_offset``; agents get no llm-bawt history). Pure data hygiene
  that makes ``messages.session_id`` FK-validatable (TASK-630).
- **Idempotent**: legacy session INSERT is ``ON CONFLICT DO NOTHING``; the
  backfill UPDATE only touches ``session_id IS NULL`` rows, so re-runs no-op.

Modes (mutually exclusive):

``--dry-run``
    Reports per-bot NULL-session counts and the legacy session ids that would
    be created, then runs the FULL create+backfill inside ONE transaction and
    ROLLS IT BACK — asserting the update count equals the pre-count and that
    no already-linked row is touched. Zero persistent impact.

``--execute`` (requires ``--yes``)
    Per bot, in its own transaction: upsert the legacy session, UPDATE the
    bot's NULL-session messages onto it, record count parity, and commit.
    A count mismatch makes the final report unsuccessful but does not roll
    back committed bots. It only writes ``session_id`` on rows that had none
    and never rewrites a valid link.

``--verify``
    Reports remaining NULL-session messages (target: 0) and FK-readiness:
    every ``messages.session_id`` must resolve to a ``sessions.id`` (no
    dangling pointers), which is the precondition for TASK-630's FK VALIDATE.

Usage::

    uv run python -m llm_bawt.memory.migrations_session_backfill --dry-run
    uv run python -m llm_bawt.memory.migrations_session_backfill --execute --yes
    uv run python -m llm_bawt.memory.migrations_session_backfill --verify
"""

import argparse
import json
import logging
import sys
import uuid
from typing import Any

from sqlalchemy import text

logger = logging.getLogger(__name__)

LEGACY_METADATA = '{"legacy_backfill": true, "task": "TASK-284"}'


def _legacy_id(bot_id: str) -> str:
    """Deterministic legacy session id (<= 36 chars for varchar(36))."""
    sid = f"legacy-{bot_id}"
    if len(sid) > 36:
        sid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"llm-bawt:legacy-session:{bot_id}"))
    return sid


def _null_session_counts(conn) -> dict[str, int]:
    rows = conn.execute(
        text(
            "SELECT bot_id, count(*) AS c FROM messages "
            "WHERE session_id IS NULL GROUP BY bot_id ORDER BY c DESC"
        )
    ).fetchall()
    return {r.bot_id: int(r.c) for r in rows}


def _upsert_legacy_session(conn, bot_id: str, user_id: str) -> str:
    """Create (idempotently) the completed legacy session for a bot; return id.

    started_at/ended_at span the bot's NULL-session message timestamps so the
    thread's time range reflects the real history it absorbs.
    """
    sid = _legacy_id(bot_id)
    conn.execute(
        text(
            """
            INSERT INTO sessions
                (id, bot_id, user_id, started_at, ended_at, status, session_metadata)
            SELECT
                :sid, :bot_id, :user_id,
                COALESCE(to_timestamp(MIN(m.timestamp)) AT TIME ZONE 'UTC', CURRENT_TIMESTAMP),
                COALESCE(to_timestamp(MAX(m.timestamp)) AT TIME ZONE 'UTC', CURRENT_TIMESTAMP),
                'archived', CAST(:meta AS jsonb)
            FROM messages m
            WHERE m.bot_id = :bot_id AND m.session_id IS NULL
            ON CONFLICT (id) DO NOTHING
            """
        ),
        {"sid": sid, "bot_id": bot_id, "user_id": user_id, "meta": LEGACY_METADATA},
    )
    existing = conn.execute(text("""
        SELECT bot_id, user_id, status, session_metadata
        FROM sessions WHERE id = :sid FOR UPDATE
    """), {"sid": sid}).one()
    metadata = existing.session_metadata or {}
    if (existing.bot_id != bot_id or existing.user_id != user_id
            or existing.status != "archived" or metadata.get("legacy_backfill") is not True):
        raise ValueError(f"Legacy session {sid} does not match the requested owner and legacy state")
    return sid


def _backfill_bot(conn, bot_id: str, sid: str) -> int:
    """Point the bot's NULL-session messages at the legacy session; return count."""
    result = conn.execute(
        text(
            "UPDATE messages SET session_id = :sid "
            "WHERE bot_id = :bot_id AND session_id IS NULL"
        ),
        {"sid": sid, "bot_id": bot_id},
    )
    return int(result.rowcount or 0)


def run_dry_run(engine, user_id: str) -> dict[str, Any]:
    with engine.connect() as conn:
        pre = _null_session_counts(conn)
        total_pre = sum(pre.values())
        already_linked = int(
            conn.execute(
                text("SELECT count(*) FROM messages WHERE session_id IS NOT NULL")
            ).scalar()
            or 0
        )

        # Simulate the full thing, then roll back. connect() autobegins the
        # transaction (SQLAlchemy 2.0), so we don't call begin() explicitly —
        # the earlier reads + these writes share one txn that we discard.
        per_bot: dict[str, dict[str, Any]] = {}
        try:
            for bot_id, cnt in pre.items():
                sid = _upsert_legacy_session(conn, bot_id, user_id)
                updated = _backfill_bot(conn, bot_id, sid)
                per_bot[bot_id] = {
                    "legacy_session": sid,
                    "null_before": cnt,
                    "updated": updated,
                    "parity": updated == cnt,
                }
            remaining = int(
                conn.execute(
                    text("SELECT count(*) FROM messages WHERE session_id IS NULL")
                ).scalar()
                or 0
            )
            # Prove no previously-linked row changed: linked count must be
            # exactly pre-linked + total backfilled.
            linked_after = int(
                conn.execute(
                    text("SELECT count(*) FROM messages WHERE session_id IS NOT NULL")
                ).scalar()
                or 0
            )
        finally:
            conn.rollback()

        all_parity = all(b["parity"] for b in per_bot.values())
        no_untouched = linked_after == already_linked + total_pre
        return {
            "ok": all_parity and remaining == 0 and no_untouched,
            "mode": "dry-run (rolled back)",
            "user_id": user_id,
            "total_null_before": total_pre,
            "already_linked": already_linked,
            "simulated_remaining_null": remaining,
            "linked_after_equals_pre_plus_backfill": no_untouched,
            "per_bot": per_bot,
        }


def run_execute(engine, user_id: str) -> dict[str, Any]:
    with engine.connect() as conn:
        pre = _null_session_counts(conn)
    per_bot: dict[str, dict[str, Any]] = {}
    ok = True
    for bot_id, cnt in pre.items():
        with engine.begin() as conn:
            sid = _upsert_legacy_session(conn, bot_id, user_id)
            updated = _backfill_bot(conn, bot_id, sid)
            parity = updated == cnt
            ok = ok and parity
            per_bot[bot_id] = {
                "legacy_session": sid,
                "null_before": cnt,
                "updated": updated,
                "parity": parity,
            }
            logger.info(
                "backfill %s: %d messages -> %s (parity=%s)",
                bot_id, updated, sid, parity,
            )
    with engine.connect() as conn:
        remaining = int(
            conn.execute(
                text("SELECT count(*) FROM messages WHERE session_id IS NULL")
            ).scalar()
            or 0
        )
    return {
        "ok": ok and remaining == 0,
        "mode": "execute",
        "user_id": user_id,
        "remaining_null": remaining,
        "per_bot": per_bot,
    }


def run_verify(engine) -> dict[str, Any]:
    with engine.connect() as conn:
        remaining = int(
            conn.execute(
                text("SELECT count(*) FROM messages WHERE session_id IS NULL")
            ).scalar()
            or 0
        )
        # FK-readiness: any message.session_id with no matching sessions.id?
        dangling = int(
            conn.execute(
                text(
                    "SELECT count(*) FROM messages m "
                    "WHERE m.session_id IS NOT NULL "
                    "AND NOT EXISTS (SELECT 1 FROM sessions s WHERE s.id = m.session_id)"
                )
            ).scalar()
            or 0
        )
        legacy = conn.execute(
            text(
                "SELECT id, bot_id, user_id, status FROM sessions "
                "WHERE session_metadata->>'legacy_backfill' = 'true' ORDER BY id"
            )
        ).fetchall()
    return {
        "ok": remaining == 0 and dangling == 0,
        "mode": "verify",
        "remaining_null_session_messages": remaining,
        "dangling_session_pointers": dangling,
        "fk_ready": dangling == 0,
        "legacy_sessions": [
            {"id": r.id, "bot_id": r.bot_id, "user_id": r.user_id, "status": r.status}
            for r in legacy
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TASK-284 NULL messages.session_id backfill onto per-bot legacy threads."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--yes", action="store_true", help="Confirm a mutating --execute run."
    )
    parser.add_argument(
        "--user", default=None, help="Legacy session owner (default: config DEFAULT_USER)."
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    from ..utils.config import Config
    from ..utils.db import get_shared_engine

    config = Config()
    engine = get_shared_engine(config, application_name="llm-bawt-session-backfill")
    if engine is None:
        print("ERROR: Postgres credentials not configured", file=sys.stderr)
        return 2

    user_id = (args.user or getattr(config, "DEFAULT_USER", "") or "").strip()
    if not user_id and not args.verify:
        print("ERROR: no --user and no DEFAULT_USER configured", file=sys.stderr)
        return 2

    if args.execute and not args.yes:
        print("Refusing to --execute without --yes.", file=sys.stderr)
        return 2

    if args.dry_run:
        result = run_dry_run(engine, user_id)
    elif args.execute:
        result = run_execute(engine, user_id)
    else:
        result = run_verify(engine)

    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
