"""TASK-284 step 13: backfill summary provenance (``source_session_ids``).

New summaries stamp provenance at insert time (``summarize_session``
via ``resolve_source_session_ids``), but ~3.2k historical
``role='summary'`` rows predate that. Worse, the step-17 NULL-session backfill
had no role filter, so it blanket-stamped summary rows onto ``legacy-<bot>``
threads regardless of where their SOURCE messages actually live.

This migration recomputes provenance from each summary's own
``summary_metadata.message_ids``:

- ``summary_metadata.source_session_ids`` := sorted DISTINCT non-NULL
  ``session_id`` of the source rows (``[]`` when indeterminable — sources
  deleted or metadata absent). The key's PRESENCE marks the row processed,
  making re-runs no-ops (idempotent).
- ``session_id`` column := the single source thread when provenance resolves
  to exactly one; otherwise the existing stamp is left alone (never nulled —
  FK cleanliness for TASK-630 is preserved either way).

Raw context reads always filter ``role <> 'summary'``, so re-stamping the
column can never leak a summary into a scoped raw transcript (verified in
``storage._get_messages_raw``).

Modes (mutually exclusive)::

    uv run python -m llm_bawt.memory.migrations_summary_provenance --dry-run
    uv run python -m llm_bawt.memory.migrations_summary_provenance --execute --yes
    uv run python -m llm_bawt.memory.migrations_summary_provenance --verify

``--dry-run`` runs the full UPDATE in one transaction and rolls it back,
reporting per-bot counts + how many rows would change ``session_id``.
"""

import argparse
import json
import logging
import sys
from typing import Any

from sqlalchemy import text

logger = logging.getLogger(__name__)

# Recompute source_session_ids from each unprocessed summary's message_ids,
# stamp it into summary_metadata, and correct the session_id column when the
# provenance is a single thread. Presence of the key = processed (idempotent).
_BACKFILL_SQL = text(
    """
    WITH src AS (
        SELECT
            s.id AS summary_id,
            s.bot_id AS bot_id,
            s.session_id AS old_session_id,
            COALESCE(
                (
                    SELECT array_agg(DISTINCT m.session_id)
                    FROM messages m
                    WHERE m.bot_id = s.bot_id
                      AND m.role <> 'summary'
                      AND m.session_id IS NOT NULL
                      AND m.id IN (
                          SELECT jsonb_array_elements_text(
                              s.summary_metadata -> 'message_ids')
                      )
                ),
                '{}'::varchar[]
            ) AS sids
        FROM messages s
        WHERE s.role = 'summary'
          AND NOT (COALESCE(s.summary_metadata, '{}'::jsonb) ? 'source_session_ids')
    )
    UPDATE messages s
    SET summary_metadata = COALESCE(s.summary_metadata, '{}'::jsonb)
            || jsonb_build_object('source_session_ids', to_jsonb(src.sids)),
        session_id = CASE
            WHEN array_length(src.sids, 1) = 1 THEN src.sids[1]
            ELSE s.session_id
        END
    FROM src
    WHERE s.id = src.summary_id
      AND s.bot_id = src.bot_id
      AND s.role = 'summary'
    RETURNING s.bot_id,
              (src.old_session_id IS DISTINCT FROM s.session_id) AS restamped
    """
)


def _pending_counts(conn) -> dict[str, int]:
    rows = conn.execute(
        text(
            "SELECT bot_id, count(*) AS c FROM messages "
            "WHERE role = 'summary' "
            "AND NOT (COALESCE(summary_metadata, '{}'::jsonb) ? 'source_session_ids') "
            "GROUP BY bot_id ORDER BY c DESC"
        )
    ).fetchall()
    return {r.bot_id: int(r.c) for r in rows}


def _run_update(conn) -> dict[str, Any]:
    rows = conn.execute(_BACKFILL_SQL).fetchall()
    per_bot: dict[str, dict[str, int]] = {}
    for r in rows:
        b = per_bot.setdefault(r.bot_id, {"updated": 0, "restamped_session_id": 0})
        b["updated"] += 1
        if r.restamped:
            b["restamped_session_id"] += 1
    return {
        "total_updated": len(rows),
        "total_restamped": sum(b["restamped_session_id"] for b in per_bot.values()),
        "per_bot": per_bot,
    }


def run_dry_run(engine) -> dict[str, Any]:
    with engine.connect() as conn:
        pre = _pending_counts(conn)
        try:
            stats = _run_update(conn)
        finally:
            conn.rollback()
    total_pre = sum(pre.values())
    return {
        "ok": stats["total_updated"] == total_pre,
        "mode": "dry-run (rolled back)",
        "pending_before": pre,
        "parity": stats["total_updated"] == total_pre,
        **stats,
    }


def run_execute(engine) -> dict[str, Any]:
    with engine.connect() as conn:
        pre = _pending_counts(conn)
        stats = _run_update(conn)
        conn.commit()
        remaining = _pending_counts(conn)
    total_pre = sum(pre.values())
    return {
        "ok": stats["total_updated"] == total_pre and not remaining,
        "mode": "execute",
        "pending_before": pre,
        "remaining_after": remaining,
        **stats,
    }


def run_verify(engine) -> dict[str, Any]:
    with engine.connect() as conn:
        remaining = _pending_counts(conn)
        dist = conn.execute(
            text(
                """
                SELECT
                    count(*) AS total,
                    count(*) FILTER (WHERE jsonb_array_length(
                        summary_metadata -> 'source_session_ids') = 1) AS single_source,
                    count(*) FILTER (WHERE jsonb_array_length(
                        summary_metadata -> 'source_session_ids') > 1) AS multi_source,
                    count(*) FILTER (WHERE jsonb_array_length(
                        summary_metadata -> 'source_session_ids') = 0) AS indeterminable,
                    count(*) FILTER (WHERE session_id IS NOT NULL
                        AND NOT EXISTS (SELECT 1 FROM sessions x
                                        WHERE x.id = messages.session_id)) AS dangling
                FROM messages WHERE role = 'summary'
                """
            )
        ).one()
    return {
        "ok": not remaining and int(dist.dangling) == 0,
        "mode": "verify",
        "pending_without_provenance": remaining,
        "total_summaries": int(dist.total),
        "single_source": int(dist.single_source),
        "multi_source": int(dist.multi_source),
        "indeterminable": int(dist.indeterminable),
        "dangling_session_pointers": int(dist.dangling),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TASK-284 step 13: backfill summary source_session_ids provenance."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Confirm a mutating --execute run.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    from ..utils.config import Config
    from ..utils.db import get_shared_engine

    config = Config()
    engine = get_shared_engine(config, application_name="llm-bawt-summary-provenance")
    if engine is None:
        print("ERROR: Postgres credentials not configured", file=sys.stderr)
        return 2

    if args.execute and not args.yes:
        print("Refusing to --execute without --yes.", file=sys.stderr)
        return 2

    if args.dry_run:
        result = run_dry_run(engine)
    elif args.execute:
        result = run_execute(engine)
    else:
        result = run_verify(engine)

    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
