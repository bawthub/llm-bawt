#!/usr/bin/env python3
"""Audited backfill of NULL message authorship (TASK-1210).

Dry run (default) prints per-rule counts and writes nothing. ``--apply`` needs
``--expect N`` equal to the dry-run writable count; the batch rolls back on any
mismatch. ``--revert BATCH`` restores NULL on rows that batch set. Protected
partitions (Mira) are never read or written.

scripts/ is not mounted in the app container, so pipe it in from the checkout:

    docker exec -i llm-bawt-app /app/.venv/bin/python - < scripts/backfill_message_authors.py
    docker exec -i llm-bawt-app /app/.venv/bin/python - --apply --expect N < scripts/backfill_message_authors.py
    docker exec -i llm-bawt-app /app/.venv/bin/python - --revert BATCH < scripts/backfill_message_authors.py
"""
from __future__ import annotations

import argparse

from sqlalchemy import create_engine

from llm_bawt.message_author_backfill import WRITTEN_RULES, MessageAuthorBackfill
from llm_bawt.service.jobs.idle_context_candidates import PROTECTED_BOT_IDS
from llm_bawt.utils import db
from llm_bawt.utils.config import config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--revert", metavar="BATCH_ID")
    parser.add_argument("--expect", type=int)
    args = parser.parse_args()

    engine = create_engine(db.build_postgres_url(config))
    backfill = MessageAuthorBackfill(engine, excluded_partitions=PROTECTED_BOT_IDS)
    if args.revert:
        print(f"restored {backfill.revert(args.revert)} rows from batch {args.revert}")
        return 0

    plan = backfill.plan()
    for rule, count in sorted(plan.counts.items()):
        print(f"{rule:14} {count:>7}  {'write' if rule in WRITTEN_RULES else 'stays NULL'}")
    print(f"{'writable':14} {plan.writable:>7}  (excluded partitions: {sorted(PROTECTED_BOT_IDS)})")
    if not args.apply:
        return 0
    if args.expect is None:
        parser.error("--apply requires --expect <writable count from the dry run>")
    batch_id, updated = backfill.apply(expected=args.expect)
    print(f"applied batch {batch_id}: {updated} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
