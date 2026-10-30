#!/usr/bin/env python3
"""Audited backfill of message authorship (TASK-1210, TASK-1807).

``--kind legacy`` (default) resolves NULL authorship on pre-authorship rows.
``--kind continuation`` re-authors approval-continuation turns stored as the
requesting user to their first-party system author.

Dry run (default) prints per-rule counts and writes nothing. ``--apply`` needs
``--expect N`` equal to the dry-run writable count; the batch rolls back on any
mismatch. ``--revert BATCH`` restores the previous authorship on rows that
batch set. Private bots' partitions (``private`` tag or the legacy
``include_in_global_search=false`` column) are never read or written; if that
set cannot be read the script aborts before touching anything.

scripts/ is not mounted in the app container, so pipe it in from the checkout:

    docker exec -i llm-bawt-app /app/.venv/bin/python - [--kind continuation] < scripts/backfill_message_authors.py
    docker exec -i llm-bawt-app /app/.venv/bin/python - [--kind continuation] --apply --expect N < scripts/backfill_message_authors.py
    docker exec -i llm-bawt-app /app/.venv/bin/python - --revert BATCH < scripts/backfill_message_authors.py
"""
from __future__ import annotations

import argparse

from sqlalchemy import create_engine

from llm_bawt.message_author_backfill import (
    ApprovalContinuationAuthorBackfill,
    MessageAuthorBackfill,
)
from llm_bawt.orphan_partitions import read_protected_slugs
from llm_bawt.utils import db
from llm_bawt.utils.config import config

KINDS = {"legacy": MessageAuthorBackfill, "continuation": ApprovalContinuationAuthorBackfill}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kind", choices=sorted(KINDS), default="legacy")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--revert", metavar="BATCH_ID")
    parser.add_argument("--expect", type=int)
    args = parser.parse_args()

    engine = create_engine(db.build_postgres_url(config))
    with engine.connect() as conn:
        protected = read_protected_slugs(conn)  # raises (fail closed) if unreadable
    backfill = KINDS[args.kind](engine, excluded_partitions=protected)
    if args.revert:
        # Revert restores each audit row's own old_* values, so it is kind-agnostic.
        print(f"restored {backfill.revert(args.revert)} rows from batch {args.revert}")
        return 0

    plan = backfill.plan()
    for rule, count in sorted(plan.counts.items()):
        print(f"{rule:16} {count:>7}  {'write' if rule in plan.written_rules else 'unchanged'}")
    print(f"{'writable':16} {plan.writable:>7}  (excluded private partitions: {len(backfill.excluded)})")
    if not args.apply:
        return 0
    if args.expect is None:
        parser.error("--apply requires --expect <writable count from the dry run>")
    batch_id, updated = backfill.apply(expected=args.expect)
    print(f"applied batch {batch_id}: {updated} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
