#!/usr/bin/env python3
"""Online upgrade to stored ``content_tsv`` message search (TASK-1323).

Phases, in order:

* ``status`` (default): read-only catalog check; writes nothing.
* ``upgrade``: column, trigger, batched backfill, concurrent per-leaf GIN
  index attached to the parent. Lock-bounded, resumable, safe to rerun.
  Covers every bot's partition; the vector is only read inside SQL for
  matching and ranking, never returned to a caller.
* ``drop-legacy``: drop the old expression index, found by structure. Only
  after the app restart that loads the ``content_tsv`` queries.

scripts/ is not mounted in the app container, so pipe it in from the checkout:

    docker exec -i llm-bawt-app /app/.venv/bin/python - < scripts/upgrade_message_tsv.py
    docker exec -i llm-bawt-app /app/.venv/bin/python - --phase upgrade < scripts/upgrade_message_tsv.py
    docker exec -i llm-bawt-app /app/.venv/bin/python - --phase drop-legacy < scripts/upgrade_message_tsv.py
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import asdict

from sqlalchemy import create_engine

from llm_bawt.memory.message_tsv import (
    drop_legacy_fts_index,
    message_tsv_state,
    upgrade_message_tsv,
)
from llm_bawt.utils import db
from llm_bawt.utils.config import config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase", choices=("status", "upgrade", "drop-legacy"), default="status")
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    engine = create_engine(db.build_postgres_url(config))
    try:
        if args.phase == "upgrade":
            report = upgrade_message_tsv(engine, batch_size=args.batch_size)
            for key, value in asdict(report).items():
                print(f"{key}: {value}")
        elif args.phase == "drop-legacy":
            print(f"legacy index dropped: {drop_legacy_fts_index(engine)}")
        with engine.connect() as conn:
            state = message_tsv_state(conn)
        print(f"complete: {state.complete}  missing: {state.missing() or 'nothing'}  "
              f"index: {state.index_name}  legacy indexes: {list(state.legacy_indexes) or 'none'}")
        return 0 if state.complete or args.phase == "status" else 1
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
