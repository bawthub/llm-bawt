#!/usr/bin/env python3
"""One-time backfill: populate bot_profiles.avatar_render for existing bots.

New bots resolve their avatar at write time. Existing rows predate the
column, so this resolves each bot's current ``avatar`` into a self-hosted
``data:`` URL and stores it. Safe to re-run (idempotent-ish: it re-resolves
and overwrites). Reads new code, triggers the column migration on store
init, and writes directly via SQL — does NOT restart the running app.

    # scripts/ is not in the app image, so pipe the file in:
    docker exec -i llm-bawt-app /app/.venv/bin/python - < scripts/backfill_avatar_render.py
    # add --force to re-resolve rows that already have a render:
    docker exec -i llm-bawt-app /app/.venv/bin/python - --force < scripts/backfill_avatar_render.py
"""
from __future__ import annotations

import sys

from sqlalchemy import text

from llm_bawt.media.avatar import resolve_avatar_render
from llm_bawt.runtime_settings import BotProfileStore
from llm_bawt.utils.config import config


def main() -> int:
    force = "--force" in sys.argv[1:]
    store = BotProfileStore(config)  # __init__ runs _migrate_add_columns()
    if store.engine is None:
        print("bot_profiles DB unavailable", file=sys.stderr)
        return 2

    rows = store.list_all()
    done = skipped = failed = 0
    for row in rows:
        current = getattr(row, "avatar_render", None)
        if current and not force:
            print(f"skip {row.slug} (already rendered)")
            skipped += 1
            continue
        render = resolve_avatar_render(row.avatar)
        if render is None:
            print(f"----  {row.slug}: no render for avatar={row.avatar!r} (native fallback)")
            failed += 1
            # Still write NULL so a --force pass is consistent; leave as-is.
            continue
        with store.engine.connect() as conn:
            conn.execute(
                text("UPDATE bot_profiles SET avatar_render = :r WHERE slug = :s"),
                {"r": render, "s": row.slug},
            )
            conn.commit()
        print(f"ok    {row.slug}: {row.avatar!r} -> {render[:32]}... ({len(render)}B)")
        done += 1

    print(f"\n{done} rendered, {skipped} skipped, {failed} unresolved (of {len(rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
