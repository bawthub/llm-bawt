#!/usr/bin/env bash
# Regenerate the Alembic baseline schema reference (TASK-631).
# Connection comes from libpq env (PGHOST/PGPORT/PGUSER/PGDATABASE + PGPASSFILE);
# never pass credentials on the command line. The target is replaced only after
# pg_dump and normalization both succeed; on any failure it is left untouched.
set -euo pipefail

target="${1:?usage: alembic_schema_snapshot.sh <output.sql>}"
PG_DUMP="${PG_DUMP:-pg_dump}"
tmp_dump="$(mktemp "${target}.dump.XXXXXX")"
tmp_out="$(mktemp "${target}.tmp.XXXXXX")"
trap 'rm -f "$tmp_dump" "$tmp_out"' EXIT

"$PG_DUMP" --schema-only --no-owner --no-privileges --no-comments \
  -T 'public.messages_p_*' -T 'public.memories_p_*' -T 'public.forgotten_messages_p_*' \
  -T public.task2061_auto_title_backup >"$tmp_dump"

grep -q '^CREATE TABLE' "$tmp_dump" || { echo "alembic_schema_snapshot: dump has no tables; refusing" >&2; exit 1; }

{
  echo "-- llm-bawt baseline schema reference (TASK-631, revision 0001_baseline, master 9dfbd3c9)."
  echo "-- Documentation only; not a restore recipe. Regenerate with: make migrate-schema-snapshot"
  sed -E '/^\\(un)?restrict /d; /^-- Dumped (from|by) /d; /^SET transaction_timeout /d' "$tmp_dump"
} >"$tmp_out"

mv -f "$tmp_out" "$target"
