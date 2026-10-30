"""Export the seed A data artifact from production, read-only (TASK-1802).

Runs on the operator host through ``pgq`` (the only sanctioned DB entrypoint;
no credentials on any command line). Fails closed before copying anything if
production holds a table, copied column or runtime-settings key the allowlist
doesn't classify. All copies run in one READ ONLY REPEATABLE READ transaction
so the artifact is a consistent snapshot and the export cannot write.

Output: one CSV per copied table plus ``manifest.json`` (row counts, sha256),
which ``llm_bawt.preview.schema_init`` verifies before loading.

    python -m llm_bawt.preview.seed_export --out DIR [--pgq PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from llm_bawt.preview.seed_allowlist import (
    COPY, classify, column_list, export_query, unclassified_bot_keys,
    unclassified_global_keys,
)

MANIFEST = "manifest.json"
SEED_FORMAT = 1

TABLES_JSON_SQL = """
SELECT coalesce(json_object_agg(relname, columns), '{}') FROM (
  SELECT c.relname, json_agg(a.attname ORDER BY a.attnum) AS columns
  FROM pg_class c
  JOIN pg_namespace n ON n.oid = c.relnamespace AND n.nspname = 'public'
  JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped
  WHERE c.relkind IN ('r', 'p') AND NOT c.relispartition
  GROUP BY c.relname) t
"""
KEYS_JSON_SQL = """
SELECT json_build_object(
  'global', coalesce((SELECT json_agg(DISTINCT key) FROM public.runtime_settings WHERE scope_type = 'global'), '[]'),
  'bot', coalesce((SELECT json_agg(DISTINCT key) FROM public.runtime_settings WHERE scope_type = 'bot'), '[]'))
"""


class SeedExportError(RuntimeError):
    pass


def _pgq(pgq: str, *args: str, stdin: str | None = None) -> str:
    result = subprocess.run(
        [pgq, "llm-bawt", "-X", "-v", "ON_ERROR_STOP=1", *args],
        input=stdin, capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise SeedExportError(f"pgq failed ({result.returncode}): {result.stderr.strip()}")
    return result.stdout


def preflight(tables: dict[str, list[str]], global_keys: list[str], bot_keys: list[str]) -> list[str]:
    """Every reason the export must not run; empty means safe."""
    problems = []
    # TABLES_JSON_SQL already excludes partitions by catalog flag; never filter
    # by name here, or a real table containing "_p_" would escape review.
    result = classify(tables)
    problems += [f"unclassified table: {t}" for t in result.unclassified_tables]
    for table, (missing, extra) in result.column_drift.items():
        problems += [f"unclassified column: {table}.{c}" for c in missing]
        problems += [f"allowlisted column missing in production: {table}.{c}" for c in extra]
    problems += [f"missing copied table: {t}" for t in COPY if t not in tables]
    problems += [f"unclassified global setting: {k}" for k in unclassified_global_keys(global_keys)]
    problems += [f"unclassified bot setting: {k}" for k in unclassified_bot_keys(bot_keys)]
    return problems


def copy_script(out: Path) -> str:
    lines = ["BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY;"]
    for table in COPY:
        target = (out / f"{table}.csv").as_posix().replace("'", "''")
        query = export_query(table).replace("\n", " ")
        lines.append(f"\\copy ({query}) TO '{target}' WITH (FORMAT csv, HEADER true)")
    lines.append("COMMIT;")
    return "\n".join(lines) + "\n"


def write_manifest(out: Path) -> dict:
    tables = {}
    for table in COPY:
        path = out / f"{table}.csv"
        data = path.read_bytes()
        tables[table] = {
            "file": path.name, "rows": _csv_rows(path), "columns": column_list(table),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    manifest = {
        "format": SEED_FORMAT,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "tables": tables,
    }
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _csv_rows(path: Path) -> int:
    import csv

    with path.open(newline="") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--pgq", default="pgq")
    args = parser.parse_args(argv)

    tables = json.loads(_pgq(args.pgq, "-tAc", TABLES_JSON_SQL))
    keys = json.loads(_pgq(args.pgq, "-tAc", KEYS_JSON_SQL))
    problems = preflight(tables, keys["global"], keys["bot"])
    if problems:
        print("seed export refused; classify these in llm_bawt/preview/seed_allowlist.py:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 2

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SeedExportError(f"{out} is not empty; export into a fresh directory")
    _pgq(args.pgq, "-q", "-f", "-", stdin=copy_script(out))
    manifest = write_manifest(out)
    for table, entry in manifest["tables"].items():
        print(f"{table}: {entry['rows']} rows  sha256={entry['sha256'][:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
