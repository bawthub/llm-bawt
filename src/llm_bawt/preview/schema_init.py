"""Create the preview database schema and load the seed A artifact (TASK-1802).

Runs inside the preview profile against the isolated preview database only.
It bootstraps every store-owned table, verifies the exporter's manifest
(format, columns, sha256, row counts) and loads the copied tables in one
transaction, then writes the single ``preview_seed_meta`` row the preview app
requires before it serves chat.

Fails closed, before writing anything, when:

* the connected database is not the one named on the command line;
* the database already holds runtime history (``messages``, ``memories``,
  ``sessions``); a seed target is always a fresh database;
* the database was already seeded (re-seed means a fresh volume);
* the artifact does not match its manifest or the current allowlist.

    python -m llm_bawt.preview.schema_init --seed DIR --database NAME
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import Engine

from llm_bawt.preview.seed_allowlist import COPY, SCHEMA_ONLY, column_list
from llm_bawt.preview.seed_export import MANIFEST, SEED_FORMAT

META_TABLE = "preview_seed_meta"
HISTORY_TABLES = ("messages", "memories", "sessions")

META_DDL = f"""
CREATE TABLE public.{META_TABLE} (
  id smallint PRIMARY KEY CHECK (id = 1),
  seed_format integer NOT NULL,
  manifest_sha256 text NOT NULL,
  exported_at timestamptz NOT NULL,
  -- Rows the schema bootstrap itself wrote into schema-only tables (e.g. the
  -- System entity profile), per table. The seed audit holds those tables to
  -- exactly this baseline: nothing from production may appear there.
  bootstrap_rows jsonb NOT NULL,
  loaded_at timestamptz NOT NULL DEFAULT now()
)
"""


def _bootstrap_rows(cur) -> dict[str, int]:
    rows: dict[str, int] = {}
    for table in sorted(SCHEMA_ONLY):
        cur.execute("SELECT to_regclass(%s)", (f"public.{table}",))
        if cur.fetchone()[0] is None:
            continue
        cur.execute(f"SELECT count(*) FROM public.{table}")
        count = cur.fetchone()[0]
        if count:
            rows[table] = count
    return rows


class SeedInitError(RuntimeError):
    pass


def verify_artifact(seed: Path) -> tuple[dict, str]:
    """Return ``(manifest, manifest_sha256)`` or raise on any mismatch."""
    raw = (seed / MANIFEST).read_bytes()
    manifest = json.loads(raw)
    if manifest.get("format") != SEED_FORMAT:
        raise SeedInitError(f"seed format {manifest.get('format')!r} != {SEED_FORMAT}")
    tables = manifest.get("tables", {})
    if set(tables) != set(COPY):
        raise SeedInitError(f"seed tables {sorted(tables)} do not match the allowlist {sorted(COPY)}")
    for table, entry in tables.items():
        if entry.get("columns") != column_list(table):
            raise SeedInitError(f"{table}: seed columns differ from the allowlist")
        path = seed / f"{table}.csv"
        if entry.get("file") != path.name or path.is_symlink() or not path.is_file():
            raise SeedInitError(f"{table}: seed file missing or linked")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry.get("sha256"):
            raise SeedInitError(f"{table}: sha256 mismatch")
    return manifest, hashlib.sha256(raw).hexdigest()


def _preflight(engine: Engine, database: str) -> None:
    with engine.connect() as conn:
        current = conn.execute(text("SELECT current_database()")).scalar_one()
        if current != database:
            raise SeedInitError(f"connected to {current!r}, expected {database!r}")
        if conn.execute(text("SELECT to_regclass(:t)"), {"t": f"public.{META_TABLE}"}).scalar():
            raise SeedInitError("database is already seeded; re-seed into a fresh database")
        for table in HISTORY_TABLES:
            if not conn.execute(text("SELECT to_regclass(:t)"), {"t": f"public.{table}"}).scalar():
                continue
            if conn.execute(text(f"SELECT EXISTS (SELECT 1 FROM public.{table})")).scalar():
                raise SeedInitError(f"{table} holds rows; refusing to seed a non-fresh database")


def load_seed(config, engine: Engine, seed: Path, database: str) -> dict[str, int]:
    """Bootstrap the schema on ``engine`` and load ``seed``; return rows per table."""
    from llm_bawt.schema_bootstrap import bootstrap_full_store

    manifest, manifest_sha = verify_artifact(seed)
    _preflight(engine, database)
    bootstrap_full_store(config, engine)
    _preflight(engine, database)  # bootstrap must not have produced history

    loaded: dict[str, int] = {}
    raw = engine.raw_connection()
    try:
        cur = raw.cursor()
        baseline = _bootstrap_rows(cur)
        # Bootstrap seeds some catalog rows; the artifact is authoritative.
        for table in reversed(COPY):
            cur.execute(f"DELETE FROM public.{table}")
        for table in COPY:
            with (seed / f"{table}.csv").open(newline="") as handle:
                cur.copy_expert(
                    f"COPY public.{table} ({column_list(table)}) FROM STDIN WITH (FORMAT csv, HEADER true)",
                    handle,
                )
            cur.execute(f"SELECT count(*) FROM public.{table}")
            loaded[table] = cur.fetchone()[0]
            expected = manifest["tables"][table]["rows"]
            if loaded[table] != expected:
                raise SeedInitError(f"{table}: loaded {loaded[table]} rows, manifest says {expected}")
        cur.execute(META_DDL)
        cur.execute(
            f"INSERT INTO public.{META_TABLE} (id, seed_format, manifest_sha256, exported_at, bootstrap_rows) "
            "VALUES (1, %s, %s, %s, %s)",
            (SEED_FORMAT, manifest_sha, manifest["exported_at"], json.dumps(baseline)),
        )
        raw.commit()
    except BaseException:
        raw.rollback()
        raise
    finally:
        raw.close()
    return loaded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", required=True, type=Path)
    parser.add_argument("--database", required=True)
    args = parser.parse_args(argv)

    secret_file = os.environ.get("LLM_BAWT_POSTGRES_PASSWORD_FILE")
    if secret_file:
        os.environ["LLM_BAWT_POSTGRES_PASSWORD"] = Path(secret_file).read_text(encoding="utf-8").strip()
    from llm_bawt.utils.config import Config
    from llm_bawt.utils.db import get_shared_engine

    config = Config(_env_file=None, MCP_SERVER_URL=None, SCHEDULER_ENABLED=False)
    if config.POSTGRES_DATABASE != args.database:
        raise SeedInitError(f"configured database {config.POSTGRES_DATABASE!r} != --database {args.database!r}")
    engine = get_shared_engine(config)
    if engine is None:
        raise SeedInitError("preview database unavailable")
    for table, rows in load_seed(config, engine, args.seed.resolve(), args.database).items():
        print(f"{table}: {rows} rows")
    print(f"{META_TABLE}: seeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
