#!/usr/bin/env python3
"""TASK-1149 read-only public-schema capture and deterministic offline comparison.

Captures never provision schema or execute DDL. Live access is through pgq ONLY;
test access requires an explicit *_test database and verifies the connection.
Use a NEW output path per capture/diff. See docs/SCHEMA_CONVERGENCE.md for
accepted exceptions, private reference bootstrap and migration gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

CATALOG_SQL = """
WITH records AS (
 SELECT 'relation' AS kind, c.relname::text AS key,
   jsonb_build_object('kind',c.relkind,'persistence',c.relpersistence,'partition_bound',CASE WHEN c.relispartition THEN pg_get_expr(c.relpartbound,c.oid) END,'partition_key',pg_get_partkeydef(c.oid),'parents',(SELECT jsonb_agg(p.relname ORDER BY p.relname) FROM pg_inherits h JOIN pg_class p ON p.oid=h.inhparent WHERE h.inhrelid=c.oid),'view',CASE WHEN c.relkind IN ('v','m') THEN pg_get_viewdef(c.oid) END) AS definition
 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname='public' AND c.relkind IN ('r','p','v','m','f')
 UNION ALL
 SELECT 'column',c.relname::text||'.'||a.attname::text,
   jsonb_build_object('type',format_type(a.atttypid,a.atttypmod),'not_null',a.attnotnull,'default',pg_get_expr(d.adbin,d.adrelid),'identity',a.attidentity,'generated',a.attgenerated,'position',a.attnum)
 FROM pg_attribute a JOIN pg_class c ON c.oid=a.attrelid JOIN pg_namespace n ON n.oid=c.relnamespace LEFT JOIN pg_attrdef d ON d.adrelid=a.attrelid AND d.adnum=a.attnum
 WHERE n.nspname='public' AND c.relkind IN ('r','p','v','m','f') AND a.attnum>0 AND NOT a.attisdropped
 UNION ALL
 SELECT 'index',i.relname::text,
   jsonb_build_object('table',t.relname,'definition',pg_get_indexdef(i.oid),'valid',ix.indisvalid,'ready',ix.indisready,'parents',(SELECT jsonb_agg(p.relname ORDER BY p.relname) FROM pg_inherits h JOIN pg_class p ON p.oid=h.inhparent WHERE h.inhrelid=i.oid))
 FROM pg_index ix JOIN pg_class i ON i.oid=ix.indexrelid JOIN pg_class t ON t.oid=ix.indrelid JOIN pg_namespace n ON n.oid=t.relnamespace WHERE n.nspname='public'
 UNION ALL
 SELECT 'constraint',t.relname::text||'.'||co.conname::text,
   jsonb_build_object('type',co.contype,'definition',pg_get_constraintdef(co.oid,true),'validated',co.convalidated,'deferrable',co.condeferrable,'deferred',co.condeferred)
 FROM pg_constraint co JOIN pg_class t ON t.oid=co.conrelid JOIN pg_namespace n ON n.oid=t.relnamespace WHERE n.nspname='public'
 UNION ALL
 SELECT 'sequence',s.relname::text,
   jsonb_build_object('type',format_type(q.seqtypid,NULL),'start',q.seqstart,'increment',q.seqincrement,'min',q.seqmin,'max',q.seqmax,'cache',q.seqcache,'cycle',q.seqcycle,'owned_by',(SELECT t.relname::text||'.'||a.attname::text FROM pg_depend dep JOIN pg_class t ON t.oid=dep.refobjid JOIN pg_attribute a ON a.attrelid=t.oid AND a.attnum=dep.refobjsubid WHERE dep.objid=s.oid AND dep.refclassid='pg_class'::regclass AND dep.deptype IN ('a','i') LIMIT 1))
 FROM pg_class s JOIN pg_namespace n ON n.oid=s.relnamespace JOIN pg_sequence q ON q.seqrelid=s.oid WHERE n.nspname='public'
 UNION ALL
 SELECT 'extension',e.extname::text,jsonb_build_object('version',e.extversion,'schema',n.nspname)
 FROM pg_extension e JOIN pg_namespace n ON n.oid=e.extnamespace
 UNION ALL
 SELECT 'function',p.proname::text||'('||pg_get_function_identity_arguments(p.oid)||')', jsonb_build_object('definition',pg_get_functiondef(p.oid))
 FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace WHERE n.nspname='public' AND p.prokind IN ('f','p') AND NOT EXISTS (SELECT 1 FROM pg_depend d WHERE d.classid='pg_proc'::regclass AND d.objid=p.oid AND d.refclassid='pg_extension'::regclass AND d.deptype='e')
 UNION ALL
 SELECT 'trigger',t.relname::text||'.'||tr.tgname::text,jsonb_build_object('definition',pg_get_triggerdef(tr.oid,true),'enabled',tr.tgenabled)
 FROM pg_trigger tr JOIN pg_class t ON t.oid=tr.tgrelid JOIN pg_namespace n ON n.oid=t.relnamespace WHERE n.nspname='public' AND NOT tr.tgisinternal
 UNION ALL
 SELECT 'enum',ty.typname::text,jsonb_build_object('labels',jsonb_agg(en.enumlabel ORDER BY en.enumsortorder))
 FROM pg_type ty JOIN pg_namespace n ON n.oid=ty.typnamespace JOIN pg_enum en ON en.enumtypid=ty.oid WHERE n.nspname='public' GROUP BY ty.typname
 UNION ALL
 SELECT 'owner_acl',c.relname::text,jsonb_build_object('owner',pg_get_userbyid(c.relowner),'acl',c.relacl::text)
 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname='public' AND c.relkind IN ('r','p','S','i')
 UNION ALL
 SELECT 'dependency',c.relname::text||'.'||d.objsubid::text||'.'||d.refclassid::regclass::text||'.'||pg_describe_object(d.refclassid,d.refobjid,d.refobjsubid)||'.'||d.deptype::text,
   jsonb_build_object('ref_class',d.refclassid::regclass::text,'ref_object',pg_describe_object(d.refclassid,d.refobjid,d.refobjsubid),'type',d.deptype,'count',count(*))
 FROM pg_depend d JOIN pg_class c ON c.oid=d.objid JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname='public' AND d.classid='pg_class'::regclass AND d.deptype IN ('n','a','i')
 GROUP BY c.relname,d.objsubid,d.refclassid,d.refobjid,d.refobjsubid,d.deptype
)
SELECT jsonb_build_object('kind',kind,'key',key,'definition',definition)::text FROM records ORDER BY kind,key
"""

# Whitelist only semantic equivalences established for this task, not arbitrary
# PostgreSQL type rewrites. Column order never carries behavior.
TEXT_EQUIVALENT = {"entity_profiles.summary"}


def parse_catalog(lines: list[str]) -> dict[tuple[str, str], dict]:
    records: dict[tuple[str, str], dict] = {}
    for line in lines:
        row = json.loads(line)
        identity = (row["kind"], row["key"])
        if identity in records:
            raise ValueError(f"duplicate catalog key: {identity}")
        records[identity] = row["definition"]
    if not records:
        raise ValueError("empty catalog snapshot")
    return records


def compare(reference: dict, operational: dict) -> list[dict]:
    differences = []
    for kind, key in sorted(reference.keys() | operational.keys()):
        expected, actual = reference.get((kind, key)), operational.get((kind, key))
        if kind == "column" and expected is not None and actual is not None:
            expected, actual = expected.copy(), actual.copy()
            expected.pop("position", None)
            actual.pop("position", None)
            if key in TEXT_EQUIVALENT:
                for value in (expected, actual):
                    if value["type"] in ("text", "character varying"):
                        value["type"] = "text"
        if expected == actual:
            continue
        differences.append({"kind": kind, "key": key,
                            "state": "reference_only" if actual is None else
                                     "operational_only" if expected is None else "definition_mismatch",
                            "reference": expected, "operational": actual})
    return differences


def _run(command: list[str], *, env: dict | None = None) -> list[str]:
    result = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
    return result.stdout.splitlines()


def capture_live() -> tuple[list[str], dict]:
    env = dict(os.environ, PGOPTIONS="-c default_transaction_read_only=on -c statement_timeout=120000 -c lock_timeout=1000")
    def pgq(sql: str) -> list[str]:
        return _run(["pgq", "llm-bawt", "-X", "-A", "-t", "-v", "ON_ERROR_STOP=1", "-c", sql], env=env)
    # The catalog query is one SQL statement (one MVCC snapshot). Both
    # statements use the pgq target, whose route is asserted before capture.
    identity = pgq("SELECT current_database(), current_user, current_setting('server_version_num')")
    if len(identity) != 1 or not identity[0].startswith("askllm|askllm|"):
        raise RuntimeError("pgq did not select the expected operational database/user")
    lines = pgq(CATALOG_SQL)
    parse_catalog(lines)
    return lines, {"identity": identity[0], "mode": "operational", "read_only": True}


def capture_test(expected_db: str) -> tuple[list[str], dict]:
    if not expected_db.endswith("_test") or os.getenv("LLM_BAWT_POSTGRES_DATABASE") != expected_db:
        raise RuntimeError("test capture requires a matching explicit *_test database")
    from sqlalchemy import text
    from llm_bawt.utils.config import Config
    from llm_bawt.utils.db import build_engine_from_config

    config = Config()
    if config.POSTGRES_DATABASE != expected_db:
        raise RuntimeError("test database config mismatch")
    engine = build_engine_from_config(config, application_name="schema-catalog-read")
    if engine is None or engine.url.database != expected_db:
        raise RuntimeError("test connection unavailable or misrouted")
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            db_name, version = conn.execute(text("SELECT current_database(), current_setting('server_version_num')")).one()
            if db_name != expected_db:
                raise RuntimeError("connection selected a different database")
            lines = [row[0] for row in conn.execute(text(CATALOG_SQL))]
        parse_catalog(lines)
        return lines, {"identity": f"{db_name}|{version}", "mode": "test", "read_only": True}
    finally:
        engine.dispose()


def write_snapshot(path: Path, lines: list[str], metadata: dict) -> None:
    # This is provenance, not a claim that the source is committed. Hash the
    # actual source bytes so an uncommitted bootstrap is still identifiable.
    root = Path(__file__).resolve().parents[1]
    source_digest = hashlib.sha256()
    for directory in (root / "src/llm_bawt", root / "src/agent_bridge"):
        for source in sorted(directory.rglob("*.py")):
            source_digest.update(str(source.relative_to(root)).encode() + b"\0")
            source_digest.update(source.read_bytes())
    metadata["source_sha256"] = source_digest.hexdigest()
    metadata["catalog_tool_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata["source_revision"] = _run(["git", "-C", str(root), "rev-parse", "HEAD"])[0]
    metadata["source_dirty"] = bool(_run(["git", "-C", str(root), "status", "--porcelain"]))
    # Exclusive creation; a bot cannot silently replace earlier evidence.
    with path.open("x", encoding="utf-8") as output:
        output.write("\n".join(lines) + "\n")
    metadata["catalog_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata["record_count"] = len(lines)
    with _meta_path(path).open("x", encoding="utf-8") as output:
        output.write(json.dumps(metadata, sort_keys=True, indent=2) + "\n")


def _meta_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("capture", help="capture metadata without schema provisioning")
    capture.add_argument("--mode", choices=("operational", "test"), required=True)
    capture.add_argument("--expected-db", help="mandatory *_test database for test mode")
    capture.add_argument("--out", type=Path, required=True)
    diff = sub.add_parser("diff", help="compare existing JSONL snapshots offline")
    diff.add_argument("--reference", type=Path, required=True)
    diff.add_argument("--operational", type=Path, required=True)
    diff.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or _meta_path(args.out).exists():
        raise FileExistsError("refusing to overwrite catalog evidence")
    if args.command == "capture":
        if args.mode == "operational":
            if args.expected_db:
                parser.error("--expected-db applies only to test capture")
            lines, metadata = capture_live()
        else:
            if not args.expected_db:
                parser.error("--expected-db is required for test capture")
            lines, metadata = capture_test(args.expected_db)
        write_snapshot(args.out, lines, metadata)
        print(f"{len(lines)} read-only records -> {args.out}")
    else:
        reference = parse_catalog(args.reference.read_text(encoding="utf-8").splitlines())
        operational = parse_catalog(args.operational.read_text(encoding="utf-8").splitlines())
        differences = compare(reference, operational)
        with args.out.open("x", encoding="utf-8") as output:
            for row in differences:
                output.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"{len(differences)} semantic differences -> {args.out}")


if __name__ == "__main__":
    main()
