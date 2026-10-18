"""TASK-1149: bounded shared-schema SQL rehearsal, strictly *_test only.

This is an executable *disposable rehearsal*, not an operational migration
entrypoint. It deliberately rejects askllm. Snark owns separately reviewed
operational execution, including backup, availability and authorization gates.
No runtime partitions, vector upgrade, application rows or live services are used.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine


MISSING_INDEXES = (
    ("ix_bot_profiles_endpoint_id", "bot_profiles", "endpoint_id"),
    ("ix_bot_profiles_prompt_override_id", "bot_profiles", "prompt_override_id"),
    ("ix_entity_profiles_email", "entity_profiles", "email"),
    ("ix_tool_call_records_tool_use_id", "tool_call_records", "tool_use_id"),
    ("ix_turn_logs_ended_at", "turn_logs", "ended_at"),
)
CONSTRAINT_RENAMES = (
    ("agent_runs", "openclaw_runs_pkey", "agent_runs_pkey", "PRIMARY KEY (run_id)"),
    ("agent_events", "openclaw_events_pkey", "agent_events_pkey", "PRIMARY KEY (id)"),
    ("agent_events", "openclaw_events_event_dedupe_key_key",
     "agent_events_event_dedupe_key_key", "UNIQUE (event_dedupe_key)"),
    ("agent_session_state", "openclaw_session_state_pkey", "agent_session_state_pkey",
     "PRIMARY KEY (session_key)"),
)
WIDTHS = (("tool_use_id", 128), ("parent_tool_use_id", 128),
          ("approval_request_id", 128), ("approval_status", 24))
FLAGS = (("tts_mode", "false"), ("include_summaries", "true"),
         ("include_in_global_search", "true"))
OLD_DISPATCH = "('QUEUED', 'DISPATCHING')"
NEW_DISPATCH = "('QUEUED', 'STEERING', 'DISPATCHING')"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"schema rehearsal preflight: {message}")


def _column(conn, table: str, column: str):
    return conn.execute(text("""
        SELECT format_type(a.atttypid,a.atttypmod), a.attnotnull,
               pg_get_expr(d.adbin,d.adrelid)
        FROM pg_attribute a JOIN pg_class c ON c.oid=a.attrelid
        JOIN pg_namespace n ON n.oid=c.relnamespace
        LEFT JOIN pg_attrdef d ON d.adrelid=a.attrelid AND d.adnum=a.attnum
        WHERE n.nspname='public' AND c.relname=:table AND a.attname=:column
          AND a.attnum>0 AND NOT a.attisdropped AND c.relkind='r'
    """), {"table": table, "column": column}).one()


def _index(conn, name: str):
    return conn.execute(text("""
        SELECT t.relname, pg_get_indexdef(i.oid), x.indisvalid, x.indisready,
               pg_get_userbyid(i.relowner)
        FROM pg_class i JOIN pg_namespace n ON n.oid=i.relnamespace
        JOIN pg_index x ON x.indexrelid=i.oid JOIN pg_class t ON t.oid=x.indrelid
        WHERE n.nspname='public' AND i.relname=:name
    """), {"name": name}).one_or_none()


def _preflight(conn, expected_db: str) -> None:
    db, owner, version = conn.execute(text("SELECT current_database(),current_user,current_setting('server_version_num')")).one()
    _require(db == expected_db and db.endswith("_test") and int(version) >= 160000,
             "actual database identity or PostgreSQL 16+ mismatch")
    for table in ("bot_profiles", "chat_pending_questions", "tool_call_records", "agent_events",
                  "agent_runs", "agent_session_state", "entity_profiles", "turn_logs", "inter_bot_deliveries"):
        relation = conn.execute(text("""
            SELECT relkind, pg_get_userbyid(relowner) FROM pg_class
            WHERE oid=to_regclass(:table)
        """), {"table": f"public.{table}"}).one()
        _require(relation == ("r", owner), f"{table} must be an owned ordinary table")
    for flag, default in FLAGS:
        _require(_column(conn, "bot_profiles", flag) == ("boolean", False, default),
                 f"bot_profiles.{flag} legacy definition changed")
        count = conn.execute(text(f"SELECT count(*) FROM public.bot_profiles WHERE {flag} IS NULL")).scalar_one()
        _require(count == 0, f"bot_profiles.{flag} contains NULL")
    _require(_column(conn, "chat_pending_questions", "origin_harness") ==
             ("character varying(32)", False, "'claude'::character varying"),
             "origin_harness legacy definition changed")
    _require(conn.execute(text("SELECT count(*) FROM public.chat_pending_questions WHERE origin_harness IS NULL")).scalar_one() == 0,
             "origin_harness contains NULL")
    for column, width in WIDTHS:
        _require(_column(conn, "tool_call_records", column) == (f"character varying({width})", False, None),
                 f"tool_call_records.{column} legacy width changed")
    for name, _, _ in MISSING_INDEXES:
        _require(conn.execute(text("SELECT to_regclass(:name)"), {"name": f"public.{name}"}).scalar_one() is None,
                 f"index name {name} already occupied")
    old = _index(conn, "ix_inter_bot_delivery_dispatch")
    _require(old is not None and old[0] == "inter_bot_deliveries" and old[2:] == (True, True, owner)
             and "target_bot_id, ordinal" in old[1] and "'QUEUED'" in old[1]
             and "'DISPATCHING'" in old[1] and "'STEERING'" not in old[1],
             "old dispatch index definition, owner or validity changed")
    for name in ("ix_inter_bot_delivery_dispatch_next", "ix_inter_bot_delivery_dispatch_previous",
                 "agent_events_id_seq"):
        _require(conn.execute(text("SELECT to_regclass(:name)"), {"name": f"public.{name}"}).scalar_one() is None,
                 f"replacement name {name} already occupied")
    for table, old_name, new_name, definition in CONSTRAINT_RENAMES:
        row = conn.execute(text("""
            SELECT pg_get_constraintdef(co.oid,true), co.convalidated,
                   x.indisvalid, x.indisready
            FROM pg_constraint co JOIN pg_class t ON t.oid=co.conrelid
            JOIN pg_namespace n ON n.oid=t.relnamespace
            JOIN pg_index x ON x.indexrelid=co.conindid
            WHERE n.nspname='public' AND t.relname=:table AND co.conname=:name
        """), {"table": table, "name": old_name}).one_or_none()
        _require(row == (definition, True, True, True), f"{table}.{old_name} changed")
        _require(conn.execute(text("""
            SELECT count(*) FROM pg_constraint WHERE conrelid=to_regclass(:table) AND conname=:name
        """), {"table": f"public.{table}", "name": new_name}).scalar_one() == 0,
                 f"{table}.{new_name} already exists")
        _require(_index(conn, new_name) is None, f"index {new_name} already exists")
    _require(conn.execute(text("SELECT pg_get_serial_sequence('public.agent_events','id')")).scalar_one()
             == "public.openclaw_events_id_seq", "event sequence ownership changed")


def _bounded(conn) -> None:
    conn.exec_driver_sql("SET LOCAL lock_timeout = '1s'")
    conn.exec_driver_sql("SET LOCAL statement_timeout = '30s'")


def _concurrent(engine: Engine, sql: str) -> None:
    """One statement outside a transaction; restore session deadman values."""
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        previous = conn.execute(text("SELECT current_setting('lock_timeout'),current_setting('statement_timeout')")).one()
        original_error = None
        try:
            conn.exec_driver_sql("SET lock_timeout = '1s'")
            conn.exec_driver_sql("SET statement_timeout = '30s'")
            conn.exec_driver_sql(sql)
        except BaseException as exc:
            original_error = exc
            raise
        finally:
            try:
                conn.execute(text("SELECT set_config('lock_timeout',:value,false)"), {"value": previous[0]})
                conn.execute(text("SELECT set_config('statement_timeout',:value,false)"), {"value": previous[1]})
            except BaseException as restore_error:
                conn.invalidate()
                if original_error is None:
                    raise
                original_error.add_note(f"timeout restoration failed; connection invalidated: {restore_error}")


def upgrade_shared_schema(engine: Engine, *, expected_db: str) -> None:
    """Run only against a fresh synthetic *_test legacy DB; stop on any drift."""
    _require(expected_db.endswith("_test") and engine.url.database == expected_db,
             "expected disposable database URL mismatch")
    with engine.connect() as conn:
        conn.exec_driver_sql("SET TRANSACTION READ ONLY")
        _preflight(conn, expected_db)
    with engine.begin() as conn:
        _bounded(conn)
        for flag, default in FLAGS:
            conn.exec_driver_sql(f"ALTER TABLE public.bot_profiles ALTER COLUMN {flag} SET DEFAULT {default}")
            conn.exec_driver_sql(f"ALTER TABLE public.bot_profiles ALTER COLUMN {flag} SET NOT NULL")
    with engine.begin() as conn:
        _bounded(conn)
        conn.exec_driver_sql("ALTER TABLE public.chat_pending_questions ALTER COLUMN origin_harness SET DEFAULT 'claude'")
        conn.exec_driver_sql("ALTER TABLE public.chat_pending_questions ALTER COLUMN origin_harness SET NOT NULL")
    with engine.begin() as conn:
        _bounded(conn)
        for column, _ in WIDTHS:
            conn.exec_driver_sql(f"ALTER TABLE public.tool_call_records ALTER COLUMN {column} TYPE VARCHAR")
    for name, table, column in MISSING_INDEXES:
        _concurrent(engine, f"CREATE INDEX CONCURRENTLY {name} ON public.{table} ({column})")
        with engine.connect() as conn:
            index = _index(conn, name)
            _require(index is not None and index[0] == table and index[2:4] == (True, True),
                     f"built index {name} invalid; manual inspection required")
    _concurrent(engine, """CREATE INDEX CONCURRENTLY ix_inter_bot_delivery_dispatch_next
        ON public.inter_bot_deliveries (target_bot_id, ordinal)
        WHERE status IN ('QUEUED', 'STEERING', 'DISPATCHING')""")
    with engine.begin() as conn:
        _bounded(conn)
        current = _index(conn, "ix_inter_bot_delivery_dispatch_next")
        previous = _index(conn, "ix_inter_bot_delivery_dispatch")
        _require(current is not None and current[2:4] == (True, True) and "'STEERING'" in current[1]
                 and previous is not None and previous[2:4] == (True, True) and "'STEERING'" not in previous[1],
                 "dispatch swap index changed")
        conn.exec_driver_sql("ALTER INDEX public.ix_inter_bot_delivery_dispatch RENAME TO ix_inter_bot_delivery_dispatch_previous")
        conn.exec_driver_sql("ALTER INDEX public.ix_inter_bot_delivery_dispatch_next RENAME TO ix_inter_bot_delivery_dispatch")
    _concurrent(engine, "DROP INDEX CONCURRENTLY public.ix_inter_bot_delivery_dispatch_previous")
    with engine.begin() as conn:
        _bounded(conn)
        for table, old, new, _ in CONSTRAINT_RENAMES:
            conn.exec_driver_sql(f"ALTER TABLE public.{table} RENAME CONSTRAINT {old} TO {new}")
        conn.exec_driver_sql("ALTER SEQUENCE public.openclaw_events_id_seq RENAME TO agent_events_id_seq")
