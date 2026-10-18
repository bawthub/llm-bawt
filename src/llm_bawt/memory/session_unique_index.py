"""TASK-1149: explicit active-session uniqueness contract and upgrade path.

Fresh bootstrap creates the index only on a newly created sessions table. An
existing table missing it needs the separate, reviewed concurrent upgrade;
never try a blocking index build or silent data repair during app startup.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine


INDEX_NAME = "idx_sessions_one_active_per_bot_user"
INDEX_SQL = (
    f"CREATE UNIQUE INDEX {INDEX_NAME} ON public.sessions "
    "USING btree (bot_id, user_id) NULLS NOT DISTINCT WHERE status='active'"
)
# PostgreSQL 16's catalog normalization; matches the already-correct live
# definition captured in the TASK-1149 Phase 1 read-only catalog snapshot.
EXPECTED_DEFINITION = (
    f"CREATE UNIQUE INDEX {INDEX_NAME} ON public.sessions "
    "USING btree (bot_id, user_id) NULLS NOT DISTINCT "
    "WHERE ((status)::text = 'active'::text)"
)


def active_session_index_present(conn: Connection) -> bool:
    """Reject wrong/invalid same-name indexes rather than trusting IF NOT EXISTS."""
    result = conn.execute(text("""
        SELECT pg_get_indexdef(i.indexrelid) AS definition,
               i.indisvalid, i.indisready, i.indisunique, i.indnullsnotdistinct
        FROM pg_class index_rel
        JOIN pg_namespace n ON n.oid = index_rel.relnamespace
        JOIN pg_index i ON i.indexrelid = index_rel.oid
        WHERE n.nspname = 'public' AND index_rel.relname = :name
    """), {"name": INDEX_NAME}).mappings().one_or_none()
    if result is None:
        # A same-name non-index object still blocks creation; fail explicitly.
        name_taken = conn.execute(text("SELECT to_regclass(:name)"),
                                  {"name": f"public.{INDEX_NAME}"}).scalar_one()
        if name_taken is not None:
            raise RuntimeError(f"{INDEX_NAME} exists but is not an index")
        return False
    if (result["definition"] != EXPECTED_DEFINITION or not result["indisvalid"]
            or not result["indisready"] or not result["indisunique"]
            or not result["indnullsnotdistinct"]):
        raise RuntimeError(f"{INDEX_NAME} is invalid or differs from the reviewed one-active contract")
    return True


def ensure_fresh_sessions_index(conn: Connection, *, table_was_new: bool) -> None:
    """Bootstrap fresh tables; hold preexisting unupgraded tables for review."""
    if active_session_index_present(conn):
        return
    if not table_was_new:
        raise RuntimeError(f"existing sessions lacks {INDEX_NAME}; run the reviewed TASK-1149 upgrade")
    conn.execute(text(INDEX_SQL))
    if not active_session_index_present(conn):
        raise RuntimeError("fresh sessions uniqueness index did not validate")


def upgrade_sessions_one_active_index(engine: Engine) -> bool:
    """Explicit upgrade, not invoked by startup. True only if an index was built.

    Concurrent build keeps ordinary reads/writes available. Duplicate active
    pairs (including NULL user_id) cause PostgreSQL to fail and may leave an
    INVALID index. That requires operator review; this function never drops
    an index, mutates sessions, or retries an ambiguous build.
    """
    with engine.connect() as conn:
        if active_session_index_present(conn):
            return False
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        previous_lock, previous_statement = conn.execute(text("""
            SELECT current_setting('lock_timeout'), current_setting('statement_timeout')
        """)).one()
        primary_error: BaseException | None = None
        try:
            conn.execute(text("SET lock_timeout = '1s'"))
            conn.execute(text("SET statement_timeout = '30s'"))
            conn.execute(text(INDEX_SQL.replace("CREATE UNIQUE INDEX ",
                                                "CREATE UNIQUE INDEX CONCURRENTLY ", 1)))
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                # RESET would discard the app's on-connect deadman limits and
                # return an unbounded session to the shared connection pool.
                conn.execute(text("SELECT set_config('lock_timeout', :value, false)"),
                             {"value": previous_lock})
                conn.execute(text("SELECT set_config('statement_timeout', :value, false)"),
                             {"value": previous_statement})
            except BaseException as restore_error:
                # A partially restored session may not go back into the pool.
                conn.invalidate(restore_error)
                if primary_error is not None:
                    primary_error.add_note(f"Session timeout restoration failed; connection invalidated: {restore_error}")
                else:
                    raise
    with engine.connect() as conn:
        if not active_session_index_present(conn):
            raise RuntimeError("concurrent sessions uniqueness index did not validate")
    return True
