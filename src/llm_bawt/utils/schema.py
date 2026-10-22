"""Concurrency-safe runtime schema bootstrap helpers.

Runtime DDL is still a compatibility bridge for deployments that predate a
given table or column.  This module keeps that bridge off steady-state request
paths by making each logical schema unit single-flight per SQLAlchemy engine.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Hashable, Mapping
from weakref import WeakKeyDictionary, WeakSet

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

logger = logging.getLogger(__name__)

_ADVISORY_LOCK_NAME = "llm-bawt-runtime-schema"

# One process-local order for every runtime schema unit.  The PostgreSQL
# advisory lock below extends the same ordering across app processes.
_bootstrap_lock = threading.RLock()
_guards: WeakSet[SchemaBootstrapGuard] = WeakSet()


class SchemaBootstrapGuard:
    """Run a transactional schema callback once per engine and logical scope."""

    def __init__(self) -> None:
        self._completed: WeakKeyDictionary[Engine, set[Hashable]] = WeakKeyDictionary()
        _guards.add(self)

    def run(
        self,
        engine: Engine,
        logical_scope: Hashable,
        bootstrap: Callable[[Connection], None],
    ) -> bool:
        """Run ``bootstrap`` once, returning whether this call performed it.

        Completion is recorded only after ``engine.begin()`` exits cleanly and
        commits.  Exceptions therefore leave the scope retryable.
        """

        with _bootstrap_lock:
            completed = self._completed.setdefault(engine, set())
            if logical_scope in completed:
                return False

            try:
                with engine.begin() as conn:
                    if conn.dialect.name == "postgresql":
                        conn.execute(
                            text(
                                "SELECT pg_advisory_xact_lock("
                                "hashtext(:schema_lock_name))"
                            ),
                            {"schema_lock_name": _ADVISORY_LOCK_NAME},
                        )
                    bootstrap(conn)
            except Exception:
                logger.exception(
                    "Runtime schema bootstrap failed for scope %r", logical_scope
                )
                raise

            # The transaction has committed successfully at this point.
            completed.add(logical_scope)
            logger.info("Runtime schema ready: %s", logical_scope)
            return True

    def reset_for_tests(self, engine: Engine | None = None) -> None:
        """Forget completion state. Intended only for isolated tests."""

        with _bootstrap_lock:
            if engine is None:
                self._completed.clear()
            else:
                self._completed.pop(engine, None)


def ensure_relations(conn: Connection, relations: Mapping[str, str]) -> list[str]:
    """Run each relation's CREATE only when the catalog lacks it; return the names created.

    ``relations`` maps a table or index name to its ``CREATE ... IF NOT EXISTS``
    statement, in creation order. One catalog read decides, because a no-op
    ``CREATE INDEX IF NOT EXISTS`` still takes SHARE on its table before the
    existence check and queues every writer behind any open transaction.
    """
    names = list(relations)
    present = {row[0] for row in conn.execute(text(
        "SELECT name FROM unnest(CAST(:names AS text[])) AS name WHERE to_regclass(name) IS NOT NULL"
    ), {"names": names})}
    created = [name for name in names if name not in present]
    for name in created:
        conn.execute(text(relations[name]))
    return created


def ensure_columns(conn: Connection, table: str, columns: Mapping[str, tuple[str, str | None]]) -> list[str]:
    """Add missing columns / set drifted defaults on ``table``; return the DDL run.

    ``columns`` maps name -> (SQL type, default expression or None), the default
    spelled exactly as ``pg_get_expr`` reports it (``now()``). One catalog read
    decides everything, so an already-migrated table gets no ALTER: even a
    no-op ``ADD COLUMN IF NOT EXISTS`` requests ACCESS EXCLUSIVE and queues
    every reader behind any open transaction on the table. A column added here
    starts NULL on existing rows; the default applies to new rows only.
    """
    present = dict(conn.execute(text("""
        SELECT a.attname, pg_get_expr(d.adbin, d.adrelid)
        FROM pg_attribute a LEFT JOIN pg_attrdef d ON d.adrelid = a.attrelid AND d.adnum = a.attnum
        WHERE a.attrelid = to_regclass(:table) AND a.attnum > 0 AND NOT a.attisdropped
    """), {"table": table}).all())
    ddl: list[str] = []
    for name, (sql_type, default) in columns.items():
        if name not in present:
            ddl.append(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {name} {sql_type}")
        if default is not None and present.get(name) != default:
            ddl.append(f"ALTER TABLE {table} ALTER COLUMN {name} SET DEFAULT {default}")
    for statement in ddl:
        conn.execute(text(statement))
    return ddl


def reset_schema_guards_for_tests(engine: Engine | None = None) -> None:
    """Reset all registered guards without touching or disposing engines."""

    with _bootstrap_lock:
        for guard in list(_guards):
            guard.reset_for_tests(engine)
