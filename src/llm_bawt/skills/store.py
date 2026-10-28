"""Skill exception rows (TASK-1309).

SQL only. Every write runs in :meth:`SkillAssignmentStore.write`, one
transaction holding a transaction-scoped advisory lock, so the cap check a
caller makes on the rows it read there cannot race another write.

Rows are exceptions to "every skill enabled": ``enable | disable | inject``
at bot or tag scope. The adopt-era schema (``off | available | injected``,
``off`` bot-only, a ``skill_managed_bots`` table) is migrated in place by
:meth:`SkillAssignmentStore.migrate`, once, with rows the service computes
(:func:`.resolver.legacy_rows`), because the conversion needs the skill
catalog and the bot list.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import text

from ..utils.schema import SchemaBootstrapGuard, ensure_relations
from .resolver import LEGACY_MODES, MODES, Assignment

_ALL_MODES = ", ".join(f"'{m}'" for m in (*MODES, *LEGACY_MODES))
_MODE_CHECK = "skill_assignments_mode_vocab"

_RELATIONS = {
    "skill_assignments": f"""
        CREATE TABLE IF NOT EXISTS skill_assignments (
            scope_type TEXT NOT NULL CHECK (scope_type IN ('bot', 'tag')),
            scope_id TEXT NOT NULL,
            skill_ref TEXT NOT NULL,
            mode TEXT NOT NULL CONSTRAINT {_MODE_CHECK} CHECK (mode IN ({_ALL_MODES})),
            position INTEGER NOT NULL DEFAULT 0 CHECK (position >= 0),
            updated_by TEXT,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (scope_type, scope_id, skill_ref)
        )""",
}

_LOCK = "llm_bawt.skill_assignments"
_schema_guard = SchemaBootstrapGuard()


_LEGACY_CHECKS = ("skill_assignments_mode_check", "skill_assignments_off_is_bot_only")


def _bootstrap(conn) -> None:
    ensure_relations(conn, _RELATIONS)


class SkillAssignmentStore:
    def __init__(self, engine) -> None:
        self.engine = engine

    def ensure_schema(self) -> None:
        _schema_guard.run(self.engine, "skill-assignments",
                          _bootstrap)

    @contextmanager
    def read(self) -> Iterator:
        self.ensure_schema()
        with self.engine.connect() as conn:
            yield conn

    @contextmanager
    def write(self) -> Iterator:
        """One serialized write transaction."""
        self.ensure_schema()
        with self.engine.begin() as conn:
            conn.execute(text("SELECT pg_advisory_xact_lock(hashtext(:name))"), {"name": _LOCK})
            yield conn

    # -- reads (on a caller's connection) ----------------------------------

    @staticmethod
    def rows(conn) -> list[Assignment]:
        result = conn.execute(text("""
            SELECT scope_type, scope_id, skill_ref, mode, position, updated_by
            FROM skill_assignments ORDER BY scope_type, scope_id, position, skill_ref"""))
        return [Assignment(**row) for row in result.mappings()]

    @staticmethod
    def pending(conn) -> bool:
        """True while the table still has the adopt-era schema."""
        return bool(conn.execute(text("""
            SELECT count(*) FROM pg_constraint
            WHERE conrelid = 'skill_assignments'::regclass AND conname = ANY(:names)"""),
            {"names": list(_LEGACY_CHECKS)}).scalar_one())

    @staticmethod
    def managed(conn) -> frozenset[str]:
        """Adopt-era managed bots (empty once migrated or never adopted)."""
        if conn.execute(text("SELECT to_regclass('skill_managed_bots')")).scalar() is None:
            return frozenset()
        return frozenset(conn.execute(text("SELECT bot_slug FROM skill_managed_bots")).scalars())

    # -- writes (inside :meth:`write`) -------------------------------------

    @staticmethod
    def upsert(conn, row: Assignment) -> None:
        conn.execute(text("""
            INSERT INTO skill_assignments
                (scope_type, scope_id, skill_ref, mode, position, updated_by, updated_at)
            VALUES (:scope_type, :scope_id, :skill_ref, :mode, :position, :updated_by, now())
            ON CONFLICT (scope_type, scope_id, skill_ref) DO UPDATE
            SET mode = EXCLUDED.mode, position = EXCLUDED.position,
                updated_by = EXCLUDED.updated_by, updated_at = now()"""),
            {"scope_type": row.scope_type, "scope_id": row.scope_id, "skill_ref": row.skill_ref,
             "mode": row.mode, "position": row.position, "updated_by": row.updated_by})

    @staticmethod
    def migrate(conn, rows: list[Assignment]) -> None:
        """Replace the adopt-era table with ``rows``, inside :meth:`write`.

        The caller re-checks :meth:`pending` under the write lock. The CHECK
        swap is the done marker; the vocab CHECK accepts both vocabularies for
        one release. ``skill_managed_bots`` is emptied, not dropped.
        """
        conn.execute(text(f"""
            ALTER TABLE skill_assignments
                DROP CONSTRAINT IF EXISTS skill_assignments_mode_check,
                DROP CONSTRAINT IF EXISTS skill_assignments_off_is_bot_only,
                DROP CONSTRAINT IF EXISTS {_MODE_CHECK},
                ADD CONSTRAINT {_MODE_CHECK} CHECK (mode IN ({_ALL_MODES}))"""))
        conn.execute(text("DELETE FROM skill_assignments"))
        for row in rows:
            SkillAssignmentStore.upsert(conn, row)
        if conn.execute(text("SELECT to_regclass('skill_managed_bots')")).scalar() is not None:
            conn.execute(text("DELETE FROM skill_managed_bots"))

    @staticmethod
    def delete(conn, scope_type: str, scope_id: str, skill_ref: str) -> bool:
        return bool(conn.execute(text("""
            DELETE FROM skill_assignments
            WHERE scope_type = :scope_type AND scope_id = :scope_id AND skill_ref = :skill_ref"""),
            {"scope_type": scope_type, "scope_id": scope_id, "skill_ref": skill_ref}).rowcount)
