"""Skill assignment rows and the set of bots whose skills are managed (TASK-1309).

SQL only. Every write runs in :meth:`SkillAssignmentStore.write`, one
transaction holding a transaction-scoped advisory lock, so the cap check a
caller makes on the rows it read there cannot race another write.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import text

from ..utils.schema import SchemaBootstrapGuard, ensure_relations
from .resolver import Assignment

_RELATIONS = {
    "skill_assignments": """
        CREATE TABLE IF NOT EXISTS skill_assignments (
            scope_type TEXT NOT NULL CHECK (scope_type IN ('bot', 'tag')),
            scope_id TEXT NOT NULL,
            skill_ref TEXT NOT NULL,
            mode TEXT NOT NULL CHECK (mode IN ('off', 'available', 'injected')),
            position INTEGER NOT NULL DEFAULT 0 CHECK (position >= 0),
            updated_by TEXT,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (scope_type, scope_id, skill_ref),
            CONSTRAINT skill_assignments_off_is_bot_only
                CHECK (mode <> 'off' OR scope_type = 'bot')
        )""",
    "skill_managed_bots": """
        CREATE TABLE IF NOT EXISTS skill_managed_bots (
            bot_slug TEXT PRIMARY KEY,
            adopted_by TEXT,
            adopted_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )""",
}

_LOCK = "llm_bawt.skill_assignments"
_schema_guard = SchemaBootstrapGuard()


class SkillAssignmentStore:
    def __init__(self, engine) -> None:
        self.engine = engine

    def ensure_schema(self) -> None:
        _schema_guard.run(self.engine, "skill-assignments",
                          lambda conn: ensure_relations(conn, _RELATIONS))

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
    def managed(conn) -> frozenset[str]:
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
    def delete(conn, scope_type: str, scope_id: str, skill_ref: str) -> bool:
        return bool(conn.execute(text("""
            DELETE FROM skill_assignments
            WHERE scope_type = :scope_type AND scope_id = :scope_id AND skill_ref = :skill_ref"""),
            {"scope_type": scope_type, "scope_id": scope_id, "skill_ref": skill_ref}).rowcount)

    @staticmethod
    def adopt(conn, bot: str, refs: list[str], actor: str | None) -> None:
        """Mark ``bot`` managed and seed ``available`` for refs it has no row for."""
        conn.execute(text("""
            INSERT INTO skill_managed_bots (bot_slug, adopted_by) VALUES (:bot, :actor)
            ON CONFLICT (bot_slug) DO NOTHING"""), {"bot": bot, "actor": actor})
        conn.execute(text("""
            INSERT INTO skill_assignments
                (scope_type, scope_id, skill_ref, mode, position, updated_by)
            SELECT 'bot', :bot, ref, 'available', 0, :actor
            FROM unnest(CAST(:refs AS text[])) AS ref
            ON CONFLICT (scope_type, scope_id, skill_ref) DO NOTHING"""),
            {"bot": bot, "refs": refs, "actor": actor})

    @staticmethod
    def release(conn, bot: str) -> bool:
        """Stop managing ``bot``: drop its rows; it returns to the ambient skills."""
        conn.execute(text("DELETE FROM skill_assignments WHERE scope_type = 'bot' AND scope_id = :bot"),
                     {"bot": bot})
        return bool(conn.execute(text("DELETE FROM skill_managed_bots WHERE bot_slug = :bot"),
                                 {"bot": bot}).rowcount)
