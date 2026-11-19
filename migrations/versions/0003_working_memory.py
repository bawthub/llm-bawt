"""Working memory family: tables, columns and indexes.

Owns the DDL of working_memory_namespaces, working_memory_items,
working_memory_task_suppressions, working_memory_task_purge_audit and
working_memory_changes. Once a database is at this revision, the runtime
bootstrap for those tables is skipped.

This calls the same idempotent bootstrap function the runtime path uses, so the
two paths cannot diverge while both exist. The family has no data transitions:
its later columns are nullable additions with no backfill. On a database whose
runtime bootstrap already ran (live), this is a no-op.

Online only: the bootstrap inspects the live catalog, so --sql cannot render it.

Revision ID: 0003_working_memory
Revises: 0002_ops_store
Create Date: 2026-11-19
"""

from __future__ import annotations

from alembic import context, op

revision = "0003_working_memory"
down_revision = "0002_ops_store"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if context.is_offline_mode():
        raise RuntimeError("0003_working_memory inspects the live catalog; run it online, not with --sql")
    from llm_bawt.working_memory.store import bootstrap_working_memory_schema

    bootstrap_working_memory_schema(op.get_bind())


def downgrade() -> None:
    # Moving back to 0002 hands ownership back to the runtime bootstrap, which is
    # idempotent against this schema. Nothing is dropped.
    pass
