"""Ops store family: tables, columns and one-time data transitions.

Owns the DDL of ops_operations, ops_jobs, ops_operation_revisions,
ops_bell_outbox, ops_release_runs, ops_release_events, quiesce_run_sightings and
ops_host_access. Once a database is at this revision, the runtime bootstraps for
those tables are skipped; the ops store's ongoing current-revision repair still
runs at startup.

This calls the same idempotent bootstrap functions the runtime path uses, so the
two paths cannot diverge while both exist. Each data transition runs only when
its column is added: the TASK-1169 execution_mode legacy mapping and the
gate_plan_key backfill. Existing values are never rewritten. On a database whose
runtime bootstrap already ran (live), this is a no-op.

Online only: the bootstraps inspect the live catalog, so --sql cannot render it.

Revision ID: 0002_ops_store
Revises: 0001_baseline
Create Date: 2026-11-19
"""

from __future__ import annotations

from alembic import context, op

revision = "0002_ops_store"
down_revision = "0001_baseline"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if context.is_offline_mode():
        raise RuntimeError("0002_ops_store inspects the live catalog; run it online, not with --sql")
    from llm_bawt.ops.host_access import bootstrap_host_access_schema
    from llm_bawt.ops.quiesce_exit import bootstrap_sighting_schema
    from llm_bawt.ops.release_store import bootstrap_release_schema
    from llm_bawt.ops.store import bootstrap_ops_store_schema

    conn = op.get_bind()
    bootstrap_ops_store_schema(conn)
    bootstrap_release_schema(conn)
    bootstrap_sighting_schema(conn)
    bootstrap_host_access_schema(conn)


def downgrade() -> None:
    # Moving back to 0001 hands ownership back to the runtime bootstraps, which
    # are idempotent against this schema. Nothing is dropped.
    pass
