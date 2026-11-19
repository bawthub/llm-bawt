"""Playground artifact table, nullable additions and indexes.

Uses the runtime's idempotent bootstrap, with no data transitions. An already
bootstrapped database needs no DDL. Online only because the bootstrap inspects
the actual catalog.

Revision ID: 0005_playground
Revises: 0004_gates
Create Date: 2026-11-19
"""

from __future__ import annotations

from alembic import context, op

revision = "0005_playground"
down_revision = "0004_gates"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if context.is_offline_mode():
        raise RuntimeError("0005_playground inspects the live catalog; run it online, not with --sql")
    from llm_bawt.playground.store import bootstrap_playground_schema

    bootstrap_playground_schema(op.get_bind())


def downgrade() -> None:
    # The prior source's runtime bootstrap can use the same table. Preserve data.
    pass
