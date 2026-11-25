"""Playground revisions: nullable ``revision``/``updated_at`` and the retired-blob table.

Uses the runtime's idempotent bootstrap, with no data transitions; existing
rows read as revision 1. ``playground_retired_blobs`` records blobs a revision
replaced until the sweep deletes them. A catalog that already has both needs
no DDL.
Online only because the bootstrap inspects the actual catalog.

Revision ID: 0007_playground_revisions
Revises: 0006_utility_usage
Create Date: 2026-11-25
"""

from __future__ import annotations

from alembic import context, op

revision = "0007_playground_revisions"
down_revision = "0006_utility_usage"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if context.is_offline_mode():
        raise RuntimeError("0007_playground_revisions inspects the live catalog; run it online, not with --sql")
    from llm_bawt.playground.store import bootstrap_playground_schema

    bootstrap_playground_schema(op.get_bind())


def downgrade() -> None:
    # Prior source ignores the two nullable columns. Preserve data.
    pass
