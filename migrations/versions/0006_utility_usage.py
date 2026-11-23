"""Utility usage table and its creation-time index.

Uses the existing idempotent bootstrap, without data transitions. A complete
runtime-bootstrapped table needs no DDL. Online only: inspects the actual catalog.

Revision ID: 0006_utility_usage
Revises: 0005_playground
Create Date: 2026-11-19
"""

from __future__ import annotations

from alembic import context, op

revision = "0006_utility_usage"
down_revision = "0005_playground"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if context.is_offline_mode():
        raise RuntimeError("0006_utility_usage inspects the live catalog; run it online, not with --sql")
    from llm_bawt.service.usage.utility_usage import bootstrap_utility_usage_schema

    bootstrap_utility_usage_schema(op.get_bind())


def downgrade() -> None:
    # Prior source can reuse the unchanged schema. Preserve all usage rows.
    pass
