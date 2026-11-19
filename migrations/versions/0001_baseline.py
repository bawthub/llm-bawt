"""Baseline: version marker for the runtime-bootstrapped schema.

This revision performs NO DDL. Stamping it records that a database already
carries the schema produced by llm-bawt's runtime bootstrap (``ensure_*`` and
``create_all`` units) as of master 9dfbd3c9. It does not verify parity with
the SQLModel metadata, and ``alembic upgrade head`` on an empty database
creates no application tables. ``migrations/baseline_schema.sql`` is reference
documentation of that starting point, not a restore recipe.

Revision ID: 0001_baseline
Revises:
Create Date: 2026-11-19
"""

from __future__ import annotations

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
