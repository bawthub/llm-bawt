"""Gate family: tables, columns and indexes.

Owns the DDL of system_gates, gate_members, gate_escalations and
ops_resume_notices. Once a database is at this revision, the runtime bootstrap
for those tables is skipped.

This calls the same idempotent bootstrap function the runtime path uses, so the
two paths cannot diverge while both exist. Its one contract step drops the six
retired gate_members control-pump columns only where they still exist, exactly
as the runtime bootstrap does; on a database whose runtime bootstrap already ran
(live), the whole revision is a no-op.

Online only: the bootstrap inspects the live catalog, so --sql cannot render it.

Revision ID: 0004_gates
Revises: 0003_working_memory
Create Date: 2026-11-19
"""

from __future__ import annotations

from alembic import context, op

revision = "0004_gates"
down_revision = "0003_working_memory"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if context.is_offline_mode():
        raise RuntimeError("0004_gates inspects the live catalog; run it online, not with --sql")
    from sqlalchemy import inspect

    from llm_bawt.service.turn_gates import _MEMBER_RETIRED_COLUMNS, bootstrap_gate_schema

    conn = op.get_bind()
    bootstrap_gate_schema(conn)
    # The runtime contract step skips on a lock timeout and retries at the next
    # start. Once 0004 owns the DDL that retry never runs, so the upgrade must not
    # commit with retired columns left: fail and roll back, the next start retries.
    left = {c["name"] for c in inspect(conn).get_columns("gate_members")} & set(_MEMBER_RETIRED_COLUMNS)
    if left:
        raise RuntimeError(f"0004_gates: retired gate_members columns still present {sorted(left)}; retry the upgrade")


def downgrade() -> None:
    # Moving back to 0003 hands ownership back to the runtime bootstrap, which is
    # idempotent against this schema. Nothing is dropped or restored.
    pass
