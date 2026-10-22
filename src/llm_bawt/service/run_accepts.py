"""The durable record of which bridge boot accepted a run (TASK-1233 B1 (b)).

A bridge publishes ``RUN_ACCEPTED`` {request_id, backend, boot_id, pid1}
(:class:`agent_bridge.run_contract.RunAccepted`) when it admits a send. The
stream worker that owns the run hands it to
:meth:`.turn_execution.TurnExecution.record_accept`, which writes it here. The
quiesce exit sweep (:mod:`..ops.quiesce_exit`) reads it as the run's first
holder: a later ``not_found`` from a different PID-1 boot proves the run died
with the accepting process, even when the bridge died before the sweep ever
saw the run live.

One row per ``(backend, request_id)``; the first accept wins and a repeat is
a no-op (``ON CONFLICT DO NOTHING``), so the record is idempotent per request.
It is independent of the turn row, so it outlives the row's deletion. Rows
older than :data:`ACCEPT_RETENTION_S` are deleted by the sweep. PostgreSQL only.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sqlalchemy import text

from agent_bridge.run_contract import RunAccepted, RunContractError

from ..utils.schema import SchemaBootstrapGuard, ensure_relations

ACCEPT_RETENTION_S = 86400
"""An accept older than this is deleted; far past any quiesce wait (``QUIESCE_TIMEOUT_MAX_S``)."""

_RELATIONS = {
    "bridge_run_accepts": """
        CREATE TABLE IF NOT EXISTS bridge_run_accepts (
            backend VARCHAR(64) NOT NULL,
            request_id VARCHAR(128) NOT NULL,
            boot_id VARCHAR(128) NOT NULL,
            pid1 BOOLEAN,
            accepted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (backend, request_id)
        )
    """,
    "ix_bridge_run_accepts_accepted_at": """
        CREATE INDEX IF NOT EXISTS ix_bridge_run_accepts_accepted_at
        ON bridge_run_accepts (accepted_at)
    """,
}

_schema_guard = SchemaBootstrapGuard()


def _is_postgres(engine) -> bool:
    return getattr(getattr(engine, "dialect", None), "name", None) == "postgresql"


def ensure_accept_schema(engine) -> None:
    """Create ``bridge_run_accepts`` once per engine (PostgreSQL only)."""
    if _is_postgres(engine):
        _schema_guard.run(engine, "bridge-run-accepts", lambda conn: ensure_relations(conn, _RELATIONS))


def record_accept(engine, raw: Mapping[str, Any] | None, *, request_id: str | None = None) -> bool:
    """Store a ``RUN_ACCEPTED`` payload; returns whether this call wrote the row.

    ``request_id``, when given, is the request the caller dispatched; an
    accept naming any other request is refused. Raises
    :class:`agent_bridge.run_contract.RunContractError` for a malformed or
    mismatched payload; the caller decides how loud that is.
    """
    if not raw or not _is_postgres(engine):
        return False
    accepted = RunAccepted.from_dict(raw)
    if request_id is not None and accepted.request_id != request_id:
        raise RunContractError(f"accept for {accepted.request_id!r}, dispatched {request_id!r}")
    ensure_accept_schema(engine)
    with engine.begin() as conn:
        return conn.execute(text(
            "INSERT INTO bridge_run_accepts (backend, request_id, boot_id, pid1)"
            " VALUES (:backend, :request, :boot, :pid1)"
            " ON CONFLICT (backend, request_id) DO NOTHING"
        ), {"backend": accepted.backend.lower(), "request": accepted.request_id, "boot": accepted.boot_id,
            "pid1": accepted.pid1}).rowcount == 1


def purge_accepts(engine, *, retention_s: int = ACCEPT_RETENTION_S) -> int:
    with engine.begin() as conn:
        return conn.execute(text(
            "DELETE FROM bridge_run_accepts WHERE accepted_at < now() - make_interval(secs => :s)"
        ), {"s": retention_s}).rowcount


__all__ = ["ACCEPT_RETENTION_S", "ensure_accept_schema", "purge_accepts", "record_accept"]
