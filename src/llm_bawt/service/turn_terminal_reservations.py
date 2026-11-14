"""Immutable cancellation of protocol reservations in a producer transaction."""
from __future__ import annotations

from types import SimpleNamespace

from sqlalchemy import select

from .turn_log_schema import TurnLog
from .turn_terminal_protocol import TurnTerminalProtocol
from .turn_usage import encode_token_usage


def interrupt_protocol_reservation(connection, turn_id: str, *, status: str = "aborted",
                                   error_text: str | None = None, audit_key: str = "recovery",
                                   audit: dict | None = None) -> bool:
    """The caller already fenced its producer; no independent commit or publish.

    Hold/requeue releases of genuinely unstarted reservations may still DELETE:
    the producer supplies the same durable assistant UUID on the next INSERT.
    Terminal producer outcomes instead keep this immutable no-history receipt.
    """
    row = connection.execute(select(TurnLog.__table__).where(TurnLog.id == turn_id)
                             .with_for_update()).mappings().first()
    if (row is None or row["terminal_protocol_state"] != "reserved"
            or row["status"] != "reserved" or row["ended_at"] is not None):
        return False
    # Commit needs only the canonical engine and pure usage encoder; do not
    # initialize stores/schemas while holding the producer's transaction locks.
    store = SimpleNamespace(engine=connection.engine, _encode_token_usage=encode_token_usage)
    result = TurnTerminalProtocol(store).interrupt(
        turn_id, status=status, end_reason="error" if status == "error" else "aborted",
        error_text=error_text, audit_key=audit_key,
        audit={"by": "reservation_producer", **(audit or {})}, connection=connection,
    )
    return result.won
