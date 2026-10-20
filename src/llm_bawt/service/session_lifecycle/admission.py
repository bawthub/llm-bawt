"""Accept a resolved session only after its turn passed durable admission.

Binding is later than turn-log creation in both chat paths. The bot-wide pointer
is stamped under the same target lock, but an older worker may reach this point
after a newer admission; that older worker must not start inference or move the
pointer backwards. The turn log supplies the admission-order fence.
"""
from __future__ import annotations

from sqlalchemy import text

from ...bot_session_state import BotSessionState
from ..turn_admission import AdmissionFenceLost, lock_target


def accept_session_binding(
    engine, *, turn_id: str, bot_id: str, user_id: str, session_id: str,
    claim=None,
) -> None:
    """Stamp the admitted, current binding or refuse stale/lost work.

    ``claim`` is a trusted Python-only AdmissionClaim; public JSON is never an
    acceptance authority. For an unclaimed ordinary turn, its started turn-log
    row is the authority. Only the caller may then proceed to provider inference.
    """
    if engine is None:
        raise AdmissionFenceLost("session binding admission database unavailable")
    with engine.begin() as conn:
        lock_target(conn, bot_id)
        if claim is not None and not claim.fence(conn):
            raise AdmissionFenceLost(f"claim for turn '{turn_id}' no longer held")
        lock = " FOR UPDATE" if conn.dialect.name == "postgresql" else ""
        current = conn.execute(text("""
            SELECT created_at FROM turn_logs
            WHERE id=:id AND bot_id=:bot AND user_id=:user
              AND ended_at IS NULL AND status IN ('pending', 'streaming')
        """ + lock), {"id": turn_id, "bot": bot_id, "user": user_id}).mappings().one_or_none()
        if current is None:
            raise AdmissionFenceLost(f"turn '{turn_id}' has no accepted open binding")
        newer = conn.execute(text("""
            SELECT 1 FROM turn_logs
            WHERE bot_id=:bot AND status <> 'reserved'
              AND (created_at > :started OR (created_at = :started AND id > :id))
            LIMIT 1
        """), {"bot": bot_id, "started": current["created_at"], "id": turn_id}).first()
        if newer is not None:
            raise AdmissionFenceLost(f"turn '{turn_id}' was superseded before binding")
        BotSessionState.accept(conn, bot_id=bot_id, session_id=session_id)
