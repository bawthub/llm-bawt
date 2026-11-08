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


def _delivery_stays_in_background(conn, *, bot_id: str, session_id: str) -> bool:
    """True when the bound session is inactive and not the bot's current one."""
    row = conn.execute(text("""
        SELECT s.status, b.last_session_id FROM sessions s
        LEFT JOIN bot_profiles b ON b.slug = s.bot_id
        WHERE s.id = :session AND s.bot_id = :bot
    """), {"session": session_id, "bot": bot_id}).mappings().one_or_none()
    return (row is not None and row["status"] not in ("active", "deleted")
            and row["last_session_id"] != session_id)


def accept_session_binding(
    engine, *, turn_id: str, bot_id: str, user_id: str, session_id: str,
    claim=None, record_resumption: bool = False,
) -> dict | None:
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
        lock = " FOR NO KEY UPDATE" if conn.dialect.name == "postgresql" else ""
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
        if getattr(claim, "reservation_delivery_id", None) and _delivery_stays_in_background(
                conn, bot_id=bot_id, session_id=session_id):
            # TASK-2144: a delivered turn (scheduled wake-up, bot message) bound
            # to a session that is neither active nor current runs there without
            # becoming the bot's current session, so the next ordinary prompt in
            # the main conversation is not a "resumption" of it.
            return None
        receipt = None
        if record_resumption:
            from ..session_resumption import record

            trigger = conn.execute(text(
                "SELECT trigger_message_id FROM turn_logs WHERE id=:id"
            ), {"id": turn_id}).scalar_one_or_none()
            receipt = record(conn, bot_id=bot_id, user_id=user_id, turn_id=turn_id,
                             trigger_message_id=trigger, session_id=session_id)
        BotSessionState.accept(conn, bot_id=bot_id, session_id=session_id)
    return receipt
