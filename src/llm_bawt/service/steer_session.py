"""Read-only durable steer destination, never a bridge/active-session pointer."""
from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy import bindparam, text


_OWNED = text("""
    SELECT t.id, t.bot_id, t.user_id, t.trigger_message_id, s.id AS session_id
    FROM turn_logs t
    JOIN messages m ON m.bot_id=t.bot_id AND m.id=t.trigger_message_id
    JOIN sessions s ON s.id=m.session_id AND s.bot_id=t.bot_id AND s.user_id=t.user_id
    WHERE t.id IN :ids AND s.status <> 'deleted'
""").bindparams(bindparam("ids", expanding=True))


def owned_turn_sessions(engine, turns) -> dict[str, str]:
    """Bounded batch enrichment; missing/foreign bindings stay unknown, not idle.

    Compare the queried binding with the caller's turn snapshot as well, so a
    changed trigger/owner cannot supply a session for a stale target snapshot.
    """
    expected = {turn.id: (turn.bot_id, turn.user_id, getattr(turn, "trigger_message_id", None))
                for turn in turns}
    if engine is None or not expected:
        return {}
    with engine.connect() as conn:
        return {row["id"]: str(row["session_id"])
                for row in conn.execute(_OWNED, {"ids": list(expected)}).mappings()
                if expected[row["id"]] == (row["bot_id"], row["user_id"], row["trigger_message_id"])}


def voice_steer_target_ready(conn, active, *, bot_id: str, user_id: str, session_id: str | None) -> bool:
    """Conservative voice admission readout and claim fence, under the target lock.

    Only a ready run's pinned NEXT capability counts, never the current bot
    configuration or a provisional reservation. A selected conversation cannot
    become a steer into a different (or unknown) durable conversation.
    """
    from ..steer_capabilities import NEXT, run_steer_caps

    if (active is None or active.get("status") not in {"streaming", "pending"}
            or active.get("user_id") != user_id
            or not active.get("agent_session_key") or not active.get("agent_request_id")
            or not active.get("agent_backend")
            or NEXT not in run_steer_caps(backend=active["agent_backend"], caps=active.get("agent_steer_caps"))):
        return False
    owned = conn.execute(_OWNED, {"ids": [active["id"]]}).mappings().first()
    return bool(owned and owned["bot_id"] == bot_id and owned["user_id"] == user_id
                and (session_id is None or session_id.strip() == str(owned["session_id"])))


def require_steer_session(service, request, turn, *, expected_session: str | None = None) -> str:
    """Fail before side effects if the exact target has no owned durable session."""
    try:
        found = owned_turn_sessions(getattr(service._turn_log_store, "engine", None), [turn]).get(turn.id)
    except Exception as exc:
        raise HTTPException(503, "Steer session lookup unavailable; message was not accepted") from exc
    if not found:
        raise HTTPException(409, "session_unavailable")
    selected = getattr(request, "session_id", None)
    if ((selected is not None and selected.strip() != found)
            or (expected_session is not None and expected_session != found)):
        raise HTTPException(409, "session_mismatch")
    return found


def revalidate_steer_target(service, request, original, session_id, *, allow_ended=False):
    """Re-read after async preparation; never replace a changed target with a new run."""
    current = service._turn_log_store.get_turn(original.id)
    if current is None:
        raise HTTPException(409, "active_run_mismatch")
    binding = ("bot_id", "user_id", "trigger_message_id", "agent_session_key", "agent_request_id", "agent_backend")
    if any(getattr(current, key, None) != getattr(original, key, None) for key in binding):
        raise HTTPException(409, "active_run_mismatch")
    if not allow_ended and (current.ended_at is not None or current.status not in ("streaming", "pending")):
        raise HTTPException(409, "Turn is no longer active")
    require_steer_session(service, request, current, expected_session=session_id)
    return current
