"""Resolve a real terminal-history session before the turn's user write."""
from __future__ import annotations


def bind_terminal_session(
    follower, llm_bawt, *, bot_id: str, user_id: str,
    resolved_session_id: str | None = None, is_agent: bool = False,
) -> str:
    """Bind canonical DB ownership and return the user/assistant write pin.

    Call only after admission and the existing reset/selected-thread setup.
    An agent must supply its final binding; a native continuous turn resolves
    its canonical active row directly, never the cached/fallback UUID property.
    The protocol performs the authoritative DB existence/owner check.
    """
    follower.check()
    backend = getattr(getattr(llm_bawt, "history_manager", None), "_db_backend", None)
    if backend is None or (getattr(backend, "bot_id", None), getattr(backend, "user_id", None)) != (
        bot_id, user_id,
    ):
        raise ValueError("terminal history backend does not match the turn owner")
    session_id = resolved_session_id
    if session_id is None:
        if is_agent:
            raise ValueError("agent terminal history requires the final request-local session")
        resolve = getattr(backend, "get_or_create_active_session", None)
        if not callable(resolve):
            raise ValueError("terminal history has no canonical session resolver")
        # This operation is after admission. Its errors must escape: do not
        # consult _current_session_id, retry via a cache, or invent a session.
        session_id = resolve(bot_id=bot_id, user_id=user_id)
    if (not isinstance(session_id, str) or not session_id
            or session_id != session_id.strip() or len(session_id) > 36):
        raise ValueError("canonical session resolution returned no usable session ID")
    follower.protocol.bind_history_session(follower.lease, session_id=session_id)
    follower.check()
    # The caller passes this only as history_write_session_id. Setting the
    # explicit-thread override here would also narrow continuous read context.
    return session_id
