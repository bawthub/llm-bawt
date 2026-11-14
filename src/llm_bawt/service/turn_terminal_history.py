"""History projection from the immutable winning terminal snapshot only."""
from __future__ import annotations

from ..message_authorship import AuthorReference


def project_terminal_history(snapshot: dict, rows) -> None:
    """Write the assistant/tools pair atomically using the pinned session/IDs.

    ``rows`` is the owning bot's canonical MessageRowStore, not a history
    manager with mutable active-session state. The caller marks projection
    complete separately; a crash between those steps safely repeats this pair.
    """
    if snapshot.get("version") != 1:
        raise ValueError("unsupported terminal snapshot version")
    if snapshot.get("history_disposition") == "unbound":
        completion = snapshot["completion"]
        if (snapshot.get("history_session_id") is not None
                or completion.get("session_id") is not None
                or completion.get("status") not in {"cancelled", "aborted", "timeout", "error"}
                or snapshot.get("response_text") or snapshot.get("reasoning")
                or snapshot.get("tool_context") or snapshot.get("attachments")
                or snapshot.get("tts_tail")):
            raise ValueError("unbound history projection requires an empty pre-inference failure")
        # No session was bound and no inference ran. Never fabricate a session
        # or fall back to the latest one just to materialize an empty history row.
        return
    session = snapshot["history_session_id"]
    if not session:
        raise ValueError("bound terminal history requires a pinned session")
    timestamp = snapshot["history_timestamp"]
    completion = snapshot["completion"]
    with rows.engine.begin() as connection:
        if snapshot.get("tool_context"):
            rows.insert_terminal(
                message_id=snapshot["tools_message_id"], role="system",
                content=f"[Tool Results]\n{snapshot['tool_context']}",
                timestamp=timestamp - 0.000001, session_id=session, connection=connection,
            )
        if snapshot.get("response_text") or snapshot.get("reasoning"):
            rows.insert_terminal(
                message_id=snapshot["assistant_message_id"], role="assistant",
                content=snapshot["response_text"], timestamp=timestamp, session_id=session,
                attachments=snapshot.get("attachments"), reasoning=snapshot.get("reasoning"),
                author=AuthorReference.bot(completion["bot_id"]), connection=connection,
            )
