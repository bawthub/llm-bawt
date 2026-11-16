"""Page-local SQL for /history; never hydrate/transport a whole context window."""

from fastapi import HTTPException
from sqlalchemy import text

from .history_scope import owned_message_scope


def load_history_page(
    service, bot_id: str, limit: int, before: str | None, after: str | None,
    session_id: str | None, user_id: str | None,
) -> tuple[list[dict], bool, bool] | None:
    """Return (chronological rows, has_older, has_newer), or legacy fallback.

    Continuous history preserves the context manager's unsummarized visibility.
    Explicit threads, like the existing raw reader, include summarized bubbles.
    Filter BEFORE LIMIT; one extra row replaces a full-history count.
    """
    if limit <= 0:
        return None  # Preserve the legacy unlimited-read contract.
    from ...media.assets import _build_engine
    from ...memory.postgresql import MESSAGES_PARENT, partition_name
    from .history_pages import _resolve_cursor, _hydrate_window_authors, log

    try:
        engine = _build_engine(service.config)
        if engine is None:
            return None
        table = partition_name(MESSAGES_PARENT, bot_id)
        scope = ["m.role NOT IN ('system', 'summary')"]
        if session_id:
            scope.append("m.session_id = :session_id")
        else:
            scope.append("(m.summarized IS NULL OR m.summarized = FALSE)")
        if user_id:
            scope.append(owned_message_scope())
        where = " AND ".join(scope)
        params = {"bot_id": bot_id, "user_id": user_id, "session_id": session_id}
        columns = (
            "m.id, m.role, m.content, m.timestamp, m.session_id, "
            "m.author_entity_type, m.author_entity_id"
        )
        with engine.connect() as conn:
            raw_cursor = after or before
            cursor = None
            if raw_cursor:
                try:
                    cursor = _resolve_cursor(raw_cursor, [])
                except HTTPException:
                    # IDs resolve only within the same owner/thread/visibility
                    # scope; hidden or foreign IDs remain Invalid cursor (400).
                    anchor = conn.execute(
                        text(f"SELECT m.id, m.timestamp FROM {table} m "
                             f"WHERE {where} AND m.id = :cursor_id"),
                        {**params, "cursor_id": raw_cursor.strip()},
                    ).mappings().all()
                    cursor = _resolve_cursor(raw_cursor, list(anchor))
            if cursor is not None:
                params["ts"] = cursor.timestamp
                op = ">" if after else "<"
                boundary = f"m.timestamp {op} :ts"
                if cursor.message_id is not None:
                    params["cursor_id"] = cursor.message_id
                    boundary = (f"({boundary} OR "
                                f"(m.timestamp = :ts AND m.id {op} :cursor_id))")
                where += f" AND {boundary}"
            # A blank timestamp cursor resolves to None: legacy history treats
            # that as a default backward tail, even when supplied via `after`.
            forward = bool(after and cursor is not None)
            direction = "ASC" if forward else "DESC"
            rows = conn.execute(
                text(f"SELECT {columns} FROM {table} m WHERE {where} "
                     f"ORDER BY m.timestamp {direction}, m.id {direction} LIMIT :lim"),
                {**params, "lim": limit + 1},
            ).mappings().all()
    except HTTPException:
        raise
    except Exception as exc:
        log.warning("History page SQL unavailable for %s: %s", bot_id, exc)
        return None

    more = len(rows) > limit
    page = [dict(row) for row in rows[:limit]]
    if not forward:
        page.reverse()
    return _hydrate_window_authors(page, bot_id), more and not forward, more and forward
