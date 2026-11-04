"""Which task each agent turn worked on, for BawtHub's board (TASK-1754).

Read-only. One resolver (``turn_task.resolve_turn_tasks``) serves this route,
bot live state and MCP ``bots_list_available``; BawtHub owns task titles and
statuses and joins them on its side.
"""
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Query, Response
from sqlalchemy.exc import SQLAlchemyError

from ...turn_task import resolve_turn_tasks
from ...utils.db import get_shared_engine
from ..dependencies import get_service

router = APIRouter()
_NO_STORE = {"Cache-Control": "no-store"}
MAX_HOURS = 24 * 7


@router.get("/v1/turn-tasks", tags=["System"])
def turn_tasks(
    response: Response,
    hours: float = Query(24, gt=0, le=MAX_HOURS),
    bot: list[str] | None = Query(None),
):
    """Turns open, started or ended in the last ``hours`` that resolve to a task.

    Newest first; each has ``turn_id, bot_id, task_ref, source, started_at,
    ended_at`` (``ended_at`` null = still running).
    """
    response.headers.update(_NO_STORE)
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    try:
        engine = get_shared_engine(get_service().config)
        with engine.connect() as conn:
            found = resolve_turn_tasks(conn, bots=bot or None, since=since)
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail="Turn tasks unavailable", headers=_NO_STORE) from exc
    return {"since": since.isoformat(), "turns": [item.to_dict() for item in found]}
