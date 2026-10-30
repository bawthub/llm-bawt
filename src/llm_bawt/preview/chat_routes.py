"""Explicit read/abort allowlist for isolated Preview conversations.

Production route modules are imported only inside guarded handlers, never
mounted wholesale. Every request is marked and bound to a seeded bot and user;
no handler can fall back to a production default identity or transport.
"""

from collections.abc import Callable
from typing import Literal

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict


class PreviewAbortRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn_id: str
    source: Literal["unknown", "chat_stop", "bot_list_stop"] = "unknown"


def register_chat_routes(
    app: FastAPI,
    require_preview: Callable[[Request], None],
    require_bot: Callable[[str], None],
) -> None:
    """Mount only Preview-safe routes against the process's isolated stores."""
    router = APIRouter()

    def scope(request: Request, bot_id: str, user_id: str) -> tuple[str, str]:
        require_preview(request)
        bot_id, user_id = bot_id.strip(), user_id.strip()
        if not bot_id or not user_id:
            raise HTTPException(status_code=400, detail="Preview requires bot and user namespaces")
        require_bot(bot_id)
        return bot_id, user_id

    async def owned_session(session_id: str, bot_id: str, user_id: str) -> dict:
        from ..mcp_server.storage import get_storage

        row = await get_storage().get_session(session_id, bot_id=bot_id)
        if not row or row.get("bot_id") != bot_id or row.get("user_id") != user_id:
            raise HTTPException(status_code=404, detail="Preview session not found")
        if row.get("status") == "deleted":
            raise HTTPException(status_code=410, detail="Preview session deleted")
        return row

    def session_info(row: dict) -> dict:
        meta = dict(row.get("session_metadata") or {})
        meta.setdefault("title", None)
        meta.setdefault("title_source", "default")
        meta.setdefault("agent_session_keys", {})
        meta.setdefault("forked_from", None)
        return {
            "id": row["id"], "bot_id": row["bot_id"], "user_id": row["user_id"],
            "started_at": str(row.get("started_at") or ""),
            "ended_at": str(row["ended_at"]) if row.get("ended_at") else None,
            "archived_at": str(row["archived_at"]) if row.get("archived_at") else None,
            "last_activity_at": row.get("last_activity_at"),
            "first_message_at": row.get("first_message_at"),
            "message_count": int(row.get("message_count") or 0),
            "turn_count": int(row.get("turn_count") or 0),
            "first_user_message": row.get("first_user_message"),
            "last_user_message": row.get("last_user_message"),
            "last_assistant_message": row.get("last_assistant_message"),
            "last_assistant_at": row.get("last_assistant_at"),
            "summary": row.get("summary"), "status": row["status"],
            "session_metadata": meta,
        }

    @router.get("/v1/sessions")
    async def sessions(
        request: Request, bot_id: str, user_id: str | None = None,
        user: str | None = None, status: Literal["active", "archived"] | None = None,
        limit: int = Query(50, ge=1, le=200),
    ):
        bot_id, user_id = scope(request, bot_id, user_id or user or "")
        from ..mcp_server.storage import get_storage

        rows = await get_storage().list_sessions(
            bot_id=bot_id, user_id=user_id, status=status, limit=limit,
            include_deleted=False,
        )
        return {"bot_id": bot_id, "user_id": user_id,
                "sessions": [session_info(row) for row in rows], "total_count": len(rows)}

    @router.get("/v1/sessions/{session_id}/details")
    async def session_details(
        request: Request, session_id: str, bot_id: str, user_id: str | None = None,
        user: str | None = None,
        limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0),
    ):
        bot_id, user_id = scope(request, bot_id, user_id or user or "")
        await owned_session(session_id, bot_id, user_id)
        from ..service.routes.session_continuity import session_details as load_details

        return await load_details(session_id, bot_id, user_id, offset, limit)

    @router.get("/v1/history")
    async def history(
        request: Request, bot_id: str, user_id: str,
        session_id: str | None = None, before: str | None = None,
        after: str | None = None, limit: int = Query(50, ge=1, le=200),
    ):
        bot_id, user_id = scope(request, bot_id, user_id)
        if session_id:
            await owned_session(session_id, bot_id, user_id)
        from ..service.routes.history_pages import get_history

        return get_history(bot_id=bot_id, user_id=user_id, session_id=session_id,
                           before=before, after=after, limit=limit)

    @router.get("/v1/history/around")
    async def history_around(
        request: Request, bot_id: str, user_id: str, message_id: str,
        session_id: str | None = None,
        before: int = Query(30, ge=0, le=200), after: int = Query(10, ge=0, le=200),
    ):
        bot_id, user_id = scope(request, bot_id, user_id)
        if session_id:
            await owned_session(session_id, bot_id, user_id)
        from ..service.routes.history_pages import get_history_around

        return get_history_around(bot_id=bot_id, user_id=user_id, message_id=message_id,
                                  session_id=session_id, before=before, after=after)

    @router.get("/v1/ws")
    async def events(
        request: Request, bot_id: list[str] = Query(default=[]),
        user_id: str | None = None, consumer_id: str | None = None,
        config_only: bool = False,
    ):
        require_preview(request)
        if config_only:
            if bot_id or user_id:
                raise HTTPException(status_code=400, detail="Config-only Preview events cannot use a user scope")
        else:
            if not bot_id or not user_id:
                raise HTTPException(status_code=400, detail="Preview events require bot and user scopes")
            for bot in bot_id:
                scope(request, bot, user_id)
        if not consumer_id:
            raise HTTPException(status_code=400, detail="Preview events require a consumer id")
        from ..service.routes.event_stream import event_stream

        return await event_stream(bot_id=bot_id, user_id=user_id, consumer_id=consumer_id,
                                  config_only=config_only)

    @router.get("/v1/turn-logs")
    async def turn_logs(
        request: Request, bot_id: str, user_id: str,
        active_only: bool = False, status: str | None = None,
        since_hours: int = Query(168, ge=1, le=168),
        limit: int = Query(100, ge=1, le=200), offset: int = Query(0, ge=0),
    ):
        bot_id, user_id = scope(request, bot_id, user_id)
        from ..service.routes.turn_logs import list_turn_logs

        return list_turn_logs(
            bot_id=bot_id, user_id=user_id, model=None, request_id=None,
            status=status, active_only=active_only, stream=None, has_tools=None,
            since_hours=since_hours, limit=limit, offset=offset,
        )

    @router.get("/v1/turn-logs/{turn_id}")
    async def turn_detail(request: Request, turn_id: str, bot_id: str, user_id: str):
        bot_id, user_id = scope(request, bot_id, user_id)
        from ..service.dependencies import get_turn_log_store
        from ..service.routes.turn_logs import get_turn_log

        row = get_turn_log_store().get_turn(turn_id)
        if row is None or row.bot_id != bot_id or row.user_id != user_id:
            raise HTTPException(status_code=404, detail="Preview turn not found")
        return get_turn_log(turn_id)

    @router.get("/v1/tool-calls")
    async def tool_calls(
        request: Request, bot_id: str, user_id: str,
        message_id: str | None = None, message_ids: list[str] | None = Query(None),
        after: float | None = None, before: float | None = None,
        since_hours: int = Query(168, ge=1, le=168),
        limit: int = Query(200, ge=1, le=1000),
    ):
        bot_id, user_id = scope(request, bot_id, user_id)
        from ..service.routes.turn_logs import get_tool_call_events

        return get_tool_call_events(
            bot_id=bot_id, user_id=user_id, message_id=message_id,
            message_ids=message_ids, after=after, before=before,
            since_hours=since_hours, limit=limit,
        )

    @router.post("/v1/chat/abort")
    async def abort(request: Request, body: PreviewAbortRequest, bot_id: str, user_id: str):
        bot_id, user_id = scope(request, bot_id, user_id)
        from ..service.dependencies import get_service
        from ..service.turn_abort import abort_turn

        service = get_service()
        row = service._turn_log_store.get_turn(body.turn_id)
        if row is None or row.bot_id != bot_id or row.user_id != user_id:
            raise HTTPException(status_code=404, detail="Preview turn not found")
        detail = await abort_turn(service, row, source=body.source,
                                  peer=request.client.host if request.client else None)
        return {"ok": True, "detail": detail, "turn_id": row.id, "queue_hold": None}

    app.include_router(router)
