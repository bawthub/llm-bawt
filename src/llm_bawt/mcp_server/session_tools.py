"""MCP tools for durable conversation sessions."""

from __future__ import annotations

import logging

from .server import _get_storage, mcp

logger = logging.getLogger(__name__)


@mcp.tool(name="sessions_close")
async def close_session(session_id: str, bot_id: str = "default") -> bool:
    """Close a session by archiving it without deleting its messages."""
    return await _get_storage().close_session(session_id=session_id, bot_id=bot_id)


@mcp.tool(name="sessions_get")
async def get_session(session_id: str, bot_id: str = "default") -> dict | None:
    """Get one durable session record by id."""
    return await _get_storage().get_session(session_id=session_id, bot_id=bot_id)


@mcp.tool(name="sessions_list")
async def list_sessions(
    bot_id: str = "default",
    since: float | str | None = None,
    status: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """List sessions for a bot, newest first."""
    return await _get_storage().list_sessions(
        bot_id=bot_id, since=since, status=status, limit=limit,
    )


@mcp.tool(name="sessions_get_active")
async def get_active_session(
    bot_id: str = "default", user_id: str | None = None,
) -> dict | None:
    """Get the most-recent active session for a (bot, user)."""
    return await _get_storage().get_active_session(bot_id=bot_id, user_id=user_id)


@mcp.tool(name="sessions_get_or_create_active")
async def get_or_create_active_session(
    bot_id: str = "default", user_id: str | None = None,
) -> str:
    """Return the active thread id for a (bot, user), creating one if absent."""
    return await _get_storage().get_or_create_active_session(
        bot_id=bot_id, user_id=user_id,
    )


@mcp.tool(name="sessions_rotate")
async def rotate_session(
    bot_id: str = "default", user_id: str | None = None,
    key: str = "", accept_pointer: bool = False,
) -> str:
    """Archive the active session; accepted /new stamps the bot pointer."""
    return await _get_storage().rotate_session(
        bot_id=bot_id, user_id=user_id, accept_pointer=accept_pointer,
        **({"idempotency_key": key} if key else {}),
    )
