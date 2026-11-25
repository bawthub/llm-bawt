"""``playground_send``: send the user an artifact for their Playground inbox (TASK-2190).

The sender is the signed caller, never the ``bot_id`` argument alone: an
unverified caller or one for another bot is refused. The artifact goes to the
caller's user. Exactly one source: an existing media ``asset_id`` (its bytes
are copied, so the source stays untouched) or inline ``content`` with a
``format``.

``replaces`` revises an artifact this bot already sent to the same user
(TASK-2625): its content is swapped in place, an open viewer reloads it live,
and no new bell entry is made. An artifact in the user's Trash or deleted
forever is refused with an error; the bot sends a new one instead.
"""
from __future__ import annotations

import asyncio
from typing import Literal

from llm_bawt.bot_access import BotAccessError
from llm_bawt.integrations import notify as notify_module
from llm_bawt.playground import bell as playground_bell
from llm_bawt.playground.service import (
    PlaygroundError,
    get_playground,
    source_from_asset,
    source_from_inline,
)
from llm_bawt.playground.store import ReviseRefused, status_of

from .registry import mcp

MAX_TITLE = 200
MAX_NOTE = 300
MAX_ID = 64

_REVISE_REFUSALS = {
    "not_found": "replaces: no artifact with that id was sent by you to this user",
    "trashed": "replaces: the user moved that artifact to Trash; send a new artifact without replaces",
    "expired": "replaces: the user deleted that artifact; send a new artifact without replaces",
}


def _caller(bot_id: str, arguments: dict):
    from .bot_access_identity import trusted_caller
    try:
        caller = trusted_caller("playground_send", arguments)
    except BotAccessError:
        return "caller_unverified", None
    if caller is None or not getattr(caller, "user_id", None):
        return "caller_unverified", None
    if caller.bot_id != bot_id:
        return "caller_mismatch", None
    return None, caller


def _max_bytes() -> int:
    from llm_bawt.service.routes.uploads import max_file_upload_bytes
    return max_file_upload_bytes()


def _send(*, user_id: str, bot_id: str, title: str, note: str | None,
          asset_id: str | None, content: str | None, fmt: str | None,
          replaces: str | None = None) -> dict:
    playground = get_playground()
    if asset_id:
        from llm_bawt.media import get_media_store
        source = source_from_asset(get_media_store(), asset_id, _max_bytes())
    else:
        source = source_from_inline(content or "", fmt or "")
    if replaces:
        return playground.revise(replaces, user_id=user_id, bot_id=bot_id, title=title,
                                 note=note, source=source)
    return playground.send(user_id=user_id, bot_id=bot_id, title=title, note=note,
                           source=source)


async def _bell(row: dict) -> None:
    try:
        sender = await asyncio.to_thread(notify_module.sender_name, row["sender_bot_id"])
        async with playground_bell.bell_http() as http:
            await playground_bell.record_artifact_bell(
                http, artifact_id=row["id"], bot_id=row["sender_bot_id"],
                user_id=row["user_id"], sender=sender, title=row["title"])
    except Exception as error:  # the artifact is already stored
        playground_bell.logger.warning("playground bell %s not recorded: %s", row["id"],
                                       type(error).__name__)


@mcp.tool()
async def playground_send(
    bot_id: str,
    title: str,
    note: str | None = None,
    asset_id: str | None = None,
    content: str | None = None,
    format: Literal["html", "markdown", "text"] | None = None,
    replaces: str | None = None,
) -> dict:
    """Send the user an artifact; it shows as a card in this chat and in their Playground.

    To revise an artifact you sent, pass its id as ``replaces`` with the full new
    content: it updates in place, live if the user has it open, with no new
    notification. Iterate on one artifact this way instead of sending copies.

    Don't open, fetch or screenshot an artifact you sent. The chat card is the
    delivery, and opening it marks it read for the user.
    """
    # The signed caller check binds the call exactly as made, before cleanup.
    arguments = {"bot_id": bot_id, "title": title, "note": note, "asset_id": asset_id,
                 "content": content, "format": format}
    if replaces is not None:
        arguments["replaces"] = replaces
    bot_id = (bot_id or "").strip().lower()
    title = (title or "").strip()
    note = (note or "").strip() or None
    asset_id = (asset_id or "").strip() or None
    replaces = (replaces or "").strip() or None
    if not bot_id or len(bot_id) > 64:
        return {"error": "bot_id must be your bot slug"}
    if not title or len(title) > MAX_TITLE:
        return {"error": f"title must be 1-{MAX_TITLE} characters"}
    if note and len(note) > MAX_NOTE:
        return {"error": f"note must be at most {MAX_NOTE} characters"}
    if bool(asset_id) == bool(content):
        return {"error": "pass exactly one of asset_id or content"}
    if content and not format:
        return {"error": "format is required with content"}
    if replaces and len(replaces) > MAX_ID:
        return {"error": _REVISE_REFUSALS["not_found"]}
    refusal, caller = await asyncio.to_thread(_caller, bot_id, arguments)
    if refusal:
        return {"error": refusal}
    try:
        row = await asyncio.to_thread(
            _send, user_id=caller.user_id, bot_id=bot_id, title=title, note=note,
            asset_id=asset_id, content=content, fmt=format, replaces=replaces)
    except PlaygroundError as error:
        return {"error": str(error)}
    except ReviseRefused as refused:
        return {"error": _REVISE_REFUSALS.get(refused.code, _REVISE_REFUSALS["not_found"])}
    if not replaces:
        await _bell(row)
    return {"id": row["id"], "kind": row["kind"], "size_bytes": row["size_bytes"],
            "status": status_of(row), "revision": row.get("revision") or 1,
            "link": playground_bell.LINK,
            "title": row["title"], "note": row["note"], "sender_bot_id": row["sender_bot_id"],
            "notice": "Don't open, fetch or screenshot an artifact you sent. The chat card "
                      "is the delivery, and opening it marks it read for the user."}
