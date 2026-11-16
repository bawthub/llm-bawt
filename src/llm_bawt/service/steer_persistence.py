"""Persist an already accepted input without sending another bridge command.

The destination is validated before acceptance and passed unchanged. This
writer never resolves a fresh active session or steers the SDK.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from ..message_authorship import AuthorReference

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SteerDestination:
    bot_id: str
    user_id: str
    turn_id: str
    source_message_id: str
    session_id: str


async def persist_accepted_input(
    service, *,
    destination: SteerDestination,
    message_id: str,
    message: str,
    author: AuthorReference,
    subscriber,
    persist_anchor,
    content_offset: int | None = None,
    attachment_refs: list[dict] | None = None,
    resolve_offset: bool = True,
) -> bool:
    """Save/repair history for the fixed message ID, then publish best effort.

    The caller owns acceptance and claim-token validation. Returning True says
    only that history and any required anchor were saved, never that this
    function obtained SDK acceptance or acknowledged a durable delivery.
    Inbox repair supplies its durable offset (including an explicit unknown)
    with resolve_offset=False; it never samples a later run's text position.
    """
    persisted = False
    target = destination
    try:
        memory_client = service.get_memory_client(target.bot_id, target.user_id)
        if memory_client is None:
            raise RuntimeError("Memory client unavailable")
        from .steer_timeline import SteerTimeline

        timeline = SteerTimeline(
            service, bot_id=target.bot_id, user_id=target.user_id,
            turn_id=target.turn_id, message_id=message_id,
        )
        offset = await timeline.content_offset(subscriber, content_offset) if resolve_offset else content_offset
        await asyncio.to_thread(
            memory_client.add_message,
            "user", message, target.session_id, None, message_id,
            attachment_refs, None, author.entity_type, author.entity_id,
        )
        if offset is not None:
            anchored = await asyncio.to_thread(
                persist_anchor, service, bot_id=target.bot_id,
                message_id=message_id, source_message_id=target.source_message_id,
                content_offset=offset,
            )
            if not anchored:
                raise RuntimeError("Persisted steer message was not found for anchor update")
        persisted = True
        await timeline.publish(subscriber)
    except Exception:
        log.exception(
            "chat.steer accepted but persistence failed: turn=%s message=%s",
            target.turn_id, message_id,
        )
    return persisted
