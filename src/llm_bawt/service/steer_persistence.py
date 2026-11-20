"""Persist an already accepted input without sending another bridge command.

The destination is validated before acceptance and passed unchanged. This
writer never resolves a fresh active session or steers the SDK.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
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


def persist_input_history(
    service, *, destination: SteerDestination, message_id: str, message: str,
    author: AuthorReference, persist_anchor, content_offset: int | None,
    attachment_refs: list[dict] | None,
) -> bool:
    """Blocking message + anchor mutation, optionally inside a caller's fence."""
    memory_client = service.get_memory_client(destination.bot_id, destination.user_id)
    if memory_client is None:
        raise RuntimeError("Memory client unavailable")
    memory_client.add_message(
        "user", message, destination.session_id, None, message_id,
        attachment_refs, None, author.entity_type, author.entity_id,
    )
    # The source identifies an accepted steer even if its text position is unknown.
    if not persist_anchor(
        service, bot_id=destination.bot_id, message_id=message_id,
        source_message_id=destination.source_message_id, content_offset=content_offset,
    ):
        raise RuntimeError("Persisted steer message was not found for source update")
    return True


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
    history_fence: Callable[[Callable[[], bool]], bool] | None = None,
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
        from .steer_timeline import SteerTimeline

        timeline = SteerTimeline(
            service, bot_id=target.bot_id, user_id=target.user_id,
            turn_id=target.turn_id, message_id=message_id,
        )
        offset = await timeline.content_offset(subscriber, content_offset) if resolve_offset else content_offset

        def write():
            return persist_input_history(
                service, destination=target, message_id=message_id, message=message,
                author=author, persist_anchor=persist_anchor, content_offset=offset,
                attachment_refs=attachment_refs,
            )

        # The blocking fence and both mutations run in the SAME worker. Cancelling
        # this coroutine cannot release a DB lock while its writer still runs.
        persisted = await asyncio.to_thread(history_fence, write) if history_fence else await asyncio.to_thread(write)
        if persisted:
            await timeline.publish(subscriber)
    except Exception:
        log.exception(
            "chat.steer accepted but persistence failed: turn=%s message=%s",
            target.turn_id, message_id,
        )
    return persisted
