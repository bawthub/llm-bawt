"""Live timeline projection for persisted mid-turn messages (user or bot)."""
from __future__ import annotations

import asyncio
import logging

from sqlalchemy import text

from ..entity_presentation import EntityPresentationResolver
from ..message_authorship import AuthorReference

log = logging.getLogger(__name__)


class SteerTimeline:
    """Keep retry anchors stable and publish the canonical row to its recipient."""

    def __init__(self, service, *, bot_id: str, user_id: str, turn_id: str, message_id: str):
        self.engine = service._turn_log_store.engine
        self.bot_id = bot_id
        self.user_id = user_id
        self.turn_id = turn_id
        self.message_id = message_id

    def _message(self) -> dict | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT id, content, timestamp, session_id, "
                    "author_entity_type, author_entity_id, "
                    "interrupt_source_message_id, interrupt_content_offset "
                    "FROM messages WHERE bot_id=:bot_id AND id=:message_id"
                ),
                {"bot_id": self.bot_id, "message_id": self.message_id},
            ).mappings().first()
        return dict(row) if row is not None else None

    async def content_offset(self, subscriber, requested: int | None) -> int | None:
        existing = await asyncio.to_thread(self._message)
        if existing and existing.get("interrupt_content_offset") is not None:
            # The deterministic steer RPC can be retried after the bot has said
            # more. Never move an already-persisted message further down its turn.
            return existing["interrupt_content_offset"]
        if requested is not None:
            return requested
        # Read this turn's own newest text position. A full snapshot needs the
        # turn's start event, which long turns lose to the capped stream; that
        # left inter-bot steers unanchored and rendered below the reply
        # (TASK-1509). Offsets are UTF-16 units, the unit the UI slices by.
        offset = await subscriber.turn_text_offset(self.bot_id, self.user_id, self.turn_id)
        if offset is not None:
            return offset
        # Do not attach a message to a different turn's text or guess from the
        # often-empty in-flight turn-log response. It can still render top-level.
        log.warning("No live timeline anchor for steer %s in %s", self.message_id, self.turn_id)
        return None

    async def publish(self, subscriber) -> None:
        """Publishing is best effort; it must never undo successful persistence."""
        try:
            message = await asyncio.to_thread(self._message)
            if not message:
                raise RuntimeError("Persisted steer message was not found")
            author = AuthorReference(
                message["author_entity_type"], message["author_entity_id"]
            )
            presentations = await asyncio.to_thread(
                EntityPresentationResolver(self.engine).resolve_many_safe, [author]
            )
            event = {
                "_type": "steer_message",
                "bot_id": self.bot_id,
                "user_id": self.user_id,
                "turn_id": self.turn_id,
                "message_id": self.message_id,
                "session_id": message["session_id"],
                "content": message["content"],
                "author": presentations[(author.entity_type, author.entity_id)],
                "interrupt_source_message_id": message["interrupt_source_message_id"],
                "interrupt_content_offset": message["interrupt_content_offset"],
                "ts": message["timestamp"],
            }
            if not await subscriber.publish_tool_event(self.bot_id, self.user_id, event):
                log.warning("Could not publish timeline event for steer %s", self.message_id)
        except Exception:
            log.exception("Could not publish timeline event for steer %s", self.message_id)
