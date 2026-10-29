"""Who actually spoke a stored message, for the maintenance pipeline.

Inter-bot deliveries are persisted with provider ``role="user"`` (the SDK needs
a user turn), but their durable authorship (``author_entity_type='bot'``,
written from the delivery record) says a bot wrote them. Fact extraction and
summarization must honor that authorship, never the role alone, and never
content that merely claims to come from a bot.

Rows with NULL authorship keep their role-based meaning: the audited backfill
already resolved every legacy row it could prove, so NULL on a user row is the
pre-existing (role-trusting) behaviour, not new inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

USER = "user"
ASSISTANT = "assistant"
BOT = "bot"


@dataclass(frozen=True)
class Speaker:
    kind: str  # USER | ASSISTANT | BOT | the raw role for anything else
    bot_id: str | None = None

    @property
    def is_human_user(self) -> bool:
        return self.kind == USER


def _field(msg: Any, name: str) -> Any:
    if isinstance(msg, Mapping):
        return msg.get(name)
    return getattr(msg, name, None)


def speaker_of(msg: Any) -> Speaker:
    """Classify a message dict/object by role plus durable authorship."""
    role = str(_field(msg, "role") or "user").lower()
    if role == ASSISTANT:
        return Speaker(ASSISTANT)
    if role != USER:
        return Speaker(role)
    if str(_field(msg, "author_entity_type") or "").lower() == BOT:
        bot_id = str(_field(msg, "author_entity_id") or "").strip().lower()
        return Speaker(BOT, bot_id or "unknown")
    return Speaker(USER)


def summary_label(speaker: Speaker) -> str:
    """Transcript label for summarization input."""
    if speaker.kind == USER:
        return "User"
    if speaker.kind == ASSISTANT:
        return "Assistant"
    if speaker.kind == BOT:
        return f"Bot '{speaker.bot_id}'"
    return speaker.kind


def human_user_turns_present(messages: list[Any]) -> bool:
    return any(speaker_of(m).is_human_user for m in messages)


def extraction_input(messages: list[Any]) -> list[Any]:
    """Messages fact extraction may see: bot-origin deliveries are dropped.

    Extraction stores facts about the user; a delivery is another bot's speech,
    so it is neither a user fact nor useful context worth paying for.
    """
    return [m for m in messages if speaker_of(m).kind != BOT]
