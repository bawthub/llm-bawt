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

from ..mcp_result_envelope import is_mcp_result_envelope

USER = "user"
ASSISTANT = "assistant"
BOT = "bot"
TOOL = "tool"
SYSTEM_AUTHOR_TYPE = "system"


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
    author_type, author_id = _field(msg, "author_entity_type"), _field(msg, "author_entity_id")
    author = _field(msg, "author")  # AuthorReference object or its to_dict()
    if author is not None and author_type is None:
        author_type, author_id = _field(author, "entity_type"), _field(author, "entity_id")
    if str(author_type or "").lower() == BOT:
        return Speaker(BOT, str(author_id or "").strip().lower() or "unknown")
    # TASK-1807: app-owned approval-continuation turns (tool results, harness
    # approval outcomes) are stamped with a first-party system author.
    if str(author_type or "").lower() == SYSTEM_AUTHOR_TYPE:
        return Speaker(TOOL)
    # The continuation writer historically stamped these user-role rows with
    # user authorship. Recognize its full app-owned frame so old rows are safe
    # too; ordinary content claiming to be a bot/tool remains a user turn.
    if is_mcp_result_envelope(_field(msg, "content")):
        return Speaker(TOOL)
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
    """Exclude inter-bot speech and internal tool results from user facts."""
    return [m for m in messages if speaker_of(m).kind not in (BOT, TOOL)]
