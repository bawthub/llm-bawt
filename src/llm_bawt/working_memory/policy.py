"""One server-side policy for who may use working memory (design section 7).

Opt-in, protected status and the durable backend are read from the bot's
profile, its tags and bot-scoped runtime setting, never from model input.
Private bots (BOT_TAGS.md §3) are refused before any of their rows is read;
an unreadable tag snapshot treats every bot as private.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from sqlalchemy import text

from ..bot_tags import BehaviorTag, BotTags
from ..bot_types import durable_agent_backend
from .contracts import WorkingMemoryError

logger = logging.getLogger(__name__)

SETTING_KEY = "working_memory_enabled"
SUPPORTED_BACKEND = "claude-code"


@dataclass(frozen=True)
class BotPolicy:
    bot_id: str
    protected: bool
    exists: bool
    enabled: bool
    backend: str | None

    @property
    def tools_allowed(self) -> bool:
        return not self.protected and self.exists and self.enabled and self.backend == SUPPORTED_BACKEND


def is_protected(bot_id: str, engine=None, *, conn=None) -> bool:
    """Whether ``bot_id`` is private; a failed tag read answers True.

    ``conn`` reads the tags on the caller's connection (never a second one).
    """
    return BotTags(engine).has(bot_id, BehaviorTag.PRIVATE, conn=conn)


def _enabled(raw: str | None) -> bool:
    """Only a stored JSON ``true`` enables; anything else, or unreadable, is off."""
    if raw is None:
        return False
    try:
        return json.loads(raw) is True
    except (TypeError, ValueError):
        return False


class WorkingMemoryPolicy:
    """Operation-specific gates over one engine (design 7, operation table)."""

    def __init__(self, engine) -> None:
        self.engine = engine

    def describe(self, bot_id: str, conn=None) -> BotPolicy:
        """The bot's policy; tags and rows are read on one connection, tags first."""
        slug = bot_id.strip().lower()
        if conn is None:
            try:
                own = self.engine.connect()
            except Exception:
                logger.warning("working memory policy unreadable for %s; treating it as private", slug)
                return BotPolicy(slug, True, False, False, None)
            with own:
                return self.describe(slug, own)
        if is_protected(slug, conn=conn):
            return BotPolicy(slug, True, False, False, None)
        profile = conn.execute(text(
            "SELECT agent_backend, harness FROM bot_profiles WHERE slug = :slug"
        ), {"slug": slug}).mappings().first()
        raw = conn.execute(text(
            "SELECT value_json FROM runtime_settings"
            " WHERE scope_type = 'bot' AND scope_id = :slug AND key = :key"
        ), {"slug": slug, "key": SETTING_KEY}).scalar()
        backend = durable_agent_backend(SimpleNamespace(**profile)) if profile else None
        return BotPolicy(slug, False, profile is not None, _enabled(raw), backend)

    def require_tools(self, bot_id: str, conn=None) -> BotPolicy:
        """MCP tools: refused for protected, unknown, not opted-in or non-claude-code bots."""
        policy = self.describe(bot_id, conn)
        if policy.protected:
            raise WorkingMemoryError("protected_bot", "working memory is not available for this bot")
        if not policy.tools_allowed:
            raise WorkingMemoryError(
                "not_enabled",
                "working memory is not enabled for this bot (bot-scoped working_memory_enabled on a claude-code bot)",
            )
        return policy

    def require_owner_access(self, bot_id: str) -> None:
        """REST owner path: protected bots are refused; disabled bots stay manageable."""
        if is_protected(bot_id, self.engine):
            raise WorkingMemoryError("protected_bot", "working memory is not available for this bot")

    def validate_setting_write(self, scope_type: str, scope_id: str, key: str, value: Any) -> None:
        """Refuse an invalid ``working_memory_enabled`` write before any row is applied (7)."""
        if key != SETTING_KEY:
            return
        if scope_type != "bot":
            raise WorkingMemoryError("invalid_setting", f"{SETTING_KEY} is bot-scoped only; a global value is refused")
        if not isinstance(value, bool):
            raise WorkingMemoryError("invalid_setting", f"{SETTING_KEY} must be true or false")
        if not value:
            return
        policy = self.describe(scope_id)
        if policy.protected:
            raise WorkingMemoryError("invalid_setting", f"{SETTING_KEY} cannot be enabled for a protected bot")
        if not policy.exists or policy.backend != SUPPORTED_BACKEND:
            raise WorkingMemoryError(
                "invalid_setting",
                f"{SETTING_KEY} can only be enabled for a bot on the {SUPPORTED_BACKEND} backend",
            )


__all__ = ["SETTING_KEY", "SUPPORTED_BACKEND", "BotPolicy", "WorkingMemoryPolicy", "is_protected"]
