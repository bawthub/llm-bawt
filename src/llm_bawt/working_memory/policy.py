"""One server-side policy for who may use working memory (design section 7).

Opt-in, protected status and the durable backend are read from the bot's
profile and bot-scoped runtime setting, never from model input. Protected bots
are refused before any of their rows is read.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from sqlalchemy import text

from ..bot_types import durable_agent_backend
from ..service.jobs.idle_context_candidates import PROTECTED_BOT_IDS
from .contracts import WorkingMemoryError

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


def is_protected(bot_id: str) -> bool:
    return bot_id.strip().lower() in PROTECTED_BOT_IDS


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
        slug = bot_id.strip().lower()
        if is_protected(slug):
            return BotPolicy(slug, True, False, False, None)
        if conn is None:
            with self.engine.connect() as own:
                return self.describe(slug, own)
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

    @staticmethod
    def require_owner_access(bot_id: str) -> None:
        """REST owner path: protected bots are refused; disabled bots stay manageable."""
        if is_protected(bot_id):
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
