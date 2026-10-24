"""The two entry points onto the store: a bot's trusted turn and Nick's REST path.

Both run every operation through one :class:`WorkingMemoryPolicy` (design 7).
A bot's identity comes only from its signed turn capability; the owner path
names the namespace explicitly. Protected bots are refused before the store is
touched, and the bot path re-checks opt-in under the namespace lock.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..task_turn_context import TaskTurnContext
from .contracts import WorkingMemoryError
from .policy import WorkingMemoryPolicy, is_protected
from .store import Actor, WorkingMemoryStore


class WorkingMemoryService:
    def __init__(self, engine) -> None:
        self.store = WorkingMemoryStore(engine)
        self.policy = WorkingMemoryPolicy(engine)

    # -- bot, through the trusted turn capability (6.1) ----------------------

    def _bot_gate(self, context: TaskTurnContext):
        if is_protected(context.bot_id, self.policy.engine):
            raise WorkingMemoryError("protected_bot", "working memory is not available for this bot")
        return lambda conn: self.policy.require_tools(context.bot_id, conn)

    @staticmethod
    def _bot_actor(context: TaskTurnContext) -> Actor:
        return Actor("bot", context.bot_id.strip().lower(), context.session_id, context.turn_id)

    def bot_list(self, context: TaskTurnContext, *, include_archived: bool = False) -> dict[str, Any]:
        gate = self._bot_gate(context)
        return self.store.list_items(
            context.bot_id, context.user_id, include_archived=include_archived, authorize=gate,
        )

    def bot_create(self, context: TaskTurnContext, fields: Mapping[str, Any], idempotency_key: Any) -> dict[str, Any]:
        gate = self._bot_gate(context)
        return self.store.create(
            context.bot_id, context.user_id, fields, idempotency_key=idempotency_key,
            actor=self._bot_actor(context), authorize=gate,
        )

    def bot_update(self, context: TaskTurnContext, item_id: Any, expected_revision: Any,
                   changes: Mapping[str, Any]) -> dict[str, Any]:
        gate = self._bot_gate(context)
        return self.store.update(
            context.bot_id, context.user_id, item_id, expected_revision=expected_revision,
            changes=changes, actor=self._bot_actor(context), authorize=gate,
        )

    # -- owner, over REST (6.2) -----------------------------------------------
    # Disabled bots stay readable and editable so Nick can clean up (7).

    def owner_list(self, bot_id: str, user_id: str) -> dict[str, Any]:
        self.policy.require_owner_access(bot_id)
        return self.store.list_items(bot_id, user_id, include_archived=True, include_removed=True)

    def owner_update(self, bot_id: str, user_id: str, item_id: Any, expected_revision: Any,
                     changes: Mapping[str, Any]) -> dict[str, Any]:
        self.policy.require_owner_access(bot_id)
        return self.store.update(
            bot_id, user_id, item_id, expected_revision=expected_revision, changes=changes,
            actor=Actor("user", str(user_id).strip()),
        )

    def owner_delete(self, bot_id: str, user_id: str, item_id: Any, *, hard: bool) -> dict[str, Any]:
        self.policy.require_owner_access(bot_id)
        return self.store.delete(bot_id, user_id, item_id, hard=hard, actor=Actor("user", str(user_id).strip()))


__all__ = ["WorkingMemoryService"]
