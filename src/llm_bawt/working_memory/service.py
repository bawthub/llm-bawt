"""The two entry points onto the store: a bot's trusted turn and Nick's REST path.

Both run every operation through one :class:`WorkingMemoryPolicy` (design 7).
A bot's identity comes only from its signed turn capability; the owner path
names the namespace explicitly. Protected bots are refused before the store is
touched, and the bot path re-checks opt-in under the namespace lock.
"""
from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..task_turn_context import TaskTurnContext
from .contracts import WorkingMemoryError, normalize_create
from .policy import WorkingMemoryPolicy, is_protected
from .store import Actor, WorkingMemoryStore
from .task_links import TaskCatalog, TaskInfo

logger = logging.getLogger(__name__)


class WorkingMemoryService:
    def __init__(self, engine, catalog: TaskCatalog | None = None) -> None:
        self.store = WorkingMemoryStore(engine)
        self.policy = WorkingMemoryPolicy(engine)
        self.catalog = catalog or TaskCatalog()

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
        self.policy.require_tools(context.bot_id)  # before any namespace read
        self.reconcile(context.bot_id, context.user_id)
        return self.store.list_items(
            context.bot_id, context.user_id, include_archived=include_archived, authorize=gate,
        )

    def reconcile(self, bot_id: str, user_id: str) -> None:
        """Bounded lazy task refresh; failure leaves the last verified snapshot unchanged."""
        if not self.policy.describe(bot_id).tools_allowed:
            return
        gate = lambda conn: self.policy.require_tools(bot_id, conn)
        try:
            self.store.prepare_links(bot_id, user_id, self.catalog.get, authorize=gate)
            snapshot = self.store.render_snapshot(bot_id, user_id)
            links = [r for r in snapshot["items"] if r["kind"] == "initiative_link"]
            ready = [row for row in links[:20] if row["canonical_task_id"]]
            # The bounded fetches run off the namespace/session locks; a slow
            # BawtHub cannot serially stall twenty task lookups on each turn.
            with ThreadPoolExecutor(max_workers=5) as pool:
                fetches = [(row, pool.submit(self.catalog.get, row["canonical_task_id"])) for row in ready]
                for row, future in fetches:
                    try:
                        info = future.result()
                        self.store.refresh_task(bot_id, user_id, row["id"], info,
                                                expected_revision=row["revision"], authorize=gate)
                    except Exception as exc:  # fail unchanged for one unreachable task
                        logger.warning("task link %s reconciliation skipped: %s", row["id"], exc)
        except Exception as exc:
            logger.warning("task links reconciliation unavailable: %s", exc)

    def link_associated(self, context: TaskTurnContext, task_id: str) -> dict[str, Any]:
        """Best-effort *after* a signed, successful BawtHub association."""
        gate = self._bot_gate(context)
        self.policy.require_tools(context.bot_id)  # private/off before task or namespace reads
        info = self.catalog.get(task_id)
        if info.id != task_id:
            raise WorkingMemoryError("task_unavailable", "associated task identity changed")
        self.store.prepare_links(context.bot_id, context.user_id, self.catalog.get, authorize=gate)
        if info.status in ("COMPLETED", "CANCELLED") or (
                info.assignee is not None and info.assignee.lower() != context.bot_id.lower()):
            return {"state": "task_not_active", "taskId": task_id}
        return self.store.create(
            context.bot_id, context.user_id,
            {"kind": "initiative_link", "objective": info.title[:160],
             "initiative_task_id": info.id, "status": "active"},
            idempotency_key=str(uuid.uuid4()), actor=Actor("bot", context.bot_id.lower()),
            authorize=gate, task=info, auto=True,
        )

    def bot_create(self, context: TaskTurnContext, fields: Mapping[str, Any], idempotency_key: Any) -> dict[str, Any]:
        gate = self._bot_gate(context)
        payload = normalize_create(fields)
        if payload["kind"] != "initiative_link":
            return self.store.create(context.bot_id, context.user_id, fields,
                                     idempotency_key=idempotency_key,
                                     actor=self._bot_actor(context), authorize=gate)
        self.policy.require_tools(context.bot_id)  # before task or namespace reads
        prior = self.store.replay_legacy_link(context.bot_id, context.user_id, fields,
                                              idempotency_key=idempotency_key, authorize=gate)
        if prior is not None:
            return prior
        info = self.catalog.get(payload["initiative_task_id"])
        self.store.prepare_links(context.bot_id, context.user_id, self.catalog.get, authorize=gate)
        return self.store.create(
            context.bot_id, context.user_id,
            {**fields, "initiative_task_id": info.id}, idempotency_key=idempotency_key,
            actor=self._bot_actor(context), authorize=gate, task=info,
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
        # Owner cleanup must work after opt-out. Resolve only this item's
        # legacy reference off-lock; never require the bot's tools gate or
        # migrate unrelated items before allowing its removal.
        ref = self.store.legacy_task_ref(bot_id, user_id, item_id)
        task = None
        if ref:
            try:
                task = self.catalog.get(ref)
            except WorkingMemoryError as exc:
                if exc.code not in ("task_unavailable", "invalid_request"):
                    raise
                # An unidentifiable hard purge leaves an unaudited purge row,
                # fencing future link creation in this namespace.
        return self.store.delete(bot_id, user_id, item_id, hard=hard,
                                 actor=Actor("user", str(user_id).strip()), legacy_task=task)


__all__ = ["WorkingMemoryService"]
