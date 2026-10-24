"""Leader-only side work driven by each dispatcher sweep (TASK-1327 step 0).

The delivery dispatcher's job is claiming and running deliveries. Everything
else its leader does on the same cadence lives here: new-session receipts and
their outbox, Nextcloud Talk replies, the quiesce control pump. A sweep calls
``run_pass`` once; ``stop`` tears the work down before leadership is released.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

Emit = Callable[[Any], Awaitable[None]]


class LeaderSidePumps:
    """Single owner of the dispatcher leader's non-claim pumps."""

    def __init__(self, service: Any, store: Any, *, talk_replies: Any = None,
                 quiesce_control: Any = None) -> None:
        self.service = service
        self.store = store
        self.talk_replies = talk_replies
        self.quiesce_control = quiesce_control
        self.talk_reply_task: asyncio.Task | None = None
        self._new_session_outbox = None

    @classmethod
    def build(cls, service: Any, store: Any) -> "LeaderSidePumps":
        from .quiesce_control import QuiesceControlRunner, build_pump
        from .talk_reply_outbox import TalkReplyOutbox

        engine = store.engine
        return cls(
            service, store,
            talk_replies=TalkReplyOutbox(engine) if engine is not None else None,
            # TASK-1233 B3: ask each quiesce-held running turn to stop.
            quiesce_control=QuiesceControlRunner(lambda: build_pump(service, engine)),
        )

    async def run_pass(self, emit: Emit) -> None:
        """One sweep's side work. Raises like the sweep it runs in."""
        if getattr(self.store, "engine", None) is not None:
            await self._settle_session_receipts(emit)
        if self.talk_replies is not None and (self.talk_reply_task is None or self.talk_reply_task.done()):
            task = asyncio.create_task(
                self.talk_replies.drain(self.store, self.service._turn_log_store),
                name="talk-reply-drain",
            )
            task.add_done_callback(_talk_reply_done)
            self.talk_reply_task = task
        if self.quiesce_control is not None:
            self.quiesce_control.maybe_start()

    async def _settle_session_receipts(self, emit: Emit) -> None:
        from .session_lifecycle.outbox import NewSessionOutbox
        from .session_lifecycle.policy_receipt import BotPolicyReceipt
        from .session_lifecycle.receipt import NewSessionReceipt

        engine = self.store.engine
        for delivery in await asyncio.to_thread(NewSessionReceipt.reconcile_typed, engine):
            await emit(delivery)
        for delivery in await asyncio.to_thread(BotPolicyReceipt.reconcile, engine):
            await emit(delivery)
        if self._new_session_outbox is None:
            self._new_session_outbox = NewSessionOutbox(engine)
        await self._new_session_outbox.drain(getattr(self.service, "_redis_subscriber", None))

    def running_tasks(self) -> list[asyncio.Task]:
        """Tasks the dispatcher cancels with its own workers on stop."""
        task = self.talk_reply_task
        return [task] if task is not None and not task.done() else []

    async def stop(self) -> None:
        """Stop the self-managed runners (the caller cancels ``running_tasks``)."""
        if self.quiesce_control is not None:
            await self.quiesce_control.stop()


def _talk_reply_done(task: asyncio.Task) -> None:
    if not task.cancelled() and task.exception():
        logger.error("Talk reply sweep failed", exc_info=task.exception())
