"""Leader-only side work driven by each dispatcher sweep (TASK-1327 step 0).

The delivery dispatcher's job is claiming and running deliveries. Everything
else its leader does on the same cadence lives here: new-session receipts and
their outbox, Nextcloud Talk replies, the quiesce control producer and the
cancelled-delivery sender notices (TASK-1327). A sweep calls
``run_pass`` once; ``stop`` tears the work down before leadership is released.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

Emit = Callable[[Any], Awaitable[None]]

# Never later than the dispatcher's own sweep would come round anyway.
CONTROL_RECHECK_CAP_SECONDS = 15.0


class LeaderSidePumps:
    """Single owner of the dispatcher leader's non-claim pumps."""

    def __init__(self, service: Any, store: Any, *, talk_replies: Any = None,
                 quiesce_control: Any = None, cancel_notices: Any = None) -> None:
        self.service = service
        self.store = store
        self.talk_replies = talk_replies
        self.quiesce_control = quiesce_control
        self.cancel_notices = cancel_notices
        self.talk_reply_task: asyncio.Task | None = None
        self._new_session_outbox = None
        self._control_recheck: asyncio.TimerHandle | None = None

    @classmethod
    def build(cls, service: Any, store: Any) -> "LeaderSidePumps":
        from .delivery_cancel_notice import CancelNoticePump
        from .quiesce_control import build_producer
        from .talk_reply_outbox import TalkReplyOutbox

        engine = store.engine
        return cls(
            service, store,
            talk_replies=TalkReplyOutbox(engine) if engine is not None else None,
            # TASK-1431 S3: link each quiesce-held running run to its control row.
            quiesce_control=build_producer(store),
            cancel_notices=CancelNoticePump(store) if engine is not None else None,
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
            await self._produce_controls()
        if self.cancel_notices is not None:
            await self._send_cancel_notices(emit)

    async def _produce_controls(self) -> None:
        """Link held runs to control rows; wake the dispatcher for what needs it.

        A new row is claimed at once. An unpinned member, or a control row
        backing off, gets one timed wake (never later than the sweep), so a
        retry keeps its own cadence without a task of its own. A failed pass
        is logged and retried on the next sweep; it never skips the pumps
        after it.
        """
        try:
            result = await asyncio.to_thread(self.quiesce_control.run_pass)
        except Exception:
            logger.exception("Quiesce control producer pass failed")
            return
        dispatcher = getattr(self.service, "_inter_bot_dispatcher", None)
        if dispatcher is None:
            return
        if result.enqueued:
            dispatcher.wake()
        if result.retry_in is not None:
            self._cancel_control_recheck()
            delay = min(max(result.retry_in, 0.5), CONTROL_RECHECK_CAP_SECONDS)
            self._control_recheck = asyncio.get_running_loop().call_later(delay, dispatcher.wake)

    def _cancel_control_recheck(self) -> None:
        handle, self._control_recheck = self._control_recheck, None
        if handle is not None:
            handle.cancel()

    async def _send_cancel_notices(self, emit: Emit) -> None:
        notices = (await asyncio.to_thread(self.cancel_notices.run_pass)).notified
        for notice in notices:
            await emit(notice)
        dispatcher = getattr(self.service, "_inter_bot_dispatcher", None)
        if notices and dispatcher is not None:
            dispatcher.wake()  # claim the new notices now, not on the next interval

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
        """Stop the self-managed timers (the caller cancels ``running_tasks``)."""
        self._cancel_control_recheck()


def _talk_reply_done(task: asyncio.Task) -> None:
    if not task.cancelled() and task.exception():
        logger.error("Talk reply sweep failed", exc_info=task.exception())
