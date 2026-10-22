"""Request-scope admission for a bridge ``chat.send`` (TASK-1232 P2).

The command listener starts one task per command in stream order. A send
calls :func:`admit_send` as its first step, before any await, so it is
registered with the :class:`~agent_bridge.session_queue.SessionQueue` before
a ``chat.abort`` read after it can run. That abort then cancels the send;
an abort read *before* the send leaves a tombstone, and the send is refused
here before any side effect (reset, seed, model call).

The Claude bridge has its own admission (``claude_code_bridge.send_admission``)
because it parses the send first and owns an awaited preflight.

Run contract (TASK-1233, ``agent_bridge.run_contract``): for bridges using
this helper, :func:`admit_send` owns the run's two contract events. It
publishes ``RUN_ACCEPTED`` on admission and the run's one ``done`` sentinel
after ``run`` has finished and cleaned up, so the sentinel's teardown report
describes a run that has already stopped. ``run`` itself never publishes
``done``. A refused (tombstoned) send gets a ``done`` with nothing attempted.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .publisher import COMMANDS_STREAM
from .run_contract import UNATTEMPTED, TeardownReport
from .session_queue import Admission, RequestPhase, SessionQueue

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunOutcome:
    """How an admitted send's ``run`` ended, for its teardown report."""

    reached_running: bool
    """The send held the session lock, so its model work may have started."""
    cancelled: bool
    """``run`` ended by cancellation (Stop, or the bridge shutting down)."""

    def default_report(self) -> TeardownReport:
        """The report for a bridge with nothing more specific to say.

        ``run`` has returned or raised, so the bridge's own work for it has
        stopped. A run cancelled while it held the lock is the one case that
        proves nothing: the cancellation may have cut off cleanup of work that
        lives outside this task, so a bridge must supply an ``exit_probe`` to
        claim more than ``exited=False`` there.
        """
        return TeardownReport(exited=not (self.cancelled and self.reached_running))


ExitProbe = Callable[[RunOutcome], TeardownReport]


def default_exit_probe(outcome: RunOutcome) -> TeardownReport:
    return outcome.default_report()


async def admit_send(
    queue: SessionQueue,
    session_key: str,
    request_id: str,
    msg_id: str,
    async_redis: Any,
    *,
    consumer_group: str,
    publisher: Any,
    run: Callable[[], Awaitable[None]],
    exit_probe: ExitProbe = default_exit_probe,
) -> None:
    """Admit one send, run it, publish its terminal, and release it.

    A command without a request id has no run stream; it isn't registered
    and ``run`` rejects it. ``exit_probe`` is always called once per admitted
    run, so a bridge may drop per-run bookkeeping there.
    """
    if not request_id:
        await run()
        return
    admission = queue.accept(session_key, request_id)
    if admission is Admission.TOMBSTONED:
        logger.info("Refusing stopped send: request_id=%s session=%s", request_id, session_key)
        publisher.publish_run_done(request_id, teardown=UNATTEMPTED)
        await async_redis.xack(COMMANDS_STREAM, consumer_group, msg_id)
        return
    if admission is Admission.DUPLICATE:
        logger.warning(
            "Dropping duplicate send for a live request: request_id=%s session=%s",
            request_id, session_key,
        )
        await async_redis.xack(COMMANDS_STREAM, consumer_group, msg_id)
        return
    publisher.publish_run_accepted(request_id, session_key)
    cancelled = False
    try:
        await run()
    except BaseException as exc:
        cancelled = isinstance(exc, asyncio.CancelledError)
        raise
    finally:
        outcome = RunOutcome(
            reached_running=queue.request_phase(session_key, request_id) is RequestPhase.RUNNING,
            cancelled=cancelled,
        )
        queue.release(session_key, request_id)
        try:
            teardown = exit_probe(outcome)
        except Exception:
            logger.exception("Exit probe failed: request_id=%s", request_id)
            teardown = TeardownReport(exited=False)
        publisher.publish_run_done(request_id, teardown=teardown)


__all__ = ["ExitProbe", "RunOutcome", "admit_send", "default_exit_probe"]
