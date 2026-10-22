"""Request-scope admission for a bridge ``chat.send`` (TASK-1232 P2).

The command listener starts one task per command in stream order. A send
calls :func:`admit_send` as its first step, before any await, so it is
registered with the :class:`~agent_bridge.session_queue.SessionQueue` before
a ``chat.abort`` read after it can run. That abort then cancels the send;
an abort read *before* the send leaves a tombstone, and the send is refused
here before any side effect (reset, seed, model call).

The Claude bridge has its own admission (``claude_code_bridge.send_admission``)
because it parses the send first and owns an awaited preflight. Bridges using
this helper publish their refusal with the same terminal they already publish
for a send cancelled while queued: ``run_done``.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from .publisher import COMMANDS_STREAM
from .session_queue import Admission, SessionQueue

logger = logging.getLogger(__name__)


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
) -> None:
    """Admit one send, run it, and release its registration on every exit.

    A command without a request id isn't registered; ``run`` rejects it.
    """
    if not request_id:
        await run()
        return
    admission = queue.accept(session_key, request_id)
    if admission is Admission.TOMBSTONED:
        logger.info("Refusing stopped send: request_id=%s session=%s", request_id, session_key)
        publisher.publish_run_done(request_id)
        await async_redis.xack(COMMANDS_STREAM, consumer_group, msg_id)
        return
    if admission is Admission.DUPLICATE:
        logger.warning(
            "Dropping duplicate send for a live request: request_id=%s session=%s",
            request_id, session_key,
        )
        await async_redis.xack(COMMANDS_STREAM, consumer_group, msg_id)
        return
    try:
        await run()
    finally:
        queue.release(session_key, request_id)


__all__ = ["admit_send"]
