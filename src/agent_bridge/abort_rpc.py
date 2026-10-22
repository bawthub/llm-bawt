"""Bridge side of the ``chat.abort`` RPC (TASK-1232 P3).

Every bridge answers ``chat.abort`` with exactly one v1
:class:`~agent_bridge.abort_contract.AbortReply`, whatever happens: a missing
target, a cancel that raised, or a backend that refused. :func:`answer_abort`
owns that rule so no bridge can fall back to the pre-contract
``{"ok": false, "error": ...}`` shape, which the app could only read as a
legacy ``error`` with no backend identity.

A bridge supplies only the cancel itself, as an ``abort(session_key,
request_id)`` coroutine that returns the :class:`AbortState` and a detail.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any

from .abort_contract import AbortReply, AbortState, process_boot_id

if TYPE_CHECKING:
    from .session_queue import SessionQueue

logger = logging.getLogger(__name__)

# The request id recorded when the caller did not name a usable one.
UNKNOWN_REQUEST_ID = "unknown"

AbortHandler = Callable[[str, str], Awaitable[tuple[AbortState, str]]]


def abort_target(params: Mapping[str, Any]) -> tuple[str, str]:
    """Return ``(session_key, request_id)``; abort is always request-scoped."""
    if not isinstance(params, Mapping):
        raise ValueError("chat.abort params must be an object")
    session_key = params.get("sessionKey")
    request_id = params.get("requestId")
    if (
        not isinstance(session_key, str) or not session_key.strip()
        or not isinstance(request_id, str) or not request_id.strip()
    ):
        raise ValueError("chat.abort requires sessionKey and requestId")
    return session_key, request_id


def abort_reply(backend: str, request_id: str, state: AbortState, detail: str = "") -> dict[str, Any]:
    """The v1 wire dict for one abort outcome from this process."""
    return AbortReply.build(
        backend=backend,
        boot_id=process_boot_id(),
        request_id=request_id,
        state=state,
        detail=detail,
    ).to_dict()


def cancel_request_handler(queue: "SessionQueue") -> AbortHandler:
    """The abort for a bridge whose send stops on task cancellation alone."""

    async def abort(session_key: str, request_id: str) -> tuple[AbortState, str]:
        # TASK-1232 P2: the stop is recorded whether or not the send is here
        # yet, so a send that arrives later is refused at receipt.
        if queue.stop_request(session_key, request_id):
            return AbortState.CANCELLED, "task_cancelled"
        return AbortState.TOMBSTONED, "request_tombstoned"

    return abort


async def answer_abort(backend: str, params: Mapping[str, Any], abort: AbortHandler) -> dict[str, Any]:
    """Run one ``chat.abort`` and return its v1 reply; never raises ``Exception``.

    Cancellation of the RPC handler itself still propagates.
    """
    named = params.get("requestId") if isinstance(params, Mapping) else None
    request_id = named.strip() if isinstance(named, str) and named.strip() else UNKNOWN_REQUEST_ID
    try:
        session_key, request_id = abort_target(params)
        state, detail = await abort(session_key, request_id)
    except Exception as exc:
        logger.warning("chat.abort failed: backend=%s request=%s: %s", backend, request_id, exc)
        return abort_reply(backend, request_id, AbortState.ERROR, f"{type(exc).__name__}: {exc}")
    return abort_reply(backend, request_id, state, detail)


__all__ = [
    "UNKNOWN_REQUEST_ID", "abort_reply", "abort_target", "answer_abort", "cancel_request_handler",
]
