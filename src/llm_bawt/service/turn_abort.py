"""Request-scoped abort orchestration; the turn audit owns terminal truth.

TASK-1232 P4, Nick: "stop should stop. no questions." A user Stop always ends
the turn through the shared settle (``turn_termination.settle_open_turn``).
The bridge's answer decides only what the ``user_stop`` audit records
(``bridge_state``, ``confirmed``), never whether the row closes. This replaces,
for user Stop only, the TASK-1015 rule that refused to finalize without an ack;
automatic sweeps still leave ambiguous rows alone.
"""
from __future__ import annotations

import asyncio
import time
import uuid

from sqlalchemy import text

from .logging import get_service_logger
from .turn_termination import TurnAbortCoordinator

log = get_service_logger(__name__)

# A signalled in-process worker normally finalizes within moments, committing
# its own partial reply (TASK-286). Past this bound Stop settles without it;
# a later finalize cannot reopen the row (update_turn is terminal-monotonic).
WORKER_FINALIZE_GRACE_SECONDS = 5.0
_WORKER_POLL_SECONDS = 0.02
# Margin over the bridge's teardown bound, so the reply that carries the
# teardown outcome lands before the app gives up (Vex S3).
_ABORT_RPC_MARGIN_SECONDS = 5.0

# bridge_state values that mean execution itself confirmed it stopped.
CONFIRMED_STATES = frozenset({"cancelled", "worker_finalized"})


async def abort_turn(service, turn, *, source: str, peer: str | None) -> str:
    """Stop ``turn`` and return ``aborted:<bridge_state>`` or ``already_completed``."""
    store = service._turn_log_store
    current = store.get_turn(turn.id)
    if current is not None and current.status == "reserved" and current.ended_at is None:
        stopped = await _stop_reservation(service, turn.id, source=source, peer=peer)
        if stopped is not None:
            return stopped
        # Consumed between our read and the target lock (Vex B1): it is a
        # started turn now, so stop it as one instead of reporting completion.
        current = store.get_turn(turn.id)
        if current is None or current.ended_at is not None:
            return "already_completed"
        turn.agent_session_key = current.agent_session_key
        turn.agent_request_id = current.agent_request_id
    coordinator = TurnAbortCoordinator(store)
    # This transaction is the cancellation linearization point. It must happen
    # BEFORE an RPC can close the SDK stream and deliver EOF/ResultMessage.
    if not coordinator.request(turn.id, source=source, peer=peer):
        return "already_completed"
    # The bridge cancels its pending Future with this run. Convert the still-
    # live row to normal deferral so a later answer dispatches a continuation.
    question_store = getattr(service, "_pending_question_store", None)
    if question_store is not None:
        question_store.expire_live_for_turn(turn.id)

    from .turn_execution import turn_executions
    execution = turn_executions.get(turn.id)
    if execution is not None and not execution.request_cancel():
        # The native worker (or pre-dispatch agent worker) owns teardown and
        # commits its own terminal. Wait for it, bounded; never forever.
        finalized = await _await_terminal(store, turn.id, WORKER_FINALIZE_GRACE_SECONDS)
        stop = {"bridge_state": "worker_finalized" if finalized else "worker_unfinalized"}
    else:
        if execution is not None:
            # Refresh from the request-local handle, not a stale route DB snapshot.
            turn.agent_session_key = execution.session_key
            turn.agent_request_id = execution.request_id
        try:
            stop = await _abort_bridge_run(service, turn, execution)
        except Exception as exc:  # Stop always stops; an unknown answer is recorded
            log.exception("Abort RPC path failed for turn %s", turn.id)
            stop = {"bridge_state": "rpc_unconfirmed", "detail": type(exc).__name__}
    stop["confirmed"] = stop["bridge_state"] in CONFIRMED_STATES
    if not stop["confirmed"]:
        # The settle releases admission while the old run may still be live,
        # so the next turn can overlap it on the same session (Vex S1).
        stop["admission_released_unconfirmed"] = True
    stop["source"] = source
    coordinator.record_outcome(turn.id, stop["bridge_state"])
    coordinator.settle(turn.id, stop)
    _wake_dispatcher(service)
    await publish_abort_snapshot(service, turn.id)
    return f"aborted:{stop['bridge_state']}"


async def _stop_reservation(service, turn_id: str, *, source: str, peer: str | None) -> str | None:
    """A reserved row has no run: revoke its producer, then settle it.

    None (nothing written) when the row is no longer an open reservation.
    """
    from .turn_admission import revoke_reservation

    producer = await asyncio.to_thread(
        revoke_reservation, service._turn_log_store.engine, turn_id,
        disposition="user_stop",
        audit={"bridge_state": "reserved", "confirmed": False, "source": source, "peer": peer},
    )
    if producer is None:
        return None
    _wake_dispatcher(service)
    await publish_abort_snapshot(service, turn_id)
    return "aborted:reserved"


async def _await_terminal(store, turn_id: str, bound_seconds: float) -> bool:
    deadline = time.monotonic() + max(0.0, bound_seconds)
    while True:
        row = store.get_turn(turn_id)
        if row is None or row.ended_at is not None:
            return True
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(_WORKER_POLL_SECONDS)


async def _abort_bridge_run(service, turn, execution) -> dict:
    """Ask the bridge to abort the exact request; describe what it said."""
    if not turn.agent_session_key:
        return {"bridge_state": "unbound"}
    if not turn.agent_request_id:
        return {"bridge_state": "missing_request_id"}
    from ..agent_backends.agent_bridge import get_agent_subscriber

    backend = execution.backend if execution is not None else _bot_backend(service, turn.bot_id)
    subscriber = get_agent_subscriber()
    if subscriber is None or not backend:
        return {"bridge_state": "bridge_unreachable", "backend": backend}
    try:
        from agent_bridge.abort_contract import ABORT_TEARDOWN_BOUND_S, AbortReply, AbortReplyError

        result = await subscriber.send_rpc(
            "chat.abort",
            {"sessionKey": turn.agent_session_key, "requestId": turn.agent_request_id},
            f"abort_{uuid.uuid4().hex}",
            timeout_s=ABORT_TEARDOWN_BOUND_S + _ABORT_RPC_MARGIN_SECONDS, backend=backend,
        )
    except Exception as exc:
        log.warning("Abort acknowledgement unavailable for turn %s: %s", turn.id, exc)
        return {"bridge_state": "rpc_unconfirmed", "backend": backend, "detail": type(exc).__name__}
    try:
        reply = AbortReply.from_wire(result, backend=backend)
    except AbortReplyError as exc:
        return {"bridge_state": "error", "backend": backend, "detail": f"malformed reply: {exc}"}
    return {"bridge_state": reply.state.value, "backend": reply.backend, "boot_id": reply.boot_id,
            "contract": reply.contract, "detail": reply.detail}


def _bot_backend(service, bot_id: str | None) -> str | None:
    from ..bots import BotManager

    bot = BotManager(service.config).get_bot(bot_id) if bot_id else None
    return getattr(bot, "agent_backend", None) if bot else None


def _wake_dispatcher(service) -> None:
    # A settled row releases admission; let queued deliveries drain now.
    dispatcher = getattr(service, "_inter_bot_dispatcher", None)
    if dispatcher is not None:
        dispatcher.wake()


async def publish_abort_snapshot(service, turn_id: str) -> None:
    try:
        await _publish_abort_snapshot(service, turn_id)
    except Exception:
        # The durable terminal row remains authoritative; a Redis outage must
        # not turn an acknowledged Stop into HTTP 500. History repairs replay.
        log.exception("Abort snapshot publish failed for turn %s", turn_id)


async def _publish_abort_snapshot(service, turn_id: str) -> None:
    store = service._turn_log_store
    row = store.get_turn(turn_id)
    subscriber = getattr(service, "_redis_subscriber", None)
    if row is None or subscriber is None or not row.bot_id:
        return
    session_id = None
    if row.trigger_message_id:
        with store.engine.connect() as conn:
            session_id = conn.execute(text(
                "SELECT session_id FROM messages WHERE bot_id=:bot AND id=:id"
            ), {"bot": row.bot_id, "id": row.trigger_message_id}).scalar()
    try:
        changed_files = store.changed_files_summary(row.id)
    except Exception:
        changed_files = None
    await subscriber.publish_tool_event(row.bot_id, row.user_id or "nick", {
        "_type": "turn_complete", "turn_id": row.id,
        "trigger_message_id": row.trigger_message_id,
        "assistant_message_id": row.assistant_message_id,
        "session_id": str(session_id) if session_id else None,
        "bot_id": row.bot_id, "user_id": row.user_id,
        # The row's own terminal (abort-class rows are aborted/aborted, A1).
        "status": {"aborted": "cancelled", "ok": "completed"}.get(row.status, row.status),
        "end_reason": row.end_reason or "aborted",
        "response_text": row.response_text, "changed_files": changed_files,
        "ts": time.time(),
    })
