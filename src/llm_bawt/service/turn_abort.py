"""Request-scoped abort orchestration; the turn audit owns terminal truth."""
from __future__ import annotations

import time
import uuid

from sqlalchemy import text

from .logging import get_service_logger
from .turn_termination import TurnAbortCoordinator

log = get_service_logger(__name__)


async def abort_turn(service, turn, *, source: str, peer: str | None) -> str:
    coordinator = TurnAbortCoordinator(service._turn_log_store)
    # This transaction is the cancellation linearization point. It must happen
    # BEFORE an RPC can close the SDK stream and deliver EOF/ResultMessage.
    if not coordinator.request(turn.id, source=source, peer=peer):
        return "already_completed"

    from .turn_execution import turn_executions
    execution = turn_executions.get(turn.id)
    if execution is not None:
        needs_rpc = execution.request_cancel()
        if not needs_rpc:
            # The native worker (or pre-dispatch agent worker) owns teardown.
            # Accepted is not terminal: its finalizer will close tools/streams.
            coordinator.record_outcome(turn.id, "worker_signalled")
            return "cancelling:worker_signalled"
        # Refresh from the request-local handle, not a stale route DB snapshot.
        turn.agent_session_key = execution.session_key
        turn.agent_request_id = execution.request_id

    outcome = "execution_not_confirmed"
    acknowledged = False
    try:
        if turn.agent_session_key and turn.agent_request_id:
            from ..agent_backends.agent_bridge import get_agent_subscriber
            from ..bots import BotManager

            bot = BotManager(service.config).get_bot(turn.bot_id) if execution is None else None
            backend = execution.backend if execution else (getattr(bot, "agent_backend", None) if bot else None)
            subscriber = get_agent_subscriber()
            if subscriber is not None and backend:
                result = await subscriber.send_rpc(
                    "chat.abort",
                    {"sessionKey": turn.agent_session_key, "requestId": turn.agent_request_id},
                    f"abort_{uuid.uuid4().hex}", timeout_s=25, backend=backend,
                )
                acknowledged = result.get("cancelled") is True or result.get("aborted") is True
                outcome = str(result.get("detail") or result.get("error") or ("acknowledged" if acknowledged else "unknown"))
        elif turn.agent_session_key:
            outcome = "missing_request_id"
    except Exception as exc:
        outcome = f"rpc_unconfirmed:{type(exc).__name__}"
        log.warning("Abort acknowledgement unavailable for turn %s: %s", turn.id, exc)
    finally:
        coordinator.record_outcome(turn.id, outcome)
    if acknowledged:
        service._turn_log_store.update_turn(
            turn_id=turn.id, status="aborted", end_reason="aborted",
            error_text="Aborted via chat.abort; detached tool outcome may be unknown",
        )
    current = service._turn_log_store.get_turn(turn.id)
    if current is not None and current.status in ("aborted", "cancelled"):
        # An identical terminal snapshot repairs a missed event. It never
        # finalizes an unacknowledged worker merely because an RPC timed out.
        await publish_abort_snapshot(service, turn.id)
        return f"aborted:{outcome}"
    from fastapi import HTTPException
    raise HTTPException(status_code=503, detail=f"Cancellation requested; execution unconfirmed ({outcome})")


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
        "status": "cancelled", "end_reason": "aborted",
        "response_text": row.response_text, "changed_files": changed_files,
        "ts": time.time(),
    })
