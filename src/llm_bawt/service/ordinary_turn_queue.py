"""Durable PostgreSQL admission for every ordinary agent START.

The full request is committed to the existing delivery outbox before any bridge
work, thread rotation, or UI turn_start. Even an idle target is dispatched by
the fenced outbox owner: an HTTP worker cannot strand an admitted direct start
between its turn-log insert and bridge publication.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
from uuid import uuid4

from sqlalchemy import text

from agent_bridge.maintenance import is_new_command

from ..bot_types import durable_agent_backend
from ..message_authorship import AuthorReference
from ..request_identity import AdmissionUnavailable, canonical_uuid
from .delivery_reply import IN_FLIGHT_RECEIPT, QUEUED_RECEIPT
from .delivery_wait_reason import wait_reason as read_wait_reason
from .turn_admission import lock_target


@dataclass(frozen=True)
class OrdinaryTurnDecision:
    state: str  # queued | already_started
    turn_id: str
    user_message_id: str
    assistant_message_id: str
    delivery_id: str | None = None
    delivery_status: str | None = None
    bot_id: str | None = None
    user_id: str | None = None
    # TASK-1676: why a QUEUED row waits (``delivery_wait_reason``), else None.
    wait_reason: dict[str, Any] | None = None

    def receipt(self) -> dict[str, Any]:
        result = {
            "object": QUEUED_RECEIPT if self.delivery_id else IN_FLIGHT_RECEIPT,
            "turn_id": self.turn_id,
            "user_message_id": self.user_message_id,
            "assistant_message_id": self.assistant_message_id,
        }
        if self.bot_id:
            result["bot_id"] = self.bot_id
        if self.user_id:
            result["user_id"] = self.user_id
        if self.delivery_id:
            result["delivery_id"] = self.delivery_id
            result["status"] = self.delivery_status or "QUEUED"
            if self.wait_reason is not None:
                result["wait_reason"] = self.wait_reason
        elif self.delivery_status:
            # An exact retry can arrive after its directly started turn ended.
            # Do not describe a terminal row as work that is still in flight.
            result["status"] = self.delivery_status
        return result


def admit_agent_request(
    service: Any, request: Any, *, bot_id: str, user_id: str, user_prompt: str,
    chat_reset: bool = False,
) -> OrdinaryTurnDecision:
    """Submit one ordinary request through the durable dispatcher."""
    # Preserve omission provenance before the queued request is normalized.
    queued_request = request
    queue = getattr(service, "_ordinary_turn_queue", None)
    if queue is None:
        dispatcher = getattr(service, "_inter_bot_dispatcher", None)
        queue = OrdinaryTurnQueue(service._turn_log_store, getattr(dispatcher, "store", None))
    return queue.decide(
        request=queued_request, bot_id=bot_id, user_id=user_id, user_prompt=user_prompt,
        chat_reset=chat_reset,
    )


def admit_ordinary_agent_turn(
    service: Any, bot: Any, request: Any, *, is_agent_backend: bool,
    admission_claim: Any, bot_id: str, user_id: str, user_prompt: str,
    trigger_message_id: str,
) -> OrdinaryTurnDecision | None:
    """The one ordinary-start admission branch shared by stream and nonstream.

    Returns None for ordinary chat, non-durable harnesses, explicitly selected
    threads, or work already holding a trusted durable admission claim. A
    continuous chat /new uses the same WAIT queue as an agent /new.
    """
    if admission_claim is not None:
        return None
    if is_agent_backend:
        if durable_agent_backend(bot) is None:
            return None
    elif not is_new_command(user_prompt) or getattr(request, "session_id", None):
        return None
    # Direct paths minted a provisional trigger above; durable admission must
    # not present that value as a caller-supplied UUID when the request omitted it.
    decision = admit_agent_request(
        service, request, bot_id=bot_id, user_id=user_id, user_prompt=user_prompt,
        chat_reset=not is_agent_backend,
    )
    dispatcher = getattr(service, "_inter_bot_dispatcher", None)
    if dispatcher is not None and decision.delivery_id:
        record = dispatcher.store.get(decision.delivery_id) if decision.delivery_status == "QUEUED" else None
        if record is not None and record.status == "QUEUED":
            dispatcher.announce_admitted(record)
        else:
            dispatcher.wake(bot_id)
    return decision


class OrdinaryTurnQueue:
    """Enqueue the full request under the target lock, including idle work."""

    def __init__(self, turns: Any, deliveries: Any):
        if (turns is None or deliveries is None or turns.engine is None
                or deliveries.engine is not turns.engine):
            raise AdmissionUnavailable("ordinary agent admission needs the shared delivery database")
        self.turns = turns
        self.deliveries = deliveries

    def decide(
        self, *, request: Any, bot_id: str, user_id: str, user_prompt: str,
        chat_reset: bool = False,
    ) -> OrdinaryTurnDecision:
        message_id = getattr(request, "user_message_id", None)
        payload = request.model_dump(exclude_none=True)
        payload.update(bot_id=bot_id, user=user_id, prefer_steer=False)
        # This server-owned field is intentionally excluded by Pydantic's public
        # model_dump, but an accepted deferred run must see the same context.
        context = getattr(request, "client_system_context", None)
        if context is not None:
            payload["client_system_context"] = context
        sender = getattr(request, "_internal_inter_bot_sender_id", None)
        sender = sender.strip().lower() if isinstance(sender, str) and sender.strip() else None
        author = AuthorReference.bot(sender) if sender else AuthorReference.user(user_id)
        sender_id = sender or bot_id
        key = f"ordinary:{canonical_uuid(message_id, 'user_message_id')}" if message_id else None
        with self.turns.engine.begin() as conn:
            lock_target(conn, bot_id)
            metadata = {"ordinary_chat": {
                "session_id": getattr(request, "session_id", None),
                "stream": bool(getattr(request, "stream", False)),
            }}
            # Preserve the exact visible command. Only a trusted claimed
            # follow-up may override the bridge's outbound message with its task.
            from .session_lifecycle.execution import NewSessionExecution

            follow_up = NewSessionExecution.typed_follow_up(user_prompt)
            if (user_prompt.strip() == "/new" or follow_up is not None) and not getattr(request, "session_id", None):
                from .session_lifecycle.intent import (
                    Admission, Continuity, NewSessionIntent, Origin,
                )
                if follow_up is not None:
                    payload["inter_bot_typed_follow_up"] = True

                old_id = conn.execute(text("""
                    SELECT id FROM sessions WHERE bot_id=:bot AND user_id=:user
                      AND status='active' AND ended_at IS NULL
                """), {"bot": bot_id, "user": user_id}).scalar_one_or_none()
                intent = NewSessionIntent(
                    bot_id=bot_id, user_id=user_id, author=author,
                    continuity=Continuity.CONFIGURED, admission=Admission.WAIT,
                    origin=Origin.CHAT_COMMAND,
                    idempotency_key=key or f"ordinary:{uuid4()}",
                    expected_session_id=old_id, follow_up=follow_up,
                )
                metadata["new_session_intent"] = {
                    **asdict(intent), "author": author.to_dict(),
                }
                if chat_reset:
                    metadata["new_session_execution"] = "chat"
            # The resolver is the transaction's last action: its identity locks
            # remain held until engine.begin commits on exit.
            record, _ = self.deliveries.enqueue(
                sender_bot_id=sender_id, target_bot_id=bot_id, author=author,
                message=user_prompt, payload=payload, connection=conn,
                idempotency_key=key, user_message_id=message_id,
                assistant_message_id=getattr(request, "assistant_message_id", None),
                metadata=metadata,
                producer="ordinary",
            )
            return OrdinaryTurnDecision(
                "queued", record.turn_id, record.user_message_id,
                record.assistant_message_id, record.id, record.status,
                bot_id=bot_id, user_id=user_id,
                wait_reason=read_wait_reason(conn, bot_id) if record.status == "QUEUED" else None,
            )
