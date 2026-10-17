"""Durable PostgreSQL admission for every ordinary agent START.

The full request is committed to the existing delivery outbox before any bridge
work, thread rotation, or UI turn_start. Even an idle target is dispatched by
the fenced outbox owner: an HTTP worker cannot strand an admitted direct start
between its turn-log insert and bridge publication.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..bot_types import durable_agent_backend
from ..message_authorship import AuthorReference
from ..request_identity import AdmissionUnavailable, canonical_uuid
from .delivery_reply import IN_FLIGHT_RECEIPT, QUEUED_RECEIPT
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

    def receipt(self) -> dict[str, str]:
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
        elif self.delivery_status:
            # An exact retry can arrive after its directly started turn ended.
            # Do not describe a terminal row as work that is still in flight.
            result["status"] = self.delivery_status
        return result


def admit_agent_request(
    service: Any, request: Any, *, bot_id: str, user_id: str, user_prompt: str,
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
    )


def admit_ordinary_agent_turn(
    service: Any, bot: Any, request: Any, *, is_agent_backend: bool,
    admission_claim: Any, bot_id: str, user_id: str, user_prompt: str,
    trigger_message_id: str,
) -> OrdinaryTurnDecision | None:
    """The one ordinary-start admission branch shared by stream and nonstream.

    Returns None when the request runs directly: chat models, non-durable
    harnesses, or work already holding a trusted durable admission claim.
    """
    if not is_agent_backend or admission_claim is not None or durable_agent_backend(bot) is None:
        return None
    # Direct paths minted a provisional trigger above; durable admission must
    # not present that value as a caller-supplied UUID when the request omitted it.
    decision = admit_agent_request(
        service, request, bot_id=bot_id, user_id=user_id, user_prompt=user_prompt,
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
            # The resolver is the transaction's last action: its identity locks
            # remain held until engine.begin commits on exit.
            record, _ = self.deliveries.enqueue(
                sender_bot_id=sender_id, target_bot_id=bot_id, author=author,
                message=user_prompt, payload=payload, connection=conn,
                idempotency_key=key, user_message_id=message_id,
                assistant_message_id=getattr(request, "assistant_message_id", None),
                metadata={"ordinary_chat": {
                    "session_id": getattr(request, "session_id", None),
                    "stream": bool(getattr(request, "stream", False)),
                }},
                producer="ordinary",
            )
            return OrdinaryTurnDecision(
                "queued", record.turn_id, record.user_message_id,
                record.assistant_message_id, record.id, record.status,
                bot_id=bot_id, user_id=user_id,
            )
