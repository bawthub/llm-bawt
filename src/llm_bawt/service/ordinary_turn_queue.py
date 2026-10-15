"""Durable PostgreSQL admission for every ordinary agent START.

The full request is committed to the existing delivery outbox before any bridge
work, thread rotation, or UI turn_start. Even an idle target is dispatched by
the fenced outbox owner: an HTTP worker cannot strand an admitted direct start
between its turn-log insert and bridge publication.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text as sa_text

from ..bot_types import durable_agent_backend
from ..message_authorship import AuthorReference
from .delivery_reply import IN_FLIGHT_RECEIPT, QUEUED_RECEIPT
from .turn_admission import AdmissionFenceLost, AdmissionUnavailable, lock_target


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


def stable_ordinary_ids(request: Any, bot_id: str, user_id: str) -> tuple[str, str, str]:
    """A retry with the same trigger reuses all three normal-turn identities."""
    raw = getattr(request, "user_message_id", None)
    try:
        message_id = str(uuid.UUID(raw)) if raw else str(uuid.uuid4())
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError("user_message_id must be a UUID for agent admission") from exc
    assistant = getattr(request, "assistant_message_id", None)
    try:
        assistant_id = str(uuid.UUID(assistant)) if assistant else str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"ordinary-assistant:{bot_id}:{user_id}:{message_id}")
        )
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError("assistant_message_id must be a UUID for agent admission") from exc
    return f"turn-{uuid.UUID(message_id).hex}", message_id, assistant_id


def admit_agent_request(
    service: Any, request: Any, *, bot_id: str, user_id: str, user_prompt: str,
) -> OrdinaryTurnDecision:
    """Submit one ordinary request through the durable dispatcher."""
    turn_id, message_id, assistant_id = stable_ordinary_ids(request, bot_id, user_id)
    queued_request = request.model_copy(update={
        "user_message_id": message_id, "assistant_message_id": assistant_id,
    })
    queue = getattr(service, "_ordinary_turn_queue", None)
    if queue is None:
        dispatcher = getattr(service, "_inter_bot_dispatcher", None)
        queue = OrdinaryTurnQueue(service._turn_log_store, getattr(dispatcher, "store", None))
    return queue.decide(
        request=queued_request, bot_id=bot_id, user_id=user_id, user_prompt=user_prompt,
        turn_id=turn_id, user_message_id=message_id, assistant_message_id=assistant_id,
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
    decision = admit_agent_request(
        service, request.model_copy(update={"user_message_id": trigger_message_id}),
        bot_id=bot_id, user_id=user_id, user_prompt=user_prompt,
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
        turn_id: str, user_message_id: str, assistant_message_id: str,
    ) -> OrdinaryTurnDecision:
        payload = request.model_dump(exclude_none=True)
        payload.update(bot_id=bot_id, user=user_id, user_message_id=user_message_id,
                       assistant_message_id=assistant_message_id, prefer_steer=False)
        # This server-owned field is intentionally excluded by Pydantic's public
        # model_dump, but an accepted deferred run must see the same context.
        context = getattr(request, "client_system_context", None)
        if context is not None:
            payload["client_system_context"] = context
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
        sender = getattr(request, "_internal_inter_bot_sender_id", None)
        sender = sender.strip().lower() if isinstance(sender, str) and sender.strip() else None
        author = AuthorReference.bot(sender) if sender else AuthorReference.user(user_id)
        sender_id = sender or bot_id
        key = f"ordinary:{user_message_id}"
        with self.turns.engine.begin() as conn:
            lock_target(conn, bot_id)
            # A retry of a queued send must reattach before looking at idle state;
            # otherwise it could start ahead of its own prior FIFO row.
            old = conn.execute(sa_text("""
                SELECT * FROM inter_bot_deliveries
                WHERE author_entity_type=:author_type AND author_entity_id=:author_id
                  AND sender_bot_id=:sender AND target_bot_id=:bot AND idempotency_key=:key
            """), {"author_type": author.entity_type, "author_id": author.entity_id,
                   "sender": sender_id, "bot": bot_id, "key": key}).mappings().first()
            if old:
                metadata = old["metadata_json"] or {}
                if metadata.get("ordinary_chat", {}).get("digest") != digest:
                    raise AdmissionFenceLost("same message id has a different queued request")
                return OrdinaryTurnDecision(
                    "queued", old["turn_id"], user_message_id,
                    old["payload_json"]["assistant_message_id"], old["id"], old["status"],
                    bot_id=bot_id, user_id=user_id,
                )
            started = conn.execute(sa_text(
                "SELECT bot_id, user_id, status, ended_at, request_json, assistant_message_id "
                "FROM turn_logs WHERE id=:id"
            ), {"id": turn_id}).mappings().first()
            if started:
                request_json = started["request_json"] or "{}"
                recorded = json.loads(request_json) if isinstance(request_json, str) else request_json
                if (started["bot_id"] != bot_id or started["user_id"] != user_id
                        or recorded.get("ordinary_request_digest") != digest):
                    raise AdmissionFenceLost("same message id already belongs to another turn")
                return OrdinaryTurnDecision(
                    "already_started", turn_id, user_message_id,
                    started["assistant_message_id"] or assistant_message_id,
                    delivery_status=(str(started["status"]).upper() if started["ended_at"] is not None else None),
                    bot_id=bot_id, user_id=user_id,
                )
            # An idle target still needs durable publication ownership. Claiming
            # this row later is the only path that reserves a turn and talks to
            # the bridge; an HTTP-process crash cannot strand a direct start.
            record, duplicate = self.deliveries.enqueue(
                sender_bot_id=sender_id, target_bot_id=bot_id, author=author,
                message=user_prompt, payload=payload, connection=conn,
                idempotency_key=key, user_message_id=user_message_id,
                assistant_message_id=assistant_message_id,
                metadata={"ordinary_chat": {"digest": digest,
                                            "session_id": getattr(request, "session_id", None),
                                            "stream": bool(getattr(request, "stream", False))}},
            )
            if duplicate:
                # An external same-key producer raced us before it acquired the
                # target lock; never accept a different request by that key.
                recorded = (record.metadata or {}).get("ordinary_chat", {})
                if recorded.get("digest") != digest:
                    raise AdmissionFenceLost("same message id has a different queued request")
            return OrdinaryTurnDecision(
                "queued", record.turn_id, user_message_id,
                assistant_message_id, record.id, record.status,
                bot_id=bot_id, user_id=user_id,
            )
