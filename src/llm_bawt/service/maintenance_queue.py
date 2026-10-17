"""Retain maintenance-time user submissions in the existing durable outbox."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid

from fastapi import HTTPException
from sqlalchemy import text

from agent_bridge.maintenance import is_compact_command
from ..inter_bot_preclaim import load_claim_head
from ..maintenance_delivery import MaintenanceBarrier, as_dict
from ..message_authorship import AuthorReference
from ..request_identity import AdmissionUnavailable, IdentityConflict, InvalidIdentity


class MaintenanceQueue:
    def __init__(self, service, *, dispatcher=None):
        self.service = service
        self.dispatcher = dispatcher or getattr(service, "_inter_bot_dispatcher", None)
        self.store = self.dispatcher.store if self.dispatcher else None

    async def enqueue_user(self, request, turn):
        """Called before ended-at validation so an HTTP retry reattaches safely."""
        if self.store is None or self.store.engine is None:
            raise HTTPException(503, "Maintenance queue unavailable; message was not accepted")
        from ..bots import BotManager
        bot = BotManager(self.service.config).get_bot(turn.bot_id)
        if getattr(bot, "agent_backend", None) != "claude-code":
            raise HTTPException(409, "Compaction maintenance requires a Claude backend")
        if not request.message.strip():
            raise HTTPException(422, "Maintenance message must not be empty")
        try:
            message_id = str(uuid.UUID(request.message_id))
        except ValueError as exc:
            raise HTTPException(400, "Maintenance message requires a UUID") from exc
        client = self.service.get_memory_client(turn.bot_id, turn.user_id)
        trigger = await asyncio.to_thread(client.get_message_by_id, turn.trigger_message_id) if client else None
        if not isinstance(trigger, dict) or not trigger.get("session_id"):
            raise HTTPException(503, "Maintenance source session unavailable; message was not accepted")
        metadata = {"maintenance_user_queue": True}
        if is_compact_command(turn.user_prompt):
            metadata["maintenance_barrier"] = {"turn_id": turn.id, "published": False}
        try:
            record, duplicate = self.store.enqueue(
                sender_bot_id="unknown", target_bot_id=turn.bot_id,
                author=AuthorReference.user(turn.user_id),
                message=request.message, user_message_id=message_id,
                idempotency_key=f"maintenance-user:{turn.id}:{message_id}",
                payload={
                    "bot_id": turn.bot_id, "user": turn.user_id,
                    "session_id": str(trigger["session_id"]),
                    "messages": [{"role": "user", "content": request.message}],
                    "prefer_steer": False, "stream": True,
                },
                metadata=metadata, session_policy="continue", producer="maintenance",
            )
        except (IdentityConflict, InvalidIdentity, AdmissionUnavailable) as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc
        # Durable outbox is the acceptance boundary. Normal dispatch persists
        # this SAME UUID; eager history persistence makes reload safe while queued.
        persisted = False
        try:
            await asyncio.to_thread(
                client.add_message, "user",
                self.store.payload(record.id)["messages"][-1]["content"],
                str(trigger["session_id"]), None, record.user_message_id,
                None, None, "user", turn.user_id,
            )
            persisted = True
        except Exception:
            logging.getLogger(__name__).exception("Queued maintenance message history will be repaired by dispatch")
        finally:
            self.dispatcher.wake(turn.bot_id)
        if not duplicate:
            await self.dispatcher._emit(record)
        return {
            "ok": True, "detail": "queued_after_maintenance", "queued": True,
            "turn_id": record.turn_id, "message_id": record.user_message_id,
            "delivery_id": record.id, "persisted": persisted,
            "can_escalate": False, "can_cancel": False,
        }

    def _pending(self, target):
        with self.store.engine.begin() as conn:
            if conn.dialect.name == "postgresql":
                conn.execute(text("SELECT pg_advisory_xact_lock(hashtext(:target))"), {"target": target})
            head = load_claim_head(conn, target, self.store.config)
            if not head or head["status"] != "QUEUED":
                return None
            source = MaintenanceBarrier(conn, target).prepare(head)
            if source is None or source.get("ended_at") is None:
                return None
            source["session_id"] = conn.execute(text(
                "SELECT session_id FROM messages WHERE bot_id=:bot AND id=:id"
            ), {"bot": target, "id": source.get("trigger_message_id")}).scalar()
            return head["id"], source

    def _release(self, target, delivery_id, source_id):
        with self.store.engine.begin() as conn:
            if conn.dialect.name == "postgresql":
                conn.execute(text("SELECT pg_advisory_xact_lock(hashtext(:target))"), {"target": target})
            head = conn.execute(text(
                "SELECT * FROM inter_bot_deliveries WHERE id=:id AND status='QUEUED'"
            ), {"id": delivery_id}).mappings().first()
            if head:
                MaintenanceBarrier(conn, target).released(head, source_id)

    async def publish_ready(self, target):
        """Await Redis XADD before releasing a successor; None means failure.

        A crash between XADD and the DB receipt republishes the SAME terminal
        snapshot. UI turn/message identities make that idempotent; work is never
        replayed here. The existing queue owns its bounded exactly-once dispatch.
        """
        pending = await asyncio.to_thread(self._pending, target)
        if pending is None:
            return
        delivery_id, source = pending
        status = source.get("status")
        event = {
            "_type": "turn_complete", "turn_id": source["id"],
            "trigger_message_id": source.get("trigger_message_id"),
            "assistant_message_id": source.get("assistant_message_id"),
            "bot_id": target, "user_id": source.get("user_id"),
            "session_id": str(source["session_id"]) if source.get("session_id") else None,
            "status": ("completed" if status in {"ok", "completed"} else
                       "cancelled" if status in {"aborted", "cancelled"} else status),
            "end_reason": source.get("end_reason"), "error_text": source.get("error_text"),
            "response_text": source.get("response_text"), "model": source.get("model"),
            "token_usage": as_dict(source.get("token_usage_json")), "ts": time.time(),
        }
        subscriber = self.service._redis_subscriber
        published = await subscriber.publish_tool_event(target, source["user_id"], event)
        if not published:
            raise RuntimeError("Maintenance completion publication unconfirmed; successor remains queued")
        await asyncio.to_thread(self._release, target, delivery_id, source["id"])
