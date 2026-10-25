"""Durable delivery receipt and public representation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .agent_context import SessionPolicy
from .message_authorship import AuthorReference, normalize_author


def _steer_uncertain(record: "DeliveryRecord") -> dict[str, Any] | None:
    # Deferred: the steer fence imports DeliveryRecord from this module.
    from .inter_bot_delivery_steer_fence import steer_uncertain

    return steer_uncertain(record)


@dataclass(frozen=True)
class DeliveryRecord:
    id: str
    sender_bot_id: str
    author_entity_type: str
    author_entity_id: str
    target_bot_id: str
    message: str
    user_message_id: str
    turn_id: str
    status: str
    attempt_count: int
    max_attempts: int
    idempotency_key: str | None
    project_id: str | None
    task_id: str | None
    message_kind: str | None
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    available_at: datetime
    dispatch_started_at: datetime | None
    delivered_at: datetime | None
    next_retry_at: datetime | None
    last_error: str | None
    response_model: str | None
    response_chars: int | None
    claim_token: str | None
    transport_accepted_at: datetime | None
    target_turn_id: str | None
    delivery_mode: str | None
    reply_mode: str
    session_policy: str
    reset_status: str | None
    reset_reason: str | None
    old_session_id: str | None
    new_session_id: str | None
    reset_at: datetime | None
    overflow_recovery_count: int
    assistant_message_id: str | None = None
    # TASK-1300: bumped by the database on every UPDATE (commit order).
    revision: int = 0
    # TASK-1431: 'control' rows steer one pinned run past a held FIFO head.
    lane: str = "normal"

    @classmethod
    def from_mapping(cls, row: Any) -> "DeliveryRecord":
        metadata = row.get("metadata_json")
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except Exception:
                metadata = {}
        payload = row.get("payload_json")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except (ValueError, TypeError):
                payload = {}
        reply_mode = payload.get("reply_mode") if isinstance(payload, dict) else None
        return cls(
            id=row["id"],
            sender_bot_id=row["sender_bot_id"],
            author_entity_type=row.get("author_entity_type") or "bot",
            author_entity_id=row.get("author_entity_id") or row["sender_bot_id"],
            target_bot_id=row["target_bot_id"],
            message=row["message"],
            user_message_id=row["user_message_id"],
            turn_id=row["turn_id"],
            status=row["status"],
            attempt_count=int(row["attempt_count"] or 0),
            max_attempts=int(row["max_attempts"] or 0),
            idempotency_key=row.get("idempotency_key"),
            project_id=row.get("project_id"),
            task_id=row.get("task_id"),
            message_kind=row.get("message_kind"),
            metadata=metadata if isinstance(metadata, dict) else {},
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            available_at=row["available_at"],
            dispatch_started_at=row.get("dispatch_started_at"),
            delivered_at=row.get("delivered_at"),
            next_retry_at=row.get("next_retry_at"),
            last_error=row.get("last_error"),
            response_model=row.get("response_model"),
            response_chars=row.get("response_chars"),
            claim_token=row.get("claim_token"),
            transport_accepted_at=row.get("transport_accepted_at"),
            target_turn_id=row.get("target_turn_id"),
            delivery_mode=row.get("delivery_mode"),
            reply_mode=reply_mode if reply_mode in {"async", "waited"} else "async",
            session_policy=row.get("session_policy") or SessionPolicy.CONTINUE.value,
            reset_status=row.get("reset_status"),
            reset_reason=row.get("reset_reason"),
            old_session_id=row.get("old_session_id"),
            new_session_id=row.get("new_session_id"),
            reset_at=row.get("reset_at"),
            overflow_recovery_count=int(row.get("overflow_recovery_count") or 0),
            assistant_message_id=payload.get("assistant_message_id") if isinstance(payload, dict) else None,
            revision=int(row.get("revision") or 0),
            lane=row.get("lane") or "normal",
        )

    @property
    def author(self) -> AuthorReference:
        author = normalize_author(self.author_entity_type, self.author_entity_id)
        if author is None:  # Database constraints keep this unreachable.
            raise ValueError("delivery author is required")
        return author

    COMPACT_PREVIEW_CHARS = 120
    COMPACT_ERROR_CHARS = 200

    def to_compact(self) -> dict[str, Any]:
        """Bounded list row (TASK-1239): ~15 scalar fields, no metadata.

        Full receipt detail stays behind ``GET /v1/inter-bot-deliveries/{id}``.
        Null fields are dropped so idle rows stay short.
        """
        def clip(value: str | None, size: int) -> str | None:
            if not value:
                return None
            flat = " ".join(value.split())
            return flat if len(flat) <= size else flat[: size - 1] + "…"

        row = {
            "delivery_id": self.id,
            "status": self.status,
            "sender_bot_id": self.sender_bot_id,
            "target_bot_id": self.target_bot_id,
            "author": None if self.author_entity_type == "bot" else self.author_entity_id,
            "message_kind": self.message_kind,
            "delivery_mode": self.delivery_mode,
            "task_id": self.task_id,
            "idempotency_key": self.idempotency_key,
            "attempts": f"{self.attempt_count}/{self.max_attempts}",
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
            "target_turn_id": self.target_turn_id,
            "last_error": clip(self.last_error, self.COMPACT_ERROR_CHARS),
            "preview": clip(self.message, self.COMPACT_PREVIEW_CHARS),
        }
        return {key: value for key, value in row.items() if value is not None}

    def to_api(self, *, duplicate: bool = False) -> dict[str, Any]:
        def iso(value: datetime | None) -> str | None:
            return value.isoformat() if value else None

        return {
            "delivery_id": self.id,
            "status": self.status,
            "sender_bot_id": self.sender_bot_id,
            "author": self.author.to_dict(),
            "target_bot_id": self.target_bot_id,
            "user_message_id": self.user_message_id,
            "turn_id": self.turn_id,
            "target_turn_id": self.target_turn_id,
            "delivery_mode": self.delivery_mode,
            "reply_mode": self.reply_mode,
            "session_policy": self.session_policy,
            "reset_status": self.reset_status,
            "reset_reason": self.reset_reason,
            "old_session_id": self.old_session_id,
            "new_session_id": self.new_session_id,
            "reset_at": iso(self.reset_at),
            "retained_history": (
                True
                if self.session_policy == SessionPolicy.RESET_RETAIN_HISTORY.value
                else False
                if self.session_policy == SessionPolicy.RESET_WITHOUT_HISTORY.value
                else None
            ),
            "overflow_recovery_count": self.overflow_recovery_count,
            "attempt_count": self.attempt_count,
            "max_attempts": self.max_attempts,
            "idempotency_key": self.idempotency_key,
            "project_id": self.project_id,
            "task_id": self.task_id,
            "message_kind": self.message_kind,
            "metadata": {k: v for k, v in self.metadata.items()
                         if k != "_request_identity_producer"},
            "created_at": iso(self.created_at),
            "updated_at": iso(self.updated_at),
            "available_at": iso(self.available_at),
            "dispatch_started_at": iso(self.dispatch_started_at),
            "delivered_at": iso(self.delivered_at),
            "next_retry_at": iso(self.next_retry_at),
            "last_error": self.last_error,
            "response_model": self.response_model,
            "response_chars": self.response_chars,
            # TASK-1300: a DELIVERED steer whose command may not have run.
            "steer_uncertain": _steer_uncertain(self),
            "revision": self.revision,
            "duplicate": duplicate,
        }

