"""Durable admission identity, scoped-key replay, and producer-specific validation.

Lock inventory: identity uses the two-int4 PostgreSQL advisory space
``(IDENTITY_NS, signed_sha256(tagged_input))``. Dispatcher/ordinary target locks
use the disjoint single-bigint space; scheduler job and manual-retry row locks
are acquired by their producers before calling this leaf. Session-reset lineage
also uses two-int4 locks, but its current path never enters this resolver.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

from sqlalchemy import text as sa_text
from sqlalchemy.engine import Connection

from .inter_bot_delivery_record import DeliveryRecord

IDENTITY_NS = 1144


class IdentityConflict(ValueError):
    """A supplied UUID/key/scope or immutable request disagrees (HTTP 409)."""

    status_code = 409


class ReplayUnverifiable(IdentityConflict):
    """Historical identity exists but its comparison evidence is missing."""


class AdmissionUnavailable(RuntimeError):
    """An insert lost a uniqueness race without a verifiable winner (HTTP 503)."""

    status_code = 503


class InvalidIdentity(ValueError):
    """A supplied identity is not a UUID (HTTP 400)."""

    status_code = 400


def canonical_uuid(value: str, field: str) -> str:
    try:
        return str(uuid.UUID(value))
    except (ValueError, AttributeError, TypeError) as exc:
        raise InvalidIdentity(f"{field} must be a UUID") from exc


def stable_ids(token: uuid.UUID) -> tuple[str, str, str]:
    return f"delivery-{token.hex}", str(token), f"turn-delivery-{token.hex}"


def assistant_id(message_id: str) -> str:
    """One default for NEW durable rows, regardless of producer."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"delivery-assistant:{message_id}"))


def identity_hash(value: str) -> int:
    """Stable signed int4; monkeypatchable to exercise actual-pair collisions."""
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "big", signed=True)


def scoped_key(values: dict[str, Any]) -> str:
    return "key:" + "\0".join(str(values[name]) for name in (
        "author_type", "author_id", "sender", "target", "idem",
    ))


def lock_identity(conn: Connection, values: dict[str, Any], explicit: str | None) -> None:
    """Acquire sorted, de-duplicated numeric identity locks as a transaction leaf.

    Transaction-scoped locks survive this helper's return. A caller that supplies
    ``connection=`` must acquire NO target, job or existing-row lock after enqueue
    and before the enclosing COMMIT/ROLLBACK. Unique-index checks can still wait.
    """
    tags = ([f"uuid:{explicit}"] if explicit else [])
    if values["idem"]:
        tags.append(scoped_key(values))
    for namespace, hashed in sorted({(IDENTITY_NS, identity_hash(tag)) for tag in tags}):
        conn.execute(sa_text("SELECT pg_advisory_xact_lock(:namespace, :hashed)"),
                     {"namespace": namespace, "hashed": hashed})


class RequestIdentityResolver:
    """Resolve retained delivery evidence before any insert or candidate mint."""

    def __init__(self, conn: Connection):
        self.conn = conn

    def _rows(self, values: dict[str, Any], explicit: str | None):
        by_uuid = self.conn.execute(sa_text(
            "SELECT * FROM inter_bot_deliveries WHERE user_message_id=:message_id"
        ), {"message_id": explicit}).mappings().first() if explicit else None
        by_key = self.conn.execute(sa_text("""
            SELECT * FROM inter_bot_deliveries
            WHERE author_entity_type=:author_type AND author_entity_id=:author_id
              AND sender_bot_id=:sender AND target_bot_id=:target
              AND idempotency_key=:idem
        """), values).mappings().first() if values["idem"] else None
        return by_uuid, by_key

    @staticmethod
    def _digest(payload: dict[str, Any], stored_assistant: str | None = None) -> str:
        comparison = dict(payload)
        if stored_assistant is not None and comparison.get("assistant_message_id") is None:
            comparison["assistant_message_id"] = stored_assistant
        return hashlib.sha256(json.dumps(comparison, sort_keys=True, default=str).encode()).hexdigest()

    def _validate(self, row: Any, values: dict[str, Any], payload: dict[str, Any],
                  producer: str) -> DeliveryRecord:
        if (row["author_entity_type"] != values["author_type"] or
                row["author_entity_id"] != values["author_id"] or
                row["sender_bot_id"] != values["sender"] or
                row["target_bot_id"] != values["target"]):
            raise IdentityConflict("message UUID belongs to another author, sender or target")
        if values["idem"] is not None and row["idempotency_key"] != values["idem"]:
            raise IdentityConflict("message UUID belongs to a different scoped key")
        if values["idem"] is None and producer in {"maintenance", "schedule", "retry"}:
            raise ReplayUnverifiable("required producer key is missing")
        metadata = row["metadata_json"] or {}
        provenance = metadata.get("_request_identity_producer") or (
            "ordinary" if "ordinary_chat" in metadata else
            "maintenance" if "maintenance_user_queue" in metadata else
            "schedule" if "prompt_schedule" in metadata else
            "retry" if "manual_retry_of" in metadata else "bot"
        )
        if producer != provenance:
            if (producer == "ordinary" and provenance == "bot"
                    and (row["idempotency_key"] or "").startswith("ordinary:")):
                raise ReplayUnverifiable("ordinary replay comparison evidence is unavailable")
            raise IdentityConflict("delivery belongs to another producer")
        if producer == "ordinary":
            evidence = metadata.get("ordinary_chat")
            stored_payload = row["payload_json"]
            stored_assistant = stored_payload.get("assistant_message_id") if isinstance(stored_payload, dict) else None
            if not isinstance(evidence, dict) or not evidence.get("digest") or not stored_assistant:
                raise ReplayUnverifiable("ordinary replay comparison evidence is unavailable")
            if self._digest(payload, stored_assistant) != evidence["digest"]:
                raise IdentityConflict("same message id has a different queued request")
        elif producer == "maintenance":
            if (row["user_message_id"] != values["message_id"] or
                    row["message"] != values["message"]):
                raise IdentityConflict("maintenance message or source turn changed")
        elif row["session_policy"] != values["session_policy"]:
            raise IdentityConflict(
                f"idempotency key already exists with session_policy={row['session_policy']!r}"
            )
        return DeliveryRecord.from_mapping(row)

    def _winner(self, by_uuid: Any, by_key: Any, values: dict[str, Any],
                payload: dict[str, Any], producer: str, explicit: str | None) -> DeliveryRecord | None:
        if by_uuid is not None and by_key is not None and by_uuid["id"] != by_key["id"]:
            raise IdentityConflict("UUID and scoped key belong to different deliveries")
        if explicit and by_uuid is not None and values["idem"] and by_key is None:
            raise IdentityConflict("UUID already exists under another scoped key")
        if explicit and by_uuid is None and by_key is not None:
            raise IdentityConflict("scoped key belongs to another message UUID")
        row = by_uuid if explicit else by_key
        if row is None:
            return None
        if explicit and row["user_message_id"] != explicit:
            raise IdentityConflict("scoped key belongs to another message UUID")
        return self._validate(row, values, payload, producer)

    def admit(self, *, values: dict[str, Any], payload: dict[str, Any],
              metadata: dict[str, Any], producer: str,
              explicit_user: str | None, explicit_assistant: str | None) -> tuple[DeliveryRecord, bool]:
        """One bounded resolution, then INSERT or one winner re-read; never remint."""
        if producer not in {"ordinary", "maintenance", "schedule", "bot", "retry"}:
            raise ValueError("unknown trusted delivery producer")
        explicit = canonical_uuid(explicit_user, "user_message_id") if explicit_user is not None else None
        assistant = canonical_uuid(explicit_assistant, "assistant_message_id") if explicit_assistant else None
        values = {**values, "message_id": explicit}
        payload = dict(payload)
        if producer == "ordinary":
            if explicit:
                payload["user_message_id"] = explicit
            if assistant:
                payload["assistant_message_id"] = assistant
        lock_identity(self.conn, values, explicit)
        by_uuid, by_key = self._rows(values, explicit)
        existing = self._winner(by_uuid, by_key, values, payload, producer, explicit)
        if existing:
            return existing, True

        # Keep the same submitted comparison envelope for both winner passes;
        # candidate identity/defaults below must not change replay validation.
        submitted_payload = payload
        # No stored identity exists. Mint ONLY here (or use the caller's UUID).
        token = uuid.UUID(explicit) if explicit else uuid.uuid4()
        delivery_id, message_id, turn_id = stable_ids(token)
        payload = dict(payload)
        payload.update(user_message_id=message_id,
                       assistant_message_id=assistant or assistant_id(message_id),
                       inter_bot_delivery_id=delivery_id,
                       inter_bot_turn_id=turn_id,
                       inter_bot_bridge_request_id=f"req_delivery_{token.hex}")
        metadata = dict(metadata)
        metadata["_request_identity_producer"] = producer  # overwrite untrusted metadata
        if producer == "ordinary":
            ordinary = dict(metadata.get("ordinary_chat") or {})
            ordinary["assistant_supplied"] = explicit_assistant is not None
            # Comparison excludes generated transport fields, as with admission.
            comparison = {k: v for k, v in payload.items()
                          if k not in {"inter_bot_delivery_id", "inter_bot_turn_id",
                                       "inter_bot_bridge_request_id"}}
            ordinary["digest"] = self._digest(comparison)
            metadata["ordinary_chat"] = ordinary
        values = {**values, "id": delivery_id, "message_id": message_id,
                  "turn_id": turn_id,
                  "payload": json.dumps(payload, ensure_ascii=False, default=str),
                  "metadata": json.dumps(metadata, ensure_ascii=False, default=str)}
        row = self.conn.execute(sa_text("""
            INSERT INTO inter_bot_deliveries (
                id, sender_bot_id, author_entity_type, author_entity_id,
                target_bot_id, message, payload_json, metadata_json,
                user_message_id, turn_id, idempotency_key, project_id, task_id,
                message_kind, status, max_attempts, session_policy, reset_status, reset_reason
            ) VALUES (
                :id, :sender, :author_type, :author_id, :target, :message,
                CAST(:payload AS jsonb), CAST(:metadata AS jsonb), :message_id,
                :turn_id, :idem, :project, :task, :kind, 'QUEUED', :max_attempts,
                :session_policy,
                CASE WHEN :session_policy='continue' THEN NULL ELSE 'PENDING' END,
                :reset_reason
            ) ON CONFLICT DO NOTHING RETURNING *
        """), values).mappings().first()
        if row is not None:
            return DeliveryRecord.from_mapping(row), False
        by_uuid, by_key = self._rows(values, explicit or message_id)
        # An unkeyed, generated-token collision cannot be claimed as this operation.
        winner = self._winner(by_uuid if explicit else None, by_key, values,
                              submitted_payload, producer, explicit)
        if winner is None:
            raise AdmissionUnavailable(f"identity insert lost without winner: candidate={delivery_id}")
        return winner, True
