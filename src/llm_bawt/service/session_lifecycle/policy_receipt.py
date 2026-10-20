"""Receipt for a reset-policy delivery followed by its unchanged task reply."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import text

from .intent import NewSessionResult, Origin, Outcome, Stage
from .outbox import NewSessionOutbox
from .receipt import RESET_CONFIRMATION


class BotPolicyReceipt:
    """Commit one provider-backed reset receipt, never rewriting the task answer."""

    @classmethod
    def reconcile(cls, engine) -> list:
        """Settle delivered follow-ups after a dispatcher crash, without replay."""
        from ...inter_bot_delivery_record import DeliveryRecord
        from .recovery import NewSessionRecovery

        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT * FROM inter_bot_deliveries WHERE status='DELIVERED'
                  AND delivery_mode='turn'
                  AND session_policy IN ('reset_retain_history', 'reset_without_history')
                  AND metadata_json->'new_session_receipt' IS NULL
                ORDER BY updated_at LIMIT 50
            """)).mappings().all()
        settled = []
        for row in rows:
            record = DeliveryRecord.from_mapping(row)
            result = cls.ensure(engine, record)
            if result is None:
                current = NewSessionRecovery.fail_missing_policy_receipt(
                    engine, record.id,
                    "reset follow-up finished without exact provider binding and persisted receipt",
                )
            else:
                with engine.connect() as conn:
                    latest = conn.execute(text("""
                        SELECT * FROM inter_bot_deliveries WHERE id=:id
                    """), {"id": record.id}).mappings().one()
                current = DeliveryRecord.from_mapping(latest)
            if current is not None:
                settled.append(current)
        return settled

    @staticmethod
    def ensure(engine, record) -> NewSessionResult | None:
        """Validate exact turn, lineage and provider key before durable completion."""
        if (record.session_policy not in {"reset_retain_history", "reset_without_history"}
                or record.delivery_mode != "turn"
                or record.reset_status not in {"ROTATED", "COMPLETED"}
                or not record.new_session_id):
            return None
        ack = record.metadata.get("new_session_provider_ack")
        if not isinstance(ack, dict) or ack.get("provider") not in {"claude-code", "codex"}:
            return None
        raw = record.metadata.get("new_session_intent")
        if isinstance(raw, dict) and raw.get("origin") in {Origin.CONTEXT_API, Origin.SESSION_API}:
            return BotPolicyReceipt._ensure_standalone(engine, record, ack)
        if record.session_policy == "reset_retain_history" and ack["provider"] != "claude-code":
            return None
        provider_id = ack.get("provider_session_id")
        if (not isinstance(provider_id, str) or not provider_id
                or ack.get("thread_session_id") != record.new_session_id):
            return None
        from ...memory.postgresql import _sanitize_table_name
        from ...memory.session_reset_lineage import SessionResetLineage

        NewSessionOutbox(engine)
        storage_bot = _sanitize_table_name(record.target_bot_id)
        receipt_id = str(uuid5(NAMESPACE_URL, f"new-session-policy-receipt:{record.id}"))
        message_id = str(uuid5(NAMESPACE_URL, f"new-session-policy-message:{record.id}"))
        with engine.begin() as conn:
            row = conn.execute(text("""
                SELECT d.*, t.status AS turn_status, t.ended_at AS turn_ended_at,
                       t.error_text AS turn_error, t.user_id AS turn_user_id,
                       t.response_text AS turn_response_text,
                       t.trigger_message_id AS turn_trigger_id,
                       t.assistant_message_id AS turn_assistant_id
                FROM inter_bot_deliveries d JOIN turn_logs t ON t.id=d.turn_id
                WHERE d.id=:id FOR UPDATE OF d
            """), {"id": record.id}).mappings().one_or_none()
            if (row is None or row["status"] != "DELIVERED"
                    or row["transport_accepted_at"] is None
                    or row["target_bot_id"] != record.target_bot_id
                    or row["turn_id"] != record.turn_id
                    or row["user_message_id"] != record.user_message_id
                    or row["reset_status"] not in {"ROTATED", "COMPLETED"}
                    or row["turn_status"] not in {"ok", "completed"}
                    or row["turn_ended_at"] is None or row["turn_error"]
                    or row["turn_trigger_id"] != row["user_message_id"]
                    or row["new_session_id"] != record.new_session_id
                    or row["old_session_id"] != record.old_session_id
                    or (row["metadata_json"] or {}).get("new_session_provider_ack") != ack):
                return None
            payload = row["payload_json"] or {}
            messages = payload.get("messages") or []
            supplied = messages[-1].get("content") if messages and isinstance(messages[-1], dict) else None
            assistant_id = payload.get("assistant_message_id")
            if (not isinstance(supplied, str) or not supplied
                    or not assistant_id or assistant_id != row["turn_assistant_id"]):
                return None
            user = conn.execute(text("""
                SELECT session_id, content, timestamp, author_entity_type, author_entity_id
                FROM messages WHERE bot_id=:bot AND id=:id AND role='user'
            """), {"bot": storage_bot, "id": row["user_message_id"]}).mappings().one_or_none()
            if (user is None or user["session_id"] != record.new_session_id
                    or user["content"] != supplied
                    or user["author_entity_type"] != record.author.entity_type
                    or user["author_entity_id"] != record.author.entity_id):
                return None
            answer = conn.execute(text("""
                SELECT session_id, content FROM messages
                WHERE bot_id=:bot AND id=:id AND role='assistant'
            """), {"bot": storage_bot, "id": assistant_id}).mappings().one_or_none()
            if (answer is None or answer["session_id"] != record.new_session_id
                    or answer["content"] != row["turn_response_text"]):
                return None
            successor = conn.execute(text("""
                SELECT bot_id, user_id, session_metadata FROM sessions WHERE id=:id
            """), {"id": record.new_session_id}).mappings().one_or_none()
            if (successor is None or successor["bot_id"] != record.target_bot_id
                    or successor["user_id"] != row["turn_user_id"]):
                return None
            lineage = SessionResetLineage.metadata(successor["session_metadata"])
            predecessors = [record.old_session_id] if record.old_session_id else []
            if (lineage.get("rotation_delivery_id") != record.id
                    or lineage.get("reset_predecessor_ids") != predecessors
                    or (successor["session_metadata"] or {}).get(
                        "agent_session_keys", {}).get(ack["provider"].replace("-", "_")) != provider_id):
                return None
            for predecessor in predecessors:
                prior = conn.execute(text("""
                    SELECT bot_id, user_id, session_metadata FROM sessions WHERE id=:id
                """), {"id": predecessor}).mappings().one_or_none()
                if (prior is None or prior["bot_id"] != record.target_bot_id
                        or prior["user_id"] != row["turn_user_id"]
                        or SessionResetLineage.metadata(prior["session_metadata"])
                        .get("reset_successor_id") != record.new_session_id):
                    return None
            prior_receipt = (row["metadata_json"] or {}).get("new_session_receipt")
            if prior_receipt is not None:
                if (row["reset_status"] != "COMPLETED"
                        or not isinstance(prior_receipt, dict)
                        or prior_receipt.get("id") != receipt_id
                        or prior_receipt.get("message_id") != message_id
                        or prior_receipt.get("new_session_id") != record.new_session_id
                        or prior_receipt.get("provider_session_id") != provider_id):
                    return None
                receipt = conn.execute(text("""
                    SELECT content FROM messages WHERE bot_id=:bot AND id=:id
                      AND role='assistant' AND session_id=:session
                """), {"bot": storage_bot, "id": message_id,
                        "session": record.new_session_id}).scalar_one_or_none()
                if receipt != RESET_CONFIRMATION:
                    return None
            else:
                if row["reset_status"] != "ROTATED":
                    return None
                conn.execute(text("""
                    INSERT INTO messages (bot_id, id, role, content, timestamp,
                                          session_id, author_entity_type,
                                          author_entity_id, processed)
                    VALUES (:bot, :id, 'assistant', :content, :timestamp,
                            :session, 'bot', :author, true)
                """), {"bot": storage_bot, "id": message_id,
                        "content": RESET_CONFIRMATION,
                        "timestamp": float(user["timestamp"]) - 0.0001,
                        "session": record.new_session_id,
                        "author": record.target_bot_id})
                receipt_data = {
                    "id": receipt_id, "message_id": message_id,
                    "new_session_id": record.new_session_id,
                    "provider_session_id": provider_id,
                    "confirmed_at": datetime.now(timezone.utc).isoformat(),
                }
                conn.execute(text("""
                    UPDATE inter_bot_deliveries SET reset_status='COMPLETED',
                        metadata_json=jsonb_set(metadata_json, '{new_session_receipt}',
                                                CAST(:receipt AS jsonb)), updated_at=now()
                    WHERE id=:id
                """), {"id": record.id, "receipt": json.dumps(receipt_data)})
                NewSessionOutbox.enqueue(
                    conn, delivery_id=record.id, bot_id=record.target_bot_id,
                    user_id=row["turn_user_id"], new_session_id=record.new_session_id,
                    message_id=message_id,
                )
        return NewSessionResult(
            outcome=Outcome.COMPLETED, stage=Stage.RECEIPT,
            delivery_id=record.id, turn_id=record.turn_id,
            old_session_id=record.old_session_id,
            new_session_id=record.new_session_id, receipt_message_id=message_id,
        )

    @staticmethod
    def _ensure_standalone(engine, record, ack) -> NewSessionResult | None:
        """A reset-only delivery has one app-authored bubble, not a task answer."""
        from ...memory.postgresql import _sanitize_table_name
        from ...memory.session_reset_lineage import SessionResetLineage
        from .lifecycle import NewSessionLifecycle

        raw = record.metadata.get("new_session_intent")
        if (record.message != "/new" or not isinstance(raw, dict)
                or ack.get("thread_session_id") != record.new_session_id):
            return None
        try:
            intent = NewSessionLifecycle._decode(raw)
        except (TypeError, ValueError):
            return None
        if (intent.origin not in {Origin.CONTEXT_API, Origin.SESSION_API}
                or intent.follow_up is not None or intent.bot_id != record.target_bot_id
                or intent.author != record.author
                or intent.expected_session_id != record.old_session_id
                or not intent.expected_session_id):
            return None
        provider = ack["provider"]
        provider_id = ack.get("provider_session_id")
        if record.session_policy == "reset_without_history":
            if provider_id is not None or ack.get("clean_start") is not True:
                return None
        elif record.session_policy == "reset_retain_history":
            if provider != "claude-code" or not isinstance(provider_id, str) or not provider_id:
                return None
        else:
            return None
        NewSessionOutbox(engine)
        storage_bot = _sanitize_table_name(intent.bot_id)
        receipt_id = str(uuid5(NAMESPACE_URL, f"new-session-policy-receipt:{record.id}"))
        with engine.begin() as conn:
            row = conn.execute(text("""
                SELECT d.*, t.status AS turn_status, t.ended_at AS turn_ended_at,
                       t.error_text AS turn_error, t.user_id AS turn_user_id,
                       t.trigger_message_id AS turn_trigger_id,
                       t.assistant_message_id AS turn_assistant_id
                FROM inter_bot_deliveries d JOIN turn_logs t ON t.id=d.turn_id
                WHERE d.id=:id FOR UPDATE OF d
            """), {"id": record.id}).mappings().one_or_none()
            if (row is None or row["status"] != "DELIVERED"
                    or row["transport_accepted_at"] is None
                    or row["reset_status"] not in {"ROTATED", "COMPLETED"}
                    or row["turn_status"] not in {"ok", "completed"}
                    or row["turn_ended_at"] is None or row["turn_error"]
                    or row["turn_user_id"] != intent.user_id
                    or row["target_bot_id"] != intent.bot_id
                    or row["user_message_id"] != row["turn_trigger_id"]
                    or row["turn_id"] != record.turn_id
                    or row["new_session_id"] != record.new_session_id
                    or row["old_session_id"] != intent.expected_session_id
                    or (row["metadata_json"] or {}).get("new_session_provider_ack") != ack
                    or (row["metadata_json"] or {}).get("new_session_intent") != raw
                    or (row["payload_json"] or {}).get("inter_bot_reset_only") is not True
                    or (row["payload_json"] or {}).get("messages") != [
                        {"role": "user", "content": "/new"}]):
                return None
            assistant_id = (row["payload_json"] or {}).get("assistant_message_id")
            if not assistant_id or assistant_id != row["turn_assistant_id"]:
                return None
            user = conn.execute(text("""
                SELECT session_id, content, author_entity_type, author_entity_id
                FROM messages WHERE bot_id=:bot AND id=:id AND role='user'
            """), {"bot": storage_bot, "id": row["user_message_id"]}).mappings().one_or_none()
            if (user is None or user["session_id"] != record.new_session_id
                    or user["content"] != "/new"
                    or user["author_entity_type"] != intent.author.entity_type
                    or user["author_entity_id"] != intent.author.entity_id):
                return None
            successor = conn.execute(text("""
                SELECT bot_id, user_id, session_metadata FROM sessions WHERE id=:id
            """), {"id": record.new_session_id}).mappings().one_or_none()
            prior = conn.execute(text("""
                SELECT bot_id, user_id, session_metadata FROM sessions WHERE id=:id
            """), {"id": record.old_session_id}).mappings().one_or_none()
            if (successor is None or prior is None
                    or (successor["bot_id"], successor["user_id"]) != (intent.bot_id, intent.user_id)
                    or (prior["bot_id"], prior["user_id"]) != (intent.bot_id, intent.user_id)
                    or SessionResetLineage.metadata(prior["session_metadata"])
                        .get("reset_successor_id") != record.new_session_id):
                return None
            lineage = SessionResetLineage.metadata(successor["session_metadata"])
            bound = (successor["session_metadata"] or {}).get(
                "agent_session_keys", {}).get(provider.replace("-", "_"))
            if (lineage.get("rotation_delivery_id") != record.id
                    or lineage.get("reset_predecessor_ids") != [record.old_session_id]
                    or (bound != provider_id if provider_id else bool(bound))):
                return None
            previous = (row["metadata_json"] or {}).get("new_session_receipt")
            if previous is not None and (row["reset_status"] != "COMPLETED"
                    or not isinstance(previous, dict)
                    or previous.get("id") != receipt_id
                    or previous.get("message_id") != assistant_id
                    or previous.get("new_session_id") != record.new_session_id
                    or previous.get("provider_session_id") != provider_id):
                return None
            reply = conn.execute(text("""
                SELECT content, author_entity_type, author_entity_id FROM messages
                WHERE bot_id=:bot AND id=:id AND role='assistant' AND session_id=:session
            """), {"bot": storage_bot, "id": assistant_id,
                    "session": record.new_session_id}).mappings().one_or_none()
            if reply is None or (previous is not None and (
                    reply["content"] != RESET_CONFIRMATION
                    or reply["author_entity_type"] != "bot"
                    or reply["author_entity_id"] != intent.bot_id)):
                return None
            if previous is None:
                if row["reset_status"] != "ROTATED":
                    return None
                conn.execute(text("""
                    UPDATE messages SET content=:content, author_entity_type='bot',
                        author_entity_id=:bot, processed=true
                    WHERE bot_id=:storage_bot AND id=:assistant
                """), {"content": RESET_CONFIRMATION, "bot": intent.bot_id,
                        "storage_bot": storage_bot, "assistant": assistant_id})
                conn.execute(text("""
                    UPDATE turn_logs SET response_text=:content WHERE id=:turn
                """), {"content": RESET_CONFIRMATION, "turn": record.turn_id})
                conn.execute(text("""
                    UPDATE inter_bot_deliveries SET reset_status='COMPLETED',
                        metadata_json=jsonb_set(metadata_json, '{new_session_receipt}',
                                                CAST(:receipt AS jsonb)), updated_at=now()
                    WHERE id=:id
                """), {"id": record.id, "receipt": json.dumps({
                    "id": receipt_id, "message_id": assistant_id,
                    "new_session_id": record.new_session_id,
                    "provider_session_id": provider_id,
                    "confirmed_at": datetime.now(timezone.utc).isoformat(),
                })})
                NewSessionOutbox.enqueue(
                    conn, delivery_id=record.id, bot_id=intent.bot_id,
                    user_id=intent.user_id, new_session_id=record.new_session_id,
                    message_id=assistant_id,
                )
        return NewSessionResult(
            outcome=Outcome.COMPLETED, stage=Stage.RECEIPT,
            delivery_id=record.id, turn_id=record.turn_id,
            old_session_id=record.old_session_id,
            new_session_id=record.new_session_id, receipt_message_id=assistant_id,
        )
