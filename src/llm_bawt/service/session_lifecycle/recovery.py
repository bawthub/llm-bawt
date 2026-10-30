"""Fail closed on accepted reset commands without replaying provider side effects."""
from __future__ import annotations

from sqlalchemy import text

from agent_bridge.maintenance import is_new_command

from ...inter_bot_delivery_ledger import DeliveryTransitionEvents
from ...inter_bot_delivery_record import DeliveryRecord


class NewSessionRecovery:
    @staticmethod
    def fail_missing_typed_receipt(engine, delivery_id: str, reason: str) -> DeliveryRecord | None:
        """Never advertise a completed typed reset without an exact receipt."""
        with engine.begin() as conn:
            row = conn.execute(text("""
                UPDATE inter_bot_deliveries SET status='FAILED',
                    next_retry_at=NULL, last_error=:reason,
                    metadata_json=jsonb_set(metadata_json, '{new_session_ambiguous}',
                                            'true'::jsonb), updated_at=now()
                WHERE id=:id AND status='DELIVERED'
                  AND transport_accepted_at IS NOT NULL
                  AND metadata_json->'new_session_intent'->>'origin'='chat_command'
                  AND metadata_json->'new_session_receipt' IS NULL
                RETURNING *
            """), {"id": delivery_id, "reason": reason[:4000]}).mappings().one_or_none()
            DeliveryTransitionEvents.stage(conn, row, reason_code="reset_receipt_missing")
        return DeliveryRecord.from_mapping(row) if row else None

    @staticmethod
    def fail_missing_policy_receipt(engine, delivery_id: str, reason: str) -> DeliveryRecord | None:
        """Terminalize a delivered follow-up with no verified provider reset."""
        with engine.begin() as conn:
            row = conn.execute(text("""
                UPDATE inter_bot_deliveries SET status='FAILED',
                    next_retry_at=NULL, last_error=:reason,
                    metadata_json=jsonb_set(metadata_json, '{new_session_ambiguous}',
                                            'true'::jsonb), updated_at=now()
                WHERE id=:id AND status='DELIVERED'
                  AND transport_accepted_at IS NOT NULL AND delivery_mode='turn'
                  AND session_policy IN ('reset_retain_history', 'reset_without_history')
                  AND metadata_json->'new_session_receipt' IS NULL
                RETURNING *
            """), {"id": delivery_id, "reason": reason[:4000]}).mappings().one_or_none()
            DeliveryTransitionEvents.stage(conn, row, reason_code="reset_receipt_missing")
        return DeliveryRecord.from_mapping(row) if row else None

    @classmethod
    def refuse_accepted_replay(cls, conn, head) -> bool:
        """A queued accepted reset has no proven cross-restart bridge dedupe."""
        if head.get("transport_accepted_at") is None or not cls.is_reset(head):
            return False
        reason = "accepted reset was requeued; outcome ambiguous without replay proof"
        DeliveryTransitionEvents.stage(conn, conn.execute(text("""
            UPDATE inter_bot_deliveries SET status='FAILED',
                next_retry_at=NULL, last_error=:reason,
                metadata_json=jsonb_set(metadata_json, '{new_session_ambiguous}',
                                        'true'::jsonb), updated_at=now()
            WHERE id=:id AND status='QUEUED' RETURNING *
        """), {"id": head["id"], "reason": reason}).mappings().first(), reason_code="reset_ambiguous")
        conn.execute(text("""
            UPDATE turn_logs SET status='error', end_reason='error',
                error_text=:reason, ended_at=COALESCE(ended_at, now())
            WHERE id=:turn AND ended_at IS NULL
        """), {"turn": head["turn_id"], "reason": reason})
        return True

    @staticmethod
    def is_reset(row) -> bool:
        if row.get("session_policy") not in (None, "continue"):
            return True
        metadata = row.get("metadata_json") or {}
        if metadata.get("new_session_intent"):
            return True
        payload = row.get("payload_json") or {}
        messages = payload.get("messages") or []
        if messages and isinstance(messages[-1], dict):
            content = messages[-1].get("content")
            return is_new_command(content if isinstance(content, str) else None)
        return False

    @staticmethod
    def mark_ambiguous(conn, *, delivery_id: str, claim_token: str | None,
                       reason: str, status: str = "DISPATCHING") -> DeliveryRecord | None:
        """Terminalize a single claim, preserving ambiguous acceptance evidence."""
        params = {"id": delivery_id, "token": claim_token or "",
                  "status": status, "reason": reason[:4000]}
        row = conn.execute(text("""
            UPDATE inter_bot_deliveries SET status='FAILED',
                claim_token=NULL, claim_owner=NULL, lease_expires_at=NULL,
                next_retry_at=NULL, last_error=:reason,
                metadata_json=jsonb_set(metadata_json, '{new_session_ambiguous}',
                                        'true'::jsonb), updated_at=now()
            WHERE id=:id AND status=:status AND claim_token=:token
              AND transport_accepted_at IS NOT NULL RETURNING *
        """), params).mappings().one_or_none()
        DeliveryTransitionEvents.stage(conn, row, reason_code="reset_ambiguous")
        if row is not None:
            conn.execute(text("""
                UPDATE turn_logs SET status='error', end_reason='error',
                    error_text=:reason, ended_at=COALESCE(ended_at, now())
                WHERE id=:turn AND ended_at IS NULL
            """), {"turn": row["turn_id"], "reason": reason[:4000]})
        return DeliveryRecord.from_mapping(row) if row else None
