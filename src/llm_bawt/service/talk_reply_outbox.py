"""Durable Talk reply subscriptions, drained by the existing delivery leader.

This is an outbound notification receipt, not a second agent-work queue. An
outbound POST cannot be replayed after an unknown result: claim before sending
and leave an interrupted/failed send UNCERTAIN for operator reconciliation.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from sqlalchemy import text

from ..utils.schema import SchemaBootstrapGuard
from .delivery_reply import resolve_delivery_reply, resolve_direct_turn_reply

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TalkReplySubscription:
    message_id: str
    bot_id: str
    user_id: str
    source_type: str
    source_id: str
    backend: str
    conversation_token: str
    reply_to: int | None


class TalkReplyOutbox:
    _schema_guard = SchemaBootstrapGuard()

    def __init__(self, engine):
        if engine is None:
            raise RuntimeError("Talk reply subscriptions require the delivery database")
        self.engine = engine
        self._schema_guard.run(engine, "talk-reply-subscriptions", self._bootstrap)

    @staticmethod
    def _bootstrap(conn):
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS talk_reply_subscriptions (
                message_id VARCHAR(96) PRIMARY KEY,
                bot_id VARCHAR(128) NOT NULL,
                user_id VARCHAR(128) NOT NULL,
                source_type VARCHAR(16) NOT NULL CHECK (source_type IN ('delivery','turn')),
                source_id VARCHAR(128) NOT NULL,
                backend TEXT NOT NULL,
                conversation_token TEXT NOT NULL,
                reply_to BIGINT,
                state VARCHAR(16) NOT NULL DEFAULT 'PENDING'
                  CHECK (state IN ('PENDING','SENDING','SENT','UNCERTAIN')),
                last_error TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                next_check_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """))
        conn.execute(text("""
            ALTER TABLE talk_reply_subscriptions
            ADD COLUMN IF NOT EXISTS next_check_at TIMESTAMPTZ NOT NULL DEFAULT now()
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_talk_reply_subscriptions_ready
            ON talk_reply_subscriptions (next_check_at) WHERE state='PENDING'
        """))

    def register(self, subscription: TalkReplySubscription) -> tuple[str, bool]:
        """Persist before the webhook ACK; retries must match every destination field."""
        values = vars(subscription)
        with self.engine.begin() as conn:
            inserted = conn.execute(text("""
                INSERT INTO talk_reply_subscriptions
                  (message_id,bot_id,user_id,source_type,source_id,backend,conversation_token,reply_to)
                VALUES (:message_id,:bot_id,:user_id,:source_type,:source_id,:backend,:conversation_token,:reply_to)
                ON CONFLICT (message_id) DO NOTHING
            """), values)
            row = conn.execute(text("""
                SELECT bot_id,user_id,source_type,source_id,backend,conversation_token,reply_to,state
                FROM talk_reply_subscriptions WHERE message_id=:message_id
            """), {"message_id": subscription.message_id}).mappings().one()
            if any(row[key] != value for key, value in values.items() if key != "message_id"):
                raise ValueError("Talk message id already belongs to a different reply destination")
            return row["state"], inserted.rowcount == 1

    def pending(self, *, limit: int = 20) -> list[TalkReplySubscription]:
        with self.engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT message_id,bot_id,user_id,source_type,source_id,backend,conversation_token,reply_to
                FROM talk_reply_subscriptions
                WHERE state='PENDING' AND next_check_at <= now()
                ORDER BY next_check_at LIMIT :limit
            """), {"limit": limit}).mappings().all()
        return [TalkReplySubscription(**row) for row in rows]

    def defer(self, message_id: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(text("""
                UPDATE talk_reply_subscriptions
                SET next_check_at=now() + interval '15 seconds'
                WHERE message_id=:id AND state='PENDING'
            """), {"id": message_id})

    def claim(self, message_id: str) -> bool:
        # The singleton leader is not the only fence: a second process can take
        # leadership mid-POST. Only one transaction may enter the outbound call.
        with self.engine.begin() as conn:
            return bool(conn.execute(text("""
                UPDATE talk_reply_subscriptions SET state='SENDING',updated_at=now()
                WHERE message_id=:id AND state='PENDING' RETURNING message_id
            """), {"id": message_id}).first())

    def settle(self, message_id: str, state: str, error: str | None = None) -> None:
        if state not in {"SENT", "UNCERTAIN"}:
            raise ValueError("invalid outbound settlement")
        with self.engine.begin() as conn:
            conn.execute(text("""
                UPDATE talk_reply_subscriptions
                SET state=:state,last_error=:error,updated_at=now()
                WHERE message_id=:id AND state='SENDING'
            """), {"id": message_id, "state": state, "error": error})

    async def drain(self, deliveries, turns) -> None:
        """Send only an exact, finished answer; never infer text from a transport ACK."""
        from ..integrations.nextcloud.manager import get_nextcloud_manager
        from ..integrations.nextcloud.webhook import send_nextcloud_message

        for subscription in await asyncio.to_thread(self.pending):
            try:
                if subscription.source_type == "delivery":
                    receipt = await asyncio.to_thread(
                        resolve_delivery_reply, deliveries, turns, subscription.source_id,
                        bot_id=subscription.bot_id, user_id=subscription.user_id,
                        user_message_id=subscription.message_id,
                    )
                else:
                    receipt = await asyncio.to_thread(
                        resolve_direct_turn_reply, turns, subscription.source_id,
                        bot_id=subscription.bot_id, user_id=subscription.user_id,
                        user_message_id=subscription.message_id,
                    )
                if receipt is None:
                    # Absence may be a transaction race; never invent a reply.
                    await asyncio.to_thread(self.defer, subscription.message_id)
                    continue
                if receipt["status"] in {"FAILED", "CANCELLED"}:
                    # A terminal failure has no answer text; report only the
                    # verified outcome, not the raw exception or a fake reply.
                    answer = (
                        "Your request ended without a verified answer "
                        f"(reference {subscription.source_id}; status {receipt['status']})."
                    )
                else:
                    answer = receipt.get("response_text") if receipt["status"] == "DELIVERED" else None
                if not isinstance(answer, str) or not answer:
                    await asyncio.to_thread(self.defer, subscription.message_id)
                    continue
                bot_config = get_nextcloud_manager().get_bot_by_conversation(subscription.conversation_token)
                if not bot_config or bot_config.llm_bawt_bot != subscription.bot_id:
                    logger.error("Talk reply destination no longer matches bot for %s", subscription.message_id)
                    await asyncio.to_thread(self.defer, subscription.message_id)
                    continue
                if not await asyncio.to_thread(self.claim, subscription.message_id):
                    continue
                # Crash after claim or network ambiguity => SENDING/UNCERTAIN,
                # never automatic resend without an idempotent Talk send API.
                try:
                    success = await send_nextcloud_message(
                        nextcloud_url=subscription.backend,
                        conversation_token=subscription.conversation_token,
                        message=answer, secret=bot_config.secret,
                        reply_to=subscription.reply_to,
                    )
                except Exception as exc:
                    logger.exception("Talk outbound result uncertain for %s", subscription.message_id)
                    await asyncio.to_thread(self.settle, subscription.message_id, "UNCERTAIN", str(exc))
                else:
                    await asyncio.to_thread(
                        self.settle, subscription.message_id,
                        "SENT" if success else "UNCERTAIN",
                        None if success else "Talk outbound result not confirmed",
                    )
            except Exception:
                logger.exception("Talk reply subscription check failed for %s", subscription.message_id)
                try:
                    await asyncio.to_thread(self.defer, subscription.message_id)
                except Exception:
                    logger.exception("Could not defer Talk reply check for %s", subscription.message_id)
