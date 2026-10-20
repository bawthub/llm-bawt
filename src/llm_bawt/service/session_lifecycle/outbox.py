"""Transactional reset event outbox; Redis fanout is never completion evidence."""
from __future__ import annotations

import json
import logging

from sqlalchemy import text

from ...utils.schema import SchemaBootstrapGuard
from ..runtime_switch_rotation import thread_switched_event

log = logging.getLogger(__name__)


class NewSessionOutbox:
    _schema_guard = SchemaBootstrapGuard()

    def __init__(self, engine):
        self.engine = engine
        if engine is not None:
            self._schema_guard.run(engine, "new-session-outbox", self._bootstrap)

    @staticmethod
    def _bootstrap(conn) -> None:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS new_session_event_outbox (
                delivery_id VARCHAR(96) PRIMARY KEY,
                bot_id VARCHAR(128) NOT NULL,
                user_id VARCHAR(128) NOT NULL,
                new_session_id VARCHAR(36) NOT NULL,
                payload_json JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                published_at TIMESTAMPTZ
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_new_session_event_outbox_pending
            ON new_session_event_outbox (created_at) WHERE published_at IS NULL
        """))

    @staticmethod
    def enqueue(conn, *, delivery_id: str, bot_id: str, user_id: str,
                new_session_id: str, message_id: str) -> None:
        event = thread_switched_event(bot_id, user_id, new_session_id, "new_session")
        event.update(event_id=f"new-session:{delivery_id}", receipt_message_id=message_id)
        conn.execute(text("""
            INSERT INTO new_session_event_outbox
                (delivery_id, bot_id, user_id, new_session_id, payload_json)
            VALUES (:id, :bot, :user, :session, CAST(:payload AS jsonb))
            ON CONFLICT (delivery_id) DO NOTHING
        """), {"id": delivery_id, "bot": bot_id, "user": user_id,
               "session": new_session_id, "payload": json.dumps(event)})

    async def drain(self, subscriber, *, limit: int = 50) -> int:
        """Publish pending events; stale successors never switch the UI back."""
        if self.engine is None or subscriber is None:
            return 0
        with self.engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT delivery_id, bot_id, user_id, new_session_id, payload_json
                FROM new_session_event_outbox WHERE published_at IS NULL
                ORDER BY created_at, delivery_id LIMIT :limit
            """), {"limit": limit}).mappings().all()
        sent = 0
        for row in rows:
            try:
                with self.engine.connect() as conn:
                    current = conn.execute(text("""
                        SELECT last_session_id FROM bot_profiles WHERE slug=:bot
                    """), {"bot": row["bot_id"]}).scalar_one_or_none()
                if current == row["new_session_id"]:
                    stream_id = await subscriber.publish_tool_event(
                        row["bot_id"], row["user_id"], row["payload_json"],
                    )
                    if not stream_id:
                        raise RuntimeError("new-session event publication was not acknowledged")
                # A newer accepted session supersedes this event; the receipt
                # remains visible in history but a late event cannot rewind UI.
                with self.engine.begin() as conn:
                    conn.execute(text("""
                        UPDATE new_session_event_outbox SET published_at=now()
                        WHERE delivery_id=:id AND published_at IS NULL
                    """), {"id": row["delivery_id"]})
                sent += 1
            except Exception:
                log.warning("new-session event publication deferred for %s", row["delivery_id"],
                            exc_info=True)
                break
        return sent
