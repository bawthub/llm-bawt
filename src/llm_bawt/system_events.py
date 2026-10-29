"""One durable, user-scoped ledger for operational events (TASK-1745).

Domain writers append in their existing transaction. Redis is an outbox projection,
never the owner of a receipt; a crash after XADD and before stamping may replay a
row, so consumers deduplicate by its immutable database id.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import (
    JSON, BigInteger, CheckConstraint, Column, DateTime, Identity, Index,
    Integer, MetaData, String, Table, UniqueConstraint, delete, func,
    insert, select, text, update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Connection, Engine

from agent_bridge.publisher import SYSTEM_EVENTS_PREFIX

from .bot_tags.facade import BotTags
from .bot_tags.vocabulary import BehaviorTag
from .profiles import EntityProfile, EntityType
from .utils.schema import SchemaBootstrapGuard

logger = logging.getLogger(__name__)

USER_FEED = "user_feed"
AUDIT_ONLY = "audit_only"
MAX_EVENTS = 200
RETENTION_DAYS = 7

_metadata = MetaData()
_system_events = Table(
    "system_events", _metadata,
    Column("id", BigInteger().with_variant(Integer(), "sqlite"), Identity(), primary_key=True),
    Column("event_key", String(256), nullable=False),
    Column("kind", String(64), nullable=False),
    Column("source", String(32), nullable=False),
    Column("user_id", String(128), nullable=True),
    Column("visibility", String(16), nullable=False),
    Column("bot_id", String(128), nullable=True),
    Column("actor_type", String(32), nullable=True),
    Column("actor_id", String(128), nullable=True),
    Column("subject_type", String(32), nullable=True),
    Column("subject_id", String(128), nullable=True),
    Column("tone", String(16), nullable=False),
    Column("summary", String(500), nullable=False),
    Column("payload", JSONB().with_variant(JSON(), "sqlite"), nullable=False, server_default="{}"),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("published_at", DateTime(timezone=True), nullable=True),
    UniqueConstraint("event_key", name="uq_system_events_key"),
    CheckConstraint("visibility IN ('user_feed', 'audit_only')", name="ck_system_events_visibility"),
    CheckConstraint("visibility = 'audit_only' OR user_id IS NOT NULL", name="ck_system_events_feed_owner"),
    CheckConstraint("tone IN ('info', 'success', 'warning', 'error')", name="ck_system_events_tone"),
    Index("idx_system_events_user_id_id", "user_id", "id"),
    Index("idx_system_events_outbox", "published_at", "id"),
)


def _receipt(row: Any) -> dict[str, Any]:
    """Only allowlisted fields ever leave the ledger through the user API/SSE."""
    item = row._mapping if hasattr(row, "_mapping") else row
    created = item["created_at"]
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return {
        "_type": "system_event", "id": str(item["id"]),
        "kind": item["kind"], "source": item["source"],
        "user_id": item["user_id"], "tone": item["tone"],
        "summary": item["summary"], "subject_type": item["subject_type"],
        "subject_id": item["subject_id"], "created_at": created.isoformat(),
    }


class SystemEventEmitter:
    """Owns schema, transactional append, scoped reads and outbox projection."""

    _schema_guard = SchemaBootstrapGuard()
    failed_appends = 0  # Process-local health signal; failures also log with key.

    def __init__(self, engine: Engine):
        self.engine = engine
        self._schema_guard.run(engine, "system-events-v1", lambda conn: _metadata.create_all(conn))

    def append(
        self, conn: Connection, *, event_key: str, kind: str, source: str,
        user_id: str | None, tone: str, summary: str,
        bot_id: str | None = None, actor_type: str | None = None,
        actor_id: str | None = None, subject_type: str | None = None,
        subject_id: str | None = None, payload: dict[str, Any] | None = None,
        visible: bool = True,
    ) -> int | None:
        """Append in the caller's transaction, without poisoning it on bad metadata.

        Missing owners are audit-only, never guessed. A savepoint isolates an
        unexpected projection failure from the producer's domain transaction.
        """
        if not event_key or len(event_key) > 256 or not kind or len(kind) > 64 or not source or len(source) > 32:
            type(self).failed_appends += 1
            logger.error("system event rejected: invalid event identity")
            return None
        if not isinstance(summary, str) or len(summary) > 500 or tone not in ("info", "success", "warning", "error"):
            type(self).failed_appends += 1
            logger.error("system event rejected: invalid display projection (key=%s)", event_key)
            return None
        if payload and (set(payload) - {"phase", "requester", "operation", "reason_code"}):
            type(self).failed_appends += 1
            logger.error("system event rejected: unallowlisted payload keys (key=%s)", event_key)
            return None
        try:
            if conn.dialect.name == "sqlite" and not conn.connection.driver_connection.in_transaction:
                # sqlite3's legacy mode does not BEGIN for SAVEPOINT; releasing
                # it would otherwise commit before the caller's outer transaction.
                conn.exec_driver_sql("BEGIN")
        except Exception:
            type(self).failed_appends += 1
            logger.exception("system event transaction unavailable (key=%s)", event_key)
            return None
        if bot_id:
            try:
                with conn.begin_nested():
                    if BotTags(self.engine).has(bot_id, BehaviorTag.PRIVATE, conn=conn):
                        visible = False  # Unknown/unreadable bot tags fail closed.
            except Exception:
                type(self).failed_appends += 1
                logger.exception("system event bot visibility unavailable; recording audit-only")
                visible = False
        if visible and user_id:
            try:
                with conn.begin_nested():
                    owner = conn.execute(select(EntityProfile.id).where(
                        EntityProfile.entity_type == EntityType.USER,
                        EntityProfile.entity_id == user_id,
                    ).limit(1)).first()
                if owner is None:
                    type(self).failed_appends += 1
                    logger.warning("system event owner not found; recording audit-only (key=%s)", event_key)
                    visible = False
            except Exception:
                type(self).failed_appends += 1
                logger.exception("system event owner check unavailable; recording audit-only (key=%s)", event_key)
                visible = False
        visibility = USER_FEED if visible and user_id else AUDIT_ONLY
        if visibility == AUDIT_ONLY:
            user_id, bot_id, actor_type, actor_id, subject_type, subject_id, payload = (
                None, None, None, None, None, None, {},
            )
            summary = "Restricted operational event"  # Do not retain a private summary.
        try:
            with conn.begin_nested():
                if conn.dialect.name == "postgresql":
                    # Hold until the outer transaction commits. The sequence id
                    # is therefore allocated in commit order, even across writers.
                    conn.execute(text("SELECT pg_advisory_xact_lock(hashtext(:lock))"),
                                 {"lock": "system-events-id-order"})
                existing = conn.execute(select(_system_events.c.id).where(
                    _system_events.c.event_key == event_key)).scalar_one_or_none()
                if existing is not None:
                    return int(existing)
                result = conn.execute(insert(_system_events).values(
                    event_key=event_key, kind=kind, source=source, user_id=user_id,
                    visibility=visibility, bot_id=bot_id, actor_type=actor_type,
                    actor_id=actor_id, subject_type=subject_type, subject_id=subject_id,
                    tone=tone, summary=summary, payload=payload or {},
                ).returning(_system_events.c.id))
                return int(result.scalar_one())
        except Exception:  # noqa: BLE001 - domain CAS must not be rolled back by projection
            type(self).failed_appends += 1
            logger.exception("system event append failed; domain transition continues (key=%s)", event_key)
            return None

    def list_page(
        self, user_id: str, *, after: int | None = None,
        before: int | None = None, limit: int = MAX_EVENTS,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Read enough safe rows for a page, even when a bot became private.

        Fetch one extra visible row to determine has_more without guessing from
        the page length; only server IDs cross the cursor boundary.
        """
        if not user_id or not 1 <= limit <= MAX_EVENTS or after is not None and before is not None:
            raise ValueError("invalid system events cursor or user scope")
        stmt = select(_system_events).where(
            _system_events.c.user_id == user_id,
            _system_events.c.visibility == USER_FEED,
        )
        ascending = after is not None
        cursor = after if ascending else before
        results = []
        with self.engine.connect() as conn:
            while len(results) <= limit:
                page = stmt
                if cursor is not None:
                    page = page.where(_system_events.c.id > cursor if ascending else
                                      _system_events.c.id < cursor)
                page = page.order_by(_system_events.c.id.asc() if ascending else
                                     _system_events.c.id.desc())
                rows = conn.execute(page.limit(MAX_EVENTS)).all()
                if not rows:
                    break
                tags = (BotTags(self.engine).snapshot(conn)
                        if any(row._mapping["bot_id"] for row in rows) else None)
                # A bot becoming private after emission cannot remain visible.
                results.extend(row for row in rows if not tags or not row._mapping["bot_id"]
                               or not tags.has(row._mapping["bot_id"], BehaviorTag.PRIVATE))
                cursor = rows[-1]._mapping["id"]
                if len(rows) < MAX_EVENTS:
                    break
        has_more = len(results) > limit
        results = results[:limit]
        if not ascending:
            results.reverse()
        return [_receipt(row) for row in results], has_more

    def get_user_event(self, user_id: str, event_id: int) -> dict[str, Any] | None:
        """Revalidate a Redis replay against current visibility before SSE."""
        if not user_id or event_id <= 0:
            return None
        with self.engine.connect() as conn:
            row = conn.execute(select(_system_events).where(
                _system_events.c.id == event_id,
                _system_events.c.user_id == user_id,
                _system_events.c.visibility == USER_FEED,
            )).first()
            if row is None:
                return None
            bot_id = row._mapping["bot_id"]
            if bot_id and BotTags(self.engine).has(bot_id, BehaviorTag.PRIVATE, conn=conn):
                return None
            return _receipt(row)

    def list_user(
        self, user_id: str, *, after: int | None = None,
        before: int | None = None, limit: int = MAX_EVENTS,
    ) -> list[dict[str, Any]]:
        return self.list_page(user_id, after=after, before=before, limit=limit)[0]

    def publish_batch(self, redis: Any, *, limit: int = 50, report_only: bool = False) -> int:
        """Bounded transactional outbox drain. Run off the asyncio event loop.

        A successful XADD followed by a crash before published_at intentionally
        replays; clients dedupe on id. A Redis error leaves the row pending.
        """
        if not 1 <= limit <= 200:
            raise ValueError("invalid outbox batch size")
        sent = 0
        with self.engine.begin() as conn:
            stmt = select(_system_events).where(
                _system_events.c.visibility == USER_FEED,
                _system_events.c.published_at.is_(None),
            ).order_by(_system_events.c.id.asc()).limit(limit)
            if conn.dialect.name == "postgresql":
                stmt = stmt.with_for_update(skip_locked=True)
            rows = conn.execute(stmt).all()
            tags = BotTags(self.engine).snapshot(conn) if any(r._mapping["bot_id"] for r in rows) else None
            if tags is not None and not tags.available:
                # Policy outage is not proof that formerly public rows became
                # private. Retry instead of irrevocably stamping them published.
                logger.warning("system event outbox waiting for bot visibility policy")
                return 0
            for row in rows:
                if report_only:
                    sent += 1
                    continue
                receipt = _receipt(row)
                is_private = bool(tags and row._mapping["bot_id"] and
                                  tags.has(row._mapping["bot_id"], BehaviorTag.PRIVATE))
                if not is_private:
                    try:
                        import json
                        redis.xadd(f"{SYSTEM_EVENTS_PREFIX}{receipt['user_id']}",
                                   {"payload": json.dumps(receipt, ensure_ascii=False)},
                                   maxlen=10000, approximate=True)
                    except Exception:  # noqa: BLE001 - retry after reconnect
                        logger.exception("system event outbox publish failed (id=%s)", receipt["id"])
                        break
                conn.execute(update(_system_events).where(
                    _system_events.c.id == int(receipt["id"]),
                    _system_events.c.published_at.is_(None),
                ).values(published_at=datetime.now(timezone.utc)))
                sent += 1
        return sent

    def prune(self, *, report_only: bool = False) -> int:
        """Prune only acknowledged or audit-only receipts after seven days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
        predicate = (_system_events.c.created_at < cutoff) & (
            (_system_events.c.visibility == AUDIT_ONLY) | _system_events.c.published_at.is_not(None))
        with self.engine.begin() as conn:
            if report_only:
                return int(conn.execute(select(func.count()).select_from(_system_events).where(predicate)).scalar_one())
            return int(conn.execute(delete(_system_events).where(predicate)).rowcount)

    def pending_health(self) -> tuple[int, float]:
        """Count and oldest lag in seconds for the internal health endpoint."""
        with self.engine.connect() as conn:
            count, oldest = conn.execute(select(func.count(), func.min(_system_events.c.created_at)).where(
                _system_events.c.visibility == USER_FEED,
                _system_events.c.published_at.is_(None),
            )).one()
        if oldest is None:
            return 0, 0.0
        if oldest.tzinfo is None:
            oldest = oldest.replace(tzinfo=timezone.utc)
        return int(count), max(0.0, (datetime.now(timezone.utc) - oldest).total_seconds())

    def pending_count(self) -> int:
        return self.pending_health()[0]
