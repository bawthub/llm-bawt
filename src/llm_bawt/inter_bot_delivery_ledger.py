"""Durable system-event receipts for inter-bot delivery transitions (TASK-1745).

Every status-changing ``UPDATE inter_bot_deliveries`` (and the enqueue INSERT)
stages its winning row with :meth:`DeliveryTransitionEvents.stage` on the same
connection. The staged rows are appended to the event ledger from the
connection's ``commit`` hook, so the append is the transaction's last write
whoever owns the transaction. The emitter takes a transaction-scoped advisory
lock to keep event ids in commit order; taking it after every domain write and
row lock means a multi-row transaction (recovery sweep, cancel cascade) cannot
deadlock against a claim that already holds the event lock. A rollback, or a
rolled-back savepoint, drops what it staged.

The audience rule is shared with the live ``inter_bot_delivery`` projection
(:func:`delivery_audience`), so a receipt lands in the same user's Events
panel that the live notice did.
"""

from __future__ import annotations

import logging
import re
import threading
import weakref
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy.engine import Connection, Engine

from .bot_tags.facade import BotTags
from .bot_tags.vocabulary import BehaviorTag
if TYPE_CHECKING:
    # Runtime imports are deferred: ``inter_bot_delivery_record`` imports
    # ``agent_context``, which stages delivery transitions through this module.
    from .inter_bot_delivery_record import DeliveryRecord
    from .system_events import SystemEventEmitter

logger = logging.getLogger(__name__)

# A reason code is a fixed slug, never free-form error text (payload allowlist).
_REASON_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_TONES = {
    "QUEUED": "info", "STEERING": "info", "DISPATCHING": "info",
    "DELIVERED": "success", "FAILED": "error", "CANCELLED": "warning",
}
# User-authored (ordinary chat) deliveries get a receipt only when they fail.
_USER_RECEIPT_STATUSES = frozenset({"FAILED", "CANCELLED"})


@dataclass(frozen=True)
class DeliveryAudience:
    """Who a delivery event belongs to and which bot it is filed under."""

    user_id: str | None
    bot_id: str
    ordinary_chat_user: str | None = None


def delivery_audience(record: DeliveryRecord, *, default_user: str | None) -> DeliveryAudience:
    """The owning user of ``record``'s lifecycle events.

    Ordinary chat sends belong to their user author (or the user a
    system-authored restart child carries); maintenance user-queue and prompt
    schedule rows to their user author; every other bot-to-bot relay to the
    service's configured default user. Precedence matches the live projection:
    maintenance user-queue, then ordinary chat, then prompt schedule.
    """
    chat_user = None
    if record.metadata.get("ordinary_chat"):
        # Deferred: the retry module imports the delivery store.
        from .inter_bot_delivery_retry import retry_original_author

        if record.author_entity_type == "user":
            chat_user = record.author_entity_id
        else:
            original = retry_original_author(record)
            if original is not None and original.entity_type == "user":
                chat_user = original.entity_id
    if record.metadata.get("maintenance_user_queue"):
        return DeliveryAudience(record.author_entity_id, record.target_bot_id, chat_user)
    if chat_user:
        return DeliveryAudience(chat_user, record.target_bot_id, chat_user)
    if schedule := record.metadata.get("prompt_schedule"):
        # A bot-created wake-up is authored by that bot but owned by the user
        # it was scheduled for (TASK-1790); older rows carry only the author.
        return DeliveryAudience(schedule.get("owner_user_id") or record.author_entity_id,
                                record.target_bot_id)
    return DeliveryAudience(default_user, record.sender_bot_id)


def configured_default_user() -> str | None:
    """The runtime-configured default user the live projection also routes to."""
    from .utils.config import config

    return getattr(config, "DEFAULT_USER", None) or None


class DeliveryTransitionEvents:
    """Stage winning delivery transitions; append them to the ledger at commit."""

    # Engines whose event table is known to exist. Weak keys, and a value that
    # never references its key: a disposed engine (tests, reconnects) is freed,
    # and a recycled id() never inherits another engine's state.
    _ready: "weakref.WeakKeyDictionary[Engine, bool]" = weakref.WeakKeyDictionary()
    _lock = threading.Lock()

    @classmethod
    def stage(cls, conn: Connection, *rows: DeliveryRecord | Mapping[str, Any] | None,
              reason_code: str | None = None) -> None:
        """Record rows a status-changing write just won, for this transaction's commit.

        Pass the ``RETURNING *`` row (or its record). ``None`` (a lost CAS) is
        skipped. Never raises: a staging fault costs a receipt, not a delivery.
        """
        try:
            from .inter_bot_delivery_record import DeliveryRecord
            from .system_events import CommitStagedAppends

            records = [r if isinstance(r, DeliveryRecord) else DeliveryRecord.from_mapping(r)
                       for r in rows if r is not None]
            if not records:
                return
            staged = [(r, reason_code) for r in records]
            CommitStagedAppends.stage(conn, lambda c: cls.flush(c, staged))
        except Exception:  # noqa: BLE001 - receipts never block a delivery
            cls._count_failure()
            logger.exception("could not stage a delivery event")

    @classmethod
    def install(cls, engine: Engine) -> None:
        """Build the event table for ``engine`` now, outside any transaction.

        A store calls this at construction, so a commit never has to. Never
        raises: without the table, staged rows are skipped at commit.
        """
        from .system_events import CommitStagedAppends

        try:
            CommitStagedAppends.install(engine)
        except Exception:  # noqa: BLE001 - receipts never block a delivery
            cls._count_failure()
            logger.exception("delivery event hooks unavailable; deliveries continue without receipts")
            return
        cls.emitter(engine)

    @staticmethod
    def _count_failure() -> None:
        from .system_events import SystemEventEmitter

        SystemEventEmitter.failed_appends += 1

    @classmethod
    def emitter(cls, engine: Engine | None) -> "SystemEventEmitter | None":
        """An emitter whose table exists; an unavailable ledger never blocks deliveries."""
        from .system_events import SystemEventEmitter

        if engine is None:
            return None
        if cls._ready.get(engine):
            return SystemEventEmitter(engine, bootstrap=False)
        try:
            found = SystemEventEmitter(engine)
        except Exception:  # noqa: BLE001 - deliveries continue without receipts
            SystemEventEmitter.failed_appends += 1
            logger.exception("delivery event ledger unavailable; deliveries continue")
            return None
        with cls._lock:
            cls._ready[engine] = True
        return found

    @classmethod
    def _commit_emitter(cls, conn: Connection) -> "SystemEventEmitter | None":
        """The engine's emitter, without schema work on a second connection.

        Inside a commit, a bootstrap would open another connection and could
        wait on the very locks this transaction holds. An engine no store
        warmed gets an emitter only when ``conn`` already sees the table.
        """
        from .system_events import SystemEventEmitter

        if not cls._ready.get(conn.engine):
            try:
                with conn.begin_nested():
                    exists = SystemEventEmitter.table_exists(conn)
            except Exception:  # noqa: BLE001 - deliveries continue without receipts
                exists = False
            if not exists:
                cls._count_failure()
                logger.warning("delivery events: event ledger table not ready on this engine; skipped")
                return None
            with cls._lock:
                cls._ready[conn.engine] = True
        return SystemEventEmitter(conn.engine, bootstrap=False)

    @classmethod
    def flush(
        cls, conn: Connection, staged: Iterable[tuple[DeliveryRecord, str | None]], *,
        default_user: str | None = None,
    ) -> None:
        """Append one receipt per staged row, in (id, revision) order.

        Runs from the commit hook, after every domain write and row lock, so
        concurrent multi-row flushes take the event lock last and in the same
        order. A row staged twice at one revision appends once.
        """
        rows = sorted({(r.id, r.revision): (r, code) for r, code in staged}.values(),
                      key=lambda item: (item[0].id, item[0].revision))
        if not rows:
            return
        emitter = cls._commit_emitter(conn)
        if emitter is None:
            return
        user = default_user if default_user is not None else configured_default_user()
        if not user:
            logger.warning("delivery events: no configured default user; bot relays are audit-only")
        for record, code in rows:
            cls._append(emitter, conn, record, default_user=user, reason_code=code)

    @staticmethod
    def _append(emitter: "SystemEventEmitter", conn: Connection, record: DeliveryRecord, *,
                default_user: str | None, reason_code: str | None) -> None:
        # TASK-1786: a user's own chat turn rides a delivery row but is not a
        # relay; the chat already shows it. Only its failure is worth a receipt.
        user_authored = record.author_entity_type == "user"
        if user_authored and record.status not in _USER_RECEIPT_STATUSES:
            return
        audience = delivery_audience(record, default_user=default_user)
        # A bot-created wake-up's sender is the scheduler; the bot is its author.
        scheduled = bool(record.metadata.get("prompt_schedule"))
        origin = (record.author_entity_id if scheduled and record.author_entity_type == "bot"
                  else record.sender_bot_id)
        # The emitter checks ``bot_id``; the other party must be public too.
        other = record.target_bot_id if audience.bot_id == record.sender_bot_id else origin
        visible = True
        if other and other != audience.bot_id:
            try:
                with conn.begin_nested():
                    if BotTags(emitter.engine).has(other, BehaviorTag.PRIVATE, conn=conn):
                        visible = False
            except Exception:  # noqa: BLE001 - fail closed to audit-only
                logger.exception("delivery event bot visibility unavailable; recording audit-only")
                visible = False
        payload = {"phase": record.status}
        if reason_code and _REASON_CODE.fullmatch(reason_code):
            payload["reason_code"] = reason_code
        emitter.append(
            conn,
            event_key=f"inter_bot:{record.id}:{record.status}:{record.revision}",
            kind="inter_bot_delivery", source="delivery",
            user_id=audience.user_id,
            tone=_TONES.get(record.status, "info"),
            summary=f"{'You' if user_authored else origin} → {record.target_bot_id}"
                    f"{' (scheduled)' if scheduled else ''} · {record.status.lower()}"[:500],
            bot_id=audience.bot_id, actor_type=record.author_entity_type or "bot",
            actor_id=(record.author_entity_id or record.sender_bot_id)[:128],
            subject_type="delivery", subject_id=record.id,
            payload=payload, visible=visible,
        )
