"""Sender notices for cancelled or user-stopped deliveries (TASK-1327/1822).

An unstarted cancellation gets a notice that work will not run; a delivery
stopped after a bound run began gets a separate notice that work may already
have happened. This leader-only pump gives each eligible terminal row exactly
one notice DELIVERY to its sender, covering user Stop, queue-hold Cancel all,
the cancel API and recovery. It runs on each dispatcher pass.

* Eligible (``_PENDING``): CANCELLED or a DELIVERED row stamped
  ``stopped_after_start``, bot-authored, sender is not the target, not itself
  a notice (so a cancelled notice is never notified about), and not a
  scheduled prompt (its receipt belongs to the scheduling API). Rows sent by
  or to a private bot (a ``private`` tag, the legacy column false, or no
  profile at all; BOT_TAGS.md §5.3) are never read, stamped or notified.
* Every row the pump decides on is stamped ``cancel_notified_at`` (the audit
  stamp), notified or not. A sender that is not an agent bot, or has no user
  to scope the notice to, is stamped and skipped.
* Exactly one: the stamp and the notice enqueue commit in one transaction,
  and the notice's idempotency key ``cancel-notice:<id>`` makes a retried
  pass a no-op. One notice ROW per cancellation; the notice may itself be
  held, cancelled or fail.
* A row whose pass fails (lock conflict, prompt registry down or blank) is
  rolled back whole and backs off (``RETRY_BASE_S`` doubling to
  ``RETRY_MAX_S``), and the next candidate read skips it, so rows that keep
  failing never starve newer ones. The backoff lives in the leader's pump;
  a new leader term retries them at once.
* Lock order (B2): both bots' ``lock_target`` in sorted order, then the source
  row ``FOR UPDATE``, the same target-then-row order ``cancel()`` takes. A
  deadlock or serialization failure rolls back the whole row (never a stamp
  without its notice) and the next pass retries it.
* ``delivery_cancel_notice_mode``: ``notify`` (default), ``report`` (reads the
  pending count and logs it when it changes; writes nothing) or ``off``.

The column is added once (``ensure_cancel_notice_schema``). On that add, the
rows already pending are stamped, so history never notifies.
"""
from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text as sa_text

from ..bot_identities import SYSTEM_BOT_ID
from ..bot_tags import BehaviorTag, sql_allowed
from ..bot_tags.catalog import ASSIGNMENT_TABLE, owned
from ..bot_types import normalize_bot_type
from ..inter_bot_delivery_record import DeliveryRecord
from ..inter_bot_delivery_user import resolve_delivery_user
from ..utils.db import is_postgresql
from ..utils.schema import SchemaBootstrapGuard, ensure_columns, ensure_relations

log = logging.getLogger(__name__)

NOTICE_KIND = "DELIVERY_CANCELLED"
NOTICE_KEY_PREFIX = "cancel-notice:"
BATCH_SIZE = 20
RETRY_BASE_S = 30.0
RETRY_MAX_S = 900.0
# Doublings that reach RETRY_MAX_S; the exponent saturates here, so a row that
# fails for days never overflows the float (Loopy r2).
_MAX_DOUBLINGS = math.ceil(math.log2(RETRY_MAX_S / RETRY_BASE_S))
EXCERPT_CHARS = 200
MODE_SETTING = "delivery_cancel_notice_mode"
NOTIFY, REPORT, OFF = "notify", "report", "off"
MODES = (NOTIFY, REPORT, OFF)
_NOTIFIED, _SKIPPED, _GONE = "notified", "skipped", "gone"

# The one eligibility predicate, shared by the candidate read, the locked
# re-check, the one-time backfill and report. The original cancel partial
# index exists on deployed DBs; a separate partial index covers started stops.
_PENDING = f"""
    (status='CANCELLED' OR (status='DELIVERED' AND metadata_json->>'stopped_after_start'='true'))
    AND cancel_notified_at IS NULL
    AND author_entity_type='bot' AND sender_bot_id <> target_bot_id
    AND message_kind IS DISTINCT FROM '{NOTICE_KIND}'
    AND metadata_json->'prompt_schedule' IS NULL
"""
_SENDER_ALLOWED, _SENDER_PARAMS = sql_allowed(
    "inter_bot_deliveries.sender_bot_id", {BehaviorTag.PRIVATE}, prefix="snd")
_TARGET_ALLOWED, _TARGET_PARAMS = sql_allowed(
    "inter_bot_deliveries.target_bot_id", {BehaviorTag.PRIVATE}, prefix="tgt")
_UNPROTECTED = f"{_SENDER_ALLOWED} AND {_TARGET_ALLOWED}"
_UNPROTECTED_PARAMS = {**_SENDER_PARAMS, **_TARGET_PARAMS}

_schema_guard = SchemaBootstrapGuard()
_COLUMNS = {"cancel_notified_at": ("TIMESTAMPTZ", None)}
_RELATIONS = {
    "ix_inter_bot_delivery_cancel_notice": f"""
        CREATE INDEX IF NOT EXISTS ix_inter_bot_delivery_cancel_notice
        ON inter_bot_deliveries (updated_at, ordinal) WHERE {_PENDING}
    """,
    "ix_inter_bot_delivery_stopped_notice": """
        CREATE INDEX IF NOT EXISTS ix_inter_bot_delivery_stopped_notice
        ON inter_bot_deliveries (updated_at, ordinal)
        WHERE status='DELIVERED' AND metadata_json->>'stopped_after_start'='true'
          AND cancel_notified_at IS NULL
    """,
}


def ensure_cancel_notice_schema(engine) -> None:
    """Add ``cancel_notified_at`` and its index once per engine (PostgreSQL only)."""
    if is_postgresql(engine):
        _schema_guard.run(engine, "inter-bot-delivery-cancel-notice", _bootstrap)


def _bootstrap(conn) -> None:
    # Under the guard's global advisory lock: a second process sees the column
    # present, runs no DDL and stamps nothing. NULL is the live pending state,
    # so the backfill runs only with the ADD that created the column.
    added = ensure_columns(conn, "inter_bot_deliveries", _COLUMNS)
    if any("ADD COLUMN" in statement for statement in added):
        if not (owned(conn, "bot_profiles") and owned(conn, ASSIGNMENT_TABLE)):
            # No tag data to tell private rows apart: stamp nothing (a private
            # row is never stamped). Only a database with no history gets here.
            log.warning("cancel_notified_at added before bot tags exist; no historical rows stamped")
        else:
            stamped = conn.execute(sa_text(
                f"UPDATE inter_bot_deliveries SET cancel_notified_at=now() WHERE {_PENDING} AND {_UNPROTECTED}"
            ), _UNPROTECTED_PARAMS).rowcount
            log.info("cancel_notified_at added; %d historical cancelled deliveries stamped (never notified)",
                     stamped)
    ensure_relations(conn, _RELATIONS)


@dataclass
class PassResult:
    notified: list[DeliveryRecord] = field(default_factory=list)
    skipped: int = 0
    errors: int = 0


class CancelNoticePump:
    """One sender notice per eligible cancelled delivery."""

    def __init__(self, store: Any, *, batch_size: int = BATCH_SIZE,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.store = store
        self.batch_size = batch_size
        self._clock = clock
        self._last_report: int | None = None
        # delivery id -> (consecutive failures, retry not before), on ``clock``.
        self._backoff: dict[str, tuple[int, float]] = {}

    @property
    def engine(self):
        return self.store.engine

    def _params(self, **extra: Any) -> dict[str, Any]:
        return {**_UNPROTECTED_PARAMS, **extra}

    def mode(self) -> str:
        from ..runtime_setting_resolution import resolve_global_runtime_setting

        try:
            mode = str(resolve_global_runtime_setting(self.store.config, MODE_SETTING) or NOTIFY).strip().lower()
        except Exception:
            log.exception("Unreadable %s; cancel notices report only this pass", MODE_SETTING)
            return REPORT
        if mode not in MODES:
            log.warning("Unknown %s %r; cancel notices report only", MODE_SETTING, mode)
            return REPORT
        return mode

    def report(self) -> dict[str, Any]:
        """The rows a ``notify`` pass would decide on. Read-only."""
        with self.engine.connect() as conn:
            row = conn.execute(sa_text(
                f"SELECT count(*) AS pending, min(updated_at) AS oldest FROM inter_bot_deliveries"
                f" WHERE {_PENDING} AND {_UNPROTECTED}"
            ), self._params()).mappings().one()
        return {"pending": int(row["pending"]), "oldest": row["oldest"].isoformat() if row["oldest"] else None}

    def run_pass(self) -> PassResult:
        """One bounded pass; the notices it enqueued (the caller announces them)."""
        result = PassResult()
        if not is_postgresql(self.engine):
            return result
        mode = self.mode()
        if mode == REPORT:
            report = self.report()
            if report["pending"] != self._last_report:
                log.info("cancel notices (report only): %d pending, oldest %s", report["pending"], report["oldest"])
                self._last_report = report["pending"]
            return result
        if mode == OFF:
            return result
        for candidate in self._candidates():
            try:
                outcome, notice = self._decide(candidate)
            except Exception:  # rolled back whole: retried after its backoff
                delay = self._defer(candidate["id"])
                log.exception("cancel notice for delivery %s failed; retrying in %.0fs", candidate["id"], delay)
                result.errors += 1
                continue
            self._backoff.pop(candidate["id"], None)
            if notice is not None:
                result.notified.append(notice)
            elif outcome == _SKIPPED:
                result.skipped += 1
        if result.notified or result.skipped or result.errors:
            log.info("cancel notices: %d notified, %d skipped, %d errors",
                     len(result.notified), result.skipped, result.errors)
        return result

    def _defer(self, delivery_id: str) -> float:
        failures = self._backoff.get(delivery_id, (0, 0.0))[0] + 1
        delay = min(RETRY_MAX_S, RETRY_BASE_S * 2 ** min(failures - 1, _MAX_DOUBLINGS))
        self._backoff[delivery_id] = (failures, self._clock() + delay)
        return delay

    def _candidates(self) -> list[dict[str, Any]]:
        now = self._clock()
        waiting = sorted(row_id for row_id, (_, retry_at) in self._backoff.items() if retry_at > now)
        with self.engine.connect() as conn:
            return [dict(row) for row in conn.execute(sa_text(
                f"SELECT id, sender_bot_id, target_bot_id FROM inter_bot_deliveries"
                f" WHERE {_PENDING} AND {_UNPROTECTED} AND id <> ALL(:waiting)"
                f" ORDER BY updated_at, ordinal LIMIT :limit"
            ), self._params(limit=self.batch_size, waiting=waiting)).mappings()]

    def _decide(self, candidate: dict[str, Any]) -> tuple[str, DeliveryRecord | None]:
        """Stamp one row and notify its sender, in one transaction."""
        from .turn_admission import lock_target  # turn_admission imports the delivery store

        with self.engine.begin() as conn:
            for bot_id in sorted({candidate["target_bot_id"], candidate["sender_bot_id"]}):
                lock_target(conn, bot_id)
            row = conn.execute(sa_text(
                f"SELECT * FROM inter_bot_deliveries WHERE id=:id AND {_PENDING} AND {_UNPROTECTED} FOR UPDATE"
            ), self._params(id=candidate["id"])).mappings().first()
            if row is None:  # decided by another pass since the candidate read
                return _GONE, None
            conn.execute(sa_text(
                "UPDATE inter_bot_deliveries SET cancel_notified_at=now() WHERE id=:id"
            ), {"id": row["id"]})
            user = self._notice_user(conn, row)
            if user is None:
                return _SKIPPED, None
            return _NOTIFIED, self._enqueue_notice(conn, row, user)

    def _notice_user(self, conn, row) -> str | None:
        """The notice's user scope, or None when the sender gets no notice."""
        profile = conn.execute(sa_text(
            "SELECT bot_type, agent_backend FROM bot_profiles WHERE slug=:bot"
        ), {"bot": row["sender_bot_id"]}).mappings().first()
        if profile is None or normalize_bot_type(profile["bot_type"], profile["agent_backend"]) != "agent":
            return None
        payload = row["payload_json"] if isinstance(row["payload_json"], dict) else {}
        try:
            return resolve_delivery_user(self.store.config, payload.get("user"))
        except ValueError:
            log.warning("cancelled delivery %s has no user to scope a notice to; skipped", row["id"])
            return None

    def _enqueue_notice(self, conn, row, user: str) -> DeliveryRecord:
        sender = row["sender_bot_id"]
        started = (row.get("metadata_json") or {}).get("stopped_after_start") is True
        message = notice_text(sender, {
            "delivery_id": row["id"], "target_bot_id": row["target_bot_id"],
            "task_id": row["task_id"] or "none", "reason": row["last_error"] or "not recorded",
            "excerpt": excerpt(row["message"]),
        }, started=started)
        record, _ = self.store.enqueue(
            sender_bot_id=SYSTEM_BOT_ID, target_bot_id=sender, message=message,
            idempotency_key=f"{NOTICE_KEY_PREFIX}{row['id']}", message_kind=NOTICE_KIND,
            project_id=row["project_id"], task_id=row["task_id"],
            # A notice about work already lost: held by a gate, never refused.
            on_gate="hold", connection=conn,
            metadata={"cancelled_delivery_id": row["id"], "target_bot_id": row["target_bot_id"],
                      "reason": row["last_error"]},
            payload={
                "bot_id": sender, "user": user, "session_id": None,
                "messages": [{"role": "user", "content": message}],
                "prefer_steer": True, "reply_mode": "async",
                "extract_memory": False, "augment_memory": False, "stream": True,
            },
        )
        return record


def excerpt(message: str | None) -> str:
    flat = " ".join((message or "").split())
    return flat if len(flat) <= EXCERPT_CHARS else flat[:EXCERPT_CHARS - 1] + "…"


def notice_text(bot_id: str, variables: dict[str, Any], *, started: bool = False) -> str:
    """Render the appropriate sender notice from the live prompt registry.

    The seed is never sent in place of a missing/blank body. The caller's
    transaction then rolls back, leaving the row unstamped for retry.
    """
    from ..prompt_registry import get_prompt_resolver
    from .delivery_cancel_notice_prompt import PROMPT_KEY, STARTED_PROMPT_KEY

    key = STARTED_PROMPT_KEY if started else PROMPT_KEY
    rendered = get_prompt_resolver().render(
        key=key, variables=variables, scope_type="bot", scope_id=bot_id)
    if not (rendered or "").strip():
        raise ValueError(f"prompt {key} rendered blank for {bot_id}")
    return rendered
