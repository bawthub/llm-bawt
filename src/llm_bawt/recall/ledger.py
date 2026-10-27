"""Reference-only record of what one recall read, for the UI drill-down (TASK-1519).

A recall returns at most a few quotes, but the person reading the result wants
to see which records were actually searched, including when the answer was
``insufficient_evidence`` or ``no_match``. The bot payload must not grow for
that (TASK-1555), so the service writes the list here, keyed by the hash of
the result handle, and the UI fetches it on demand.

Entries are references only (source, id, provenance, flags): no message,
memory or task text is stored. The drill-down re-reads content live under the
current policy, so a bot made private later disappears from the list.
"""
from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import JSONB

from llm_bawt.utils.bounded_worker import BoundedWorker
from llm_bawt.utils.schema import SchemaBootstrapGuard

from .contracts import Evidence, RecallScope

logger = logging.getLogger(__name__)

LEDGER_TABLE = "recall_evidence_ledgers"
LEDGER_TTL = timedelta(days=30)
# One recall reads at most 33 rows today (18 messages, 6 memories, 6 tool
# receipts, 3 tasks); the ceiling only bounds a future source.
MAX_LEDGER_ENTRIES = 256
LEDGER_WRITE_SECONDS = 2.0
PURGE_BATCH = 200
# Writes never share the recall read pool; when both slots are busy the
# write is skipped (the drill-down then says the list is unavailable).
LEDGER_WORKER = BoundedWorker(2, "recall-ledger")
# Fields copied from Evidence; never `text`.
ENTRY_FIELDS = ("source", "id", "source_bot", "session_id", "at", "speaker", "follows",
                "window_kind", "start_char")

_DDL = f"""
CREATE TABLE IF NOT EXISTS {LEDGER_TABLE} (
  handle_sha256 CHAR(64)     PRIMARY KEY,
  bot_id        VARCHAR(128) NOT NULL,
  user_id       VARCHAR(128) NOT NULL,
  session_id    VARCHAR(128) NOT NULL,
  coverage      JSONB        NOT NULL CHECK (jsonb_typeof(coverage) = 'object'),
  entries       JSONB        NOT NULL CHECK (jsonb_typeof(entries) = 'array'),
  omitted       INTEGER      NOT NULL DEFAULT 0 CHECK (omitted >= 0),
  created_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
  expires_at    TIMESTAMPTZ  NOT NULL
)
"""
_INDEX_DDL = f"CREATE INDEX IF NOT EXISTS {LEDGER_TABLE}_expires_idx ON {LEDGER_TABLE} (expires_at)"

_INSERT_SQL = text(f"""
    INSERT INTO {LEDGER_TABLE}
        (handle_sha256, bot_id, user_id, session_id, coverage, entries, omitted, expires_at)
    VALUES (:digest, :bot, :user, :session, :coverage, :entries, :omitted, :expires)
    ON CONFLICT (handle_sha256) DO NOTHING
""").bindparams(bindparam("coverage", type_=JSONB), bindparam("entries", type_=JSONB))
_PURGE_SQL = text(f"""
    DELETE FROM {LEDGER_TABLE} WHERE handle_sha256 IN (
        SELECT handle_sha256 FROM {LEDGER_TABLE} WHERE expires_at <= now()
        ORDER BY expires_at LIMIT {PURGE_BATCH})
""")
_READ_SQL = text(f"""
    SELECT bot_id, user_id, session_id, coverage, entries, omitted
    FROM {LEDGER_TABLE}
    WHERE handle_sha256 = :digest AND bot_id = :bot AND user_id = :user AND expires_at > now()
""")


def handle_digest(handle: str) -> str:
    """The raw handle is a bearer continuation; only its digest is stored."""
    return hashlib.sha256(handle.encode("utf-8")).hexdigest()


def ledger_entry(row: Evidence) -> dict[str, Any]:
    """References and provenance for one accepted row; flags are set by the service."""
    entry = {field: getattr(row, field) for field in ENTRY_FIELDS}
    entry["admitted"] = False
    entry["cited"] = False
    return entry


@dataclass(frozen=True)
class LedgerRecord:
    bot_id: str
    user_id: str
    session_id: str
    coverage: dict
    entries: list[dict]
    omitted: int


class LedgerSink(ABC):
    """Where the service sends the reference list of one completed recall."""

    @abstractmethod
    def write(self, handle: str, scope: RecallScope, coverage: dict, entries: list[dict],
              omitted: int) -> None:
        """Persist one recall's references. Called off the event loop; may raise."""


class PostgresLedger(LedgerSink):
    _guard = SchemaBootstrapGuard()

    def __init__(self, engine, *, ttl: timedelta = LEDGER_TTL):
        self.engine = engine
        self.ttl = ttl

    def _ensure(self) -> None:
        def bootstrap(conn):
            conn.execute(text(_DDL))
            conn.execute(text(_INDEX_DDL))
        self._guard.run(self.engine, "recall-evidence-ledger-v1", bootstrap)

    def write(self, handle: str, scope: RecallScope, coverage: dict, entries: list[dict],
              omitted: int) -> None:
        if self.engine is None:
            return
        self._ensure()
        with self.engine.begin() as conn:
            conn.execute(text("SET LOCAL statement_timeout = '2000ms'"))
            conn.execute(_INSERT_SQL, {
                "digest": handle_digest(handle), "bot": scope.bot_id, "user": scope.user_id,
                "session": scope.session_id, "coverage": coverage, "entries": entries,
                "omitted": omitted, "expires": datetime.now(timezone.utc) + self.ttl})
            # Bounded housekeeping on the write path; no timer needed.
            conn.execute(_PURGE_SQL)

    def read(self, handle: str, bot_id: str, user_id: str) -> LedgerRecord | None:
        """Owner-checked, unexpired record or None. Never says which check failed."""
        if self.engine is None:
            return None
        self._ensure()
        with self.engine.connect() as conn:
            with conn.begin():
                conn.execute(text("SET TRANSACTION READ ONLY"))
                conn.execute(text("SET LOCAL statement_timeout = '2000ms'"))
                row = conn.execute(_READ_SQL, {"digest": handle_digest(handle), "bot": bot_id,
                                               "user": user_id}).mappings().first()
        if row is None:
            return None
        return LedgerRecord(row["bot_id"], row["user_id"], row["session_id"],
                            dict(row["coverage"]), list(row["entries"]), int(row["omitted"]))
