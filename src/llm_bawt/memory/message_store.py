"""Canonical PostgreSQL message table definitions and row persistence.

Extracted from :mod:`llm_bawt.memory.postgresql` for TASK-717.  The wider
memory backend still owns sessions, memories, summarization, and search; this
module owns only the two message tables and the permanent-row upsert contract.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    JSON,
    MetaData,
    String,
    Table,
    Text,
    and_,
    case,
    cast,
    column,
    func,
    literal,
    or_,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from ..message_authorship import AuthorReference
from .message_tsv import ensure_message_tsv

logger = logging.getLogger(__name__)

MESSAGES_PARENT = "messages"
FORGOTTEN_PARENT = "forgotten_messages"

_message_table_cache: dict[str, Table] = {}
_forgotten_table_cache: dict[str, Table] = {}


def ensure_message_parent_tables(conn) -> None:
    """Create the canonical and forgotten partitioned message parents."""
    # Unqualified, like the DDL below, so it follows the connection's search_path.
    messages_was_new = conn.execute(
        text("SELECT to_regclass(:name) IS NULL"), {"name": MESSAGES_PARENT}
    ).scalar_one()
    conn.execute(text(f"""
        CREATE TABLE IF NOT EXISTS {MESSAGES_PARENT} (
            bot_id VARCHAR(64) NOT NULL,
            id VARCHAR(36) NOT NULL,
            role VARCHAR(20) NOT NULL,
            content TEXT NOT NULL,
            timestamp DOUBLE PRECISION NOT NULL,
            session_id VARCHAR(36),
            processed BOOLEAN DEFAULT FALSE,
            summarized BOOLEAN DEFAULT FALSE,
            summary_metadata JSONB,
            recalled_history BOOLEAN DEFAULT FALSE,
            attachments JSONB NOT NULL DEFAULT '[]'::jsonb,
            reasoning TEXT,
            interrupt_source_message_id VARCHAR(36),
            interrupt_content_offset INTEGER,
            author_entity_type VARCHAR(16),
            author_entity_id VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (bot_id, id)
        ) PARTITION BY LIST (bot_id)
    """))
    conn.execute(text(f"""
        CREATE TABLE IF NOT EXISTS {FORGOTTEN_PARENT} (
            bot_id VARCHAR(64) NOT NULL,
            id VARCHAR(36) NOT NULL,
            role VARCHAR(20) NOT NULL,
            content TEXT NOT NULL,
            timestamp DOUBLE PRECISION NOT NULL,
            session_id VARCHAR(36),
            processed BOOLEAN DEFAULT FALSE,
            interrupt_source_message_id VARCHAR(36),
            interrupt_content_offset INTEGER,
            author_entity_type VARCHAR(16),
            author_entity_id VARCHAR(100),
            created_at TIMESTAMP,
            forgotten_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (bot_id, id)
        ) PARTITION BY LIST (bot_id)
    """))
    for table_name in (MESSAGES_PARENT, FORGOTTEN_PARENT):
        conn.execute(text(
            f"ALTER TABLE {table_name} "
            "ADD COLUMN IF NOT EXISTS author_entity_type VARCHAR(16)"
        ))
        conn.execute(text(
            f"ALTER TABLE {table_name} "
            "ADD COLUMN IF NOT EXISTS author_entity_id VARCHAR(100)"
        ))
        conn.execute(text(
            f"ALTER TABLE {table_name} "
            "ADD COLUMN IF NOT EXISTS interrupt_source_message_id VARCHAR(36)"
        ))
        conn.execute(text(
            f"ALTER TABLE {table_name} "
            "ADD COLUMN IF NOT EXISTS interrupt_content_offset INTEGER"
        ))
    # TASK-1323: stored content_tsv for FTS. Built only on a new parent; an
    # existing one needs the reviewed online upgrade (logged, never blocking).
    ensure_message_tsv(conn, table_was_new=messages_was_new)


def message_parent_indexes() -> list[str]:
    """Return parent index DDL shared by existing and future partitions."""
    return [
        f"CREATE INDEX IF NOT EXISTS idx_{MESSAGES_PARENT}_timestamp ON {MESSAGES_PARENT}(timestamp)",
        f"CREATE INDEX IF NOT EXISTS idx_{MESSAGES_PARENT}_session_ts ON {MESSAGES_PARENT}(session_id, timestamp DESC)",
        f"CREATE INDEX IF NOT EXISTS idx_{MESSAGES_PARENT}_processed ON {MESSAGES_PARENT}(processed)",
        f"CREATE INDEX IF NOT EXISTS {MESSAGES_PARENT}_content_trgm_idx ON {MESSAGES_PARENT} USING gin (content gin_trgm_ops)",
        # FTS uses the stored content_tsv index (memory.message_tsv).
    ]


def get_message_table(
    metadata: MetaData,
    table_name: str,
    utcnow: Callable[[], datetime],
) -> Table:
    """Return the SQLAlchemy table for one bot's canonical partition."""
    if table_name in _message_table_cache:
        return _message_table_cache[table_name]

    table = Table(
        table_name,
        metadata,
        Column("id", String(36), primary_key=True),
        Column("role", String(20), nullable=False),
        Column("content", Text, nullable=False),
        Column("timestamp", Float, nullable=False),
        Column("session_id", String(36), nullable=True),
        Column("processed", Boolean, default=False),
        Column("summarized", Boolean, default=False),
        Column("recalled_history", Boolean, default=False),
        Column("summary_metadata", JSON, nullable=True),
        Column(
            "attachments",
            JSON,
            nullable=False,
            server_default=text("'[]'::jsonb"),
            default=list,
        ),
        Column("reasoning", Text, nullable=True),
        Column("interrupt_source_message_id", String(36), nullable=True),
        Column("interrupt_content_offset", Integer, nullable=True),
        Column("author_entity_type", String(16), nullable=True),
        Column("author_entity_id", String(100), nullable=True),
        Column("created_at", DateTime, default=utcnow),
        extend_existing=True,
    )
    _message_table_cache[table_name] = table
    return table


def get_forgotten_message_table(
    metadata: MetaData,
    table_name: str,
    utcnow: Callable[[], datetime],
) -> Table:
    """Return the SQLAlchemy table for one bot's forgotten partition."""
    if table_name in _forgotten_table_cache:
        return _forgotten_table_cache[table_name]

    table = Table(
        table_name,
        metadata,
        Column("id", String(36), primary_key=True),
        Column("role", String(20), nullable=False),
        Column("content", Text, nullable=False),
        Column("timestamp", Float, nullable=False),
        Column("session_id", String(36), nullable=True),
        Column("processed", Boolean, default=False),
        Column("interrupt_source_message_id", String(36), nullable=True),
        Column("interrupt_content_offset", Integer, nullable=True),
        Column("author_entity_type", String(16), nullable=True),
        Column("author_entity_id", String(100), nullable=True),
        Column("created_at", DateTime, default=utcnow),
        Column("forgotten_at", DateTime, default=utcnow),
        extend_existing=True,
    )
    _forgotten_table_cache[table_name] = table
    return table


class MessageRowStore:
    """Persist canonical rows while preserving stable-id upsert semantics."""

    def __init__(self, engine: Engine, table: Table, table_name: str):
        self.engine = engine
        self.table = table
        self.table_name = table_name

    def upsert(
        self,
        *,
        message_id: str,
        role: str,
        content: str,
        timestamp: float,
        session_id: str | None = None,
        attachments: list[dict] | None = None,
        reasoning: str | None = None,
        author: AuthorReference | None = None,
        extract_memory: bool = True,
    ) -> None:
        """Insert or refresh mutable fields atomically, without an existence read.

        Conflict updates preserve role/session/created_at. The incoming INSERT
        candidate must still satisfy DB column constraints, including on retry;
        unlike the former UPDATE branch, malformed immutable inputs can fail.
        """
        if not content or content.isspace():
            logger.warning("Skipping empty content for message ID: %s", message_id)
            return

        dialect = self.engine.dialect.name
        if dialect == "postgresql":
            from sqlalchemy.dialects.postgresql import insert

            # The model describes one partition; its physical inherited PK is
            # (bot_id, id), with bot_id supplied by the partition's DB default.
            conflict = [column("bot_id"), self.table.c.id]
        elif dialect == "sqlite":
            from sqlalchemy.dialects.sqlite import insert

            conflict = [self.table.c.id]
        else:
            raise NotImplementedError(f"message upsert is unsupported for {dialect}")

        values = {
            "content": content,
            # Inspect the STORED role, never a retry's requested role.
            "timestamp": case(
                (self.table.c.role == "user", self.table.c.timestamp),
                else_=timestamp,
            ),
        }
        if attachments is not None:
            values["attachments"] = attachments
        if reasoning is not None:
            values["reasoning"] = reasoning
        if author is not None:
            values.update(author_entity_type=author.entity_type, author_entity_id=author.entity_id)
        if not extract_memory:
            values.update(processed=True, summary_metadata=self._exclusion_metadata(dialect))
        statement = insert(self.table).values(
            id=message_id, role=role, content=content, timestamp=timestamp,
            session_id=session_id, attachments=attachments if attachments is not None else [],
            reasoning=reasoning,
            author_entity_type=(author.entity_type if author else None),
            author_entity_id=(author.entity_id if author else None),
            processed=not extract_memory,
            **({"summary_metadata": {"extract_memory": False}} if not extract_memory else {}),
            created_at=datetime.now(timezone.utc),
        ).on_conflict_do_update(index_elements=conflict, set_=values)
        with Session(self.engine) as session:
            try:
                if not extract_memory:
                    # RETURNING is part of the ONE DML statement, not an owner
                    # or row-existence SELECT. Preserve Python's old malformed
                    # truthy non-mapping failure BEFORE commit, rolling back all
                    # updates. Falsy JSON values were normalized by the SQL CASE.
                    metadata = session.execute(
                        statement.returning(self.table.c.summary_metadata)
                    ).scalar_one()
                    if not isinstance(metadata, dict):
                        raise TypeError("stored summary metadata must be a mapping")
                else:
                    session.execute(statement)
                session.commit()
                logger.debug("Added message %s to %s", message_id, self.table_name)
            except Exception:
                session.rollback()
                logger.exception("Failed to add message %s", message_id)
                raise

    def _exclusion_metadata(self, dialect: str):
        """Shallow-merge exclusion, preserving the old `metadata or {}` contract.

        Truthy non-objects are returned unchanged for validation/rollback above;
        never silently repair malformed stored metadata into an eligible object.
        """
        metadata = self.table.c.summary_metadata
        flag = {"extract_memory": False}
        if dialect == "postgresql":
            from sqlalchemy.dialects.postgresql import JSONB

            stored = cast(metadata, JSONB)
            object_ = func.jsonb_typeof(stored) == "object"
            falsy = or_(
                metadata.is_(None),
                func.jsonb_typeof(stored) == "null",
                stored.in_([literal(v, type_=JSONB) for v in (False, 0, "", [])]),
            )
            return case(
                (object_, stored.op("||")(literal(flag, type_=JSONB))),
                (falsy, literal(flag, type_=JSONB)),
                else_=stored,
            )
        kind = func.json_type(metadata)
        value = func.json_extract(metadata, "$")
        falsy = or_(
            metadata.is_(None),
            kind.in_(["null", "false"]),
            and_(kind.in_(["integer", "real"]), value == 0),
            and_(kind == "text", value == ""),
            and_(kind == "array", func.json_array_length(metadata) == 0),
        )
        encoded_flag = literal(flag, type_=JSON)
        return case(
            (kind == "object", func.json_patch(metadata, encoded_flag)),
            (falsy, encoded_flag),
            else_=metadata,
        )
