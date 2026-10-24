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
    insert,
    select,
    text,
    update,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from ..message_authorship import AuthorReference

logger = logging.getLogger(__name__)

MESSAGES_PARENT = "messages"
FORGOTTEN_PARENT = "forgotten_messages"

_message_table_cache: dict[str, Table] = {}
_forgotten_table_cache: dict[str, Table] = {}


def ensure_message_parent_tables(conn) -> None:
    """Create the canonical and forgotten partitioned message parents."""
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


def message_parent_indexes() -> list[str]:
    """Return parent index DDL shared by existing and future partitions."""
    return [
        f"CREATE INDEX IF NOT EXISTS idx_{MESSAGES_PARENT}_timestamp ON {MESSAGES_PARENT}(timestamp)",
        f"CREATE INDEX IF NOT EXISTS idx_{MESSAGES_PARENT}_session_ts ON {MESSAGES_PARENT}(session_id, timestamp DESC)",
        f"CREATE INDEX IF NOT EXISTS idx_{MESSAGES_PARENT}_processed ON {MESSAGES_PARENT}(processed)",
        f"CREATE INDEX IF NOT EXISTS {MESSAGES_PARENT}_content_trgm_idx ON {MESSAGES_PARENT} USING gin (content gin_trgm_ops)",
        # TASK-1323: word/identifier FTS. The expression must match the
        # queries' to_tsvector('english', content) exactly to be usable.
        f"CREATE INDEX IF NOT EXISTS {MESSAGES_PARENT}_content_fts_idx ON {MESSAGES_PARENT} USING gin (to_tsvector('english', content))",
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
        """Insert one row or refresh mutable payload fields on an existing id."""
        if not content or content.isspace():
            logger.warning("Skipping empty content for message ID: %s", message_id)
            return

        with Session(self.engine) as session:
            try:
                existing = session.execute(
                    select(self.table).where(self.table.c.id == message_id)
                ).first()
                if existing:
                    values: dict = {"content": content}
                    # TASK-1234: a user row's timestamp is first-write-wins, so
                    # a retried delivery cannot reorder it after later replies.
                    # Assistant partial->final upserts keep refreshing theirs.
                    if existing.role != "user":
                        values["timestamp"] = timestamp
                    if not extract_memory:
                        values["processed"] = True
                        values["summary_metadata"] = {
                            **(existing.summary_metadata or {}), "extract_memory": False,
                        }
                    if attachments is not None:
                        values["attachments"] = attachments
                    if reasoning is not None:
                        values["reasoning"] = reasoning
                    if author is not None:
                        values["author_entity_type"] = author.entity_type
                        values["author_entity_id"] = author.entity_id
                    session.execute(
                        update(self.table)
                        .where(self.table.c.id == message_id)
                        .values(**values)
                    )
                else:
                    session.execute(
                        insert(self.table).values(
                            id=message_id,
                            role=role,
                            content=content,
                            timestamp=timestamp,
                            session_id=session_id,
                            attachments=attachments if attachments is not None else [],
                            reasoning=reasoning,
                            author_entity_type=(author.entity_type if author else None),
                            author_entity_id=(author.entity_id if author else None),
                            processed=not extract_memory,
                            # Reuse existing message metadata for durable exclusion
                            # provenance; processed alone also means "extracted".
                            **({"summary_metadata": {"extract_memory": False}}
                               if not extract_memory else {}),
                            created_at=datetime.now(timezone.utc),
                        )
                    )
                session.commit()
                logger.debug("Added message %s to %s", message_id, self.table_name)
            except Exception:
                session.rollback()
                logger.exception("Failed to add message %s", message_id)
                raise
