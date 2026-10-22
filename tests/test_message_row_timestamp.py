"""TASK-1234 Fix 3: a retry must not move a user message's timestamp.

Incident a67418db: Nick's user row moved from ~20:30:09 to 20:39:01 when the
durable delivery was retried, landing it after Al's 20:37:16 reply. The user
row's timestamp is first-write-wins; assistant partial->final upserts keep
refreshing theirs.
"""
from __future__ import annotations

import pytest
from sqlalchemy import (Boolean, Column, DateTime, Float, JSON, MetaData, String,
                        Table, create_engine, select)
from sqlalchemy.pool import StaticPool

from llm_bawt.memory.message_store import MessageRowStore

ORIGINAL = 1792629009.0   # 20:30:09, first write
AL_REPLY = 1792629436.3   # 20:37:16, another bot's row in between
RETRY = 1792629541.2      # 20:39:01, the retried write


@pytest.fixture
def rows():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    table = Table("messages_ts_test", MetaData(),
                  Column("id", String, primary_key=True), Column("role", String),
                  Column("content", String), Column("timestamp", Float),
                  Column("session_id", String), Column("attachments", JSON),
                  Column("reasoning", String), Column("author_entity_type", String),
                  Column("author_entity_id", String), Column("processed", Boolean, default=False),
                  Column("summary_metadata", JSON), Column("created_at", DateTime))
    table.metadata.create_all(engine)
    yield MessageRowStore(engine, table, table.name)
    engine.dispose()


def _timeline(store: MessageRowStore) -> list[tuple[str, float, str]]:
    with store.engine.connect() as conn:
        result = conn.execute(
            select(store.table.c.id, store.table.c.timestamp, store.table.c.content)
            .order_by(store.table.c.timestamp)
        ).all()
    return [tuple(row) for row in result]


def test_retried_user_row_keeps_original_timestamp_and_order(rows):
    rows.upsert(message_id="a67418db", role="user", content="do the thing", timestamp=ORIGINAL)
    rows.upsert(message_id="819d8f89", role="assistant", content="al reply", timestamp=AL_REPLY)
    rows.upsert(message_id="a67418db", role="user", content="do the thing (edited)", timestamp=RETRY)

    assert _timeline(rows) == [
        ("a67418db", ORIGINAL, "do the thing (edited)"),
        ("819d8f89", AL_REPLY, "al reply"),
    ]


def test_assistant_partial_to_final_still_refreshes_timestamp(rows):
    rows.upsert(message_id="asst-1", role="assistant", content="partial", timestamp=ORIGINAL)
    rows.upsert(message_id="other", role="user", content="next", timestamp=AL_REPLY)
    rows.upsert(message_id="asst-1", role="assistant", content="final answer", timestamp=RETRY)

    assert _timeline(rows) == [
        ("other", AL_REPLY, "next"),
        ("asst-1", RETRY, "final answer"),
    ]
