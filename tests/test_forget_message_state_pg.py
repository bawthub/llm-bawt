"""TASK-2251: forget and restore keep every durable message column."""

from __future__ import annotations

import json
import time
import uuid

import pytest
from sqlalchemy import text

from llm_bawt.memory.message_store import (
    FORGOTTEN_PARENT,
    MESSAGE_STATE_COLUMNS,
    ensure_message_parent_tables,
)
from llm_bawt.memory.postgresql import PostgreSQLMemoryBackend
from llm_bawt.message_authorship import AuthorReference
from llm_bawt.runtime_settings import BotProfileStore
from tests.private_postgres import private_app_pg  # noqa: F401  (pytest fixture)

pytestmark = pytest.mark.integration


@pytest.fixture
def backend(private_app_pg):  # noqa: F811
    slug = f"forget-{uuid.uuid4().hex[:12]}"
    BotProfileStore(private_app_pg).upsert({"slug": slug, "name": slug, "system_prompt": "fixture"})
    return PostgreSQLMemoryBackend(private_app_pg, bot_id=slug)


def _add(backend, content, *, ts, anchor=None):
    """One message with every durable column set to a non-default value."""
    mid = str(uuid.uuid4())
    backend.add_message(
        mid, "assistant", content, ts,
        attachments=[{"asset_id": f"a-{content}", "kind": "image", "name": "card.png"}],
        reasoning=f"thinking about {content}",
        author=AuthorReference("bot", backend.bot_id),
        extract_memory=False,
    )
    with backend.engine.begin() as conn:
        conn.execute(text(f"""
            UPDATE {backend._messages_table_name}
            SET processed = TRUE, summarized = TRUE, recalled_history = TRUE,
                summary_metadata = CAST(:meta AS jsonb),
                interrupt_source_message_id = :anchor, interrupt_content_offset = 7
            WHERE id = :id
        """), {"id": mid, "anchor": anchor or mid,
               "meta": json.dumps({"opt_out": True, "summary_id": "s1"})})
    return mid


def _state(backend, table, mid):
    cols = ", ".join(MESSAGE_STATE_COLUMNS)
    with backend.engine.connect() as conn:
        row = conn.execute(text(f"SELECT {cols} FROM {table} WHERE id = :id"), {"id": mid}).mappings().one_or_none()
    return dict(row) if row else None


def _assert_round_trip(backend, ids, before):
    for mid in ids:
        assert _state(backend, backend._messages_table_name, mid) is None
        assert _state(backend, backend._forgotten_table_name, mid) == before[mid]
    assert backend.restore_ignored_messages() == len(ids)
    for mid in ids:
        assert _state(backend, backend._messages_table_name, mid) == before[mid]
        assert _state(backend, backend._forgotten_table_name, mid) is None


def test_forget_by_id_and_restore_keep_every_column(backend):
    now = time.time()
    first = _add(backend, "one", ts=now - 10)
    second = _add(backend, "two", ts=now - 5, anchor=first)
    before = {m: _state(backend, backend._messages_table_name, m) for m in (first, second)}
    assert before[second]["attachments"][0]["asset_id"] == "a-two"
    assert before[second]["summary_metadata"] == {"opt_out": True, "summary_id": "s1"}

    assert backend.ignore_message_by_id(second) is True
    assert backend.ignore_message_by_id(first[:8]) is True  # prefix path
    _assert_round_trip(backend, [first, second], before)


def test_forget_recent_count_and_restore_keep_every_column(backend):
    now = time.time()
    ids = [_add(backend, f"m{i}", ts=now - 30 + i) for i in range(3)]
    before = {m: _state(backend, backend._messages_table_name, m) for m in ids}

    assert backend.ignore_recent_messages(2) == 2
    assert _state(backend, backend._messages_table_name, ids[0]) == before[ids[0]]
    _assert_round_trip(backend, ids[1:], before)


def test_forget_time_window_and_restore_keep_every_column(backend):
    now = time.time()
    old = _add(backend, "old", ts=now - 3600)
    recent = [_add(backend, f"r{i}", ts=now - 60 + i) for i in range(2)]
    before = {m: _state(backend, backend._messages_table_name, m) for m in recent}

    assert backend.ignore_messages_since_minutes(5) == 2
    assert _state(backend, backend._messages_table_name, old) is not None
    _assert_round_trip(backend, recent, before)


def test_restore_keeps_forgotten_copy_when_id_is_already_canonical(backend):
    mid = _add(backend, "dup", ts=time.time())
    before = _state(backend, backend._messages_table_name, mid)
    assert backend.ignore_message_by_id(mid) is True
    # The same id reappears in history before restore.
    backend.add_message(mid, "assistant", "rewritten", time.time(), extract_memory=False)

    assert backend.restore_ignored_messages() == 0
    assert _state(backend, backend._forgotten_table_name, mid) == before
    assert _state(backend, backend._messages_table_name, mid)["content"] == "rewritten"

    # Forgetting it again refreshes the forgotten copy rather than losing the canonical row.
    assert backend.ignore_message_by_id(mid) is True
    assert _state(backend, backend._forgotten_table_name, mid)["content"] == "rewritten"


def test_failed_forget_changes_nothing(backend, monkeypatch):
    mid = _add(backend, "keep", ts=time.time())
    before = _state(backend, backend._messages_table_name, mid)
    # A broken forgotten table makes the single move statement fail.
    monkeypatch.setattr(backend, "_forgotten_table_name", "forgotten_messages_p_missing")
    assert backend.ignore_message_by_id(mid) is False
    assert _state(backend, backend._messages_table_name, mid) == before


def test_existing_forgotten_parent_gains_the_state_columns(backend):
    added = ("summarized", "summary_metadata", "recalled_history", "attachments", "reasoning")
    with backend.engine.begin() as conn:
        for col in added:
            conn.execute(text(f"ALTER TABLE {FORGOTTEN_PARENT} DROP COLUMN {col}"))
        conn.execute(text(f"""
            INSERT INTO {backend._forgotten_table_name} (id, role, content, timestamp)
            VALUES ('legacy-row', 'user', 'from before', 1.0)
        """))
        ensure_message_parent_tables(conn)
        present = set(conn.execute(text("""
            SELECT column_name FROM information_schema.columns WHERE table_name = :t
        """), {"t": FORGOTTEN_PARENT}).scalars())
    assert set(added) <= present
    legacy = _state(backend, backend._forgotten_table_name, "legacy-row")
    assert legacy["attachments"] == [] and legacy["reasoning"] is None
    assert backend.restore_ignored_messages() == 1
    assert _state(backend, backend._messages_table_name, "legacy-row")["content"] == "from before"
