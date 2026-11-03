"""Execution tracking upgrades on a disposable PostgreSQL database only."""
from __future__ import annotations

import pytest
from sqlalchemy import text

from llm_bawt.service.tool_call_store import ToolCallStore
from tests.full_store_fixture import full_store_db  # noqa: F401

pytestmark = pytest.mark.integration


def test_tracking_additive_upgrade_keeps_legacy_untracked(full_store_db):
    _, engine = full_store_db
    store = ToolCallStore(engine)
    store.ensure_schema()
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE tool_call_records DROP COLUMN exec_tracked"))
        conn.execute(text("INSERT INTO tool_call_records (tool_name, iteration, created_at) "
                          "VALUES ('legacy', 1, now())"))
    store._schema_guard.reset_for_tests(engine)
    store.ensure_schema()
    with engine.begin() as conn:
        assert conn.execute(text("SELECT exec_tracked FROM tool_call_records "
                                 "WHERE tool_name='legacy'")).scalar_one() is False
        column = conn.execute(text(
            "SELECT is_nullable, column_default FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name='tool_call_records' "
            "AND column_name='exec_tracked'"
        )).one()
        assert tuple(column) == ("NO", "false")
        conn.execute(text("UPDATE tool_call_records SET exec_tracked=true WHERE tool_name='legacy'"))
    store._schema_guard.reset_for_tests(engine)
    store.ensure_schema()
    with engine.connect() as conn:
        assert conn.execute(text("SELECT exec_tracked FROM tool_call_records "
                                 "WHERE tool_name='legacy'")).scalar_one() is True
