"""TASK-2449: a fresh install has shared memory schema before any bot writes.

Read-only GETs never provision schema, so the app-startup helper must create
``sessions`` and the parents with no bot identity and no bot partitions.
"""
from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text

from tests.private_postgres import PORT, private_url

pytestmark = [pytest.mark.integration]


@pytest.fixture()
def fresh_config(private_pg, private_module_db, monkeypatch):
    from llm_bawt.utils import db
    from tests.pg_isolation import require_test_db_config

    url = private_url(private_pg, private_module_db)
    monkeypatch.setattr(db, "build_postgres_url", lambda config: url)
    monkeypatch.setenv("LLM_BAWT_POSTGRES_PASSWORD", uuid.uuid4().hex)
    monkeypatch.setenv("LLM_BAWT_POSTGRES_PORT", PORT)
    try:
        yield require_test_db_config()
    finally:
        db.reset_shared_engines()


def test_startup_helper_provisions_shared_schema_without_a_bot(fresh_config):
    from llm_bawt.memory.postgresql import provision_shared_memory_schema
    from llm_bawt.utils.db import get_shared_engine

    engine = get_shared_engine(fresh_config)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT to_regclass('public.sessions')")).scalar() is None

    # The exact call app startup makes; twice for restart idempotence.
    provision_shared_memory_schema(fresh_config)
    provision_shared_memory_schema(fresh_config)

    with engine.connect() as conn:
        assert conn.execute(text("SELECT to_regclass('public.sessions')")).scalar() == "sessions"
        assert conn.execute(text("SELECT to_regclass('public.messages')")).scalar() == "messages"
        assert conn.execute(text(
            "SELECT count(*) FROM pg_inherits i JOIN pg_class p ON p.oid = i.inhparent "
            "WHERE p.relname = 'messages'"
        )).scalar() == 0
        # The first-page read for a bot that has never chatted now succeeds.
        assert conn.execute(text(
            "SELECT count(*) FROM sessions WHERE bot_id = 'newbot'"
        )).scalar() == 0
