"""A newly created bot can be read empty and written to (private PostgreSQL)."""
from __future__ import annotations

import asyncio
import time
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError

from llm_bawt.memory.postgresql import MESSAGES_PARENT, PARENT_TABLES, PostgreSQLMemoryBackend, partition_name
from llm_bawt.runtime_settings import BotProfileStore
from llm_bawt.service.routes import settings as settings_routes
from llm_bawt.service.routes.history_pages import _fetch_attachments_via_shared_engine
from llm_bawt.service.routes.history_timeline_store import timeline_rows
from llm_bawt.utils.db import get_shared_engine
from tests.pg_isolation import require_test_db_config

pytestmark = pytest.mark.integration


@pytest.fixture
def new_bot(private_app_pg):
    config = require_test_db_config()
    PostgreSQLMemoryBackend.provision_shared_schema(get_shared_engine(config), 1024)
    engine = get_shared_engine(config)
    bot = f"newbot{uuid.uuid4().hex[:12]}"
    BotProfileStore(config).upsert({"slug": bot, "name": "Fresh bot"})
    try:
        yield SimpleNamespace(config=config, engine=engine, bot=bot,
                              table=partition_name(MESSAGES_PARENT, bot))
    finally:
        with engine.begin() as conn:
            for parent in PARENT_TABLES:
                conn.execute(text(f"DROP TABLE IF EXISTS {partition_name(parent, bot)}"))
            conn.execute(text("DELETE FROM bot_profiles WHERE slug=:bot"), {"bot": bot})


def _timeline(rig):
    now = time.time()
    return timeline_rows(rig.engine, rig.table, rig.bot, "owner", None, "UTC",
                         now - 86400, now, recent=True)


def test_profile_row_alone_leaves_history_reads_failing(new_bot):
    """The reported failure: a profile with no partition breaks direct reads.

    The timeline has its own missing-partition guard and answers empty; other
    history reads address the partition directly and still fail.
    """
    with new_bot.engine.connect() as conn:
        assert conn.execute(text("SELECT to_regclass(:p)"), {"p": new_bot.table}).scalar() is None
    with pytest.raises(ProgrammingError, match="does not exist"):
        _fetch_attachments_via_shared_engine(new_bot.config, new_bot.bot, ["m1"])
    assert _timeline(new_bot) is not None


def test_created_bot_reads_empty_then_accepts_first_write(new_bot):
    asyncio.run(settings_routes._provision_bot_storage(SimpleNamespace(config=new_bot.config), new_bot.bot))

    assert _timeline(new_bot) is not None
    assert _fetch_attachments_via_shared_engine(new_bot.config, new_bot.bot, ["m1"]) == {"m1": []}
    with new_bot.engine.connect() as conn:
        assert conn.execute(text(f"SELECT count(*) FROM {new_bot.table}")).scalar() == 0
        sessions = conn.execute(text("SELECT count(*) FROM sessions WHERE bot_id=:b"),
                                {"b": new_bot.bot}).scalar()
        assert sessions == 0

    with new_bot.engine.begin() as conn:
        conn.execute(text("INSERT INTO messages (bot_id, id, role, content, timestamp) "
                          "VALUES (:b, :id, 'user', 'hello', :ts)"),
                     {"b": new_bot.bot, "id": str(uuid.uuid4()), "ts": time.time()})
    with new_bot.engine.connect() as conn:
        assert conn.execute(text(f"SELECT content FROM {new_bot.table}")).scalar() == "hello"


def test_provisioning_is_idempotent(new_bot):
    service = SimpleNamespace(config=new_bot.config)
    asyncio.run(settings_routes._provision_bot_storage(service, new_bot.bot))
    asyncio.run(settings_routes._provision_bot_storage(service, new_bot.bot))
    with new_bot.engine.connect() as conn:
        for parent in PARENT_TABLES:
            assert conn.execute(text("SELECT to_regclass(:p)"),
                                {"p": partition_name(parent, new_bot.bot)}).scalar()
