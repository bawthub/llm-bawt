"""Playground store on PostgreSQL (TASK-2190/2195): mailbox, purge, column upgrade."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text, update

from llm_bawt.media.object_store import FsBlobBackend
from llm_bawt.playground import store as store_module
from llm_bawt.playground.service import Playground, Source
from llm_bawt.playground.store import PlaygroundStore, artifacts, public_view
from llm_bawt.utils.db import get_shared_engine
from tests.pg_isolation import require_test_db_config

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.fixture
def pg(tmp_path):
    store = PlaygroundStore(get_shared_engine(require_test_db_config()))
    with store.engine.begin() as conn:
        conn.execute(text("DELETE FROM playground_artifacts"))
    yield Playground(store, FsBlobBackend(tmp_path))
    with store.engine.begin() as conn:
        conn.execute(text("DELETE FROM playground_artifacts"))


def _send(pg):
    return pg.send(user_id="u", bot_id="vex", title="t", note=None,
                   source=Source(kind="audio", mime="audio/wav", filename=None, data=b"RIFF"))


def test_mailbox_search_and_trash_purge_on_postgres(pg, tmp_path):
    first, second = _send(pg), _send(pg)
    _, token = pg.open(first["id"], "u")
    pg.open(first["id"], "u")
    assert pg.store.viewable(first["id"], "u", token) is None  # reopen mints a new token
    assert public_view(pg.store.get(first["id"], "u"))["status"] == "opened"
    assert (tmp_path / first["blob_key"]).exists()
    assert [r["id"] for r in pg.store.list("u", query="T")] == [second["id"], first["id"]]

    pg.store.set_state(second["id"], "u", "trash")
    with pg.store.engine.begin() as conn:
        conn.execute(update(artifacts).where(artifacts.c.id == second["id"]).values(
            trashed_at=datetime.now(timezone.utc) - timedelta(days=31)))
    assert pg.sweep(30)["purged"] == 1
    assert not (tmp_path / second["blob_key"]).exists()
    assert [row["id"] for row in pg.store.list("u", view="all", status="expired")] == [second["id"]]


def test_columns_are_added_to_an_old_table_with_rows_intact(pg):
    """The v1 table (no archive/trash columns) upgrades in place; rows keep their state."""
    engine = pg.store.engine
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE playground_artifacts DROP COLUMN IF EXISTS archived_at"))
        conn.execute(text("ALTER TABLE playground_artifacts DROP COLUMN IF EXISTS trashed_at"))
        conn.execute(text(
            "INSERT INTO playground_artifacts (id, user_id, sender_bot_id, title, kind, mime, size_bytes,"
            " inline_content, created_at, opened_at) VALUES"
            " ('pg_old0000000001', 'u', 'vex', 'old unread', 'text', 'text/plain', 1, 'a', now(), NULL),"
            " ('pg_old0000000002', 'u', 'vex', 'old read', 'text', 'text/plain', 1, 'b', now(), now())"))
    with engine.begin() as conn:
        store_module._bootstrap(conn)
    store = PlaygroundStore(engine)
    states = {r["id"]: public_view(r)["status"] for r in store.list("u")}
    assert states == {"pg_old0000000001": "unopened", "pg_old0000000002": "opened"}
