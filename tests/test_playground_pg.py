"""Playground store on PostgreSQL (TASK-2190): open once, close, sweep."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text, update

from llm_bawt.media.object_store import FsBlobBackend
from llm_bawt.playground.service import Playground, Source
from llm_bawt.playground.store import OpenRefused, PlaygroundStore, artifacts, public_view
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


def test_one_time_open_close_and_sweep_on_postgres(pg, tmp_path):
    first, second = _send(pg), _send(pg)
    _, token = pg.open(first["id"], "u")
    assert pg.store.viewable(first["id"], "u", token) is not None
    with pytest.raises(OpenRefused):
        pg.open(first["id"], "u")
    assert not (tmp_path / first["blob_key"]).exists()
    assert public_view(pg.store.get(first["id"], "u"))["status"] == "expired"

    pg.open(second["id"], "u")
    with pg.store.engine.begin() as conn:
        conn.execute(update(artifacts).where(artifacts.c.id == second["id"]).values(
            opened_at=datetime.now(timezone.utc) - timedelta(hours=1)))
    assert pg.sweep(30)["expired"] == 1
    assert not (tmp_path / second["blob_key"]).exists()
    assert [row["id"] for row in pg.store.list("u", status="expired")] == [second["id"], first["id"]]
