"""Preview grant writer uses a real private PostgreSQL runtime-settings row."""

import json

import pytest
from sqlalchemy import text

from llm_bawt.preview.credential_publisher import _upsert_grant
from llm_bawt.runtime_settings import RuntimeSettingsStore
from llm_bawt.service.preview_credential import GRANT_KEY, PreviewGrant
from tests.full_store_fixture import full_store_db  # noqa: F401 (private PG fixture)

pytestmark = pytest.mark.integration


def test_exact_run_replaced_atomically_then_revoked(full_store_db):  # noqa: F811
    config, engine = full_store_db
    RuntimeSettingsStore(config)  # schema + unique (scope, key) index
    first = PreviewGrant("00000000-0000-4000-8000-000000000001", "claude", 7, 2_000_000_000)
    second = PreviewGrant("00000000-0000-4000-8000-000000000002", "openai_chatgpt", 8, 2_000_000_100)
    _upsert_grant(engine, first)
    _upsert_grant(engine, second)
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT scope_type,scope_id,value_json FROM runtime_settings "
                                 "WHERE key=:key"), {"key": GRANT_KEY}).all()
    assert len(rows) == 1
    assert rows[0][:2] == ("global", "*")
    assert json.loads(rows[0].value_json) == second.__dict__
    _upsert_grant(engine, None)
    with engine.connect() as conn:
        count = conn.execute(text("SELECT count(*) FROM runtime_settings WHERE key=:key"),
                             {"key": GRANT_KEY}).scalar_one()
    assert count == 0
