"""Exact-bot cleanup, enum storage and real session FKs on private PostgreSQL."""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

from llm_bawt import runtime_settings
from tests.private_postgres import private_url

pytestmark = pytest.mark.integration


@pytest.fixture
def purge_db(private_pg, private_module_db, monkeypatch):
    url = private_url(private_pg, private_module_db)
    admin = create_engine(url)
    schema = "purge_" + uuid.uuid4().hex
    with admin.begin() as conn:
        conn.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
    engine = create_engine(url, connect_args={"options": f"-csearch_path={schema}"})
    with engine.begin() as conn:
        for ddl in (
            "CREATE TYPE entitytype AS ENUM ('BOT','USER','bot','user')",
            "CREATE TABLE sessions (id TEXT PRIMARY KEY, bot_id TEXT, user_id TEXT, status TEXT)",
            "CREATE TABLE bot_profiles (slug TEXT PRIMARY KEY, last_session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL)",
            "CREATE TABLE messages (id TEXT PRIMARY KEY, bot_id TEXT, session_id TEXT REFERENCES sessions(id))",
            "CREATE TABLE memories (bot_id TEXT)",
            "CREATE TABLE forgotten_messages (bot_id TEXT)",
            "CREATE TABLE runtime_settings (scope_type TEXT, scope_id TEXT)",
            "CREATE TABLE entity_profiles (entity_type entitytype, entity_id TEXT)",
            "CREATE TABLE entity_profile_attributes (entity_type entitytype, entity_id TEXT)",
            "CREATE TABLE prompt_template_versions (scope_type TEXT, scope_id TEXT)",
            "CREATE TABLE prompt_templates (scope_type TEXT, scope_id TEXT)",
            "CREATE TABLE turn_logs (id TEXT PRIMARY KEY, bot_id TEXT)",
            "CREATE TABLE tool_call_records (id INTEGER PRIMARY KEY, bot_id TEXT, turn_id TEXT REFERENCES turn_logs(id) ON DELETE SET NULL)",
            # The legacy payload store has no FK or cascade in the real schema.
            "CREATE TABLE tool_call_result_payloads (tool_call_record_id INTEGER PRIMARY KEY, content_text TEXT)",
            "CREATE TABLE scheduled_jobs (id TEXT PRIMARY KEY, bot_id TEXT)",
            "CREATE TABLE job_runs (id TEXT PRIMARY KEY, job_id TEXT REFERENCES scheduled_jobs(id), bot_id TEXT, status TEXT)",
            "CREATE TABLE inter_bot_deliveries (id TEXT PRIMARY KEY, sender_bot_id TEXT, target_bot_id TEXT, status TEXT, metadata_json JSONB, transport_accepted_at TIMESTAMP)",
            "CREATE TABLE new_session_event_outbox (id TEXT PRIMARY KEY, bot_id TEXT, session_id TEXT, published_at TIMESTAMP)",
            "CREATE TABLE bot_profile_changes (bot_slug TEXT)",
        ):
            conn.exec_driver_sql(ddl)

    def purge(slug="source"):
        # Only the existing purge's engine construction is redirected. Every SQL
        # operation and commit runs on this test-owned private PostgreSQL schema.
        with monkeypatch.context() as patch:
            patch.setattr(sqlalchemy, "create_engine", lambda *a, **kw: SimpleNamespace(
                connect=engine.connect, dispose=lambda: None))
            patch.setattr(runtime_settings, "has_database_credentials", lambda config: True)
            return runtime_settings.purge_bot_data(SimpleNamespace(), slug)

    try:
        yield engine, purge
    finally:
        engine.dispose()
        with admin.begin() as conn:
            conn.exec_driver_sql(f'DROP SCHEMA "{schema}" CASCADE')
        admin.dispose()


def seed(engine, entity_type="BOT", keep_profile=False):
    with engine.begin() as conn:
        conn.exec_driver_sql("""INSERT INTO sessions VALUES
            ('s-active','source','owner','active'),
            ('s-archived','source','other-user','archived'),
            ('s-deleted','source',NULL,'deleted'),
            ('s-dash','dash','owner','active')""")
        conn.exec_driver_sql("INSERT INTO bot_profiles VALUES ('dash','s-dash')")
        if keep_profile:
            conn.exec_driver_sql("INSERT INTO bot_profiles VALUES ('source','s-active')")
        conn.exec_driver_sql("""INSERT INTO messages VALUES
            ('m-source','source','s-active'),('m-dash','dash','s-dash')""")
        for table in ("memories", "forgotten_messages"):
            conn.exec_driver_sql(f"INSERT INTO {table} VALUES ('source'),('dash')")
        for table in ("runtime_settings", "prompt_templates", "prompt_template_versions"):
            conn.exec_driver_sql(f"INSERT INTO {table} VALUES ('bot','source'),('bot','dash'),('user','source')")
        for table in ("entity_profiles", "entity_profile_attributes"):
            conn.execute(text(f"INSERT INTO {table} VALUES (:type,'source'),('USER','source'),('user','source'),('BOT','dash')"),
                         {"type": entity_type})
        conn.exec_driver_sql("INSERT INTO turn_logs VALUES ('t-source','source'),('t-dash','dash')")
        conn.exec_driver_sql("INSERT INTO tool_call_records VALUES (1,'source','t-source'),(2,'dash','t-dash')")
        conn.exec_driver_sql("INSERT INTO tool_call_result_payloads VALUES (1,'source payload'),(2,'dash payload')")
        conn.exec_driver_sql("INSERT INTO scheduled_jobs VALUES ('shared','*'),('own','source')")
        conn.exec_driver_sql("""INSERT INTO job_runs VALUES
            ('r-source','shared','source','SUCCESS'),
            ('r-running','shared','source','RUNNING'),
            ('r-dash','shared','dash','SUCCESS'),('r-own','own','source','SUCCESS')""")
        conn.exec_driver_sql("""INSERT INTO inter_bot_deliveries VALUES
            ('d-in','dash','source','DELIVERED','{}',CURRENT_TIMESTAMP),
            ('d-out','source','dash','FAILED','{}',NULL)""")
        conn.exec_driver_sql("""INSERT INTO new_session_event_outbox VALUES
            ('o-source','source','s-active',CURRENT_TIMESTAMP),
            ('o-dash','dash','s-dash',NULL)""")
        conn.exec_driver_sql("INSERT INTO bot_profile_changes VALUES ('source'),('dash')")


@pytest.mark.parametrize("entity_type", ["BOT", "bot"])
@pytest.mark.parametrize("keep_profile", [False, True])
def test_purge_removes_bot_profile_data_and_sessions_without_requiring_definition(
    purge_db, entity_type, keep_profile,
):
    engine, purge = purge_db
    seed(engine, entity_type, keep_profile)
    result = purge(" SOURCE ")
    assert result["deleted_rows"]["sessions"] == 3
    assert result["deleted_rows"]["entity_profiles"] == 1
    assert result["deleted_rows"]["entity_profile_attributes"] == 1
    with engine.connect() as conn:
        assert conn.exec_driver_sql("SELECT id FROM sessions").scalars().all() == ["s-dash"]
        assert conn.exec_driver_sql("SELECT id FROM messages").scalars().all() == ["m-dash"]
        assert conn.exec_driver_sql("SELECT last_session_id FROM bot_profiles WHERE slug='dash'").scalar_one() == "s-dash"
        if keep_profile:
            assert conn.exec_driver_sql("SELECT last_session_id FROM bot_profiles WHERE slug='source'").scalar_one() is None
        else:
            assert conn.exec_driver_sql("SELECT count(*) FROM bot_profiles WHERE slug='source'").scalar_one() == 0
        for table in ("entity_profiles", "entity_profile_attributes"):
            assert set(conn.exec_driver_sql(f"SELECT entity_type::text,entity_id FROM {table}").all()) == {
                ("BOT", "dash"), ("USER", "source"), ("user", "source"),
            }
        for table in ("runtime_settings", "prompt_templates", "prompt_template_versions"):
            assert conn.exec_driver_sql(f"SELECT scope_type,scope_id FROM {table} ORDER BY 1,2").all() == [
                ("bot", "dash"), ("user", "source"),
            ]
    # Standalone repair works after profile deletion and is idempotent.
    again = purge()
    assert all(n == 0 for n in again["deleted_rows"].values())


def test_purge_removes_only_owned_tool_payloads_before_deleting_records(purge_db):
    engine, purge = purge_db
    seed(engine)
    result = purge()
    assert result["deleted_rows"]["tool_call_result_payloads"] == 1
    with engine.connect() as conn:
        assert conn.exec_driver_sql("SELECT * FROM tool_call_result_payloads").all() == [(2, "dash payload")]
        assert conn.exec_driver_sql("SELECT id FROM tool_call_records").scalars().all() == [2]
        assert conn.exec_driver_sql("SELECT id FROM turn_logs").scalars().all() == ["t-dash"]


def test_purge_does_not_require_optional_payload_table(purge_db):
    engine, purge = purge_db
    seed(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP TABLE tool_call_result_payloads")
    assert purge()["deleted_rows"]["tool_call_records"] == 1


def test_purge_preserves_shared_job_history_and_cross_bot_receipts(purge_db):
    engine, purge = purge_db
    seed(engine)
    tables = ("inter_bot_deliveries", "new_session_event_outbox", "bot_profile_changes")
    with engine.connect() as conn:
        before = {t: conn.exec_driver_sql(f"SELECT * FROM {t} ORDER BY 1").all() for t in tables}
    result = purge()
    assert result["deleted_rows"]["job_runs"] == 1
    with engine.connect() as conn:
        assert conn.exec_driver_sql("SELECT id FROM scheduled_jobs").scalars().all() == ["shared"]
        assert conn.exec_driver_sql("SELECT id,status FROM job_runs ORDER BY id").all() == [
            ("r-dash", "SUCCESS"), ("r-running", "RUNNING"), ("r-source", "SUCCESS"),
        ]
        for table in tables:
            assert conn.exec_driver_sql(f"SELECT * FROM {table} ORDER BY 1").all() == before[table]


def test_cross_bot_session_reference_rolls_back_instead_of_erasing_other_bot(purge_db):
    engine, purge = purge_db
    seed(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql("INSERT INTO messages VALUES ('foreign-reference','dash','s-active')")
    with pytest.raises(IntegrityError):
        purge()
    with engine.connect() as conn:
        assert conn.exec_driver_sql("SELECT count(*) FROM sessions").scalar_one() == 4
        assert conn.exec_driver_sql("SELECT count(*) FROM messages").scalar_one() == 3
        assert conn.exec_driver_sql("SELECT count(*) FROM entity_profiles").scalar_one() == 4
        assert conn.exec_driver_sql("SELECT count(*) FROM tool_call_result_payloads").scalar_one() == 2
