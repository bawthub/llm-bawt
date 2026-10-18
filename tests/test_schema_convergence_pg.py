"""Fresh creation and existing-table behavior on disposable private PostgreSQL."""

from __future__ import annotations

import json
import uuid
import runpy
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.exc import IntegrityError

from tests.pg_isolation import require_test_db_config
from tests.private_postgres import PORT, private_url

pytestmark = pytest.mark.integration


@pytest.fixture
def schema_db(private_pg, monkeypatch):
    from llm_bawt.memory.postgresql import PostgreSQLMemoryBackend
    from llm_bawt.profiles import ProfileManager
    from llm_bawt.runtime_settings import BotProfileStore
    from llm_bawt.service.chat_pending_questions import PendingQuestionStore
    from llm_bawt.service.turn_logs import TurnLogStore
    from llm_bawt.utils import db

    name = f"schema_contract_{uuid.uuid4().hex[:12]}_test"
    admin = create_engine(private_url(private_pg, "postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    url = private_url(private_pg, name)
    monkeypatch.setattr(db, "build_postgres_url", lambda config: url)
    monkeypatch.setenv("LLM_BAWT_POSTGRES_HOST", str(private_pg))
    monkeypatch.setenv("LLM_BAWT_POSTGRES_PORT", PORT)
    monkeypatch.setenv("LLM_BAWT_POSTGRES_USER", "test_user")
    monkeypatch.setenv("LLM_BAWT_POSTGRES_DATABASE", name)
    monkeypatch.setenv("LLM_BAWT_POSTGRES_PASSWORD", uuid.uuid4().hex)
    engine = create_engine(url)
    try:
        config = require_test_db_config()
        PostgreSQLMemoryBackend(config, bot_id="schema_proof")
        TurnLogStore(config)
        assert BotProfileStore(config).engine is not None
        ProfileManager(config)
        assert PendingQuestionStore(config).engine is not None
        yield config, engine
    finally:
        engine.dispose()
        db.reset_shared_engines()
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


def test_fresh_column_contract_from_real_catalog(schema_db):
    _, engine = schema_db
    names = {
        "bot_profiles": ("endpoint_id", "tts_mode", "include_summaries", "include_in_global_search"),
        "chat_pending_questions": ("origin_harness",),
        "entity_profiles": ("summary",),
        "tool_call_records": (
            "tool_use_id", "parent_tool_use_id", "approval_request_id", "approval_status",
            "result_payload_available",
        ),
    }
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT c.relname, a.attname, format_type(a.atttypid,a.atttypmod),
                   a.attnotnull, pg_get_expr(d.adbin,d.adrelid)
            FROM pg_attribute a
            JOIN pg_class c ON c.oid=a.attrelid
            JOIN pg_namespace n ON n.oid=c.relnamespace AND n.nspname='public'
            LEFT JOIN pg_attrdef d ON d.adrelid=a.attrelid AND d.adnum=a.attnum
            WHERE c.relname IN :tables AND a.attnum > 0 AND NOT a.attisdropped
        """).bindparams(bindparam("tables", expanding=True)),
            {"tables": list(names)}).all()
    columns = {(table, column): (typ, not_null, default) for table, column, typ, not_null, default in rows}
    assert columns[("bot_profiles", "endpoint_id")][:2] == ("bigint", False)
    for name, default in (("tts_mode", "false"), ("include_summaries", "true"),
                          ("include_in_global_search", "true")):
        assert columns[("bot_profiles", name)] == ("boolean", True, default)
    assert columns[("entity_profiles", "summary")][:2] == ("text", False)
    assert columns[("chat_pending_questions", "origin_harness")] == (
        "character varying(32)", True, "'claude'::character varying",
    )
    for name in names["tool_call_records"][:-1]:
        assert columns[("tool_call_records", name)][:2] == ("character varying", False)
    assert columns[("tool_call_records", "result_payload_available")] == (
        "boolean", True, "false",
    )


def test_fresh_foreign_keys_and_delete_paths(schema_db):
    _, engine = schema_db
    with engine.connect() as conn:
        constraints = {
            name: (delete_action, deferrable, validated)
            for name, delete_action, deferrable, validated in conn.execute(text("""
                SELECT conname, confdeltype::text, condeferrable, convalidated
                FROM pg_constraint
                WHERE conname IN ('fk_messages_session', 'fk_toolcalls_turn')
                  AND conrelid IN ('public.messages'::regclass, 'public.tool_call_records'::regclass)
            """))
        }
    assert constraints == {
        "fk_messages_session": ("a", False, True),
        "fk_toolcalls_turn": ("n", False, True),
    }
    session_id, message_id, turn_id = (str(uuid.uuid4()) for _ in range(3))
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO sessions (id, bot_id) VALUES (:id, 'schema_proof')"), {"id": session_id})
        for value in (session_id, None):
            conn.execute(text("""
                INSERT INTO messages (bot_id, id, role, content, timestamp, session_id)
                VALUES ('schema_proof', :id, 'user', 'synthetic', 1.0, :session)
            """), {"id": message_id if value else str(uuid.uuid4()), "session": value})
        conn.execute(text("UPDATE sessions SET status='archived' WHERE id=:id"), {"id": session_id})
        from llm_bawt.service.turn_log_schema import TurnLog
        conn.execute(TurnLog.__table__.insert().values(
            id=turn_id, created_at=datetime.now(timezone.utc),
        ))
        tool_id = conn.execute(text("""
            INSERT INTO tool_call_records (turn_id, tool_name, iteration, created_at)
            VALUES (:turn, 'synthetic', 1, CURRENT_TIMESTAMP) RETURNING id
        """), {"turn": turn_id}).scalar_one()
    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO messages (bot_id, id, role, content, timestamp, session_id)
            VALUES ('schema_proof', :id, 'user', 'synthetic', 1.0, :session)
        """), {"id": str(uuid.uuid4()), "session": str(uuid.uuid4())})
    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(text("DELETE FROM sessions WHERE id=:id"), {"id": session_id})
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM turn_logs WHERE id=:id"), {"id": turn_id})
        assert conn.execute(text("SELECT turn_id FROM tool_call_records WHERE id=:id"),
                            {"id": tool_id}).scalar_one() is None
        assert conn.execute(text("SELECT session_id FROM messages WHERE id=:id"),
                            {"id": message_id}).scalar_one() == session_id


def test_catalog_query_is_read_only_and_has_unique_stable_keys(schema_db):
    _, engine = schema_db
    tool = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/schema_catalog.py"))
    with engine.connect() as conn:
        conn.exec_driver_sql("SET TRANSACTION READ ONLY")
        lines = [row[0] for row in conn.execute(text(tool["CATALOG_SQL"]))]
    catalog = tool["parse_catalog"](lines)
    assert ("constraint", "messages.fk_messages_session") in catalog
    assert ("constraint", "tool_call_records.fk_toolcalls_turn") in catalog
    assert tool["compare"](catalog, catalog) == []


def test_readonly_capture_checks_disposable_identity_and_records_provenance(schema_db, tmp_path):
    config, _ = schema_db
    tool = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/schema_catalog.py"))
    lines, metadata = tool["capture_test"](config.POSTGRES_DATABASE)
    assert metadata["read_only"] is True
    assert len(lines) > 0
    output = tmp_path / "catalog.jsonl"
    tool["write_snapshot"](output, lines, metadata)
    saved = json.loads((tmp_path / "catalog.jsonl.meta.json").read_text())
    assert saved["record_count"] == len(lines)
    assert len(saved["source_sha256"]) == 64
    assert len(saved["catalog_sha256"]) == 64
    with pytest.raises(FileExistsError):
        tool["write_snapshot"](output, lines, metadata)


def test_existing_table_rebootstrap_does_not_repair_missing_fk(schema_db):
    config, engine = schema_db
    from llm_bawt.memory.postgresql import PostgreSQLMemoryBackend
    from llm_bawt.service.turn_logs import TurnLogStore
    from llm_bawt.utils import db

    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE public.messages DROP CONSTRAINT fk_messages_session"))
        conn.execute(text("ALTER TABLE public.tool_call_records DROP CONSTRAINT fk_toolcalls_turn"))
    db.reset_shared_engines()
    PostgreSQLMemoryBackend(config, bot_id="schema_proof")
    TurnLogStore(config)
    with engine.connect() as conn:
        assert conn.execute(text("""
            SELECT count(*) FROM pg_constraint
            WHERE conname IN ('fk_messages_session', 'fk_toolcalls_turn')
              AND conrelid IN ('public.messages'::regclass, 'public.tool_call_records'::regclass)
        """)).scalar_one() == 0
