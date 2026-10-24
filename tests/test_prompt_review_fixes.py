"""Scheduling review regressions. SQLite memory only; no bots, models or services."""
import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import (Boolean, Column, DateTime, Float, JSON, MetaData, String,
                        Table, create_engine, event, insert, select, text)
from sqlalchemy.pool import StaticPool

from llm_bawt.inter_bot_delivery import DeliveryRecord, InterBotDeliveryStore
from llm_bawt.memory.message_store import MessageRowStore
from llm_bawt.memory.postgresql_messages import PostgreSQLMessageMixin
from llm_bawt.memory.postgresql_short_term import PostgreSQLShortTermManager
from llm_bawt.memory.summary_extraction_policy import summary_allows_extraction
from llm_bawt.prompt_purge import PromptPurgeGuard
from llm_bawt.service.prompt_execution import PromptHistoryManager, create_prompt_instance
from llm_bawt.service.routes import inter_bot_deliveries as routes


@pytest.fixture
def engine():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    with engine.connect() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON"))
    yield engine
    engine.dispose()


def delivery_row(delivery_id, metadata):
    now = datetime.now(timezone.utc)
    return dict(id=delivery_id, sender_bot_id="scheduler:owner-hash", target_bot_id="test",
                author_entity_type="user", author_entity_id="owner", message="Private prompt",
                user_message_id="message-" + delivery_id, turn_id="turn-" + delivery_id,
                metadata_json=metadata, status="QUEUED", attempt_count=0, max_attempts=5,
                created_at=now, updated_at=now, available_at=now)


@pytest.fixture
def delivery_api(engine, monkeypatch):
    metadata = MetaData()
    table = Table("inter_bot_deliveries", metadata,
                  Column("ordinal", String, primary_key=True),
                  *(Column(key, JSON if key == "metadata_json" else
                           DateTime if key.endswith("_at") else String)
                    for key in delivery_row("", {})))
    metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(insert(table), [
            dict(delivery_row("ordinary", {}), ordinal="1"),
            dict(delivery_row("private", {"prompt_schedule": {
                "schedule_id": "private-job", "occurrence_id": "private-run",
                "scheduled_for": "2026-09-15T00:00:00Z"}}), ordinal="2"),
            dict(delivery_row("malformed-private", {"prompt_schedule": {}}), ordinal="3"),
        ])
    store = InterBotDeliveryStore.__new__(InterBotDeliveryStore)
    store.engine = engine
    # Use typed rows here; raw SQLite timestamp strings aren't production datetimes.
    def get(delivery_id):
        with engine.connect() as conn:
            row = conn.execute(select(table).where(table.c.id == delivery_id)).mappings().first()
            return DeliveryRecord.from_mapping(row) if row else None
    store.get = get
    store.cancel = Mock(return_value=None)
    dispatcher = SimpleNamespace(store=store, _emit=AsyncMock())
    monkeypatch.setattr(routes, "_dispatcher", lambda: dispatcher)
    # Preserve the real SQL list/filter but adapt only SQLite timestamp values.
    original = store.list
    store.list = lambda **kw: [get(row.id) for row in original(**kw)]
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        yield client, store, dispatcher


@pytest.mark.parametrize("owner", ["owner", "another-ui-owner"])
@pytest.mark.parametrize("delivery_id", ["private", "malformed-private"])
def test_generic_delivery_read_and_cancel_never_expose_prompt_receipts(delivery_api, owner, delivery_id):
    client, store, dispatcher = delivery_api
    assert client.get(f"/v1/inter-bot-deliveries/{delivery_id}?user={owner}").status_code == 404
    assert client.post(f"/v1/inter-bot-deliveries/{delivery_id}/cancel?user={owner}").status_code == 404
    store.cancel.assert_not_called()
    dispatcher._emit.assert_not_called()
    # Internal scheduling reads are unaffected.
    assert store.get(delivery_id).status == "QUEUED"


def test_generic_delivery_list_filters_before_limit(delivery_api):
    client, store, _ = delivery_api
    response = client.get("/v1/inter-bot-deliveries?limit=1&target_bot_id=test&status=queued")
    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert [row["delivery_id"] for row in response.json()["deliveries"]] == ["ordinary"]
    assert len(store.list(limit=10)) == 3
    assert client.get("/v1/inter-bot-deliveries/ordinary").status_code == 200


def test_ordinary_delivery_cancel_remains_available(delivery_api):
    from dataclasses import replace
    client, store, dispatcher = delivery_api
    store.cancel.return_value = replace(store.get("ordinary"), status="CANCELLED")
    assert client.post("/v1/inter-bot-deliveries/ordinary/cancel").status_code == 200
    store.cancel.assert_called_once_with("ordinary")
    dispatcher._emit.assert_awaited_once()


def test_mcp_cancel_cannot_bypass_generic_route(delivery_api, monkeypatch):
    # Import the registration root first (the tool module re-exports through it).
    from llm_bawt.mcp_server import server  # noqa: F401
    from llm_bawt.mcp_server import inter_bot_tools
    client, store, _ = delivery_api
    class LocalClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def post(self, url, **kwargs):
            return client.post(url.split(inter_bot_tools._APP_BASE_URL)[1])
    monkeypatch.setattr(inter_bot_tools.httpx, "AsyncClient", LocalClient)
    result = asyncio.run(inter_bot_tools.cancel_delivery("private"))
    assert result["status_code"] == 404
    store.cancel.assert_not_called()


@pytest.fixture
def history_backend(engine):
    table = Table("messages_test", MetaData(),
                  Column("id", String, primary_key=True), Column("role", String),
                  Column("content", String), Column("timestamp", Float),
                  Column("session_id", String), Column("attachments", JSON),
                  Column("reasoning", String), Column("author_entity_type", String),
                  Column("author_entity_id", String), Column("processed", Boolean, default=False),
                  Column("summary_metadata", JSON), Column("created_at", DateTime))
    table.metadata.create_all(engine)
    backend = PostgreSQLMessageMixin()
    backend.engine, backend.messages_table, backend._messages_table_name = engine, table, table.name
    backend._message_rows = MessageRowStore(engine, table, table.name)
    manager = PostgreSQLShortTermManager.__new__(PostgreSQLShortTermManager)
    manager._backend = backend
    manager._session_id_cache = "personal"
    history = PromptHistoryManager.__new__(PromptHistoryManager)
    history.messages, history._db_backend = [], manager
    return history, backend


@pytest.mark.parametrize("extract", [False, True])
def test_prompt_request_persists_atomic_extraction_policy_for_both_roles(history_backend, extract, monkeypatch):
    from llm_bawt.service import prompt_execution
    from llm_bawt.service import prompt_capabilities
    history, backend = history_backend
    monkeypatch.setattr(prompt_capabilities, "resolve_prompt_target", lambda *a: "bound-model")
    monkeypatch.setattr(prompt_execution, "PromptLLMBawt", lambda **kw: SimpleNamespace(history_manager=history))
    service = SimpleNamespace(config=SimpleNamespace(model_copy=lambda **kw: object()))
    request = SimpleNamespace(bot_id="test", model="bound-model", user="owner",
                              augment_memory=False, extract_memory=extract)
    create_prompt_instance(service, request)
    statements = []
    event.listen(backend.engine, "before_cursor_execute",
                 lambda conn, cursor, sql, *args: statements.append(sql))
    for role in ("user", "assistant"):
        history.add_message(role, f"{role} private fact", message_id=role, session_id="automation")
    # INSERT itself carries exclusion; no later UPDATE window for the selector.
    assert not any(sql.lstrip().upper().startswith("UPDATE") for sql in statements)
    with backend.engine.connect() as conn:
        rows = conn.execute(select(backend.messages_table)).mappings().all()
    assert all(row["processed"] is (not extract) for row in rows)
    assert all(row["session_id"] == "automation" for row in rows)
    assert len(backend.get_unprocessed_messages()) == (2 if extract else 0)
    assert history._db_backend._session_id_cache == "personal"
    # Stable-id updates cannot revive an excluded row even with default consent.
    backend.add_message("user", "user", "edited", 2)
    assert len(backend.get_unprocessed_messages()) == (2 if extract else 0)


def test_excluded_message_insert_failure_leaves_no_extractable_row(history_backend):
    history, backend = history_backend
    history.extract_memory = False
    def fail_insert(conn, cursor, sql, *args):
        if sql.lstrip().upper().startswith("INSERT"):
            raise RuntimeError("injected insert failure")
    event.listen(backend.engine, "before_cursor_execute", fail_insert)
    with pytest.raises(RuntimeError, match="injected insert failure"):
        history.add_message("user", "Secret", message_id="failed", session_id="automation")
    event.remove(backend.engine, "before_cursor_execute", fail_insert)
    assert backend.get_unprocessed_messages() == []
    with backend.engine.connect() as conn:
        assert conn.execute(select(backend.messages_table)).first() is None


def test_extraction_policy_is_request_local(history_backend):
    history, backend = history_backend
    history.extract_memory = False
    history.add_message("user", "Excluded", message_id="private", session_id="automation")
    other = PromptHistoryManager.__new__(PromptHistoryManager)
    other.messages, other._db_backend = [], history._db_backend
    other.add_message("user", "Allowed", message_id="ordinary", session_id="personal")
    assert [row["id"] for row in backend.get_unprocessed_messages()] == ["ordinary"]


@pytest.mark.parametrize("excluded_role", ["user", "assistant"])
def test_summary_derived_extraction_skips_any_excluded_source(history_backend, excluded_role):
    history, backend = history_backend
    history.extract_memory = True
    history.add_message("user", "User secret", message_id="user", session_id="automation")
    history.add_message("assistant", "Assistant repeats secret", message_id="assistant", session_id="automation")
    backend._message_rows.upsert(message_id="summary", role="summary", content="User secret",
                                timestamp=3, session_id="automation")
    with backend.engine.begin() as conn:
        conn.execute(backend.messages_table.update().where(backend.messages_table.c.id == "summary")
                     .values(summary_metadata={"message_ids": ["user", "assistant"]}))
    assert summary_allows_extraction(backend, "summary")
    history.extract_memory = False
    history.add_message(excluded_role, "Private updated content", message_id=excluded_role, session_id="automation")
    assert not summary_allows_extraction(backend, "summary")
    assert not summary_allows_extraction(backend, "missing-summary")


def test_summary_extraction_entrypoint_does_not_send_opted_out_user_content(history_backend, monkeypatch):
    from llm_bawt.service.background_tasks import BackgroundTasksMixin
    from llm_bawt.memory.extraction.service import MemoryExtractionService
    history, backend = history_backend
    history.extract_memory = False
    history.add_message("user", "Never extract my secret", message_id="user", session_id="automation")
    with backend.engine.begin() as conn:
        conn.execute(insert(backend.messages_table).values(id="summary", role="summary",
                     content="Never extract my secret", summary_metadata={"message_ids": ["user"]}))
    monkeypatch.setattr("llm_bawt.memory.postgresql.PostgreSQLMemoryBackend", lambda *a, **kw: backend)
    extract = Mock(side_effect=AssertionError("Excluded content reached extraction"))
    monkeypatch.setattr(MemoryExtractionService, "extract_from_summary", extract)
    service = BackgroundTasksMixin()
    service.config = SimpleNamespace()
    service._get_background_client = lambda: (object(), "mock")
    service.get_memory_client = lambda *a: object()
    result = service._extract_from_summaries([
        {"created": True, "summary_id": "summary", "summary_text": "Never extract my secret"}
    ], "test", "owner")
    assert result["summaries_processed"] == 0
    extract.assert_not_called()


def schedule_tables(engine):
    from llm_bawt.service.scheduler import create_scheduler_tables
    from llm_bawt.service.prompt_schedule_store import PromptScheduleStore
    from llm_bawt.service.prompt_timing import PromptTiming
    create_scheduler_tables(engine)
    store = PromptScheduleStore(engine)
    now = datetime.now(timezone.utc)
    job = store.create(owner="owner", bot_id="test", name="Test", prompt="Test",
                       timing=PromptTiming(kind="once", once_at=now + timedelta(seconds=1)), now=now)
    run = store.reserve_due(now + timedelta(seconds=1))[0]
    return job, run


@pytest.mark.parametrize("state", ["pending", "queued", "running", "unknown"])
def test_purge_refuses_active_or_unknown_occurrences(engine, state):
    job, run = schedule_tables(engine)
    with engine.begin() as conn:
        conn.execute(text("UPDATE prompt_occurrences SET state=:state WHERE run_id=:run"),
                     {"state": state, "run": run})
    with pytest.raises(ValueError, match="active or unknown"):
        with engine.begin() as conn:
            guard = PromptPurgeGuard(conn)
            guard.guard({"test"})
            guard.delete_jobs("test")
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM scheduled_jobs WHERE id=:id"), {"id": job}).scalar_one() == 1


@pytest.mark.parametrize("state", ["succeeded", "failed", "skipped", "cancelled"])
def test_purge_deletes_terminal_companions_before_fk_parents(engine, state):
    schedule_tables(engine)
    with engine.begin() as conn:
        conn.execute(text("UPDATE prompt_occurrences SET state=:state"), {"state": state})
        guard = PromptPurgeGuard(conn)
        guard.guard({"test"})
        counts = guard.delete_jobs("test")
        assert counts == {"prompt_occurrences": 1, "prompt_schedules": 1, "job_runs": 1, "scheduled_jobs": 1}


@pytest.mark.parametrize("status,accepted", [("QUEUED", None), ("DISPATCHING", None),
                                             ("STEERING", "accepted"), ("FAILED", "accepted"),
                                             ("CANCELLED", "accepted"), ("UNKNOWN", None)])
def test_purge_guards_unlinked_prompt_deliveries(engine, status, accepted):
    with engine.begin() as conn:
        conn.execute(text("""CREATE TABLE inter_bot_deliveries (target_bot_id TEXT,
            metadata_json TEXT, status TEXT, transport_accepted_at TEXT)"""))
        conn.execute(text("""INSERT INTO inter_bot_deliveries VALUES
            ('test', :meta, :status, :accepted)"""),
            {"meta": json.dumps({"prompt_schedule": {"schedule_id": "job", "occurrence_id": "run"}}),
             "status": status, "accepted": accepted})
        with pytest.raises(ValueError, match="deliveries are active or unknown"):
            PromptPurgeGuard(conn).guard({"test"})
        PromptPurgeGuard(conn).guard({"another-bot"})


def test_purge_checks_delivery_receipts_after_schedule_target_changes(engine):
    job, run = schedule_tables(engine)
    with engine.begin() as conn:
        conn.execute(text("UPDATE prompt_occurrences SET state='succeeded', delivery_id='delivery'"))
        conn.execute(text("""CREATE TABLE inter_bot_deliveries (id TEXT, target_bot_id TEXT,
            metadata_json TEXT, status TEXT, transport_accepted_at TEXT)"""))
        conn.execute(text("""INSERT INTO inter_bot_deliveries VALUES
            ('delivery', 'old-target', :meta, 'QUEUED', NULL)"""),
            {"meta": json.dumps({"prompt_schedule": {"schedule_id": job, "occurrence_id": run}})})
        with pytest.raises(ValueError, match="deliveries are active or unknown"):
            PromptPurgeGuard(conn).guard({"test"})


def test_purge_child_deletes_roll_back_with_caller_transaction(engine):
    schedule_tables(engine)
    with engine.begin() as conn:
        conn.execute(text("UPDATE prompt_occurrences SET state='cancelled'"))
    with pytest.raises(RuntimeError, match="interrupted purge"):
        with engine.begin() as conn:
            guard = PromptPurgeGuard(conn)
            guard.guard({"test"})
            guard.delete_jobs("test")
            raise RuntimeError("interrupted purge")
    with engine.connect() as conn:
        for table in ("prompt_occurrences", "prompt_schedules", "job_runs", "scheduled_jobs"):
            assert conn.execute(text(f"SELECT count(*) FROM {table}")).scalar_one() == 1


@pytest.fixture
def purge_runtime(engine, monkeypatch):
    """Run actual purge entrypoints on SQLite with only PG syntax adapted."""
    import sqlalchemy
    from llm_bawt import runtime_settings
    schedule_tables(engine)
    with engine.begin() as conn:
        for table, columns in {
            "messages": "bot_id TEXT", "memories": "bot_id TEXT",
            "forgotten_messages": "bot_id TEXT", "bot_profiles": "slug TEXT",
            "runtime_settings": "scope_type TEXT, scope_id TEXT",
            "entity_profiles": "entity_type TEXT, entity_id TEXT",
            "entity_profile_attributes": "entity_type TEXT, entity_id TEXT",
            "prompt_template_versions": "scope_type TEXT, scope_id TEXT",
            "prompt_templates": "scope_type TEXT, scope_id TEXT",
            "tool_call_records": "bot_id TEXT", "turn_logs": "bot_id TEXT",
        }.items():
            conn.execute(text(f"CREATE TABLE {table} ({columns})"))
        conn.execute(text("INSERT INTO messages VALUES ('test')"))
        conn.execute(text("INSERT INTO turn_logs VALUES ('test')"))
    def translate(conn, cursor, sql, parameters, context, many):
        if "FROM information_schema.tables" in sql:
            sql = "SELECT name FROM sqlite_master WHERE type='table'"
        return sql.replace("entity_type::text", "entity_type"), parameters
    event.listen(engine, "before_cursor_execute", translate, retval=True)
    monkeypatch.setattr(sqlalchemy, "create_engine", lambda *a, **kw: SimpleNamespace(
        connect=engine.connect, dispose=lambda: None))
    monkeypatch.setattr(runtime_settings, "has_database_credentials", lambda config: True)
    from llm_bawt.utils import db
    monkeypatch.setattr(db, "get_shared_engine", lambda config: engine)
    return runtime_settings, SimpleNamespace(), engine


@pytest.mark.parametrize("cleanup", [False, True])
def test_runtime_purge_entrypoints_guard_before_any_mutation(purge_runtime, cleanup):
    runtime, config, engine = purge_runtime
    statements = []
    event.listen(engine, "before_cursor_execute", lambda conn, cursor, sql, *args: statements.append(sql))
    with pytest.raises(ValueError, match="active or unknown"):
        if cleanup:
            runtime.purge_orphaned_shared_rows(config)
        else:
            runtime.purge_bot_data(config, "test")
    assert not any(sql.lstrip().upper().startswith(("DELETE", "DROP")) for sql in statements)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM messages")).scalar_one() == 1
        assert conn.execute(text("SELECT count(*) FROM turn_logs")).scalar_one() == 1


@pytest.mark.parametrize("cleanup", [False, True])
def test_runtime_purge_entrypoints_honor_prompt_fk_order(purge_runtime, cleanup):
    runtime, config, engine = purge_runtime
    with engine.begin() as conn:
        conn.execute(text("UPDATE prompt_occurrences SET state='cancelled'"))
    result = (runtime.purge_orphaned_shared_rows(config) if cleanup else
              runtime.purge_bot_data(config, "test"))
    assert result["deleted_rows"]["prompt_occurrences"] == 1
    assert result["deleted_rows"]["prompt_schedules"] == 1
    assert result["deleted_rows"]["job_runs"] == 1
    assert result["deleted_rows"]["scheduled_jobs"] == 1


def test_orphan_cleanup_dry_run_does_not_mutate_active_prompts(purge_runtime):
    runtime, config, engine = purge_runtime
    result = runtime.purge_orphaned_shared_rows(config, dry_run=True)
    assert result["orphaned_bot_ids"] == ["test"]
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM prompt_occurrences")).scalar_one() == 1


def test_purge_optional_tables_and_other_bot_preserved(engine):
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE scheduled_jobs (id TEXT PRIMARY KEY, bot_id TEXT)"))
        conn.execute(text("CREATE TABLE job_runs (id TEXT PRIMARY KEY, job_id TEXT REFERENCES scheduled_jobs(id))"))
        conn.execute(text("INSERT INTO scheduled_jobs VALUES ('a','test'),('b','other')"))
        conn.execute(text("INSERT INTO job_runs VALUES ('ra','a'),('rb','b')"))
        guard = PromptPurgeGuard(conn)
        guard.guard({"test"})
        assert guard.delete_jobs("test") == {"job_runs": 1, "scheduled_jobs": 1}
        assert conn.execute(text("SELECT id FROM scheduled_jobs")).scalar_one() == "b"
