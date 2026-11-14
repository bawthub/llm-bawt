"""Real PG partitions + complete production HTTP/storage write SQL receipts.

No fake add_message/rotation/storage implementation: only inference, transport,
and unrelated turn-log/event infrastructure are isolated. Schema provisioning is
explicitly warmed before measuring normal writes, not silently filtered out.
"""
from __future__ import annotations

import asyncio
import io
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from starlette.testclient import TestClient

from llm_bawt import task_turn_context
from llm_bawt.bot_access import BotPolicyResolver
from llm_bawt.bot_access import service as service_codec
from llm_bawt.bot_access.refresh import PolicyRefresher
from llm_bawt.memory.session_reset_lineage import SessionResetLineage
from llm_bawt.models.message import Message
from llm_bawt.service.schemas import ChatMessage
from llm_bawt.task_turn_context import mint_for_agent_turn as real_mint
from tests.session_lifecycle_pg_fixtures import rig
from tests.test_message_native_upsert import (
    FALSY_METADATA, INVALID_METADATA, capture_sql, check_metadata_contract,
    check_new_exclusion, check_retry_contract, check_sql_null_metadata, one_upsert, row,
)
from tests.test_nonstream_turn_execution import _claimed, _isolate, _request, _service

__all__ = ["rig", "_isolate"]
pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.fixture
def real_storage(rig, monkeypatch):
    from llm_bawt.mcp_server import storage as storage_module

    storage = storage_module.MemoryStorage(rig.config)
    backend = storage.get_backend(rig.bot, provision_schema=True)
    manager = storage.get_short_term_manager(rig.bot, provision_schema=True)
    assert storage.engine is backend.engine is manager._backend.engine is rig.engine
    assert storage.engine.pool._pre_ping is True
    # Warm both steady-state connections: fresh-connection driver SET/bootstrap
    # work is outside this normal-write receipt, not misreported as zero SQL.
    with storage.engine.connect(), storage.engine.connect():
        pass
    monkeypatch.setattr(storage_module, "_storage", storage)
    return storage


@pytest.mark.parametrize("stored_role", ["user", "assistant", "system", "summary"])
def test_partition_retry_contract(real_storage, rig, stored_role):
    check_retry_contract(real_storage.get_backend(rig.bot)._message_rows, stored_role, rig.session_id)


@pytest.mark.parametrize("metadata", FALSY_METADATA + INVALID_METADATA + [
    {"nested": {"keep": True}, "extract_memory": True, "other": None},
])
def test_partition_exclusion_metadata_contract(real_storage, rig, metadata):
    check_metadata_contract(real_storage.get_backend(rig.bot)._message_rows, metadata)


def test_partition_new_exclusion_contract(real_storage, rig):
    check_new_exclusion(real_storage.get_backend(rig.bot)._message_rows)


def test_partition_sql_null_metadata_contract(real_storage, rig):
    check_sql_null_metadata(real_storage.get_backend(rig.bot)._message_rows)


def test_partition_concurrent_same_id_is_one_row_with_first_user_timestamp(real_storage, rig):
    store = real_storage.get_backend(rig.bot)._message_rows
    barrier = threading.Barrier(2)

    def write(number):
        barrier.wait(timeout=10)  # timing-ok: hang guard for two real DB writers
        store.upsert(message_id="stable", role="user", content=str(number), timestamp=number,
                     session_id=rig.session_id)

    with capture_sql(rig.engine) as sql:
        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(write, [1, 2]))
    assert len(sql) == 2 and all("ON CONFLICT (bot_id, id)" in s for s in sql), sql
    assert sql.pool_ping_sql == ["SELECT 1", "SELECT 1"]
    assert len(sql.all_statements) == 4
    actual = row(store)
    assert actual["session_id"] == rig.session_id
    assert actual["timestamp"] in {1, 2}
    assert actual["content"] in {"1", "2"}


@pytest.mark.parametrize("invalid", [{"role": "x" * 21}, {"session_id": "x" * 37}])
def test_partition_constraint_failure_rolls_back_and_connection_remains_usable(real_storage, rig, invalid):
    store = real_storage.get_backend(rig.bot)._message_rows
    store.upsert(message_id="stable", role="assistant", content="first", timestamp=10)
    first = row(store)
    # An INSERT candidate is validated before conflict handling. Invalid immutable
    # fields formerly ignored by UPDATE can now fail; do not hide that distinction.
    values = {"message_id": "stable", "role": "assistant", "content": "invalid", "timestamp": 20,
              **invalid}
    with capture_sql(rig.engine) as sql, pytest.raises(DBAPIError) as failure:
        store.upsert(**values)
    assert failure.value.orig.pgcode == "22001"  # string_data_right_truncation, never coercion
    one_upsert(sql)
    assert row(store) == first
    with capture_sql(rig.engine) as sql:
        store.upsert(message_id="stable", role="assistant", content="recovered", timestamp=30)
    one_upsert(sql)
    assert row(store)["content"] == "recovered"


def test_same_id_is_partition_scoped(real_storage, rig):
    from llm_bawt.memory.postgresql import PARENT_TABLES, partition_name

    first = real_storage.get_backend(rig.bot)._message_rows
    other_bot = rig.bot + "isolated"
    second = real_storage.get_backend(other_bot, provision_schema=True)._message_rows
    try:
        for store, content in [(first, "first bot"), (second, "second bot"), (first, "first retry")]:
            with capture_sql(rig.engine) as sql:
                store.upsert(message_id="stable", role="assistant", content=content, timestamp=10)
            one_upsert(sql)
        assert row(first)["content"] == "first retry"
        assert row(second)["content"] == "second bot"
        with rig.engine.connect() as conn:
            actual = conn.execute(text("SELECT bot_id FROM messages WHERE id='stable'")).scalars().all()
        assert set(actual) == {rig.bot, other_bot}
    finally:
        with rig.engine.begin() as conn:
            for parent in PARENT_TABLES:
                conn.execute(text(f"DROP TABLE IF EXISTS {partition_name(parent, other_bot)}"))


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
def test_precommit_error_or_cancellation_rolls_back_native_dml(real_storage, rig, failure):
    from sqlalchemy import event
    from sqlalchemy.orm import Session

    store = real_storage.get_backend(rig.bot)._message_rows
    store.upsert(message_id="stable", role="assistant", content="original", timestamp=10)
    first = row(store)

    def fail(_session):
        raise failure("precommit failure")

    event.listen(Session, "before_commit", fail)
    try:
        with capture_sql(rig.engine) as sql, pytest.raises(failure, match="precommit failure"):
            store.upsert(message_id="stable", role="assistant", content="must rollback", timestamp=20,
                         extract_memory=False)
        one_upsert(sql)
    finally:
        event.remove(Session, "before_commit", fail)
    assert row(store) == first
    store.upsert(message_id="stable", role="assistant", content="recovered", timestamp=30)
    assert row(store)["content"] == "recovered"


@pytest.fixture
def http_runtime(real_storage, rig, monkeypatch):
    from llm_bawt.mcp_server import server as tools, session_tools
    from llm_bawt.mcp_server.bot_access_identity import BotAccessMiddleware
    from llm_bawt.mcp_server.catalog import CatalogFastMCP
    from llm_bawt.mcp_server.client import MemoryClient
    from llm_bawt.mcp_server.task_association import TaskTurnCapabilityMiddleware
    from llm_bawt.runtime_settings import BotProfileStore
    from llm_bawt.utils.history import HistoryManager
    from tests.test_mcp_approval_interceptor import FakeStore

    signer = Fernet(Fernet.generate_key())
    monkeypatch.setattr(task_turn_context, "_get_fernet", lambda: signer)
    monkeypatch.setattr(service_codec, "_get_fernet", lambda: signer)
    monkeypatch.setattr(task_turn_context, "mint_for_agent_turn", real_mint)
    profiles = BotProfileStore(rig.config)
    profiles.upsert({"slug": rig.bot, "include_in_global_search": False})
    other_bot = rig.bot + "x"
    profiles.upsert({"slug": other_bot, "include_in_global_search": False})
    assert PolicyRefresher(rig.engine).refresh()
    server = CatalogFastMCP("full-storage-write-pg", bot_policy_resolver=BotPolicyResolver(rig.engine),
        approval_store_provider=lambda: FakeStore(None, []))
    for function, name in ((session_tools.get_session, "sessions_get"),
                           (session_tools.get_active_session, "sessions_get_active"),
                           (session_tools.rotate_session, "sessions_rotate"),
                           (tools.add_message, "messages_add")):
        server.add_tool(function, name=name)
    server.mark_tools_registered()
    stages = []
    authorize = server.bot_access.authorize

    async def track(name, arguments, **kwargs):
        if name == "messages_add":
            stages.append("execution" if kwargs.get("bound") is not None else "preflight")
        return await authorize(name, arguments, **kwargs)

    monkeypatch.setattr(server.bot_access, "authorize", track)
    app = BotAccessMiddleware(TaskTurnCapabilityMiddleware(server.streamable_http_app(stateless_http=True, json_response=True)))
    with TestClient(app, base_url="http://localhost:8000") as http:
        def urlopen(request, **kwargs):
            response = http.post("/mcp", content=request.data, headers=dict(request.header_items()))
            response.raise_for_status()
            return io.BytesIO(response.content)

        monkeypatch.setattr("urllib.request.urlopen", urlopen)
        config = SimpleNamespace(HISTORY_FILE=None, SYSTEM_MESSAGE="fixture")

        def client_for(bot=None, user=None):
            return MemoryClient(config, bot_id=bot or rig.bot, user_id=user or rig.user,
                                server_url="http://localhost:8000")

        client = client_for()
        history = HistoryManager(SimpleNamespace(), config, db_backend=client.get_short_term_manager(),
                                 bot_id=rig.bot, user_id=rig.user)
        yield SimpleNamespace(client=client, client_for=client_for, history=history, stages=stages,
                              other_bot=other_bot)
    with rig.engine.begin() as conn:
        conn.execute(text("DELETE FROM bot_profiles WHERE slug=:bot"), {"bot": other_bot})


def proof_for(rig):
    return real_mint(backend="claude-code", session_id=rig.session_id, turn_id="turn-" + uuid.uuid4().hex,
                     trigger_message_id=rig.message_id, bot_id=rig.bot, user_id=rig.user)


@pytest.mark.parametrize("kind", ["agent", "agent-stream", "native-nonstream", "native-stream",
                                  "native-nonstream-task", "native-stream-task"])
def test_coordinator_initial_final_full_store_sql(http_runtime, real_storage, rig, monkeypatch, kind):
    from llm_bawt.service import chat_streaming
    from llm_bawt.service.session_lifecycle.bridge_reset import BridgeSessionResetMixin
    from llm_bawt.service.session_lifecycle.receipt import RESET_CONFIRMATION
    from llm_bawt.service.turn_stream_finalize import TurnStreamFinalizer
    from llm_bawt.service.turn_stream_worker import TurnStreamWorker

    runtime = http_runtime
    agent = kind.startswith("agent")
    command = "/help" if agent else "/new task" if kind.endswith("-task") else "/new"
    service = _service(Mock(return_value=("answer", "", [])), model_type="claude-code" if agent else "openai")
    llm = service._get_llm_bawt.return_value
    llm.history_manager = runtime.history
    llm.bot.slug, llm.user_id = rig.bot, rig.user
    if agent:
        service._bind_agent_thread = BridgeSessionResetMixin._bind_agent_thread.__get__(service)
        service._resolve_thread_sdk_key = BridgeSessionResetMixin._resolve_thread_sdk_key.__get__(service)
        service._resolve_active_thread_binding.return_value = None
    measured = []
    written = []

    def persist(role, content, message_id=None):
        # Capture the WHOLE HistoryManager call, including HTTP authorization,
        # real MemoryStorage, provisioning guards, actual MessageRowStore, title.
        with capture_sql(rig.engine) as sql:
            runtime.history.add_message(role, content, message_id=message_id,
                                        session_id=llm._session_id_override)
        measured.append(sql)
        written.append((role, content, llm._session_id_override))

    def prepare(prompt, **kwargs):
        persist("user", prompt, kwargs.get("message_id"))
        return [Message(role="user", content=prompt)]

    llm.prepare_messages_for_query.side_effect = prepare
    service._finalize_turn.side_effect = lambda **kw: persist("assistant", kw["response_text"],
                                                              kw.get("assistant_message_id"))
    service._turn_log_store.get_turn = Mock(return_value=None)
    service._turn_log_store.reap_other_open_turns = Mock(return_value=[])
    llm.bot.uses_tools, llm.adapter = False, None
    llm._get_generation_kwargs = Mock(return_value={})
    llm.client.stream_raw = Mock(return_value=iter(("answer",)))
    monkeypatch.setattr(chat_streaming, "prepare_stream_request_attachments",
                        AsyncMock(return_value=(command, [], [], None)))
    monkeypatch.setattr(chat_streaming, "ToolEventCoordinator", Mock())
    monkeypatch.setattr(TurnStreamWorker, "_publish_unified", AsyncMock())
    monkeypatch.setattr(TurnStreamFinalizer, "_finalize",
        lambda finalizer, **kw: persist("assistant", finalizer.ctx.full_response_holder[0]))
    monkeypatch.setattr("llm_bawt.entity_presentation.EntityPresentationResolver",
        lambda *_a: SimpleNamespace(resolve_many_safe=lambda refs: {
            (r.entity_type, r.entity_id): r.to_dict() for r in refs}))
    request = _claimed(_request()).model_copy(update={
        "bot_id": rig.bot, "user": rig.user,
        "stream": kind in {"agent-stream", "native-stream", "native-stream-task"},
        "session_id": rig.session_id if agent else None,
        "user_message_id": str(uuid.uuid4()),
        "inter_bot_turn_id": "turn-delivery-" + uuid.uuid4().hex,
        "messages": [ChatMessage(role="user", content=command)],
    })

    async def run():
        if request.stream:
            return [chunk async for chunk in service.chat_completion_stream(request)]
        return await service.chat_completion(request)

    asyncio.run(run())
    assert [(role, content) for role, content, _ in written] == [
        ("user", command), ("assistant", "answer" if agent or kind.endswith("-task") else RESET_CONFIRMATION)]
    expected_session = rig.session_id if agent else SessionResetLineage.delivery_successor_id("delivery-test")
    assert all(sid == expected_session for _, _, sid in written)
    assert len(measured) == 2
    for statements in measured:
        one_upsert(statements)
        assert "ON CONFLICT (bot_id, id)" in statements[0]
    assert runtime.stages == ["preflight", "execution"] * 2
    # No capability survives the coordinator/executor scope. Ordinary fallback
    # ownership checks remain (two guards); no auto-title read for assistant.
    with capture_sql(rig.engine) as sql:
        runtime.client.add_message("assistant", "after coordinator", session_id=expected_session)
    assert sum("SELECT bot_id, user_id FROM sessions" in s for s in sql) == 2
    assert len(sql) == 9 and sum(s.startswith("SET ") for s in sql) == 6, sql
    assert sql.pool_ping_sql == ["SELECT 1"] * 3
    assert len(sql.all_statements) == 12
    with rig.engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM messages WHERE bot_id=:bot AND session_id=:sid"),
                            {"bot": rig.bot, "sid": expected_session}).scalar_one() == (4 if agent else 3)


@pytest.mark.parametrize("negative", ["cross-bot", "wrong-session", "wrong-user"])
def test_proof_cannot_authorize_other_owner_full_store(http_runtime, real_storage, rig, negative):
    runtime = http_runtime
    client, sid = runtime.client, rig.session_id
    if negative == "cross-bot":
        client = runtime.client_for(bot=runtime.other_bot)
    elif negative == "wrong-user":
        client = runtime.client_for(user="fixture_other_user")
    else:
        sid = str(uuid.uuid4())
        with rig.engine.begin() as conn:
            conn.execute(text("INSERT INTO sessions (id, bot_id, user_id, status) "
                              "VALUES (:id, :bot, 'fixture_other_user', 'archived')"),
                         {"id": sid, "bot": rig.bot})
    with service_codec.bind_service_turn(proof_for(rig)), capture_sql(rig.engine) as sql:
        with pytest.raises(RuntimeError, match="access_denied"):
            client.add_message("assistant", "forbidden", session_id=sid)
    assert not any(s.lstrip().startswith(("INSERT", "UPDATE")) for s in sql), sql
    assert not any("bot_profiles" in s or "bot_tags" in s for s in sql), sql


def test_real_http_duplicate_and_retry_preserve_row(http_runtime, real_storage, rig):
    runtime = http_runtime
    message_id = str(uuid.uuid4())
    with service_codec.bind_service_turn(proof_for(rig)):
        for content, ts in [("/help", 10), ("/help", 10), ("/help edited", 20)]:
            with capture_sql(rig.engine) as sql:
                runtime.client.add_message("user", content, timestamp=ts, message_id=message_id,
                                           session_id=rig.session_id)
            one_upsert(sql)
    actual = row(real_storage.get_backend(rig.bot)._message_rows, message_id)
    assert (actual["content"], actual["timestamp"], actual["session_id"]) == ("/help edited", 10, rig.session_id)


def test_real_http_constraint_failure_does_not_mutate_existing_row(http_runtime, real_storage, rig):
    runtime = http_runtime
    message_id = str(uuid.uuid4())
    store = real_storage.get_backend(rig.bot)._message_rows
    with service_codec.bind_service_turn(proof_for(rig)):
        runtime.client.add_message("assistant", "original", message_id=message_id, session_id=rig.session_id)
        first = row(store, message_id)
        with capture_sql(rig.engine) as sql, pytest.raises(RuntimeError):
            runtime.client.add_message("x" * 21, "must rollback", message_id=message_id,
                                       session_id=rig.session_id)
        one_upsert(sql)
        assert row(store, message_id) == first
        with capture_sql(rig.engine) as sql:
            runtime.client.add_message("assistant", "recovered", message_id=message_id,
                                       session_id=rig.session_id)
        one_upsert(sql)
    assert row(store, message_id)["content"] == "recovered"


@pytest.mark.parametrize("end", ["error", "cancel"])
def test_agent_error_cancel_scope_cleanup_with_real_initial_write(http_runtime, real_storage, rig, end):
    from llm_bawt.service.session_lifecycle.bridge_reset import BridgeSessionResetMixin

    runtime = http_runtime
    service = _service(Mock())
    llm = service._get_llm_bawt.return_value
    llm.history_manager = runtime.history
    llm.bot.slug, llm.user_id = rig.bot, rig.user
    service._bind_agent_thread = BridgeSessionResetMixin._bind_agent_thread.__get__(service)
    service._resolve_thread_sdk_key = BridgeSessionResetMixin._resolve_thread_sdk_key.__get__(service)
    measured = []

    def prepare(prompt, **kwargs):
        with capture_sql(rig.engine) as sql:
            runtime.history.add_message("user", prompt, session_id=llm._session_id_override)
        measured.append(sql)
        return [Message(role="user", content=prompt)]

    def execute(*_args, **_kwargs):
        if end == "error":
            raise RuntimeError("inference failed")
        _kwargs["turn_execution"].cancel_event.set()
        return "cancelled answer", "", []

    llm.prepare_messages_for_query.side_effect = prepare
    llm.execute_llm_query.side_effect = execute
    request = _claimed(_request()).model_copy(update={
        "bot_id": rig.bot, "user": rig.user, "session_id": rig.session_id,
        "user_message_id": str(uuid.uuid4()), "inter_bot_turn_id": "turn-delivery-" + uuid.uuid4().hex,
        "messages": [ChatMessage(role="user", content="/help")],
    })
    if end == "error":
        with pytest.raises(RuntimeError, match="inference failed"):
            asyncio.run(service.chat_completion(request))
    else:
        response = asyncio.run(service.chat_completion(request))
        assert response.choices[0].finish_reason == "cancelled"
    assert len(measured) == 1
    one_upsert(measured[0])
    service._finalize_turn.assert_not_called()
    with capture_sql(rig.engine) as sql:
        runtime.client.add_message("assistant", "after failure", session_id=rig.session_id)
    assert len(sql) == 9 and sum(s.startswith("SET ") for s in sql) == 6, sql
    assert sql.pool_ping_sql == ["SELECT 1"] * 3
    assert len(sql.all_statements) == 12
    assert sum("SELECT bot_id, user_id FROM sessions" in s for s in sql) == 2, sql


def test_real_http_title_and_omitted_session_reads_are_not_hidden(http_runtime, real_storage, rig):
    runtime = http_runtime
    with service_codec.bind_service_turn(proof_for(rig)), capture_sql(rig.engine) as title_sql:
        runtime.history.add_message("user", "A plain untitled message", session_id=rig.session_id)
    # TASK-2061: a user message no longer writes a stored auto title.
    assert len(title_sql) == 1, title_sql
    assert title_sql.pool_ping_sql == ["SELECT 1"]
    assert len(title_sql.all_statements) == 2
    assert sum(s.lstrip().startswith("SELECT") for s in title_sql) == 0, title_sql
    assert sum(s.lstrip().startswith("INSERT") for s in title_sql) == 1, title_sql
    assert not any("sessions" in s for s in title_sql), title_sql
    assert not any("SELECT bot_id, user_id FROM sessions" in s for s in title_sql), title_sql
    with service_codec.bind_service_turn(proof_for(rig)), capture_sql(rig.engine) as omitted_sql:
        runtime.client.add_message("assistant", "omitted session")
    assert len(omitted_sql) == 2, omitted_sql
    assert omitted_sql.pool_ping_sql == ["SELECT 1"] * 2
    assert len(omitted_sql.all_statements) == 4
    assert sum(s.lstrip().startswith("SELECT") for s in omitted_sql) == 1, omitted_sql
    assert sum(s.lstrip().startswith("INSERT") for s in omitted_sql) == 1, omitted_sql
    assert any(s.lstrip().startswith("SELECT") and "sessions" in s for s in omitted_sql), omitted_sql
    assert not any("bot_profiles" in s or "bot_tags" in s for s in title_sql + omitted_sql)
