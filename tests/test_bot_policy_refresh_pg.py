"""Disposable private-PG policy notification, process and lifecycle receipts."""
from __future__ import annotations

import multiprocessing
import select
import threading
import uuid
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine, text

from llm_bawt.bot_access import BotPolicyResolver
from llm_bawt.bot_access.refresh import PolicyRefresher, policy_refresh_lifecycle
from llm_bawt.bot_policy_events import POLICY_CHANNEL, policy_changed
from llm_bawt.bot_tags.store import BotTagStore
from llm_bawt.runtime_settings import BotProfileStore
from llm_bawt.utils import db
from tests.pg_isolation import require_test_db_config

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.fixture
def world(private_app_pg):
    url = db.build_postgres_url(require_test_db_config())
    root = create_engine(url)
    schema = "policy_" + uuid.uuid4().hex
    with root.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA {schema}"))
    engine = create_engine(url, connect_args={"options": f"-csearch_path={schema},public"})
    store = object.__new__(BotProfileStore)
    store.engine = engine
    store.config = SimpleNamespace()
    store._ensure_tables_exist()
    store.upsert({"slug": "fixture_owner", "name": "Fixture owner"})
    store.upsert({"slug": "fixture_peer", "name": "Fixture peer"})
    try:
        yield SimpleNamespace(engine=engine, store=store, url=url, schema=schema)
    finally:
        engine.dispose()
        with root.begin() as conn:
            conn.execute(text(f"DROP SCHEMA {schema} CASCADE"))
        root.dispose()


def listen(engine):
    conn = engine.connect().execution_options(isolation_level="AUTOCOMMIT")
    conn.exec_driver_sql(f"LISTEN {POLICY_CHANNEL}")
    return conn


def notified(listener):
    raw = listener.connection.driver_connection
    if not raw.notifies:
        assert select.select([raw], [], [], 5)[0], "no committed policy change notification"
    raw.poll()
    events, raw.notifies[:] = list(raw.notifies), []
    return events


def test_change_hints_commit_only_and_rollback_does_not_publish(world):
    with listen(world.engine) as listener:
        with world.engine.connect() as writer:
            tx = writer.begin()
            policy_changed(writer)
            tx.rollback()
        raw = listener.connection.driver_connection
        raw.poll()
        assert not raw.notifies
        with world.engine.begin() as writer:
            policy_changed(writer)
        assert {e.payload for e in notified(listener)} == {world.schema}


def test_actual_profile_assignment_catalog_delete_mutations_notify(world):
    with listen(world.engine) as listener:
        world.store.upsert({"slug": "fixture_peer", "include_in_global_search": False})
        notified(listener)
        world.store.upsert({"slug": "fixture_peer", "tags": []})
        notified(listener)
        tags = BotTagStore(world.engine)
        tags.create_role("fixture_role", "Fixture role")
        notified(listener)
        tags.rename_role("fixture_role", "fixture_renamed")
        notified(listener)
        tags.delete_role("fixture_renamed")
        notified(listener)
        world.store.upsert({"slug": "fixture_new", "tags": []})
        notified(listener)
        assert world.store.delete("fixture_new")
        notified(listener)


def test_individual_assignment_and_grant_changes_notify(world):
    from llm_bawt.bot_tags.grants import GrantActor, write_grant

    with listen(world.engine) as listener:
        world.store.upsert({"slug": "fixture_peer", "tags": ["no_global_search"]})
        notified(listener)
        BotTagStore(world.engine).unassign("no_global_search", "fixture_peer")
        notified(listener)
        for granted in (True, False):
            with world.engine.begin() as conn:
                result = write_grant(conn, "approve_ops", "fixture_peer", granted=granted,
                    reason="fixture change", actor=GrantActor.for_person("fixture_operator"),
                    capability=None)
                assert result.changed
            notified(listener)


def test_borrowed_create_session_notification_is_not_precommit(world):
    from sqlmodel import Session

    with listen(world.engine) as listener, Session(world.engine) as session:
        world.store.upsert({"slug": "fixture_borrowed", "tags": []}, session=session)
        raw = listener.connection.driver_connection
        raw.poll()
        assert not raw.notifies
        session.commit()
        notified(listener)


def reader_process(url, schema, pipe):
    """Independent process cache; no shared Python snapshot/engine state."""
    engine = create_engine(url, connect_args={"options": f"-csearch_path={schema},public"})
    owner = PolicyRefresher(engine)
    published = threading.Event()
    expected = [None]
    original = owner.snapshot.install

    def install(policy, *args):
        result = original(policy, *args)
        if result and expected[0] is not None:
            if ("fixture_peer" in policy.private_bots) == expected[0]:
                published.set()
        return result

    owner.snapshot.install = install
    try:
        owner.start()
        if not owner.ready.wait(10):
            pipe.send(("error", owner.snapshot.diagnostics()["state"]))
            return
        pipe.send(("ready", False))
        while True:
            command = pipe.recv()
            if command == "stop":
                return
            expected[0] = command
            published.clear()
            current = owner.snapshot.get()
            if ("fixture_peer" in current.private_bots) == command:
                published.set()
            pipe.send(("waiting", command))
            if not published.wait(10):
                pipe.send(("error", owner.snapshot.diagnostics()["state"]))
                return
            policy = BotPolicyResolver(engine).resolve_sync()
            pipe.send(("installed", "fixture_peer" in policy.private_bots))
    finally:
        owner.stop()
        engine.dispose()
        pipe.close()


def receive(pipe):
    assert pipe.poll(15), "policy child did not respond"
    return pipe.recv()


def test_cross_process_privacy_revocation_and_explicit_raw_sql_recovery(world):
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=reader_process, args=(world.url, world.schema, child))
    process.start()
    child.close()
    try:
        assert receive(parent) == ("ready", False)
        parent.send(True)
        assert receive(parent) == ("waiting", True)
        world.store.upsert({"slug": "fixture_peer", "tags": ["private"]})
        assert receive(parent) == ("installed", True)
        parent.send(False)
        assert receive(parent) == ("waiting", False)
        # Bypass helpers, hence no notification. An explicit admin refresh is
        # the documented recovery, never a periodic/request-side policy read.
        with world.engine.begin() as conn:
            conn.execute(text("DELETE FROM bot_profile_tags WHERE bot_slug='fixture_peer'"))
            conn.execute(text("UPDATE bot_profiles SET include_in_global_search=true "
                              "WHERE slug='fixture_peer'"))
        from llm_bawt.bot_access.refresh import request_policy_refresh
        request_policy_refresh(world.engine)
        assert receive(parent) == ("installed", False)
    finally:
        if process.is_alive():
            parent.send("stop")
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(5)
        parent.close()
    assert process.exitcode == 0


def test_standalone_embedded_client_real_resolver_owns_initialization_and_teardown(world, monkeypatch):
    from sqlalchemy import event
    from unittest.mock import AsyncMock
    from llm_bawt.bot_access.snapshot import policy_snapshot
    from llm_bawt.mcp_server import storage as storage_module
    from llm_bawt.mcp_server.client import MemoryClient

    monkeypatch.setattr(db, "get_shared_engine", lambda *_a, **_k: world.engine)
    monkeypatch.setattr(storage_module, "get_shared_engine", lambda *_a, **_k: world.engine)
    storage = storage_module.MemoryStorage(SimpleNamespace())
    storage.list_memory_sources = AsyncMock(return_value=[])
    storage.search_memories = AsyncMock(return_value=[])
    monkeypatch.setattr(storage_module, "get_storage", lambda: storage)
    # No FastAPI/FastMCP lifespan and NO resolver/snapshot mocking or external
    # cache installation: construction itself owns the real background worker.
    first = MemoryClient(SimpleNamespace(), bot_id="fixture_owner", user_id="fixture_user")
    second = MemoryClient(SimpleNamespace(), bot_id="fixture_owner", user_id="fixture_user")
    snapshot = policy_snapshot(world.engine)
    request_reads = []
    caller_thread = threading.get_ident()

    def query(_conn, _cursor, sql, *_args):
        if threading.get_ident() == caller_thread:
            request_reads.append(sql)

    event.listen(world.engine, "before_cursor_execute", query)
    try:
        assert snapshot.diagnostics()["state"] == "ready"
        assert first.list_memory_sources() == []
        assert first.search_memory_source("fixture_peer", "fixture query") == []
        first.close()
        first.close()  # idempotent; second client still owns the shared worker
        assert snapshot.diagnostics()["state"] == "ready"
        assert second.list_memory_sources() == []
        assert request_reads == []
    finally:
        first.close()
        second.close()
        event.remove(world.engine, "before_cursor_execute", query)
    assert snapshot.diagnostics()["state"] == "snapshot_stopped"
    with MemoryClient(SimpleNamespace(), bot_id="fixture_owner", user_id="fixture_user") as restarted:
        assert restarted.list_memory_sources() == []
    assert snapshot.diagnostics()["state"] == "snapshot_stopped"


def test_lifecycle_reference_count_and_startup_recovery(world):
    with policy_refresh_lifecycle(world.engine) as first:
        assert first.ready.wait(10)
        with policy_refresh_lifecycle(world.engine) as second:
            assert second is first
        assert first.snapshot.diagnostics()["state"] == "ready"
    assert first.snapshot.diagnostics()["state"] == "snapshot_stopped"
    # A failed connection attempt at startup must reconnect outside calls.
    wrapper = Mock()
    original = world.engine.connect
    attempts = [0]

    def connect():
        attempts[0] += 1
        if attempts[0] == 1:
            raise RuntimeError("fixture disconnected")
        return original()

    wrapper.connect.side_effect = connect
    owner = PolicyRefresher(wrapper)
    try:
        owner.start()
        assert owner.ready.wait(10)
        assert attempts[0] >= 3  # failed connect, new LISTEN, new snapshot
        assert BotPolicyResolver(wrapper).resolve_sync().require_target("fixture_peer")
    finally:
        owner.stop()


def test_committed_runtime_rotation_history_http_has_zero_authorization_selects(world, monkeypatch):
    """Real commit -> existing proof -> worker scope -> client HTTP -> TWO guards.

    Disposable PG rows only. Narrow storage to an INSERT to count authorization
    SQL independently of unrelated title/embedding/extraction write side effects.
    """
    import io
    from cryptography.fernet import Fernet
    from sqlalchemy import event
    from starlette.testclient import TestClient
    from llm_bawt import task_turn_context
    from llm_bawt.bot_access import service as service_codec
    from llm_bawt.memory.session_reset_lineage import SessionResetLineage
    from llm_bawt.mcp_server import server as tools
    from llm_bawt.mcp_server.bot_access_identity import BotAccessMiddleware
    from llm_bawt.mcp_server.catalog import CatalogFastMCP
    from llm_bawt.mcp_server.client import MemoryClient
    from llm_bawt.mcp_server.task_association import TaskTurnCapabilityMiddleware
    from llm_bawt.service.turn_stream_worker import TurnStreamWorker
    from llm_bawt.utils.history import HistoryManager
    from tests.test_mcp_approval_interceptor import FakeStore

    signer = Fernet(Fernet.generate_key())
    monkeypatch.setattr(task_turn_context, "_get_fernet", lambda: signer)
    monkeypatch.setattr(service_codec, "_get_fernet", lambda: signer)
    world.store.upsert({"slug": "fixture_owner", "include_in_global_search": False})
    with world.engine.begin() as conn:
        conn.execute(text("""CREATE TABLE sessions (
            id TEXT PRIMARY KEY, bot_id TEXT, user_id TEXT, status TEXT,
            started_at TIMESTAMP, ended_at TIMESTAMP, archived_at TIMESTAMP,
            session_metadata JSONB)"""))
        conn.execute(text("CREATE TABLE fixture_messages (id TEXT, session_id TEXT, content TEXT)"))
    owner = PolicyRefresher(world.engine)
    assert owner.refresh()
    server = CatalogFastMCP("runtime-ownership-pg", bot_policy_resolver=BotPolicyResolver(world.engine),
        approval_store_provider=lambda: FakeStore(None, []))
    successor = str(uuid.uuid4())
    stored = []

    async def rotate(**kwargs):
        assert kwargs["bot_id"] == "fixture_owner" and kwargs["user_id"] == "fixture_user"
        SessionResetLineage.rotate(world.engine, bot_id=kwargs["bot_id"], user_id=kwargs["user_id"],
            new_id=successor, working_memory_mode=kwargs["working_memory_mode"])
        return successor

    async def write(**kwargs):
        from llm_bawt.bot_access.context import current_bot_access

        stored.append((kwargs, current_bot_access().caller.kind))
        with world.engine.begin() as conn:
            conn.execute(text("INSERT INTO fixture_messages VALUES (:id, :sid, :content)"),
                {"id": str(uuid.uuid4()), "sid": kwargs["session_id"], "content": kwargs["content"]})
        return SimpleNamespace(to_dict=lambda: {"id": "fixture-message"})

    async def get_session(session_id, **kwargs):
        with world.engine.connect() as conn:
            row = conn.execute(text("SELECT * FROM sessions WHERE id=:id"), {"id": session_id}).mappings().first()
            return dict(row) if row is not None else None

    from llm_bawt.mcp_server import session_tools
    storage = SimpleNamespace(rotate_session=rotate, add_message=write, get_session=get_session)
    monkeypatch.setattr(tools, "_get_storage", lambda: storage)
    monkeypatch.setattr(session_tools, "_get_storage", lambda: storage)
    server.add_tool(session_tools.get_session, name="sessions_get")
    server.add_tool(session_tools.rotate_session, name="sessions_rotate")
    server.add_tool(tools.add_message, name="messages_add")
    server.mark_tools_registered()
    stages = []
    original = server.bot_access.authorize

    async def authorize(name, arguments, **kwargs):
        if name == "messages_add":
            stages.append("execution" if kwargs.get("bound") is not None else "preflight")
        return await original(name, arguments, **kwargs)

    monkeypatch.setattr(server.bot_access, "authorize", authorize)
    sql = []

    def query(_conn, _cursor, statement, *_args):
        sql.append(statement)

    event.listen(world.engine, "before_cursor_execute", query)
    try:
        app = BotAccessMiddleware(TaskTurnCapabilityMiddleware(server.streamable_http_app(stateless_http=True, json_response=True)))
        with TestClient(app, base_url="http://localhost:8000") as http:
            def urlopen(request, **kwargs):
                response = http.post("/mcp", content=request.data, headers=dict(request.header_items()))
                response.raise_for_status()
                return io.BytesIO(response.content)

            monkeypatch.setattr("urllib.request.urlopen", urlopen)
            config = SimpleNamespace(HISTORY_FILE=None, SYSTEM_MESSAGE="fixture")
            client = MemoryClient(config, bot_id="fixture_owner", user_id="fixture_user",
                                  server_url="http://localhost:8000")
            backend = client.get_short_term_manager()
            assert backend.rotate_session(working_memory_mode="pinned") == successor
            assert any("INSERT INTO sessions" in statement for statement in sql)
            proof = task_turn_context.mint_for_agent_turn(backend="claude-code", session_id=successor,
                turn_id="turn-" + uuid.uuid4().hex, trigger_message_id=str(uuid.uuid4()),
                bot_id="fixture_owner", user_id="fixture_user")
            history = HistoryManager(SimpleNamespace(), config, db_backend=backend,
                                     bot_id="fixture_owner", user_id="fixture_user")
            from llm_bawt.service.session_lifecycle.bridge_reset import BridgeSessionResetMixin
            llm = SimpleNamespace(bot=SimpleNamespace(slug="fixture_owner"), user_id="fixture_user",
                                  history_manager=SimpleNamespace(_db_backend=backend))
            binding = BridgeSessionResetMixin()._resolve_thread_sdk_key(
                llm, {"thread_session_id": successor}, "claude-code")
            assert binding["_service_owner"] == (successor, "fixture_owner", "fixture_user")
            worker = TurnStreamWorker(SimpleNamespace(task_turn_capability=proof, thread_binding=binding,
                bot_id="fixture_owner", user_id="fixture_user"))

            def body():
                history.add_message("user", "/new", session_id=successor)
                history.add_message("assistant", "reset confirmation", session_id=successor)

            monkeypatch.setattr(worker, "_stream_to_queue_with_context", body)
            sql.clear()
            worker._stream_to_queue()
            assert stages == ["preflight", "execution"] * 2
            assert len(sql) == 2 and all(statement.startswith("INSERT") for statement in sql)
            assert [kind for _, kind in stored] == ["service", "service"]
            # Scope exit must restore fallback. This also proves the SQL counter
            # observes both authoritative owner SELECTs, not only mock calls.
            sql.clear()
            client.add_message("user", "no proof fallback", session_id=successor)
            assert sum("SELECT bot_id, user_id FROM sessions" in statement for statement in sql) == 2
            other = MemoryClient(config, bot_id="fixture_owner", user_id="fixture_other_user",
                                 server_url="http://localhost:8000")
            with service_codec.bind_service_turn(proof), pytest.raises(RuntimeError, match="access_denied"):
                other.add_message("user", "spoof", session_id=successor)
            assert len(stored) == 3

            def failure():
                raise RuntimeError("fixture worker failure")

            monkeypatch.setattr(worker, "_stream_to_queue_with_context", failure)
            with pytest.raises(RuntimeError, match="fixture worker failure"):
                worker._stream_to_queue()
            sql.clear()
            client.add_message("user", "after failure", session_id=successor)
            assert sum("SELECT bot_id, user_id FROM sessions" in statement for statement in sql) == 2
    finally:
        event.remove(world.engine, "before_cursor_execute", query)
