"""Committed native /new -> actual coordinator -> MemoryClient HTTP -> both guards."""
from __future__ import annotations

import asyncio
import io
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import event, text
from starlette.testclient import TestClient

from llm_bawt import task_turn_context
from llm_bawt.bot_access import BotPolicyResolver
from llm_bawt.bot_access import service as service_codec
from llm_bawt.bot_access.refresh import PolicyRefresher
from llm_bawt.memory.session_reset_lineage import SessionResetLineage
from llm_bawt.models.message import Message
from llm_bawt.service.schemas import ChatMessage
from llm_bawt.service.session_lifecycle.receipt import RESET_CONFIRMATION
from tests.test_bot_policy_refresh_pg import world  # noqa: F401 - fixture
from tests.test_nonstream_turn_execution import _claimed, _isolate, _request, _service

__all__ = ["_isolate", "world"]
pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.mark.parametrize("stream", [True, False], ids=["stream", "nonstream"])
@pytest.mark.parametrize("command", ["/new", "/new task"])
def test_native_committed_reset_history_has_zero_authorization_selects(world, monkeypatch, stream, command):
    from llm_bawt.mcp_server import server as tools, session_tools
    from llm_bawt.mcp_server.bot_access_identity import BotAccessMiddleware
    from llm_bawt.mcp_server.catalog import CatalogFastMCP
    from llm_bawt.mcp_server.client import MemoryClient
    from llm_bawt.mcp_server.task_association import TaskTurnCapabilityMiddleware
    from llm_bawt.service import chat_streaming
    from llm_bawt.service.turn_stream_finalize import TurnStreamFinalizer
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
    assert PolicyRefresher(world.engine).refresh()
    server = CatalogFastMCP("native-reset-pg", bot_policy_resolver=BotPolicyResolver(world.engine),
        approval_store_provider=lambda: FakeStore(None, []))
    successor = SessionResetLineage.delivery_successor_id("delivery-test")
    stored, sql, stages = [], [], []

    async def rotate(**kwargs):
        assert (kwargs["bot_id"], kwargs["user_id"]) == ("fixture_owner", "fixture_user")
        assert kwargs["idempotency_key"] == "delivery-test"
        # Real committed lineage, with unrelated bot-wide pointer work isolated.
        SessionResetLineage.rotate(world.engine, bot_id=kwargs["bot_id"], user_id=kwargs["user_id"],
            new_id=successor, idempotency_key=kwargs["idempotency_key"],
            working_memory_mode=kwargs["working_memory_mode"])
        return successor

    async def get_session(session_id, **kwargs):
        with world.engine.connect() as conn:
            row = conn.execute(text("SELECT * FROM sessions WHERE id=:id"), {"id": session_id}).mappings().first()
            return dict(row) if row is not None else None

    async def get_active_session(**kwargs):
        return None  # first native reset, no predecessor

    async def write(**kwargs):
        from llm_bawt.bot_access.context import current_bot_access

        assert current_bot_access().caller.kind == "service"
        stored.append(kwargs)
        with world.engine.begin() as conn:
            conn.execute(text("INSERT INTO fixture_messages VALUES (:id, :sid, :content)"),
                {"id": str(uuid.uuid4()), "sid": kwargs["session_id"], "content": kwargs["content"]})
        return SimpleNamespace(to_dict=lambda: {"id": str(uuid.uuid4())})

    storage = SimpleNamespace(rotate_session=rotate, get_session=get_session,
                             get_active_session=get_active_session, add_message=write)
    monkeypatch.setattr(tools, "_get_storage", lambda: storage)
    monkeypatch.setattr(session_tools, "_get_storage", lambda: storage)
    for function, name in ((session_tools.get_session, "sessions_get"),
                           (session_tools.get_active_session, "sessions_get_active"),
                           (session_tools.rotate_session, "sessions_rotate"),
                           (tools.add_message, "messages_add")):
        server.add_tool(function, name=name)
    server.mark_tools_registered()
    original_authorize = server.bot_access.authorize

    async def authorize(name, arguments, **kwargs):
        if name == "messages_add":
            stages.append("execution" if kwargs.get("bound") is not None else "preflight")
        return await original_authorize(name, arguments, **kwargs)

    monkeypatch.setattr(server.bot_access, "authorize", authorize)

    def query(_conn, _cursor, statement, *_args):
        sql.append(statement)

    event.listen(world.engine, "before_cursor_execute", query)
    try:
        app = BotAccessMiddleware(TaskTurnCapabilityMiddleware(server.streamable_http_app(stateless_http=True, json_response=True)))
        with TestClient(app, base_url="http://localhost:8000") as http:
            headers = []

            def urlopen(request, **kwargs):
                headers.append(dict(request.header_items()))
                response = http.post("/mcp", content=request.data, headers=headers[-1])
                response.raise_for_status()
                return io.BytesIO(response.content)

            monkeypatch.setattr("urllib.request.urlopen", urlopen)
            config = SimpleNamespace(HISTORY_FILE=None, SYSTEM_MESSAGE="fixture")
            client = MemoryClient(config, bot_id="fixture_owner", user_id="fixture_user",
                                  server_url="http://localhost:8000")
            backend = client.get_short_term_manager()
            history = HistoryManager(SimpleNamespace(), config, db_backend=backend,
                                     bot_id="fixture_owner", user_id="fixture_user")
            service = _service(Mock(return_value=("task answer", "", [])), model_type="openai")
            llm = service._get_llm_bawt.return_value
            llm.history_manager = history
            llm.bot.slug, llm.user_id = "fixture_owner", "fixture_user"
            write_sql = []

            def persist(role, content):
                start = len(sql)
                history.add_message(role, content, session_id=llm._session_id_override)
                write_sql.append(sql[start:])

            def prepare(prompt, **kwargs):
                persist("user", prompt)
                return [Message(role="user", content=prompt)]

            llm.prepare_messages_for_query.side_effect = prepare
            service._finalize_turn.side_effect = lambda **kw: persist("assistant", kw["response_text"])
            service._turn_log_store.get_turn = Mock(return_value=None)
            service._turn_log_store.reap_other_open_turns = Mock(return_value=[])
            llm.bot.uses_tools, llm.adapter = False, None
            llm._get_generation_kwargs = Mock(return_value={})
            llm.client.stream_raw = Mock(return_value=iter(("task answer",)))
            monkeypatch.setattr(chat_streaming, "prepare_stream_request_attachments",
                                AsyncMock(return_value=(command, [], [], None)))
            monkeypatch.setattr(chat_streaming, "ToolEventCoordinator", Mock())
            monkeypatch.setattr(TurnStreamWorker, "_publish_unified", AsyncMock())
            monkeypatch.setattr(TurnStreamFinalizer, "_finalize",
                lambda finalizer, **kw: persist("assistant", finalizer.ctx.full_response_holder[0]))
            monkeypatch.setattr("llm_bawt.entity_presentation.EntityPresentationResolver",
                lambda *_a: SimpleNamespace(resolve_many_safe=lambda refs: {
                    (r.entity_type, r.entity_id): r.to_dict() for r in refs}))
            cleanups, native_proofs = [], []
            bind = service_codec.bind_service_turn
            arguments = {"bot_id": "fixture_owner", "user_id": "fixture_user", "session_id": successor}

            @contextmanager
            def tracked_bind(capability):
                native_proofs.append(capability)
                with bind(capability):
                    yield
                # This runs on the actual worker/executor thread after scope exit.
                token = client._service_principal.mint("messages_add", arguments)
                cleanups.append(service_codec.open_service_session_owner(token, "messages_add", arguments))

            monkeypatch.setattr(service_codec, "bind_service_turn", tracked_bind)
            request = _claimed(_request()).model_copy(update={
                "bot_id": "fixture_owner", "user": "fixture_user", "stream": stream,
                "inter_bot_turn_id": "turn-delivery-" + uuid.uuid4().hex,
                "messages": [ChatMessage(role="user", content=command)],
            })

            async def run():
                if stream:
                    return [chunk async for chunk in service.chat_completion_stream(request)]
                return await service.chat_completion(request)

            asyncio.run(run())
            assert [entry["content"] for entry in stored] == [
                command, RESET_CONFIRMATION if command == "/new" else "task answer"]
            assert all(entry["session_id"] == successor for entry in stored)
            assert stages == ["preflight", "execution"] * 2
            assert len(write_sql) == 2
            assert all(len(statements) == 1 and statements[0].startswith("INSERT") for statements in write_sql)
            assert cleanups == [None]
            assert all("x-llm-bawt-task-turn-context" not in {k.lower() for k in h} for h in headers)
            sql.clear()
            client.add_message("user", "no proof fallback", session_id=successor)
            assert sum("SELECT bot_id, user_id FROM sessions" in s for s in sql) == 2
            other = MemoryClient(config, bot_id="fixture_owner", user_id="fixture_other_user",
                                 server_url="http://localhost:8000")
            assert len(native_proofs) == 1 and native_proofs[0]
            with bind(native_proofs[0]), pytest.raises(RuntimeError, match="access_denied"):
                other.add_message("user", "spoof", session_id=successor)
            foreign_session = str(uuid.uuid4())
            with world.engine.begin() as conn:
                conn.execute(text("INSERT INTO sessions (id, bot_id, user_id, status) "
                    "VALUES (:id, 'fixture_owner', 'fixture_other_user', 'active')"), {"id": foreign_session})
            with bind(native_proofs[0]), pytest.raises(RuntimeError, match="access_denied"):
                client.add_message("user", "session spoof", session_id=foreign_session)
            assert len(stored) == 3
    finally:
        event.remove(world.engine, "before_cursor_execute", query)
