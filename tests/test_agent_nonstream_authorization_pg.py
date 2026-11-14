"""Actual agent nonstream coordinator scopes verified history at BOTH phases."""
from __future__ import annotations

import asyncio
import io
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

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
from llm_bawt.task_turn_context import mint_for_agent_turn as real_mint
from tests.test_bot_policy_refresh_pg import world
from tests.test_nonstream_turn_execution import _claimed, _isolate, _request, _service

__all__ = ["_isolate", "world"]
pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.mark.parametrize("row_kind", ["owned", "hidden", "other_user", "missing"])
def test_agent_nonstream_initial_and_final_history_http_owner_provenance(world, monkeypatch, row_kind):
    from llm_bawt.mcp_server import server as tools, session_tools
    from llm_bawt.mcp_server.bot_access_identity import BotAccessMiddleware
    from llm_bawt.mcp_server.catalog import CatalogFastMCP
    from llm_bawt.mcp_server.client import MemoryClient
    from llm_bawt.mcp_server.task_association import TaskTurnCapabilityMiddleware
    from llm_bawt.service.session_lifecycle.bridge_reset import BridgeSessionResetMixin
    from llm_bawt.utils.history import HistoryManager
    from tests.test_mcp_approval_interceptor import FakeStore

    signer = Fernet(Fernet.generate_key())
    monkeypatch.setattr(task_turn_context, "_get_fernet", lambda: signer)
    monkeypatch.setattr(service_codec, "_get_fernet", lambda: signer)
    monkeypatch.setattr(task_turn_context, "mint_for_agent_turn", real_mint)
    world.store.upsert({"slug": "fixture_owner", "include_in_global_search": False})
    with world.engine.begin() as conn:
        conn.execute(text("""CREATE TABLE sessions (
            id TEXT PRIMARY KEY, bot_id TEXT, user_id TEXT, status TEXT,
            started_at TIMESTAMP, ended_at TIMESTAMP, archived_at TIMESTAMP,
            session_metadata JSONB)"""))
        conn.execute(text("CREATE TABLE fixture_messages (id TEXT, session_id TEXT, content TEXT)"))
    successor = SessionResetLineage.delivery_successor_id("delivery-test")
    if row_kind != "missing":
        # This reset-policy successor is already committed at durable admission,
        # before nonstream execution receives its internally supplied session_id.
        SessionResetLineage.rotate(world.engine, bot_id="fixture_owner",
            user_id="fixture_other_user" if row_kind == "other_user" else "fixture_user",
            new_id=successor, idempotency_key="delivery-test", working_memory_mode="pinned")
    assert PolicyRefresher(world.engine).refresh()
    server = CatalogFastMCP("agent-nonstream-pg", bot_policy_resolver=BotPolicyResolver(world.engine),
        approval_store_provider=lambda: FakeStore(None, []))
    stored, sql, stages = [], [], []

    async def get_session(session_id, **kwargs):
        with world.engine.connect() as conn:
            row = conn.execute(text("SELECT * FROM sessions WHERE id=:id"), {"id": session_id}).mappings().first()
            # Missing SDK-binding provenance must not be invented from the id,
            # even if the later authoritative fallback can find a valid owner.
            return dict(row) if row is not None and row_kind != "hidden" else None

    async def write(**kwargs):
        from llm_bawt.bot_access.context import current_bot_access

        assert current_bot_access().caller.kind == "service"
        stored.append(kwargs)
        with world.engine.begin() as conn:
            conn.execute(text("INSERT INTO fixture_messages VALUES (:id, :sid, :content)"),
                {"id": str(uuid.uuid4()), "sid": kwargs["session_id"], "content": kwargs["content"]})
        return SimpleNamespace(to_dict=lambda: {"id": str(uuid.uuid4())})

    storage = SimpleNamespace(get_session=get_session, add_message=write)
    monkeypatch.setattr(tools, "_get_storage", lambda: storage)
    monkeypatch.setattr(session_tools, "_get_storage", lambda: storage)
    server.add_tool(session_tools.get_session, name="sessions_get")
    server.add_tool(tools.add_message, name="messages_add")
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
            service = _service(Mock(return_value=("reset answer", "", [])))
            llm = service._get_llm_bawt.return_value
            llm.history_manager = history
            llm.bot.slug, llm.user_id = "fixture_owner", "fixture_user"
            service._bind_agent_thread = BridgeSessionResetMixin._bind_agent_thread.__get__(service)
            service._resolve_thread_sdk_key = BridgeSessionResetMixin._resolve_thread_sdk_key.__get__(service)
            service._resolve_active_thread_binding.return_value = None
            measured = []

            def persist(role, content):
                start = len(sql)
                history.add_message(role, content, session_id=llm._session_id_override)
                measured.append(sql[start:])

            def prepare(prompt, **kwargs):
                persist("user", prompt)
                return [Message(role="user", content=prompt)]

            llm.prepare_messages_for_query.side_effect = prepare
            service._finalize_turn.side_effect = lambda **kw: persist("assistant", kw["response_text"])
            original_bind = service_codec.bind_service_turn
            contexts, cleanups = [], []
            arguments = {"bot_id": "fixture_owner", "user_id": "fixture_user", "session_id": successor}

            @contextmanager
            def track(capability):
                contexts.append(capability)
                try:
                    with original_bind(capability):
                        yield
                finally:
                    token = client._service_principal.mint("messages_add", arguments)
                    cleanups.append(service_codec.open_service_session_owner(token, "messages_add", arguments))

            monkeypatch.setattr(service_codec, "bind_service_turn", track)
            request = _claimed(_request()).model_copy(update={
                "bot_id": "fixture_owner", "user": "fixture_user", "session_id": successor,
                "inter_bot_turn_id": "turn-delivery-" + uuid.uuid4().hex,
                "inter_bot_session_policy": "reset_without_history",
            })
            if row_kind in {"other_user", "missing"}:
                with pytest.raises(Exception, match="access_denied"):
                    asyncio.run(service.chat_completion(request))
                assert stored == []
                assert all(context is None for context in contexts)
                assert cleanups and all(owner is None for owner in cleanups)
                return

            asyncio.run(service.chat_completion(request))
            assert [entry["content"] for entry in stored] == ["do the task", "reset answer"]
            assert all(entry["session_id"] == successor for entry in stored)
            assert stages == ["preflight", "execution"] * 2
            assert len(measured) == 2
            if row_kind == "owned":
                assert all(len(statements) == 1 and statements[0].startswith("INSERT") for statements in measured)
                assert len([cap for cap in contexts if cap]) == 2
            else:
                assert all(sum("SELECT bot_id, user_id FROM sessions" in s for s in statements) == 2
                           for statements in measured)
                assert all(context is None for context in contexts)
            assert cleanups == [None, None, None]  # initial, final, outer executor
            assert all("x-llm-bawt-task-turn-context" not in {k.lower() for k in h} for h in headers)
            # No proof is left for later writes on the cached client.
            sql.clear()
            client.add_message("user", "after coordinator", session_id=successor)
            assert sum("SELECT bot_id, user_id FROM sessions" in s for s in sql) == 2
            if row_kind == "owned":
                proof = next(cap for cap in contexts if cap)
                other = MemoryClient(config, bot_id="fixture_owner", user_id="fixture_other_user",
                                     server_url="http://localhost:8000")
                with original_bind(proof), pytest.raises(RuntimeError, match="access_denied"):
                    other.add_message("user", "spoof", session_id=successor)
            assert len(stored) == 3
    finally:
        event.remove(world.engine, "before_cursor_execute", query)
