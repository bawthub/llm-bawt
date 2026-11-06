"""TASK-1907: selected native HTTP turns through real history, MCP and PG.

Only model output, prompt augmentation, media/events and turn-log instrumentation
are isolated. Admission, adapters, coordinators, prepare/finalize, HistoryManager,
MemoryClient, both MCP authorization stages and partitioned writes are production.
Provisioning and pool bootstrap are warmed OUTSIDE the whole-request receipts.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from sqlalchemy import text
from starlette.testclient import TestClient

from llm_bawt.bot_access import service as service_codec
from llm_bawt.bots import Bot
from llm_bawt.service.background_service import BackgroundService
from llm_bawt.service.core import ServiceLLMBawt
from llm_bawt.service.schemas import ChatCompletionRequest
from tests.session_lifecycle_pg_fixtures import rig
from tests.test_message_native_upsert import capture_sql
from tests.test_message_native_upsert_pg import http_runtime, real_storage
from tests.test_nonstream_turn_execution import _service

__all__ = ["rig", "http_runtime", "real_storage"]
pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


def _receipt(sql):
    return {"application": len(sql), "pings": len(sql.pool_ping_sql),
            "total": len(sql.all_statements),
            "owner_selects": sum("SELECT bot_id, user_id FROM sessions" in s for s in sql),
            "read_bearing_owner_dml": sum(s.lstrip().startswith("INSERT") and "FROM sessions" in s for s in sql),
            "selects": sum(s.lstrip().upper().startswith("SELECT") for s in sql.all_statements),
            "statements": sql.all_statements}


def _record(record_property, scope, sql):
    record_property(scope, json.dumps(_receipt(sql)))


@pytest.fixture
def native_runtime(request, real_storage, rig, monkeypatch):
    from llm_bawt.mcp_server import server as tools
    from llm_bawt.mcp_server.catalog import CatalogFastMCP
    from llm_bawt.service import chat_streaming, dependencies
    from llm_bawt.service.routes import chat, sessions
    from llm_bawt.service.turn_stream_worker import TurnStreamWorker

    # Extend the existing real HTTP fixture before it creates its server.
    registered = CatalogFastMCP.mark_tools_registered

    def register_reads(server):
        server.add_tool(tools.get_messages, name="messages_get")
        return registered(server)

    monkeypatch.setattr(CatalogFastMCP, "mark_tools_registered", register_reads)
    # TASK-2064: this suite counts the exact SQL of each append. The generated-title
    # worker (covered by test_generated_title_pg) runs on the same engine and would
    # add its own statements to the capture.
    monkeypatch.setattr("llm_bawt.sessions.generated_title.schedule_session_title",
                        lambda *args, **kwargs: None)
    runtime = request.getfixturevalue("http_runtime")
    model_calls, appends, reads, admissions = [], [], [], []
    real_owned = sessions._owned_session_or_404
    measure_admission = [True]

    async def owned(*args, **kwargs):
        if not measure_admission[0]:
            return await real_owned(*args, **kwargs)
        with capture_sql(rig.engine) as sql:
            try:
                return await real_owned(*args, **kwargs)
            finally:
                admissions.append(sql)

    monkeypatch.setattr(sessions, "_owned_session_or_404", owned)
    llm = ServiceLLMBawt.__new__(ServiceLLMBawt)
    llm.bot_id, llm.user_id = rig.bot, rig.user
    llm.bot = Bot(slug=rig.bot, name="Snark", description="isolated native chat",
                  system_prompt="Fixture Snark", uses_tools=False)
    llm.history_manager = runtime.history
    llm.config = SimpleNamespace(SYSTEM_MESSAGE="Fixture Snark")
    llm.adapter, llm.tool_format = None, "none"
    llm.model_definition = {"type": "openai"}
    llm._get_generation_kwargs = lambda: {}
    # No memory retrieval/prompt-registry augmentation. Real preparation still
    # loads the explicit session over HTTP and assembles the real history pool.
    llm._build_context_messages = lambda *_a, **_k: runtime.history.get_context_messages()

    def query(messages, **kwargs):
        assert service_codec._runtime_turn.get() is None  # history-only authority
        model_calls.append(list(messages))
        return "native answer"

    def stream(messages, **kwargs):
        assert service_codec._runtime_turn.get() is None
        model_calls.append(list(messages))
        yield "native "
        yield "answer"

    llm.client = SimpleNamespace(model_definition=llm.model_definition,
        query=query, stream_raw=stream, get_token_usage=lambda: None)
    real_add, real_load = runtime.history.add_message, runtime.history.load_history

    def add(*args, **kwargs):
        with capture_sql(rig.engine) as sql:
            result = real_add(*args, **kwargs)
        appends.append((args[0], kwargs.get("session_id"), sql))
        return result

    def load(*args, **kwargs):
        with capture_sql(rig.engine) as sql:
            result = real_load(*args, **kwargs)
        reads.append(sql)
        return result

    monkeypatch.setattr(runtime.history, "add_message", add)
    monkeypatch.setattr(runtime.history, "load_history", load)
    service = _service(None, model_type="openai")
    service.config.DEFAULT_USER, service.config.DEFAULT_BOT = rig.user, rig.bot
    service.config.DEBUG_TURN_LOG = False
    service._get_llm_bawt.return_value = llm
    service._finalize_turn = BackgroundService._finalize_turn.__get__(service)
    service._turn_log_store.get_turn = Mock(return_value=None)
    service._turn_log_store.reap_other_open_turns = Mock(return_value=[])
    monkeypatch.setattr(chat, "get_service", lambda: service)
    monkeypatch.setattr(dependencies, "_service", service)
    monkeypatch.setattr(chat_streaming, "prepare_stream_request_attachments",
                        AsyncMock(return_value=("hello selected", [], [], None)))
    monkeypatch.setattr(chat_streaming, "ToolEventCoordinator", Mock())
    monkeypatch.setattr(TurnStreamWorker, "_publish_unified", AsyncMock())
    monkeypatch.setattr("llm_bawt.entity_presentation.EntityPresentationResolver",
        lambda *_a: SimpleNamespace(resolve_many_safe=lambda refs: {
            (r.entity_type, r.entity_id): r.to_dict() for r in refs}))
    app = FastAPI()
    app.include_router(chat.router)
    with TestClient(app) as http:
        yield SimpleNamespace(http=http, service=service, llm=llm, memory=runtime,
                              appends=appends, reads=reads, admissions=admissions,
                              measure_admission=measure_admission, model_calls=model_calls)


def _body(rig, *, stream=False, **updates):
    return {"model": "fixture-native", "bot_id": rig.bot, "user": rig.user,
            "session_id": rig.session_id, "stream": stream, "augment_memory": False,
            "user_message_id": str(uuid.uuid4()), "assistant_message_id": str(uuid.uuid4()),
            "messages": [{"role": "user", "content": "hello selected"}], **updates}


def _assert_response(response, stream):
    assert response.status_code == 200, response.text
    if stream:
        frames = [json.loads(line[6:]) for line in response.text.splitlines()
                  if line.startswith("data: ") and line != "data: [DONE]"]
        assert not any("error" in frame for frame in frames), frames
        assert "".join(frame["choices"][0]["delta"].get("content", "") for frame in frames) == "native answer"
        assert response.text.endswith("data: [DONE]\n\n")
    else:
        assert response.json()["choices"][0]["message"]["content"] == "native answer"


def _owner_checks(sql, expected):
    assert sum("SELECT bot_id, user_id FROM sessions" in s for s in sql) == expected, sql.all_statements


def _fallback(sql):
    _owner_checks(sql, 2)
    assert len(sql) == 9 and sum(s.startswith("SET ") for s in sql) == 6, sql
    assert sql.pool_ping_sql == ["SELECT 1"] * 3
    assert len(sql.all_statements) == 12


@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
@pytest.mark.parametrize("status", ["active", "archived"])
@pytest.mark.parametrize("title", [None, "settled title"], ids=["cold-title", "settled-title"])
def test_actual_http_selected_native_history_sql(native_runtime, real_storage, rig,
                                                record_property, stream, status, title):
    runtime = native_runtime
    with rig.engine.begin() as conn:
        conn.execute(text("UPDATE sessions SET status=:status, session_metadata=CAST(:meta AS jsonb) WHERE id=:sid"),
                     {"status": status, "meta": json.dumps({"title": title, "title_source": "user"} if title else {}),
                      "sid": rig.session_id})
    body = _body(rig, stream=stream)
    with capture_sql(rig.engine) as whole:
        response = runtime.http.post("/v1/chat/completions", json=body)
    _record(record_property, "whole_http_unfiltered", whole)
    _assert_response(response, stream)
    assert len(runtime.model_calls) == 1
    assert any(m.content == "test" for m in runtime.model_calls[0])  # real scoped read
    assert any(m.content == "hello selected" for m in runtime.model_calls[0])
    assert [role for role, _, _ in runtime.appends] == ["user", "assistant"]
    assert runtime.memory.stages == ["preflight", "execution"] * 2
    assert len(runtime.reads) == 1 and runtime.reads[0]  # not an empty mock read
    _record(record_property, "context_history_reads", runtime.reads[0])
    assert len(runtime.admissions) == 1
    _record(record_property, "http_owned_row_admission", runtime.admissions[0])
    # Every statement belongs to an explicit scope; nothing in the whole
    # warmed HTTP receipt (including real pings) was filtered away.
    assert len(whole.all_statements) == sum(len(sql.all_statements) for sql in
        [*runtime.admissions, *runtime.reads, *(entry[2] for entry in runtime.appends)])
    for role, sid, sql in runtime.appends:
        assert sid == rig.session_id
        _record(record_property, role + "_append", sql)
        _owner_checks(sql, 0)
        assert sum("ON CONFLICT (bot_id, id)" in s for s in sql) == 1
        # INSERT...SELECT is one READ-bearing DML statement, not zero reads.
        owner_dml = [s for s in sql if s.lstrip().startswith("INSERT") and "FROM sessions" in s]
        assert len(owner_dml) == 1
        assert "sessions.id =" in owner_dml[0] and "sessions.bot_id =" in owner_dml[0]
        assert "sessions.user_id =" in owner_dml[0]
        assert "RETURNING" in owner_dml[0] and "FOR UPDATE" not in owner_dml[0]
        assert not any(s.startswith("SET ") for s in sql)
        if role == "assistant":
            assert len(sql.all_statements) == 2  # actual UPSERT + delegated ping
        else:
            # TASK-2061 retired the auto-title writer: a title-less session no
            # longer adds a session read + UPDATE to the user append.
            assert len(sql.all_statements) == 2, sql.all_statements
            assert not any("UPDATE sessions" in s for s in sql), sql
    assert len(whole.all_statements) == 24, whole.all_statements
    assert _receipt(whole)["read_bearing_owner_dml"] == 2
    # Admission and context SELECTs remain in the WHOLE HTTP receipt; do not
    # sell the append optimization as a zero-SELECT whole request.
    assert any(s.lstrip().upper().startswith("SELECT") and s != "SELECT 1"
               for s in whole.all_statements)
    assert service_codec._runtime_turn.get() is None
    with rig.engine.connect() as conn:
        rows = conn.execute(text("SELECT id, role, content, session_id FROM messages "
                                 "WHERE bot_id=:bot ORDER BY timestamp"), {"bot": rig.bot}).mappings().all()
    assert {(r["id"], r["role"], r["content"], r["session_id"]) for r in rows} == {
        (rig.message_id, "user", "test", rig.session_id),
        (body["user_message_id"], "user", "hello selected", rig.session_id),
        (body["assistant_message_id"], "assistant", "native answer", rig.session_id)}
    # Same cached actual MemoryClient after coordinator cleanup retains fallback.
    with capture_sql(rig.engine) as after:
        runtime.memory.client.add_message("assistant", "after cleanup", session_id=rig.session_id)
    _fallback(after)
    _record(record_property, "after_cleanup_explicit_assistant", after)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("failure,code", [("missing", 404), ("foreign-bot", 404),
                                         ("foreign-user", 404), ("deleted", 410)])
def test_actual_http_admission_rejects_before_model_or_write(native_runtime, rig, failure, code, stream):
    body = _body(rig, stream=stream)
    if failure == "missing":
        body["session_id"] = str(uuid.uuid4())
    elif failure == "foreign-bot":
        body["bot_id"] = native_runtime.memory.other_bot
    elif failure == "foreign-user":
        body["user"] = "another-user"
    else:
        with rig.engine.begin() as conn:
            conn.execute(text("UPDATE sessions SET status='deleted' WHERE id=:sid"), {"sid": rig.session_id})
    with capture_sql(rig.engine) as sql:
        response = native_runtime.http.post("/v1/chat/completions", json=body)
    assert response.status_code == code, response.text
    assert response.json()["detail"] == ("Session deleted" if code == 410 else "Session not found")
    assert not native_runtime.model_calls and not native_runtime.appends
    assert not any(s.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE")) for s in sql), sql
    native_runtime.service._persist_turn_log.assert_not_called()


@pytest.mark.parametrize("stream", [False, True])
def test_no_ingress_actual_service_keeps_fallback_and_ignores_json_proof(native_runtime, rig,
                                                                       record_property, stream):
    body = _body(rig, stream=stream,
                 _http_selected_session_owner=[rig.session_id, rig.bot, rig.user])
    request = ChatCompletionRequest.model_validate_json(json.dumps(body))
    assert request._http_selected_session_owner is None

    async def run():
        if stream:
            return [chunk async for chunk in native_runtime.service.chat_completion_stream(request)]
        return await native_runtime.service.chat_completion(request)

    asyncio.run(run())
    assert len(native_runtime.model_calls) == 1
    assert [role for role, _, _ in native_runtime.appends] == ["user", "assistant"]
    for role, _, sql in native_runtime.appends:
        _owner_checks(sql, 2)
        _record(record_property, "no_ingress_" + role, sql)
        if role == "assistant":
            _fallback(sql)


@pytest.mark.parametrize("mismatch", ["bot", "user", "session", "cleared"])
def test_real_admission_tuple_disagreement_falls_back(native_runtime, rig, monkeypatch,
                                                      record_property, mismatch):
    from llm_bawt.service import chat_response_adapter

    real_response = chat_response_adapter.completion_response

    async def change_after_real_admission(service, request):
        assert request._http_selected_session_owner == (rig.session_id, rig.bot, rig.user)
        owner = list(request._http_selected_session_owner)
        if mismatch == "cleared":
            request._http_selected_session_owner = None
        else:
            owner[{"session": 0, "bot": 1, "user": 2}[mismatch]] = "wrong"
            request._http_selected_session_owner = tuple(owner)
        return await real_response(service, request)

    monkeypatch.setattr(chat_response_adapter, "completion_response", change_after_real_admission)
    response = native_runtime.http.post("/v1/chat/completions", json=_body(rig))
    _assert_response(response, False)
    for role, sid, sql in native_runtime.appends:
        assert sid == rig.session_id
        _owner_checks(sql, 2)
        _record(record_property, mismatch + "_" + role, sql)
        if role == "assistant":
            _fallback(sql)


def test_actual_http_json_spoof_cannot_replace_admitted_row(native_runtime, rig):
    response = native_runtime.http.post("/v1/chat/completions", json=_body(
        rig, _http_selected_session_owner=["wrong-session", "wrong-bot", "wrong-user"]))
    _assert_response(response, False)
    for _, sid, sql in native_runtime.appends:
        assert sid == rig.session_id
        _owner_checks(sql, 0)  # actual DB admission, never the spoofed JSON tuple


def test_actual_http_expired_initial_proof_is_not_renewed_for_final(native_runtime, rig,
                                                                  monkeypatch, record_property):
    from llm_bawt import task_turn_context
    from llm_bawt.service.session_lifecycle.execution import NativeSelectedHistory

    # Real Fernet crypto/validation, with only its clock advanced (no sleep and
    # no fabricated token). Service envelopes are minted at the new clock;
    # their embedded initial task capability is genuinely beyond its TTL.
    now = [int(time.time())]
    monkeypatch.setattr("cryptography.fernet.time", SimpleNamespace(time=lambda: now[0]))
    real_query = native_runtime.llm.client.query
    real_take = NativeSelectedHistory.take.__func__
    proofs = []

    def take(cls, *args, **kwargs):
        selected = real_take(cls, *args, **kwargs)
        proofs.append(selected)
        return selected

    def expire(messages, **kwargs):
        result = real_query(messages, **kwargs)
        now[0] += task_turn_context.TASK_TURN_CONTEXT_TTL_SECONDS + 1
        return result

    monkeypatch.setattr(NativeSelectedHistory, "take", classmethod(take))
    monkeypatch.setattr(native_runtime.llm.client, "query", expire)
    with capture_sql(rig.engine) as sql:
        response = native_runtime.http.post("/v1/chat/completions", json=_body(rig))
    _record(record_property, "expired_proof_whole_http", sql)
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "Current-turn context is invalid or expired"
    assert len(proofs) == 1 and proofs[0].capability is not None
    assert [role for role, _, _ in native_runtime.appends] == ["user"]
    assert sum("ON CONFLICT (bot_id, id)" in s for s in sql) == 1
    with rig.engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM messages WHERE bot_id=:bot AND role='assistant'"),
                            {"bot": rig.bot}).scalar_one() == 0
    assert service_codec._runtime_turn.get() is None


@pytest.mark.parametrize("stream", [False, True])
def test_real_finalizer_persists_system_tool_context_to_exact_selected_session(
        native_runtime, rig, monkeypatch, record_property, stream):
    # Supplemental finalizer-role proof: insert a simulated native tool result
    # at the model-result/finalizer boundary. No history/finalize/storage method
    # is replaced; the actual _finalize_turn invokes actual finalize_response.
    real_finalize = native_runtime.service._finalize_turn

    def with_tool_result(**kwargs):
        return real_finalize(**{**kwargs, "tool_context": "isolated tool evidence"})

    monkeypatch.setattr(native_runtime.service, "_finalize_turn", with_tool_result)
    response = native_runtime.http.post("/v1/chat/completions", json=_body(rig, stream=stream))
    _assert_response(response, stream)
    assert [role for role, _, _ in native_runtime.appends] == ["user", "system", "assistant"]
    for role, sid, sql in native_runtime.appends:
        assert sid == rig.session_id
        _owner_checks(sql, 0)
        _record(record_property, "tool_context_" + role, sql)
        if role != "user":
            assert len(sql.all_statements) == 2
    with rig.engine.connect() as conn:
        rows = conn.execute(text("SELECT role, content, session_id FROM messages "
                                 "WHERE bot_id=:bot ORDER BY timestamp"), {"bot": rig.bot}).all()
    assert rows == [("user", "test", rig.session_id), ("user", "hello selected", rig.session_id),
                    ("system", "[Tool Results]\nisolated tool evidence", rig.session_id),
                    ("assistant", "native answer", rig.session_id)]


@pytest.mark.parametrize("stream", [False, True])
def test_barrier_concurrent_owner_runtime_instances_do_not_share_proofs(
        native_runtime, rig, record_property, stream):
    from llm_bawt import task_turn_context
    from llm_bawt.utils.history import HistoryManager

    # Distinct (bot,user) cache keys use distinct actual ServiceLLMBawt and
    # MemoryClient instances. This tests ContextVar/HTTP envelope isolation,
    # not the separate same-cache mutable-override concurrency contract.
    second_user, second_sid = "concurrent-owner", str(uuid.uuid4())
    with rig.engine.begin() as conn:
        conn.execute(text("INSERT INTO sessions (id,bot_id,user_id,status) "
                          "VALUES (:sid,:bot,:user,'archived')"),
                     {"sid": second_sid, "bot": rig.bot, "user": second_user})
    barriers = {role: threading.Barrier(2) for role in ("user", "assistant")}
    seen, instances = [], {}
    for user, sid in ((rig.user, rig.session_id), (second_user, second_sid)):
        memory = native_runtime.memory.client_for(user=user)
        history = HistoryManager(SimpleNamespace(),
            SimpleNamespace(HISTORY_FILE=None, SYSTEM_MESSAGE="Fixture Snark"),
            db_backend=memory.get_short_term_manager(), bot_id=rig.bot, user_id=user)
        llm = copy(native_runtime.llm)
        llm.user_id, llm.history_manager = user, history
        llm._build_context_messages = lambda *_a, history=history, **_k: history.get_context_messages()
        real_add = history.add_message

        def add(role, content, *, user=user, sid=sid, original=real_add, **kwargs):
            token = service_codec._runtime_turn.get()
            proof = task_turn_context.open_task_turn_context(token)
            assert (proof.bot_id, proof.user_id, proof.session_id) == (rig.bot, user, sid)
            assert kwargs["session_id"] == sid
            barriers[role].wait(timeout=10)  # timing-ok: concurrent history/auth window hang guard
            result = original(role, content, **kwargs)
            assert service_codec._runtime_turn.get() == token
            seen.append((role, user, sid, token))
            return result

        history.add_message = add
        instances[user] = llm
    native_runtime.service._get_llm_bawt.side_effect = (
        lambda _model, _bot, user, _local: instances[user])
    bodies = [_body(rig, stream=stream),
              _body(rig, stream=stream, user=second_user, session_id=second_sid)]
    # capture_sql wraps the dialect ping hook; concurrent subcaptures must
    # not compete for that process-wide hook. The whole capture stays intact.
    native_runtime.measure_admission[0] = False
    # Constructed histories, signing key, server policy, partitions and initial
    # connections above are outside the concurrent whole-request receipt.
    with capture_sql(rig.engine) as sql, ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(native_runtime.http.post, "/v1/chat/completions", json=body)
                   for body in bodies]
        responses = [future.result(timeout=30) for future in futures]  # timing-ok: real HTTP workers hang guard
    _record(record_property, "concurrent_distinct_owners_whole_http", sql)
    for response in responses:
        _assert_response(response, stream)
    assert len(seen) == 4
    assert len({token for _, _, _, token in seen}) == 2
    for user in (rig.user, second_user):
        assert {role for role, owner, _, _ in seen if owner == user} == {"user", "assistant"}
        assert len({token for _, owner, _, token in seen if owner == user}) == 1
    assert service_codec._runtime_turn.get() is None
    with rig.engine.connect() as conn:
        actual = conn.execute(text("SELECT id,session_id FROM messages WHERE bot_id=:bot"),
                              {"bot": rig.bot}).all()
    actual = dict(actual)
    for body in bodies:
        assert actual[body["user_message_id"]] == body["session_id"]
        assert actual[body["assistant_message_id"]] == body["session_id"]
