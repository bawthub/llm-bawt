"""Hermetic HTTP-row provenance and narrowly scoped native history authority.

No DB, service, inference, configured signing key, or sleeps. PG integration
separately exercises the real history writers/finalizer against storage.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from cryptography.fernet import Fernet
from fastapi import HTTPException
from starlette.requests import Request

from llm_bawt import task_turn_context as turn_codec
from llm_bawt.bot_access import service as service_codec
from llm_bawt.mcp_server.client_short_term import _MCPShortTermManager
from llm_bawt.models.message import Message
from llm_bawt.service.schemas import ChatCompletionRequest, ChatMessage
from llm_bawt.service.session_lifecycle.execution import NativeSelectedHistory

BOT, USER = "caid", "nick"
SID = "10000000-0000-4000-8000-000000000001"
OTHER_SID = "10000000-0000-4000-8000-000000000002"
MID = "20000000-0000-4000-8000-000000000001"
TURN = "turn-" + "a" * 32
OWNER = (SID, BOT, USER)


def _request(**updates):
    return ChatCompletionRequest(**{
        "messages": [ChatMessage(role="user", content="hello")],
        "bot_id": BOT, "user": USER, "session_id": SID,
        "user_message_id": MID, **updates,
    })


def _proven_request(**updates):
    request = _request(**updates)
    request._http_selected_session_owner = OWNER
    return request


def _llm():
    memory = SimpleNamespace(bot_id=BOT, user_id=USER,
                             _service_principal=service_codec.ServicePrincipal(BOT, USER))
    history = SimpleNamespace(bot_id=BOT, user_id=USER,
                              _db_backend=_MCPShortTermManager(memory))
    return SimpleNamespace(bot_id=BOT, user_id=USER, history_manager=history,
                           client=SimpleNamespace(model_definition={"type": "openai"}),
                           _session_id_override=SID)


def _take(request, llm, **updates):
    return NativeSelectedHistory.take(request, llm, **{
        "bot_id": BOT, "user_id": USER, "turn_id": TURN,
        "trigger_message_id": MID, "is_agent_backend": False,
        "admission_claim": None, **updates,
    })


def _window(selected, request, llm, phase, **updates):
    return selected.history_context(request, llm, **{
        "phase": phase, "bot_id": BOT, "user_id": USER,
        "turn_id": TURN, "trigger_message_id": MID, **updates,
    })


def _arguments(**updates):
    return {"bot_id": BOT, "user_id": USER, "session_id": SID,
            "role": "user", "content": "hello", **updates}


@pytest.fixture
def signer(monkeypatch):
    signer = Fernet(Fernet.generate_key())
    monkeypatch.setattr(turn_codec, "_get_fernet", lambda: signer)
    monkeypatch.setattr(service_codec, "_get_fernet", lambda: signer)
    return signer


def test_private_http_provenance_is_not_json_or_copy_authority():
    payload = _request().model_dump()
    payload["_http_selected_session_owner"] = list(OWNER)
    request = ChatCompletionRequest.model_validate_json(json.dumps(payload))
    assert request._http_selected_session_owner is None
    request._http_selected_session_owner = OWNER
    assert "_http_selected_session_owner" not in request.model_dump()
    assert "_http_selected_session_owner" not in request.model_dump_json()
    assert "_http_selected_session_owner" not in ChatCompletionRequest.model_json_schema()["properties"]
    assert ChatCompletionRequest.model_validate_json(request.model_dump_json())._http_selected_session_owner is None
    for update in ({}, {"session_id": OTHER_SID}, {"bot_id": "other"}, {"user": "other"},
                   {"session_id": OTHER_SID, "_http_selected_session_owner": (OTHER_SID, BOT, USER)}):
        for deep in (False, True):
            copied = request.model_copy(update=update, deep=deep)
            assert copied._http_selected_session_owner is None
    assert request._http_selected_session_owner == OWNER  # copy did not consume source


@pytest.mark.parametrize("status,visibility", [("active", "private"), ("active", "public"),
                                               ("archived", "private"), ("archived", "public")])
def test_route_keeps_existing_admission_and_captures_returned_exact_row(monkeypatch, status, visibility):
    from llm_bawt.service import dependencies
    from llm_bawt.service.routes import chat, sessions

    # Deliberately different id proves capture is from the returned row, not sid.
    row = {"id": OTHER_SID, "bot_id": BOT, "user_id": USER,
           "status": status, "visibility": visibility}
    storage = SimpleNamespace(get_session=AsyncMock(return_value=row))
    service = SimpleNamespace(chat_completion=AsyncMock(return_value="completed"))
    monkeypatch.setattr(chat, "get_service", lambda: service)
    monkeypatch.setattr(chat, "_internal_inter_bot_sender", lambda _request: None)
    monkeypatch.setattr(sessions, "get_storage", lambda: storage)
    monkeypatch.setattr(sessions, "_resolve_user", lambda user: user or USER)
    monkeypatch.setattr(dependencies, "get_effective_bot_id", lambda bot: bot or BOT)
    request = _request(session_id=" " + SID + " ")
    http = Request({"type": "http", "headers": []})
    assert asyncio.run(chat.chat_completions(request, http)) == "completed"
    storage.get_session.assert_awaited_once_with(SID, bot_id=BOT)
    service.chat_completion.assert_awaited_once_with(request)
    assert service.chat_completion.call_args.args[0] is request  # adapter keeps Python provenance
    assert request._http_selected_session_owner == (OTHER_SID, BOT, USER)
    assert _take(request, _llm()) is None  # disagreement never promotes the request id


@pytest.mark.parametrize("row,code,detail", [
    (None, 404, "Session not found"),
    ({"id": SID, "bot_id": "foreign", "user_id": USER}, 404, "Session not found"),
    ({"id": SID, "bot_id": BOT, "user_id": "foreign"}, 404, "Session not found"),
    ({"id": SID, "bot_id": BOT, "user_id": USER, "status": "deleted"}, 410, "Session deleted"),
])
def test_route_failure_retains_exact_404_410_contract(monkeypatch, row, code, detail):
    from llm_bawt.service import dependencies
    from llm_bawt.service.routes import chat, sessions

    storage = SimpleNamespace(get_session=AsyncMock(return_value=row))
    service = SimpleNamespace(chat_completion=AsyncMock())
    monkeypatch.setattr(chat, "get_service", lambda: service)
    monkeypatch.setattr(chat, "_internal_inter_bot_sender", lambda _request: None)
    monkeypatch.setattr(sessions, "get_storage", lambda: storage)
    monkeypatch.setattr(sessions, "_resolve_user", lambda user: user or USER)
    monkeypatch.setattr(dependencies, "get_effective_bot_id", lambda bot: bot or BOT)
    request = _request()
    with pytest.raises(HTTPException) as error:
        asyncio.run(chat.chat_completions(request, Request({"type": "http", "headers": []})))
    assert (error.value.status_code, error.value.detail) == (code, detail)
    assert request._http_selected_session_owner is None
    service.chat_completion.assert_not_called()


def test_take_clears_once_and_mints_one_invocation_bound_token(signer, monkeypatch):
    mint = Mock(wraps=turn_codec.mint_task_turn_context)
    monkeypatch.setattr(turn_codec, "mint_task_turn_context", mint)
    request, llm = _proven_request(), _llm()
    selected = _take(request, llm)
    assert selected is not None and request._http_selected_session_owner is None
    assert _take(request, llm) is None
    principal = llm.history_manager._db_backend._memory_client._service_principal
    assert service_codec._runtime_turn.get() is None
    tokens = []
    for phase in ("initial", "final"):
        with _window(selected, request, llm, phase):
            tokens.append(service_codec._runtime_turn.get())
            envelope = principal.mint("messages_add", _arguments())
            assert service_codec.open_service_native_selected(envelope, "messages_add", _arguments()) is True
            assert service_codec.open_service_session_owner(envelope, "messages_add", _arguments()) == (BOT, USER)
            # Invocation binding remains exact; unrelated tools gain no owner.
            with pytest.raises(Exception):
                service_codec.open_service_session_owner(envelope, "messages_add", _arguments(content="changed"))
            unrelated = principal.mint("sessions_get", {"session_id": SID})
            assert service_codec.open_service_session_owner(unrelated, "sessions_get", {"session_id": SID}) is None
        assert service_codec._runtime_turn.get() is None
    assert tokens == [selected.capability, selected.capability]
    mint.assert_called_once_with(session_id=SID, bot_id=BOT, user_id=USER,
                                 turn_id=TURN, trigger_message_id=MID)
    assert "selected_history" not in vars(llm)
    assert "capability" not in vars(llm.history_manager._db_backend._memory_client)


@pytest.mark.parametrize("field,value", [
    ("_internal_inter_bot_sender_id", "sender"),
    ("_internal_approval_claim", ("approval", "claim")),
    ("inter_bot_delivery_id", "delivery"), ("inter_bot_turn_id", TURN),
    ("inter_bot_bridge_request_id", "bridge"), ("inter_bot_claim_token", "claim"),
    ("inter_bot_timeout_seconds", 30), ("inter_bot_session_policy", "continue"),
    ("inter_bot_seed_session_id", SID), ("inter_bot_reset_only", True),
    ("inter_bot_typed_follow_up", True), ("parent_turn_id", TURN),
    ("answered_question_id", "question"),
    ("continuation_payload", {"kind": "mcp_tool_result", "approval_request_id": "approval",
                              "tool_name": "test", "result": "done"}),
])
def test_nonordinary_callers_consume_but_never_mint(monkeypatch, field, value):
    request, llm = _proven_request(), _llm()
    setattr(request, field, value)
    mint = Mock(side_effect=AssertionError("excluded caller minted"))
    monkeypatch.setattr(turn_codec, "mint_task_turn_context", mint)
    assert _take(request, llm) is None
    assert request._http_selected_session_owner is None
    mint.assert_not_called()


@pytest.mark.parametrize("change", ["missing", "cleared", "session", "bot", "user", "row-owner",
                                    "row-list", "row-incomplete", "spaces", "backend-owner", "unknown-backend",
                                    "client-principal", "client-destination", "history-owner", "llm-owner",
                                    "native-model", "agent-flag", "admission-claim", "changed-copy"])
def test_missing_or_disagreeing_authority_uses_fallback(monkeypatch, change):
    request, llm = _proven_request(), _llm()
    kwargs = {}
    if change in ("missing", "cleared"):
        request._http_selected_session_owner = None
    elif change == "session":
        request.session_id = OTHER_SID
    elif change == "bot":
        request.bot_id = "other"
    elif change == "user":
        request.user = "other"
    elif change == "row-owner":
        request._http_selected_session_owner = (SID, BOT, "other")
    elif change == "row-list":
        request._http_selected_session_owner = list(OWNER)
    elif change == "row-incomplete":
        request._http_selected_session_owner = (SID, BOT, None)
    elif change == "spaces":
        request.session_id = " " + SID + " "
    elif change == "backend-owner":
        llm.history_manager._db_backend = SimpleNamespace(bot_id=BOT, user_id="other")
    elif change == "unknown-backend":
        llm.history_manager._db_backend = object()
    elif change == "client-principal":
        llm.history_manager._db_backend._memory_client._service_principal = service_codec.ServicePrincipal("other", USER)
    elif change == "client-destination":
        llm.history_manager._db_backend._memory_client.bot_id = "CAID"
    elif change == "history-owner":
        llm.history_manager.user_id = "other"
    elif change == "llm-owner":
        llm.user_id = "other"
    elif change == "native-model":
        llm.client.model_definition["type"] = "agent_backend"
    elif change == "agent-flag":
        kwargs["is_agent_backend"] = True
    elif change == "admission-claim":
        kwargs["admission_claim"] = object()
    elif change == "changed-copy":
        request = request.model_copy(update={"session_id": OTHER_SID})
    mint = Mock(side_effect=AssertionError("unproven request minted"))
    monkeypatch.setattr(turn_codec, "mint_task_turn_context", mint)
    assert _take(request, llm, **kwargs) is None
    assert request._http_selected_session_owner is None
    mint.assert_not_called()


@pytest.mark.parametrize("phase", ["initial", "final"])
@pytest.mark.parametrize("change", ["request", "override", "principal", "native", "sender", "turn", "trigger"])
def test_each_history_window_revalidates_destination_and_phase(signer, phase, change):
    request, llm = _proven_request(), _llm()
    selected = _take(request, llm)
    if phase == "final":
        with _window(selected, request, llm, "initial"):
            assert service_codec._runtime_turn.get() == selected.capability
    kwargs = {}
    if change == "request":
        request.session_id = OTHER_SID
    elif change == "override":
        llm._session_id_override = OTHER_SID
    elif change == "principal":
        llm.history_manager._db_backend._memory_client.user_id = "other"
    elif change == "native":
        llm.client.model_definition["type"] = "agent_backend"
    elif change == "sender":
        request._internal_inter_bot_sender_id = "sender"
    elif change == "turn":
        kwargs["turn_id"] = "turn-" + "b" * 32
    elif change == "trigger":
        kwargs["trigger_message_id"] = OTHER_SID
    with _window(selected, request, llm, phase, **kwargs):
        assert service_codec._runtime_turn.get() is None
    assert service_codec._runtime_turn.get() is None


@pytest.mark.parametrize("phases,expected", [
    (["final", "initial"], [False, False]),
    (["initial", "initial", "final"], [True, False, False]),
    (["initial", "final", "final"], [True, True, False]),
])
def test_history_windows_are_one_shot_lifecycle_not_a_replay_cache(signer, phases, expected):
    request, llm = _proven_request(), _llm()
    selected = _take(request, llm)
    seen = []
    for phase in phases:
        with _window(selected, request, llm, phase):
            seen.append(service_codec._runtime_turn.get() is not None)
    assert seen == expected


def test_signing_failure_falls_back_without_retrying_stale_row(monkeypatch):
    mint = Mock(side_effect=ValueError("key unavailable"))
    monkeypatch.setattr(turn_codec, "mint_task_turn_context", mint)
    request, llm = _proven_request(), _llm()
    selected = _take(request, llm)
    assert selected is not None and selected.capability is None
    for phase in ("initial", "final"):
        with _window(selected, request, llm, phase):
            assert service_codec._runtime_turn.get() is None
    mint.assert_called_once()
    assert request._http_selected_session_owner is None


def test_expired_supplied_token_fails_closed_without_remint(signer, monkeypatch):
    clock = SimpleNamespace(now=int(time.time()))

    class ClockedSigner:
        def encrypt(self, data):
            return signer.encrypt_at_time(data, current_time=clock.now)

        def decrypt(self, data, ttl):
            return signer.decrypt_at_time(data, ttl=ttl, current_time=clock.now)

    monkeypatch.setattr(turn_codec, "_get_fernet", lambda: ClockedSigner())
    mint = Mock(wraps=turn_codec.mint_task_turn_context)
    monkeypatch.setattr(turn_codec, "mint_task_turn_context", mint)
    request, llm = _proven_request(), _llm()
    selected = _take(request, llm)
    principal = llm.history_manager._db_backend._memory_client._service_principal
    with _window(selected, request, llm, "initial"):
        principal.mint("messages_add", _arguments())
    clock.now += turn_codec.TASK_TURN_CONTEXT_TTL_SECONDS + 1
    with _window(selected, request, llm, "final"):
        assert service_codec._runtime_turn.get() == selected.capability
        with pytest.raises(turn_codec.TaskTurnContextError, match="expired"):
            principal.mint("messages_add", _arguments())
    assert service_codec._runtime_turn.get() is None
    mint.assert_called_once()


def test_direct_service_or_mcp_without_ingress_never_gains_owner(signer):
    assert _take(_request(), _llm()) is None
    principal = service_codec.ServicePrincipal(BOT, USER)
    envelope = principal.mint("messages_add", _arguments())
    assert service_codec.open_service_session_owner(envelope, "messages_add", _arguments()) is None


def test_concurrent_history_contexts_are_isolated_and_restore_parent(signer):
    barrier = threading.Barrier(2)

    def run(sid, turn):
        request, llm = _proven_request(session_id=sid), _llm()
        request._http_selected_session_owner = (sid, BOT, USER)
        llm._session_id_override = sid
        selected = _take(request, llm, turn_id=turn)
        seen = []
        with service_codec.bind_service_turn("parent"):
            for phase in ("initial", "final"):
                with _window(selected, request, llm, phase, turn_id=turn):
                    barrier.wait(timeout=10)
                    assert service_codec._runtime_native_selected.get() is True
                    seen.append(turn_codec.open_task_turn_context(service_codec._runtime_turn.get()).session_id)
                    barrier.wait(timeout=10)
                assert service_codec._runtime_turn.get() == "parent"
                assert service_codec._runtime_native_selected.get() is False
        assert service_codec._runtime_turn.get() is None
        assert service_codec._runtime_native_selected.get() is False
        return seen

    with ThreadPoolExecutor(max_workers=2) as pool:
        one = pool.submit(run, SID, TURN)
        two = pool.submit(run, OTHER_SID, "turn-" + "b" * 32)
        assert one.result(timeout=15) == [SID, SID]
        assert two.result(timeout=15) == [OTHER_SID, OTHER_SID]
    assert service_codec._runtime_turn.get() is None


@pytest.mark.parametrize("failure", [None, "initial", "execute", "final"])
def test_native_nonstream_scopes_inside_executor_only(signer, monkeypatch, failure):
    # Existing harness is constructor-free and uses only request-local mocks;
    # no imported autouse fixture or production admission contract is replaced.
    from tests.test_nonstream_turn_execution import _service
    from llm_bawt.service import chat_nonstream

    monkeypatch.setattr(chat_nonstream, "get_bot", lambda _bot: None)
    service = _service(Mock(return_value=("answer", "", [])), model_type="openai")
    llm = service._get_llm_bawt.return_value
    owner_llm = _llm()
    llm.bot_id, llm.user_id, llm.history_manager = BOT, USER, owner_llm.history_manager
    caller_thread = threading.get_ident()
    observed, exits = [], []
    bind = service_codec.bind_service_turn

    @contextmanager
    def track(capability, **kwargs):
        try:
            with bind(capability, **kwargs):
                yield
        finally:
            exits.append((threading.get_ident(), service_codec._runtime_turn.get()))

    monkeypatch.setattr(service_codec, "bind_service_turn", track)

    def record(phase):
        assert threading.get_ident() != caller_thread
        observed.append((phase, service_codec._runtime_turn.get()))
        if failure == phase:
            raise RuntimeError("fixture failure")

    def prepare(*args, **kwargs):
        record("initial")
        return [Message(role="user", content="hello")]

    def execute(*args, **kwargs):
        record("execute")
        assert kwargs["task_turn_capability"] is None  # never a native inference kwarg
        return "answer", "", []

    llm.prepare_messages_for_query.side_effect = prepare
    llm.execute_llm_query = execute
    service._finalize_turn.side_effect = lambda **kwargs: record("final")
    request = _proven_request()
    if failure:
        with pytest.raises(Exception, match="fixture failure"):
            asyncio.run(service.chat_completion(request))
    else:
        asyncio.run(service.chat_completion(request))
    assert observed[0][0] == "initial" and observed[0][1] is not None
    if any(phase == "execute" for phase, _ in observed):
        assert dict(observed)["execute"] is None
    if any(phase == "final" for phase, _ in observed):
        assert dict(observed)["final"] == observed[0][1]
    assert request._http_selected_session_owner is None
    assert exits and all(thread_id != caller_thread and restored is None for thread_id, restored in exits)
    assert service_codec._runtime_turn.get() is None


@pytest.mark.parametrize("failure", [None, "initial", "execute", "final", "cancel", "disconnect"])
def test_native_stream_worker_history_only_scopes_and_cleanup(signer, monkeypatch, failure):
    from llm_bawt.service import turn_stream_worker as worker_module
    from llm_bawt.service.turn_stream_context import TurnStreamContext

    request, llm = _proven_request(), _llm()
    selected = _take(request, llm)
    observed = []
    caller_thread = threading.get_ident()
    cancel = threading.Event()
    service = SimpleNamespace(_turn_log_store=SimpleNamespace(engine=object(), get_turn=lambda _id: None),
                              _update_turn_log=Mock(), _resolve_turn_token_usage=lambda *args: None)
    llm.bot = SimpleNamespace(tts_mode=False, uses_tools=False)
    llm._get_generation_kwargs = lambda: {}
    llm.adapter = None

    def record(phase):
        assert threading.get_ident() != caller_thread
        observed.append((phase, service_codec._runtime_turn.get()))
        if phase == failure:
            raise RuntimeError("fixture failure")

    def prepare(*args, **kwargs):
        record("initial")
        if failure == "cancel":
            cancel.set()
        return [Message(role="user", content="hello")]

    def stream(*args, **kwargs):
        record("execute")
        assert "task_turn_capability" not in kwargs
        if failure == "disconnect":
            raise RuntimeError("client disconnected")  # error enqueue sees closed loop
        # Empty iterator avoids unrelated event publishing; actual worker and
        # consume_stream_chunks still execute their ordinary inference window.
        return iter(())

    llm.prepare_messages_for_query = prepare
    llm.client.stream_raw = stream
    monkeypatch.setattr(worker_module.WorkingMemoryTurn, "begin", lambda _ctx:
                        SimpleNamespace(extend_suffix=lambda suffix: suffix))

    class Finalizer:
        def __init__(self, *args, **kwargs):
            pass

        def finalize(self, **kwargs):
            record("final")

    monkeypatch.setattr(worker_module, "TurnStreamFinalizer", Finalizer)

    def call_soon(callback, *args):
        if failure == "disconnect":
            raise RuntimeError("loop is closed")
        callback(*args)

    ctx = TurnStreamContext(
        svc=service, request=request, llm_bawt=llm, loop=SimpleNamespace(call_soon_threadsafe=call_soon),
        bot_id=BOT, user_id=USER, model_alias="fixture-model", user_prompt="hello",
        user_attachments=[], user_author=None, attachments_to_persist=[], media_store=None,
        trigger_message_id=MID, assistant_message_id=None, turn_log_id=TURN,
        is_agent_backend=False, inject_seed_messages=None,
        selected_history=selected, cancel_event=cancel, done_event=threading.Event(),
        chunk_queue=SimpleNamespace(put_nowait=Mock()), tts_scrub=False, tts_scrubber=None,
        full_response_holder=[""], reasoning_holder=[""], animation_holder=[None],
        tool_context_holder=[""], tool_call_details_holder=[], timing_holder=[0.0, 0.0],
        cancelled_holder=[False], token_usage_holder=[None], agent_attachments_holder=[],
        question_id_holder=[None], approval_id_holder=[None], approval_persist_failed_holder=[None],
        _redis_sub=None, _tool_event_coordinator=None, _approval_handled=set(),
        _await_handled=set(), _upstream_model=[None],
    )
    worker = worker_module.TurnStreamWorker(ctx)

    def run():
        try:
            if failure == "final":
                with pytest.raises(RuntimeError, match="fixture failure"):
                    worker._stream_to_queue()
            else:
                worker._stream_to_queue()
        finally:
            assert service_codec._runtime_turn.get() is None

    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(run).result(timeout=15)
    phases = dict(observed)
    assert phases["initial"] == selected.capability
    assert phases["final"] == selected.capability
    if failure not in ("initial", "cancel"):
        assert "execute" in phases and phases["execute"] is None
    assert service_codec._runtime_turn.get() is None
