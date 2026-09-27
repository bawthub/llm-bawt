"""TASK-952 review: exercise real dispatch/abort/consume paths, not just helpers."""
from __future__ import annotations

import asyncio
import io
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException
from rich.console import Console
from rich.logging import RichHandler
from sqlmodel import Session, SQLModel, create_engine

from agent_bridge.events import AgentEvent, AgentEventKind
from claude_code_bridge.command_ops import ClaudeCommandMixin
from llm_bawt.agent_backends import agent_bridge
from llm_bawt.service.chat_stream_worker import consume_stream_chunks
from llm_bawt.service.turn_abort import abort_turn
from llm_bawt.service.turn_execution import TurnExecution, turn_executions
from llm_bawt.service.turn_logs import TurnLog, TurnLogStore
from llm_bawt.service.tool_call_store import ToolCallRecord
from llm_bawt.service.turn_termination import TurnAbortCoordinator


@pytest.fixture
def store(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/turns.db", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine, tables=[TurnLog.__table__, ToolCallRecord.__table__])
    store = TurnLogStore.__new__(TurnLogStore)
    store.engine = engine
    store.ttl_hours = None
    with Session(engine) as session:
        for i, name in enumerate(("A", "B")):
            session.add(TurnLog(id=name, bot_id="test", user_id="nick", status="streaming",
                               created_at=datetime.now(timezone.utc) + timedelta(seconds=i)))
        session.add(ToolCallRecord(turn_id="A", tool_name="Bash", call_id="tool-A"))
        session.commit()
    yield store
    for name in ("A", "B"):
        turn_executions.remove(name)
    engine.dispose()


def execution(store, turn="A", agent=True):
    owner = TurnExecution(turn, threading.Event(), is_agent=agent, store=store)
    turn_executions.register(owner)
    return owner


def service(store):
    return SimpleNamespace(_turn_log_store=store, config=None, _redis_subscriber=None)


@pytest.mark.parametrize("backend", ["codex", "openclaw", "local"])
def test_claude_does_not_answer_another_bridges_abort(backend):
    bridge = ClaudeCommandMixin()
    bridge._backend_name = "claude-code"
    bridge._publisher = Mock()
    bridge._session_queue = Mock()
    redis = SimpleNamespace(xack=AsyncMock())
    asyncio.run(bridge._handle_rpc({"backend": backend, "request_id": "rpc", "method": "chat.abort",
                                  "params": {"sessionKey": "test:nick", "requestId": "req-A"}}, "msg", redis))
    bridge._publisher.publish_rpc_result.assert_not_called()
    bridge._session_queue.cancel_request.assert_not_called()
    redis.xack.assert_awaited_once()


@pytest.mark.parametrize("reply", [{"ok": True}, {"ok": True, "cancelled": False},
                                  {"ok": True, "aborted": "test:nick", "detail": "no_active_task"}])
def test_ok_without_actual_cancellation_is_not_confirmation(store, monkeypatch, reply):
    owner = execution(store)
    owner.bind_bridge(backend="local", session_key="test:nick", request_id="req-A")
    owner.dispatched()
    rpc = AsyncMock(return_value=reply)
    monkeypatch.setattr(agent_bridge, "get_agent_subscriber", lambda: SimpleNamespace(send_rpc=rpc))
    with pytest.raises(HTTPException) as error:
        asyncio.run(abort_turn(service(store), store.get_turn("A"), source="chat_stop", peer=None))
    assert error.value.status_code == 503
    assert store.get_turn("A").status == "cancelling"
    assert store.get_turn("A").ended_at is None
    assert rpc.call_args.kwargs["backend"] == "local"


def test_stop_before_dispatch_prevents_command(store, monkeypatch):
    owner = execution(store)
    assert asyncio.run(abort_turn(service(store), store.get_turn("A"), source="chat_stop", peer=None)) == "cancelling:worker_signalled"
    monkeypatch.setattr(agent_bridge, "get_agent_subscriber", lambda: Mock())
    backend = agent_bridge.AgentBridgeBackend()
    from llm_bawt.service.turn_execution import TurnExecutionCancelled
    with pytest.raises(TurnExecutionCancelled):
        list(backend.stream_raw("hello", {"session_key": "test:nick", "request_id": "req-A", "turn_execution": owner}))
    assert store.get_turn("A").agent_request_id == "req-A"


def test_pre_output_abort_and_overlapping_streams_keep_own_identity(store, monkeypatch):
    """Two requests share ONE backend; neither yields until A is aborted."""
    owners = {name: execution(store, name) for name in ("A", "B")}
    sent = {name: threading.Event() for name in ("A", "B")}
    release = {name: threading.Event() for name in ("A", "B")}
    output = {name: [] for name in ("A", "B")}
    errors = []
    rpc_targets = []

    async def rpc(method, params, request_id, **kwargs):
        rpc_targets.append(params["requestId"])
        release[params["requestId"][-1]].set()
        return {"ok": True, "cancelled": True}

    root = SimpleNamespace(send_rpc=rpc, _redis=SimpleNamespace(connection_pool=SimpleNamespace(connection_kwargs={})))
    monkeypatch.setattr(agent_bridge, "get_agent_subscriber", lambda: root)

    class Subscriber:
        def __init__(self, url): pass
        async def connect(self): pass
        async def close(self): pass
        async def send_command(self, **kwargs):
            name = kwargs["request_id"][-1]
            # Identity must already be durable when Redis sees the command.
            assert store.get_turn(name).agent_request_id == kwargs["request_id"]
            sent[name].set()
        send_rpc = staticmethod(rpc)
        async def subscribe_run(self, request_id, **kwargs):
            name = request_id[-1]
            while not release[name].is_set():
                await asyncio.sleep(.001)
            yield AgentEvent(kind=AgentEventKind.ASSISTANT_DELTA, run_id=request_id,
                             session_key="test:nick", seq=1, text=name,
                             event_id=f"evt-{name}", origin="test")

    monkeypatch.setattr("agent_bridge.subscriber.RedisSubscriber", Subscriber)
    from llm_bawt.clients.agent_backend_client import AgentBackendClient
    from llm_bawt.models.message import Message
    backend = agent_bridge.AgentBridgeBackend()
    client = AgentBackendClient.__new__(AgentBackendClient)
    client._backend = backend
    client._bot_config = {"session_key": "test:nick"}
    def run(name):
        try:
            output[name].extend(client.stream_raw([Message(role="user", content=name)],
                                                 bridge_request_id=f"req-{name}", turn_execution=owners[name]))
        except Exception as error:
            errors.append(error)
    threads = [threading.Thread(target=run, args=(name,)) for name in ("A", "B")]
    try:
        threads[0].start()
        assert sent["A"].wait(3)
        threads[1].start()
        assert sent["B"].wait(3)
        assert not output["A"] and not output["B"]
        # Dispatch marks immediately after send returns; synchronize on that
        # state instead of sleeping or relying on first output.
        async def abort_ready():
            async with asyncio.timeout(3):
                while not owners["A"]._dispatched or not owners["B"]._dispatched:
                    await asyncio.sleep(.001)
            return await abort_turn(service(store), store.get_turn("A"), source="chat_stop", peer=None)
        assert asyncio.run(abort_ready()).startswith("aborted:")
        assert rpc_targets == ["req-A"]
        assert not release["B"].is_set()
        assert store.get_turn("B").status == "streaming"
    finally:
        for event in release.values():
            event.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(3)
    assert not errors
    assert output["B"] == ["B"]


def test_abort_during_command_publication_is_sent_after_chat_send(store, monkeypatch):
    owner = execution(store)
    order = []
    root = SimpleNamespace(_redis=SimpleNamespace(connection_pool=SimpleNamespace(connection_kwargs={})))
    monkeypatch.setattr(agent_bridge, "get_agent_subscriber", lambda: root)

    class Subscriber:
        def __init__(self, url): pass
        async def connect(self): pass
        async def close(self): pass
        async def send_command(self, **kwargs):
            result = await abort_turn(service(store), store.get_turn("A"), source="chat_stop", peer=None)
            assert result == "cancelling:worker_signalled"
            order.append("send")
        async def send_rpc(self, method, params, request_id, **kwargs):
            assert params["requestId"] == "req-A"
            assert kwargs["backend"] == "agent-bridge"
            order.append("abort")
            return {"cancelled": True}
        async def subscribe_run(self, request_id, **kwargs):
            return
            yield

    monkeypatch.setattr("agent_bridge.subscriber.RedisSubscriber", Subscriber)
    list(agent_bridge.AgentBridgeBackend().stream_raw("hello", {
        "session_key": "test:nick", "request_id": "req-A", "turn_execution": owner,
    }))
    assert order == ["send", "abort"]


@pytest.mark.parametrize("raise_after_abort", [False, True])
def test_real_native_worker_stop_finalizes_partial_and_closes_http(store, monkeypatch, raise_after_abort):
    from dataclasses import fields
    from llm_bawt.service.turn_stream_context import TurnStreamContext
    from llm_bawt.service.turn_stream_worker import TurnStreamWorker
    from tests.test_turn_termination import finalizer_context

    # Reuse the finalizer fixture for actual persistence and terminal emission,
    # and execute the entire worker through Stop, not just consume_stream_chunks.
    ctx = TurnStreamContext(**{field.name: None for field in fields(TurnStreamContext)})
    base = finalizer_context(store, monkeypatch)
    for key, value in vars(base).items():
        setattr(ctx, key, value)
    ctx.turn_log_id = "A"
    owner = execution(store, agent=False)
    ctx.cancel_event = owner.cancel_event
    ctx.execution = owner
    ctx.is_agent_backend = False
    ctx.done_event = threading.Event()
    ctx.svc._end_generation = lambda cancel, done, bot: done.set()
    ctx.svc._finalize_turn = lambda **kw: store.update_turn(
        turn_id="A", status=kw.get("status", "ok"), end_reason=kw.get("end_reason"), response_text=kw["response_text"],
    )
    def update_log(**kw):
        kw.pop("prepared_messages", None)
        store.update_turn(**kw)
    ctx.svc._update_turn_log = update_log
    ctx.request = SimpleNamespace(client_system_context=None, ha_mode=False, include_summaries=False,
                                  tts_mode=False, inject_user_prefix=False, session_id=None,
                                  animations=[], avatar_visible=False, inter_bot_bridge_request_id=None)
    closed = []
    def stream(*args, **kwargs):
        try:
            yield "partial"
            asyncio.run(abort_turn(service(store), store.get_turn("A"), source="chat_stop", peer=None))
            if raise_after_abort:
                raise RuntimeError("transport interrupted during cancelling")
            yield "not forwarded" * 100  # exceeds the Redis coalescing threshold
            pytest.fail("continued native model after Stop")
        finally:
            closed.append(True)
    ctx.llm_bawt = SimpleNamespace(
        bot=SimpleNamespace(tts_mode=False, uses_tools=False),
        client=SimpleNamespace(model_definition={"type": "test"}, stream_raw=stream),
        prepare_messages_for_query=lambda *a, **kw: [],
        _get_generation_kwargs=lambda: {},
    )
    ctx.full_response_holder = [""]
    ctx.bot_id = "test"
    ctx.loop = SimpleNamespace(call_soon_threadsafe=lambda fn, value: fn(value))
    chunks = []
    ctx.chunk_queue = SimpleNamespace(put_nowait=chunks.append)
    events = []
    monkeypatch.setattr("llm_bawt.service.turn_stream_finalize.put_queue_item_threadsafe", lambda loop, queue, value: chunks.append(value))
    monkeypatch.setattr(TurnStreamWorker, "_publish_event_direct", lambda self, event: events.append(event))
    TurnStreamWorker(ctx)._stream_to_queue()
    assert closed == [True]
    assert store.get_turn("A").status == "aborted"
    assert store.get_turn("A").response_text == "partial"
    assert chunks == ["partial", None]
    assert events[-1]["status"] == "cancelled"
    assert all("not forwarded" not in event.get("delta", "") for event in events)
    assert ctx.done_event.is_set() and turn_executions.get("A") is None


def test_native_stop_closes_iterator_without_draining_remaining_reply(store):
    owner = execution(store, agent=False)
    closed = []
    delivered = []
    response = [""]
    def source():
        try:
            yield "partial"
            asyncio.run(abort_turn(service(store), store.get_turn("A"), source="chat_stop", peer=None))
            yield "not forwarded"
            pytest.fail("native stream continued after Stop")
        finally:
            closed.append(True)
    loop = SimpleNamespace(call_soon_threadsafe=lambda fn, value: fn(value))
    queue = SimpleNamespace(put_nowait=delivered.append)
    assert consume_stream_chunks(source(), cancel_event=owner.cancel_event, loop=loop,
                                 chunk_queue=queue, full_response_holder=response, stop_on_cancel=True)
    assert delivered == ["partial"] and response == ["partial"] and closed == [True]
    # Acceptance does not prematurely settle tools while a provider is blocked.
    assert store.get_turn("A").status == "cancelling"
    store.update_turn(turn_id="A", status="ok", end_reason="stop")
    assert store.get_turn("A").status == "aborted"


def test_reaper_preserves_cancelling_and_active_siblings(store):
    owner = execution(store)
    assert store.reap_other_open_turns(bot_id="test", current_turn_id="B") == []
    turn_executions.remove(owner.turn_id)
    TurnAbortCoordinator(store).request("A", source="chat_stop", peer=None)
    assert store.reap_other_open_turns(bot_id="test", current_turn_id="B") == []
    assert store.get_turn("A").status == "cancelling"
    store.update_turn(turn_id="A", status="ok", end_reason="stop")
    assert store.get_turn("A").status == "aborted"


def test_reaper_settles_unknown_tools(store):
    assert store.reap_other_open_turns(bot_id="test", current_turn_id="B") == [{"id": "A", "user_id": "nick"}]
    with Session(store.engine) as session:
        row = session.get(ToolCallRecord, 1)
        assert row.ended_at is not None and row.result_text is None and row.result_complete is False


def test_terminal_approval_failure_keeps_diagnostic_enrichment(store):
    store.update_turn(turn_id="A", status="error", end_reason="approval_persist_failed")
    store.update_turn(turn_id="A", status="error", end_reason="approval_persist_failed", error_text="approval DB unavailable")
    assert store.get_turn("A").error_text == "approval DB unavailable"
    store.update_turn(turn_id="A", status="ok", end_reason="stop", error_text="wrong late overwrite")
    assert store.get_turn("A").error_text == "approval DB unavailable"


def test_redis_failure_does_not_turn_successful_stop_into_500(store, monkeypatch):
    owner = execution(store)
    owner.bind_bridge(backend="codex", session_key="test:nick", request_id="req-A")
    owner.dispatched()
    monkeypatch.setattr(agent_bridge, "get_agent_subscriber", lambda: SimpleNamespace(send_rpc=AsyncMock(return_value={"aborted": True})))
    svc = service(store)
    svc._redis_subscriber = SimpleNamespace(publish_tool_event=AsyncMock(side_effect=RuntimeError("Redis down")))
    assert asyncio.run(abort_turn(svc, store.get_turn("A"), source="chat_stop", peer=None)).startswith("aborted:")
    svc._redis_subscriber.publish_tool_event.assert_awaited_once()
    assert store.get_turn("A").status == "aborted"


def test_model_and_task_failure_logs_do_not_parse_user_markup():
    from llm_bawt.service.logging import ServiceLogger
    output = io.StringIO()
    log = ServiceLogger("abort-review")
    log._logger.handlers = [RichHandler(console=Console(file=output), markup=True)]
    log._logger.propagate = False
    log.model_error("test", "bad [/]")
    log.task_failed("task", "job", "bad [/]", 1.)
    assert output.getvalue().count("bad [/]") == 2
