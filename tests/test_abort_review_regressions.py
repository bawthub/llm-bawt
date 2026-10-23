"""TASK-952 review: exercise real dispatch/abort/consume paths, not just helpers."""
from __future__ import annotations

import asyncio
import io
import json
import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from rich.console import Console
from rich.logging import RichHandler
from sqlmodel import Session, SQLModel

from agent_bridge.abort_contract import ABORT_RPC_TIMEOUT_S
from agent_bridge.events import AgentEvent, AgentEventKind
from claude_code_bridge.command_ops import ClaudeCommandMixin
from llm_bawt.agent_backends import agent_bridge
from llm_bawt.service.chat_stream_worker import consume_stream_chunks
from llm_bawt.service.turn_abort import abort_turn
from llm_bawt.service.turn_execution import TurnExecution, turn_executions
from llm_bawt.service.turn_logs import TurnLog, TurnLogStore
from llm_bawt.service.tool_call_store import ToolCallRecord
from llm_bawt.service.turn_termination import TurnAbortCoordinator
from tests.sqlite_support import fast_sqlite_engine


@pytest.fixture
def store(tmp_path):
    engine = fast_sqlite_engine(tmp_path / "turns.db", check_same_thread=False)
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


@pytest.mark.parametrize("reply", [
    {"ok": True},
    {"ok": True, "cancelled": False},
    # claude-code's old ``aborted`` was the session key.
    {"ok": True, "aborted": "test:nick", "detail": "no_active_task"},
])
def test_pre_contract_reply_is_malformed_but_still_stops(store, monkeypatch, reply):
    """TASK-1232 #21: a reply without the v1 shape is no longer mapped. It
    still stops the turn (the P4 guarantee), but is recorded as an
    unconfirmed ``error``, never as cancelled or not_found."""
    owner = execution(store)
    owner.bind_bridge(backend="local", session_key="test:nick", request_id="req-A")
    owner.dispatched()
    rpc = AsyncMock(return_value=reply)
    monkeypatch.setattr(agent_bridge, "get_agent_subscriber", lambda: SimpleNamespace(send_rpc=rpc))
    result = asyncio.run(abort_turn(service(store), store.get_turn("A"), source="chat_stop", peer=None))
    assert result == "aborted:error"
    row = store.get_turn("A")
    assert (row.status, row.end_reason) == ("aborted", "aborted") and row.ended_at is not None
    stop = json.loads(row.request_json)["user_stop"]
    assert (stop["bridge_state"], stop["confirmed"]) == ("error", False)
    assert stop["detail"].startswith("malformed reply") and "contract" not in stop
    assert rpc.call_args.kwargs["backend"] == "local"


def test_stop_before_dispatch_prevents_command(store, monkeypatch):
    from llm_bawt.service import turn_abort

    monkeypatch.setattr(turn_abort, "WORKER_FINALIZE_GRACE_SECONDS", 0.05)
    owner = execution(store)
    # No worker runs in this test, so Stop settles after the bound.
    assert asyncio.run(abort_turn(service(store), store.get_turn("A"), source="chat_stop", peer=None)) == "aborted:worker_unfinalized"
    monkeypatch.setattr(agent_bridge, "get_agent_subscriber", lambda: Mock())
    backend = agent_bridge.AgentBridgeBackend()
    from llm_bawt.service.turn_execution import TurnExecutionCancelled
    with pytest.raises(TurnExecutionCancelled):
        list(backend.stream_raw("hello", {"session_key": "test:nick", "request_id": "req-A", "turn_execution": owner}))
    assert store.get_turn("A").agent_request_id is None


@pytest.mark.parametrize("terminal", ["error", "timeout", "aborted"])
def test_terminal_turn_cannot_publish_bridge_command(store, monkeypatch, terminal):
    from llm_bawt.service.turn_execution import TurnExecutionCancelled

    owner = execution(store)
    store.update_turn(turn_id="A", status=terminal, end_reason=terminal)
    subscriber = Mock()
    monkeypatch.setattr(agent_bridge, "get_agent_subscriber", lambda: subscriber)
    with pytest.raises(TurnExecutionCancelled, match="no longer permits"):
        list(agent_bridge.AgentBridgeBackend().stream_raw("hello", {
            "session_key": "test:nick", "request_id": "req-A", "turn_execution": owner,
        }))
    assert store.get_turn("A").agent_request_id is None
    subscriber.send_command.assert_not_called()


def test_unconsumed_reservation_cannot_bind_bridge_command(store):
    from llm_bawt.service.turn_execution import TurnExecutionCancelled

    with Session(store.engine) as session:
        row = session.get(TurnLog, "A")
        row.status = "reserved"
        session.add(row)
        session.commit()
    owner = execution(store)
    with pytest.raises(TurnExecutionCancelled, match="no longer permits"):
        owner.bind_bridge(backend="claude-code", session_key="test:nick", request_id="req-A")
    assert store.get_turn("A").agent_request_id is None


def test_cross_process_stop_cannot_publish_bridge_command(store, monkeypatch):
    from llm_bawt.service.turn_execution import TurnExecutionCancelled

    owner = execution(store)
    TurnAbortCoordinator(store).request("A", source="chat_stop", peer=None)
    # Simulate a Stop issued from another process: it wrote the DB intent but
    # did not have the in-process execution handle to signal its event.
    assert not owner.cancel_event.is_set()
    subscriber = Mock()
    monkeypatch.setattr(agent_bridge, "get_agent_subscriber", lambda: subscriber)
    with pytest.raises(TurnExecutionCancelled, match="no longer permits"):
        list(agent_bridge.AgentBridgeBackend().stream_raw("hello", {
            "session_key": "test:nick", "request_id": "req-A", "turn_execution": owner,
        }))
    assert store.get_turn("A").agent_request_id is None
    subscriber.send_command.assert_not_called()


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
    from llm_bawt.service import turn_abort

    # Stop runs inline inside the publishing worker here, so that worker can't
    # finalize during the grace; in production they run concurrently.
    monkeypatch.setattr(turn_abort, "WORKER_FINALIZE_GRACE_SECONDS", 0.0)
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
            assert result == "aborted:worker_unfinalized"
            order.append("send")
        async def send_rpc(self, method, params, request_id, **kwargs):
            assert params["requestId"] == "req-A"
            assert kwargs["backend"] == "agent-bridge"
            # Same timeout as the Stop path: it must outlive the bridge's
            # teardown bound (TASK-1232 P4).
            assert kwargs["timeout_s"] == ABORT_RPC_TIMEOUT_S
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


def _real_native_worker(store, monkeypatch, stream):
    """The real TurnStreamWorker over a native ``stream``, persisting through the finalizer fixture."""
    from dataclasses import fields
    from llm_bawt.service.turn_stream_context import TurnStreamContext
    from llm_bawt.service.turn_stream_worker import TurnStreamWorker
    from tests.test_turn_termination import finalizer_context

    ctx = TurnStreamContext(**{field.name: None for field in fields(TurnStreamContext)})
    for key, value in vars(finalizer_context(store, monkeypatch)).items():
        setattr(ctx, key, value)
    ctx.turn_log_id = "A"
    owner = execution(store, agent=False)
    ctx.cancel_event, ctx.execution, ctx.is_agent_backend = owner.cancel_event, owner, False
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
    ctx.llm_bawt = SimpleNamespace(
        bot=SimpleNamespace(tts_mode=False, uses_tools=False),
        client=SimpleNamespace(model_definition={"type": "test"}, stream_raw=stream),
        prepare_messages_for_query=lambda *a, **kw: [],
        _get_generation_kwargs=lambda: {},
    )
    ctx.full_response_holder = [""]
    ctx.bot_id = "test"
    ctx.loop = SimpleNamespace(call_soon_threadsafe=lambda fn, value: fn(value))
    chunks, events = [], []
    ctx.chunk_queue = SimpleNamespace(put_nowait=chunks.append)
    monkeypatch.setattr("llm_bawt.service.turn_stream_finalize.put_queue_item_threadsafe", lambda loop, queue, value: chunks.append(value))
    monkeypatch.setattr(TurnStreamWorker, "_publish_event_direct", lambda self, event: events.append(event))
    return TurnStreamWorker(ctx), ctx, owner, chunks, events


def _assert_worker_closed_with_partial(store, ctx, closed, chunks, events):
    assert closed == [True]
    row = store.get_turn("A")
    assert (row.status, row.end_reason) == ("aborted", "aborted") and row.response_text == "partial"
    assert chunks == ["partial", None]
    assert events[-1]["status"] == "cancelled"
    assert all("not forwarded" not in event.get("delta", "") for event in events)
    assert ctx.done_event.is_set() and turn_executions.get("A") is None


def test_real_native_worker_finalizes_within_the_grace_and_stop_confirms_it(store, monkeypatch):
    """Production shape: the worker streams on its own thread, Stop arrives from
    the event loop. The worker's own terminal must be visible to Stop's bounded
    wait, so Stop returns worker_finalized well inside the grace, not on its
    timeout fallback."""
    from llm_bawt.service import turn_abort

    closed, waiting = [], threading.Event()
    def stream(*args, **kwargs):
        try:
            yield "partial"
            waiting.set()
            # A blocked provider: its next chunk arrives only after Stop signals.
            turn_executions.get("A").cancel_event.wait(5)
            yield "not forwarded" * 100
            pytest.fail("continued native model after Stop")
        finally:
            closed.append(True)
    worker, ctx, owner, chunks, events = _real_native_worker(store, monkeypatch, stream)
    thread = threading.Thread(target=worker._stream_to_queue)
    thread.start()
    try:
        assert waiting.wait(5)
        started = time.monotonic()
        result = asyncio.run(abort_turn(service(store), store.get_turn("A"), source="chat_stop", peer=None))
        elapsed = time.monotonic() - started
    finally:
        owner.cancel_event.set()
        thread.join(5)
    assert result == "aborted:worker_finalized"
    assert elapsed < turn_abort.WORKER_FINALIZE_GRACE_SECONDS / 5
    stop = json.loads(store.get_turn("A").request_json)["user_stop"]
    assert (stop["bridge_state"], stop["confirmed"], stop["settled"]) == ("worker_finalized", True, False)
    _assert_worker_closed_with_partial(store, ctx, closed, chunks, events)


@pytest.mark.parametrize("raise_after_abort", [False, True])
def test_stop_from_inside_the_workers_own_stream_settles_on_the_fallback_and_keeps_the_partial(
        store, monkeypatch, raise_after_abort):
    """Stop runs synchronously inside the generator the worker is iterating, so
    the worker cannot finalize until Stop returns: Stop settles on its
    worker_unfinalized fallback (grace zeroed; waiting it out proves nothing),
    and the worker's late finalize still keeps the partial without reopening."""
    from llm_bawt.service import turn_abort

    monkeypatch.setattr(turn_abort, "WORKER_FINALIZE_GRACE_SECONDS", 0.0)
    closed, results = [], []
    def stream(*args, **kwargs):
        try:
            yield "partial"
            results.append(asyncio.run(abort_turn(service(store), store.get_turn("A"), source="chat_stop", peer=None)))
            if raise_after_abort:
                raise RuntimeError("transport interrupted during cancelling")
            yield "not forwarded" * 100  # exceeds the Redis coalescing threshold
            pytest.fail("continued native model after Stop")
        finally:
            closed.append(True)
    worker, ctx, _, chunks, events = _real_native_worker(store, monkeypatch, stream)
    worker._stream_to_queue()
    assert results == ["aborted:worker_unfinalized"]
    stop = json.loads(store.get_turn("A").request_json)["user_stop"]
    assert (stop["bridge_state"], stop["confirmed"]) == ("worker_unfinalized", False)
    _assert_worker_closed_with_partial(store, ctx, closed, chunks, events)


def test_agent_worker_carries_the_bridge_abort_cause_to_the_turn_row(store, monkeypatch):
    """TASK-1231B: the worker's turn_cancelled branch passes the bridge's cause on."""
    from dataclasses import fields
    from agent_bridge.abort_contract import HOST_SHUTDOWN
    from llm_bawt.inter_bot_delivery_retry import restart_abort_cause
    from llm_bawt.service.turn_stream_context import TurnStreamContext
    from llm_bawt.service.turn_stream_worker import TurnStreamWorker
    from tests.test_turn_termination import finalizer_context

    ctx = TurnStreamContext(**{field.name: None for field in fields(TurnStreamContext)})
    for key, value in vars(finalizer_context(store, monkeypatch)).items():
        setattr(ctx, key, value)
    ctx.turn_log_id = "A"
    owner = execution(store)
    ctx.cancel_event, ctx.execution, ctx.is_agent_backend = owner.cancel_event, owner, True
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
    def stream(*args, **kwargs):
        yield "partial"
        yield {"event": "turn_cancelled", "abort_cause": HOST_SHUTDOWN}
    ctx.llm_bawt = SimpleNamespace(
        bot=SimpleNamespace(tts_mode=False, uses_tools=False),
        client=SimpleNamespace(model_definition={"type": "test"}, stream_raw=stream),
        prepare_messages_for_query=lambda *a, **kw: [],
        _get_generation_kwargs=lambda: {},
    )
    ctx.full_response_holder = [""]
    ctx.bot_id = "test"
    ctx.loop = SimpleNamespace(call_soon_threadsafe=lambda fn, value: fn(value))
    ctx.chunk_queue = SimpleNamespace(put_nowait=lambda value: None)
    monkeypatch.setattr("llm_bawt.service.turn_stream_finalize.put_queue_item_threadsafe", lambda loop, queue, value: None)
    monkeypatch.setattr(TurnStreamWorker, "_publish_event_direct", lambda self, event: None)
    TurnStreamWorker(ctx)._stream_to_queue()
    row = store.get_turn("A")
    assert (row.status, row.end_reason) == ("aborted", "aborted")
    assert restart_abort_cause(row) == HOST_SHUTDOWN


def test_native_stop_closes_iterator_without_draining_remaining_reply(store, monkeypatch):
    from llm_bawt.service import turn_abort
    # The worker is blocked inside this very generator, so it cannot finalize
    # within the grace: Stop settles without it (TASK-1232 P4).
    monkeypatch.setattr(turn_abort, "WORKER_FINALIZE_GRACE_SECONDS", 0.0)
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
    stopped = store.get_turn("A")
    assert (stopped.status, stopped.end_reason) == ("aborted", "aborted")
    assert json.loads(stopped.request_json)["user_stop"]["bridge_state"] == "worker_unfinalized"
    # The worker's late finalize cannot reopen the settled row.
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


def test_reaper_bounds_unacknowledged_cancel_without_live_owner(store, monkeypatch):
    """TASK-1015: fresh intent survives; a stale ownerless Stop settles."""
    import llm_bawt.service.turn_termination as termination

    TurnAbortCoordinator(store).request("A", source="chat_stop", peer=None)
    assert store.reap_other_open_turns(bot_id="test", current_turn_id="B") == []
    monkeypatch.setattr(termination, "UNCONFIRMED_ABORT_GRACE_SECONDS", -1.0)
    owner = execution(store)  # a live owner still blocks the bound
    assert store.reap_other_open_turns(bot_id="test", current_turn_id="B") == []
    turn_executions.remove(owner.turn_id)
    reaped = store.reap_other_open_turns(bot_id="test", current_turn_id="B")
    assert [r["id"] for r in reaped] == ["A"]
    row = store.get_turn("A")
    # A1: abort-class settles are aborted/aborted; the cause is in the audit.
    assert (row.status, row.end_reason) == ("aborted", "aborted")
    assert json.loads(row.request_json)["recovery"]["cause"] == "abort_unconfirmed"
    assert row.ended_at is not None
    assert store.get_turn("A").status == "aborted"


def test_reaper_settles_unknown_tools(store):
    assert store.reap_other_open_turns(bot_id="test", current_turn_id="B") == [{"id": "A", "user_id": "nick"}]
    with Session(store.engine) as session:
        row = session.get(ToolCallRecord, 1)
        assert row.ended_at is not None and row.result_text is None and row.result_complete is False


@pytest.mark.parametrize("bound", [False, True], ids=["pre-publication", "possibly-published"])
@pytest.mark.parametrize("stopping", [False, True], ids=["running", "stopping"])
def test_reaper_cannot_clear_ambiguous_ordinary_admission(store, monkeypatch, bound, stopping):
    from llm_bawt.service import turn_termination

    with Session(store.engine) as session:
        row = session.get(TurnLog, "A")
        row.request_json = json.dumps({"ordinary_request_digest": "exact-request"})
        if bound:
            row.agent_request_id = "req-A"
        session.add(row)
        session.commit()
    if stopping:
        TurnAbortCoordinator(store).request("A", source="chat_stop", peer=None)
        monkeypatch.setattr(turn_termination, "UNCONFIRMED_ABORT_GRACE_SECONDS", -1.0)

    assert store.reap_other_open_turns(bot_id="test", current_turn_id="B") == []
    row = store.get_turn("A")
    assert row.ended_at is None
    assert row.status == ("cancelling" if stopping else "streaming")
    with Session(store.engine) as session:
        assert session.get(ToolCallRecord, 1).ended_at is None


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


# ---------------------------------------------------------------------------
# TASK-1232 P4: "stop should stop, no questions" (Nick). A user Stop settles the
# turn on every bridge outcome through the shared settle, records the bridge's
# answer in a ``user_stop`` audit, and releases admission. Abort-class rows end
# aborted/aborted (A1); the cause lives in the audit, never in end_reason.
# ---------------------------------------------------------------------------


def _normalized_abort_reply(state: str, *, cancelled: bool = False, detail: str = "") -> dict:
    # The v1 wire shape comes from the contract module, never a hand copy.
    from agent_bridge.abort_contract import AbortReply, AbortState

    reply = AbortReply.build(backend="claude-code", boot_id="boot-1", request_id="req-A",
                             state=AbortState(state), detail=detail)
    assert reply.cancelled is cancelled
    return reply.to_dict()


def _orphan_bound_to_bridge(store, monkeypatch, *, request_id: str | None = "req-A"):
    """App restarted: the row names a bridge run, but no in-process owner exists."""
    with Session(store.engine) as session:
        row = session.get(TurnLog, "A")
        row.agent_session_key, row.agent_request_id = "test:nick", request_id
        session.add(row)
        session.commit()
    from llm_bawt import bots
    monkeypatch.setattr(bots, "BotManager", lambda config: SimpleNamespace(
        get_bot=lambda bot_id: SimpleNamespace(agent_backend="claude-code")))


def _assert_user_stop_settled(store, *, bridge_state: str, confirmed: bool) -> None:
    row = store.get_turn("A")
    assert (row.status, row.end_reason) == ("aborted", "aborted")
    assert row.ended_at is not None
    audit = json.loads(row.request_json)
    assert audit["abort_request"]["source"] == "chat_stop"
    assert audit["user_stop"]["bridge_state"] == bridge_state
    assert audit["user_stop"]["confirmed"] is confirmed
    with Session(store.engine) as session:
        tool = session.get(ToolCallRecord, 1)
        assert tool.ended_at is not None and tool.result_complete is False


@pytest.mark.parametrize(("rpc", "bridge_state", "confirmed"), [
    (AsyncMock(return_value=_normalized_abort_reply("cancelled", cancelled=True)), "cancelled", True),
    (AsyncMock(return_value=_normalized_abort_reply("not_found")), "not_found", False),
    (AsyncMock(return_value=_normalized_abort_reply("tombstoned")), "tombstoned", False),
    (AsyncMock(return_value=_normalized_abort_reply("error")), "error", False),
    (AsyncMock(side_effect=asyncio.TimeoutError()), "rpc_unconfirmed", False),
    (None, "bridge_unreachable", False),
], ids=["cancelled", "not_found", "tombstoned", "bridge_error", "rpc_timeout", "unreachable"])
def test_user_stop_settles_orphan_on_every_bridge_outcome(store, monkeypatch, rpc, bridge_state, confirmed):
    """The 2026-10-21 deadlock: an orphan's Stop got 503 and the row stayed open."""
    _orphan_bound_to_bridge(store, monkeypatch)
    subscriber = SimpleNamespace(send_rpc=rpc) if rpc is not None else None
    monkeypatch.setattr(agent_bridge, "get_agent_subscriber", lambda: subscriber)
    result = asyncio.run(abort_turn(service(store), store.get_turn("A"), source="chat_stop", peer=None))
    assert result.startswith("aborted:")
    _assert_user_stop_settled(store, bridge_state=bridge_state, confirmed=confirmed)
    if rpc is not None:
        assert rpc.call_args.kwargs["backend"] == "claude-code"


def test_user_stop_waits_out_the_bridge_teardown_and_audits_its_detail(store, monkeypatch):
    """Vex S3: the RPC outlives the bridge's teardown bound, so the reply that
    carries the teardown outcome is what the user_stop audit records."""
    _orphan_bound_to_bridge(store, monkeypatch)
    rpc = AsyncMock(return_value=_normalized_abort_reply(
        "cancelled", cancelled=True, detail="teardown_killed_after_backstop"))
    monkeypatch.setattr(agent_bridge, "get_agent_subscriber", lambda: SimpleNamespace(send_rpc=rpc))
    assert asyncio.run(abort_turn(service(store), store.get_turn("A"), source="chat_stop", peer=None)) == "aborted:cancelled"
    assert rpc.call_args.kwargs["timeout_s"] == ABORT_RPC_TIMEOUT_S
    _assert_user_stop_settled(store, bridge_state="cancelled", confirmed=True)
    stop = json.loads(store.get_turn("A").request_json)["user_stop"]
    assert (stop["detail"], stop["boot_id"], stop["backend"], stop["contract"]) == (
        "teardown_killed_after_backstop", "boot-1", "claude-code", "v1")


def test_user_stop_settles_orphan_without_request_id(store, monkeypatch):
    _orphan_bound_to_bridge(store, monkeypatch, request_id=None)
    rpc = AsyncMock()
    monkeypatch.setattr(agent_bridge, "get_agent_subscriber", lambda: SimpleNamespace(send_rpc=rpc))
    result = asyncio.run(abort_turn(service(store), store.get_turn("A"), source="chat_stop", peer=None))
    assert result.startswith("aborted:")
    rpc.assert_not_called()  # no request id: there is nothing exact to abort
    _assert_user_stop_settled(store, bridge_state="missing_request_id", confirmed=False)


def test_user_stop_snapshot_publishes_the_settled_end_reason(store, monkeypatch):
    _orphan_bound_to_bridge(store, monkeypatch)
    rpc = AsyncMock(return_value=_normalized_abort_reply("not_found"))
    monkeypatch.setattr(agent_bridge, "get_agent_subscriber", lambda: SimpleNamespace(send_rpc=rpc))
    svc = service(store)
    svc._redis_subscriber = SimpleNamespace(publish_tool_event=AsyncMock())
    asyncio.run(abort_turn(svc, store.get_turn("A"), source="chat_stop", peer=None))
    event = svc._redis_subscriber.publish_tool_event.await_args.args[2]
    assert (event["status"], event["end_reason"]) == ("cancelled", "aborted")


def test_user_stop_settles_a_worker_that_never_finalizes(store, monkeypatch):
    """A signalled in-process worker that hangs is bounded, never waited on forever."""
    from llm_bawt.service import turn_abort

    monkeypatch.setattr(turn_abort, "WORKER_FINALIZE_GRACE_SECONDS", 0.05, raising=False)
    execution(store, agent=False)  # native owner: Stop signals it; no RPC
    result = asyncio.run(abort_turn(service(store), store.get_turn("A"), source="chat_stop", peer=None))
    assert result.startswith("aborted:")
    _assert_user_stop_settled(store, bridge_state="worker_unfinalized", confirmed=False)


def test_user_stop_keeps_the_workers_own_terminal_when_it_finalizes_in_time(store, monkeypatch):
    from llm_bawt.service import turn_abort

    monkeypatch.setattr(turn_abort, "WORKER_FINALIZE_GRACE_SECONDS", 0.5, raising=False)
    execution(store, agent=False)

    def finalize_soon():
        # Finalize only after Stop recorded its request: a fixed sleep let a
        # loaded run finalize first, and Stop then saw a completed turn.
        deadline = time.monotonic() + 5
        while store.get_turn("A").status != "cancelling" and time.monotonic() < deadline:
            time.sleep(0.005)
        store.update_turn(turn_id="A", status="aborted", end_reason="aborted")

    worker = threading.Thread(target=finalize_soon)
    worker.start()
    try:
        result = asyncio.run(abort_turn(service(store), store.get_turn("A"), source="chat_stop", peer=None))
    finally:
        worker.join()
    assert result == "aborted:worker_finalized"
    row = store.get_turn("A")
    assert (row.status, row.end_reason) == ("aborted", "aborted")
    stop = json.loads(row.request_json)["user_stop"]
    # The worker's own terminal won; Stop only recorded what it saw.
    assert (stop["bridge_state"], stop["confirmed"], stop["settled"]) == ("worker_finalized", True, False)


def test_late_finalize_cannot_reopen_or_erase_settle_audits(store):
    """S6 fence: a worker finalizing after a settle writes request_payload, which
    today keeps only abort_request/ordinary_request_digest and drops the rest."""
    audits = {
        "abort_request": {"source": "chat_stop", "outcome": "not_found"},
        "user_stop": {"bridge_state": "not_found", "confirmed": False},
        "recovery": {"step": "1d", "stream_last_id": "1-2"},
        "manual_settle": {"by": "snark"},
    }
    with Session(store.engine) as session:
        row = session.get(TurnLog, "A")
        row.status, row.end_reason = "aborted", "aborted"
        row.ended_at = datetime.now(timezone.utc)
        row.request_json = json.dumps(audits)
        session.add(row)
        session.commit()
    store.update_turn(turn_id="A", status="completed", end_reason="stop", response_text="late reply",
                      request_payload={"messages": [{"role": "user", "content": "hi"}]})
    row = store.get_turn("A")
    assert (row.status, row.end_reason) == ("aborted", "aborted")
    payload = json.loads(row.request_json)
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    for key, value in audits.items():
        assert payload[key] == value
