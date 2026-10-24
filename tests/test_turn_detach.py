"""Shutdown detach for agent turn workers (TASK-1286 defect #1, slice B).

On app shutdown every streaming agent worker is detached: it stops reading
its bridge run, writes no terminal state and never cancels the run, so the
next process can reattach. A delivery claim is released only once its worker
has returned AND its bridge stream thread is dead, both checked directly.
"""
from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import fields as dataclass_fields
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from sqlmodel import Session, SQLModel

from agent_bridge.subscriber import RedisSubscriber
from llm_bawt.agent_backends import agent_bridge
from llm_bawt.service.inter_bot_dispatcher import InterBotDeliveryDispatcher
from llm_bawt.service.turn_execution import (
    TurnExecution,
    TurnExecutionDetached,
    TurnExecutions,
    turn_executions,
)
from llm_bawt.service.turn_logs import TurnLog, TurnLogStore
from llm_bawt.service.tool_call_store import ToolCallRecord
from tests.sqlite_support import fast_sqlite_engine


def _until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


# ── subscribe_run(should_stop=) ─────────────────────────────────────────────


class _Redis:
    """XREAD stand-in: ``batches`` are returned in order, then empty blocks."""

    def __init__(self, batches=(), block_s=0.0, on_xread=None):
        self.batches = list(batches)
        self.block_s = block_s
        self.on_xread = on_xread
        self.calls = 0

    async def xread(self, streams, count=None, block=None):
        self.calls += 1
        if self.on_xread is not None:
            self.on_xread(self.calls)
        if self.batches:
            return self.batches.pop(0)
        await asyncio.sleep(self.block_s)
        return []


def _subscriber(redis) -> RedisSubscriber:
    sub = RedisSubscriber.__new__(RedisSubscriber)
    sub._redis = redis
    return sub


def _event(seq: int) -> tuple[str, dict]:
    import json
    return (f"{seq}-0", {"payload": json.dumps({
        "kind": "assistant_delta", "run_id": "req-1", "session_key": "s", "seq": seq,
        "text": f"t{seq}", "event_id": f"e{seq}", "origin": "test"})})


def test_subscribe_run_returns_on_detach_after_an_empty_xread_instead_of_timing_out():
    redis = _Redis(block_s=0.06)

    async def run():
        # The empty XREAD outlasts the inactivity window: only the detach check
        # at the top of the loop stands between it and a TimeoutError.
        return [event async for event in _subscriber(redis).subscribe_run(
            "req-1", timeout_s=0.05, should_stop=lambda: redis.calls >= 1)]

    assert asyncio.run(run()) == []
    assert redis.calls == 1


def test_subscribe_run_stops_before_the_next_event_once_detached():
    redis = _Redis(batches=[[("run:req-1", [_event(1), _event(2), _event(3)])]])
    seen = []

    async def run():
        async for event in _subscriber(redis).subscribe_run(
                "req-1", timeout_s=5, should_stop=lambda: bool(seen)):
            seen.append(event.text)

    asyncio.run(run())
    assert seen == ["t1"]


# ── AgentBridgeBackend.stream_raw ───────────────────────────────────────────


class _Turns:
    def bind_agent_bridge(self, **kwargs):
        pass


def _execution(turn="turn-1") -> TurnExecution:
    return TurnExecution(turn, threading.Event(), is_agent=True, store=_Turns())


def _bridge(monkeypatch, redis, *, close_gate: threading.Event | None = None, name="claude-code"):
    calls = SimpleNamespace(commands=[], rpcs=[], closed=0)
    root = SimpleNamespace(_redis=SimpleNamespace(connection_pool=SimpleNamespace(connection_kwargs={})))
    monkeypatch.setattr(agent_bridge, "get_agent_subscriber", lambda: root)

    class Subscriber(RedisSubscriber):
        def __init__(self, url):
            self._redis = redis

        async def connect(self):
            pass

        async def close(self):
            if close_gate is not None:
                while not close_gate.is_set():
                    await asyncio.sleep(0.005)
            calls.closed += 1

        async def send_command(self, **kwargs):
            calls.commands.append(kwargs["request_id"])

        async def send_rpc(self, method, params, request_id=None, **kwargs):
            calls.rpcs.append(method)
            return {"ok": True, "cancelled": True}

    monkeypatch.setattr("agent_bridge.subscriber.RedisSubscriber", Subscriber)
    backend = agent_bridge.AgentBridgeBackend()
    backend.name = name
    return backend, calls


def _consume(backend, execution, outcome, **config):
    try:
        outcome["items"] = list(backend.stream_raw("hello", {
            "session_key": "test:nick", "request_id": "req-1", "turn_execution": execution, **config,
        }))
    except Exception as exc:  # noqa: BLE001 - the test inspects the type
        outcome["error"] = exc


def test_a_silent_run_detaches_promptly_without_cancelling_the_bridge_run(monkeypatch):
    redis = _Redis(block_s=0.01)
    backend, calls = _bridge(monkeypatch, redis)
    execution, outcome = _execution(), {}
    # Daemon + a short inactivity bound: a regression that ignores detach then
    # fails this test instead of hanging the run for the 600s default.
    consumer = threading.Thread(target=_consume, args=(backend, execution, outcome),
                                kwargs={"timeout_seconds": 3}, daemon=True)
    consumer.start()
    assert _until(lambda: redis.calls >= 1)

    execution.detach()
    consumer.join(1.0)

    assert not consumer.is_alive()
    assert isinstance(outcome.get("error"), TurnExecutionDetached)
    assert calls.commands == ["req-1"] and calls.rpcs == []  # no chat.cancel
    assert calls.closed == 1
    assert not execution._stream_thread.is_alive()


def test_a_detached_worker_never_publishes_its_command(monkeypatch):
    backend, calls = _bridge(monkeypatch, _Redis())
    execution, outcome = _execution(), {}
    execution.detach()
    _consume(backend, execution, outcome)
    assert isinstance(outcome.get("error"), TurnExecutionDetached)
    assert calls.commands == [] and calls.rpcs == []


def test_a_run_whose_terminal_arrived_before_detach_ends_normally(monkeypatch):
    redis = _Redis(batches=[[("run:req-1", [_event(1), ("2-0", {"done": "1"})])]])
    backend, calls = _bridge(monkeypatch, redis)
    execution, outcome = _execution(), {}
    # Shutdown lands right after the run's terminal was read.
    execution.record_exit = lambda raw: execution.detach() or False
    _consume(backend, execution, outcome)
    assert "error" not in outcome, outcome
    assert outcome["items"] == ["t1"]
    assert execution.detach_requested()


def test_an_inactivity_timeout_after_detach_sends_no_chat_cancel(monkeypatch):
    backend, calls = _bridge(monkeypatch, _Redis())
    execution, outcome = _execution(), {}

    async def timed_out(self, request_id, **kwargs):
        execution.detach()
        raise TimeoutError("no event")
        yield  # pragma: no cover

    monkeypatch.setattr(RedisSubscriber, "subscribe_run", timed_out)
    _consume(backend, execution, outcome)
    assert isinstance(outcome.get("error"), TurnExecutionDetached)
    assert "chat.cancel" not in calls.rpcs


@pytest.mark.integration
@pytest.mark.offline_integration
def test_the_worker_does_not_return_while_the_stream_thread_hangs_in_close(monkeypatch):
    """Real threads, held past one 2s queue poll: a worker that stops waiting
    for the stream thread after a poll timeout would return inside the hold."""
    gate = threading.Event()
    redis = _Redis(block_s=0.01)
    backend, calls = _bridge(monkeypatch, redis, close_gate=gate)
    execution, outcome = _execution(), {}

    async def run():
        loop = asyncio.get_running_loop()
        future = execution.launch_worker(
            loop, lambda: _consume(backend, execution, outcome, timeout_seconds=3))
        try:
            while redis.calls < 1:
                await asyncio.sleep(0.005)
            execution.detach()
            await asyncio.sleep(2.4)
            # The subscriber stopped, but close() hangs: the worker is still
            # waiting on the stream thread, so nothing may treat it as stopped.
            assert execution._stream_thread.is_alive()
            assert not future.done() and not execution.confirmed_stopped()
        finally:
            gate.set()  # asyncio.run waits for executor threads on exit
        await asyncio.wait_for(future, 1.0)
        assert execution.confirmed_stopped()

    asyncio.run(run())
    assert isinstance(outcome.get("error"), TurnExecutionDetached)
    assert calls.rpcs == []


# ── TurnExecution proof of exit ─────────────────────────────────────────────


def test_confirmed_stopped_needs_the_worker_returned_and_the_stream_thread_dead():
    execution = _execution()
    release_worker, release_thread = threading.Event(), threading.Event()
    stream_thread = threading.Thread(target=release_thread.wait, args=(2,))
    stream_thread.start()
    execution.attach_stream_thread(stream_thread)

    async def run():
        loop = asyncio.get_running_loop()
        future = execution.launch_worker(loop, lambda: release_worker.wait(2))
        try:
            # I1: cancelling the asyncio future makes it "done" but proves nothing.
            future.cancel()
            assert future.done() and not execution.confirmed_stopped()
            release_worker.set()
            assert await asyncio.to_thread(_until, execution._worker_exited.is_set)
            assert not execution.confirmed_stopped()  # the stream thread still runs
            release_thread.set()
            stream_thread.join(1)
            assert execution.confirmed_stopped()
        finally:
            release_worker.set()  # asyncio.run waits for executor threads on exit
            release_thread.set()

    asyncio.run(run())


def test_detach_agents_touches_every_registered_agent_turn_launched_or_not():
    registry = TurnExecutions()
    launched, native, unlaunched = _execution("a"), _execution("b"), _execution("c")
    native.is_agent = False
    for execution in (launched, native, unlaunched):
        registry.register(execution)

    async def launch():
        await launched.launch_worker(asyncio.get_running_loop(), lambda: None)

    asyncio.run(launch())
    assert registry.detach_agents() == [launched, unlaunched]
    assert launched.detach_requested() and unlaunched.detach_requested()
    assert not native.detach_requested()


# ── TurnStreamWorker + finalizer ────────────────────────────────────────────


@pytest.fixture
def turn_store(tmp_path):
    engine = fast_sqlite_engine(tmp_path / "turns.db", check_same_thread=False)
    SQLModel.metadata.create_all(engine, tables=[TurnLog.__table__, ToolCallRecord.__table__])
    store = TurnLogStore.__new__(TurnLogStore)
    store.engine = engine
    store.ttl_hours = None
    with Session(engine) as session:
        session.add(TurnLog(id="A", bot_id="test", user_id="nick", status="streaming",
                            created_at=datetime.now(timezone.utc)))
        session.commit()
    return store


def _agent_worker(store, monkeypatch, stream):
    from llm_bawt.service.turn_stream_context import TurnStreamContext
    from llm_bawt.service.turn_stream_worker import TurnStreamWorker
    from tests.test_turn_termination import finalizer_context

    ctx = TurnStreamContext(**{field.name: None for field in dataclass_fields(TurnStreamContext)})
    for key, value in vars(finalizer_context(store, monkeypatch)).items():
        setattr(ctx, key, value)
    ctx.turn_log_id = "A"
    owner = TurnExecution("A", threading.Event(), is_agent=True, store=store)
    turn_executions.register(owner)
    ctx.cancel_event, ctx.execution, ctx.is_agent_backend = owner.cancel_event, owner, True
    ctx.done_event = threading.Event()
    ctx.svc._end_generation = lambda cancel, done, bot: done.set()
    ctx.svc._finalize_turn = lambda **kw: store.update_turn(
        turn_id="A", status=kw.get("status", "ok"), end_reason=kw.get("end_reason"),
        response_text=kw["response_text"],
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
        client=SimpleNamespace(model_definition={"type": "test"}, stream_raw=lambda *a, **kw: stream(owner)),
        prepare_messages_for_query=lambda *a, **kw: [],
        _get_generation_kwargs=lambda: {},
    )
    ctx.full_response_holder = [""]
    ctx.bot_id = "test"
    ctx.loop = SimpleNamespace(call_soon_threadsafe=lambda fn, value: fn(value))
    chunks, events = [], []
    ctx.chunk_queue = SimpleNamespace(put_nowait=chunks.append)
    monkeypatch.setattr("llm_bawt.service.turn_stream_finalize.put_queue_item_threadsafe",
                        lambda loop, queue, value: chunks.append(value))
    monkeypatch.setattr(TurnStreamWorker, "_publish_event_direct", lambda self, event: events.append(event))
    return TurnStreamWorker(ctx), ctx, owner, chunks, events


def test_a_detached_worker_leaves_the_turn_open_and_publishes_no_terminal(turn_store, monkeypatch):
    def stream(owner):
        yield "partial"
        owner.detach()
        raise TurnExecutionDetached()

    worker, ctx, owner, chunks, events = _agent_worker(turn_store, monkeypatch, stream)
    worker._stream_to_queue()

    row = turn_store.get_turn("A")
    assert (row.status, row.ended_at, row.error_text) == ("streaming", None, None)
    assert owner.detached_exit
    assert not any(event.get("_type") == "turn_complete" for event in events)
    assert not any(isinstance(chunk, Exception) for chunk in chunks)
    assert chunks[-1] is None  # the HTTP generator is still released
    assert ctx.done_event.is_set() and turn_executions.get("A") is None


def test_a_stream_that_finished_before_detach_still_finalizes(turn_store, monkeypatch):
    def stream(owner):
        yield "whole reply"
        owner.detach()  # shutdown arrives after the run ended on its own

    worker, ctx, owner, chunks, events = _agent_worker(turn_store, monkeypatch, stream)
    worker._stream_to_queue()

    row = turn_store.get_turn("A")
    assert row.status != "streaming" and row.response_text == "whole reply"
    assert not owner.detached_exit
    assert turn_executions.get("A") is None


# ── dispatcher.stop() ───────────────────────────────────────────────────────


class _StopStore:
    def __init__(self, order):
        self.order = order
        self.expired: list[list[str]] = []

    def expire_own_claims(self, *, claim_owner, turn_ids):
        self.order.append("expire")
        self.expired.append((claim_owner, list(turn_ids)))
        return [f"delivery-{turn}" for turn in turn_ids]

    def release_dispatcher_lock(self, lock):
        self.order.append("unlock")


@pytest.fixture
def registry(monkeypatch):
    """A private execution registry: stop() must see only this test's workers."""
    fresh = TurnExecutions()
    monkeypatch.setattr("llm_bawt.service.turn_execution.turn_executions", fresh)
    return fresh


def _stopping_dispatcher(order):
    dispatcher = InterBotDeliveryDispatcher.__new__(InterBotDeliveryDispatcher)
    dispatcher.store = _StopStore(order)
    dispatcher._stop_event = asyncio.Event()
    dispatcher._wake_event = asyncio.Event()
    dispatcher._task = dispatcher._leader_task = dispatcher.side_pumps = None
    dispatcher._recovery = None
    dispatcher._target_tasks = {}
    dispatcher._dispatch_tasks = set()
    dispatcher.claim_owner = "dispatcher-old"
    dispatcher._dispatcher_lock = object()
    return dispatcher


def test_stop_releases_only_confirmed_workers_after_cancelling_dispatch_and_before_unlock(
        monkeypatch, registry):
    monkeypatch.setattr(InterBotDeliveryDispatcher, "DETACH_CONFIRM_SECONDS", 0.2)
    order: list[str] = []
    dispatcher = _stopping_dispatcher(order)
    quick, stuck = _execution("turn-quick"), _execution("turn-stuck")
    stuck_release = threading.Event()

    async def run():
        loop = asyncio.get_running_loop()
        for execution, wait in ((quick, lambda: _until(quick.detach_requested)),
                                (stuck, lambda: stuck_release.wait(2))):
            registry.register(execution)
            execution.launch_worker(loop, wait)

        async def dispatch():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                # Recorded, not asserted: stop() gathers with return_exceptions.
                detached = quick.detach_requested() and stuck.detach_requested()
                order.append("dispatch-cancelled" if detached else "cancelled-before-detach")
                raise

        dispatcher._dispatch_tasks.add(asyncio.create_task(dispatch()))
        await asyncio.sleep(0)
        try:
            await dispatcher.stop()
        finally:
            stuck_release.set()  # asyncio.run waits for executor threads on exit

    asyncio.run(run())
    # I1: the worker still running after the bound keeps its lease.
    assert dispatcher.store.expired == [("dispatcher-old", ["turn-quick"])]
    assert order == ["dispatch-cancelled", "expire", "unlock"]


def test_stop_detaches_a_turn_registered_before_its_worker_launched(monkeypatch, registry):
    """chat_streaming registers the execution, then awaits turn_start work, then
    launches the worker. A stop() inside that pause must still reach the turn:
    the worker launched afterwards sees the detach before publishing, and the
    claim keeps its lease because no exit was proven while stop() ran."""
    monkeypatch.setattr(InterBotDeliveryDispatcher, "DETACH_CONFIRM_SECONDS", 0.1)
    order: list[str] = []
    dispatcher = _stopping_dispatcher(order)
    paused = _execution("turn-paused")
    published: list[str] = []

    def worker():
        paused.check_detached()  # agent_bridge checks this before send_command
        published.append("chat.send")

    async def run():
        registry.register(paused)
        await dispatcher.stop()
        assert paused.detach_requested()
        future = paused.launch_worker(asyncio.get_running_loop(), worker)
        with pytest.raises(TurnExecutionDetached):
            await future
        assert paused.confirmed_stopped()

    asyncio.run(run())
    assert published == []
    assert dispatcher.store.expired == [] and order == ["unlock"]


def test_stop_releases_nothing_while_a_detached_worker_is_still_finalizing(monkeypatch, registry):
    monkeypatch.setattr(InterBotDeliveryDispatcher, "DETACH_CONFIRM_SECONDS", 0.1)
    order: list[str] = []
    dispatcher = _stopping_dispatcher(order)
    finalizing = _execution("turn-finalizing")
    release = threading.Event()

    async def run():
        registry.register(finalizing)
        finalizing.launch_worker(asyncio.get_running_loop(), lambda: release.wait(2))
        try:
            await dispatcher.stop()
        finally:
            release.set()  # asyncio.run waits for executor threads on exit

    asyncio.run(run())
    assert dispatcher.store.expired == [] and order == ["unlock"]
