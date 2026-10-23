"""Unit tests for shared agent-bridge transport components.

Tests EventIngestPipeline (openclaw-flavored), EventStore, FanoutHub,
and SessionBridge — the parts that all bridge backends rely on.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_bridge.session_queue import SessionQueue
from claude_code_bridge.active_run import ClaudeActiveRun
from llm_bawt.integrations.agent_bridge_events import (
    AgentEvent,
    AgentEventKind,
    synthesize_event_id,
)
from llm_bawt.integrations.openclaw_ingest import EventIngestPipeline
from tests.test_session_search_pg import repository as pg_repository  # noqa: F401 (pytest fixture registration)


# ---------------------------------------------------------------------------
# SessionQueue tests
# ---------------------------------------------------------------------------


class TestSessionQueue:
    def test_active_client_set_get_pop(self):
        queue = SessionQueue()
        client = object()

        queue.set_active_client("main", client)

        assert queue.get_active_client("main") is client
        assert queue.pop_active_client("main") is client
        assert queue.get_active_client("main") is None

    def test_active_client_pop_is_idempotent(self):
        queue = SessionQueue()
        client = object()

        queue.set_active_client("main", client)

        assert queue.pop_active_client("main") is client
        assert queue.pop_active_client("main") is None


class TestClaudeActiveRun:
    @pytest.mark.anyio
    async def test_steer_interrupts_then_queries_same_client(self):
        client = AsyncMock()
        run = ClaudeActiveRun(client=client, request_id="req-1")

        # TASK-1300: the replacement is app-rendered registry text, sent as is.
        framed = "[app-rendered chat.steer.interrupt_replacement]\nchange direction"
        await run.steer("change direction", origin="user", framed_message=framed)

        client.interrupt.assert_awaited_once_with()
        assert client.query.await_args.args[0] == framed
        interrupted = MagicMock(subtype="error_during_execution")
        assert run.consume_replaced_result(interrupted) is True
        assert run.consume_replaced_result(interrupted) is False

    @pytest.mark.anyio
    async def test_steer_now_does_not_wait_for_blocked_sdk_message_reader(self):
        """Bridge control can advance while its model-message reader is silent.

        This does not establish whether the live CLI acknowledges interrupt
        while its *own* upstream HTTP read is blocked; that requires a live probe.
        """
        entered = asyncio.Event()
        release = asyncio.Event()

        async def silent_reader():
            entered.set()
            await release.wait()
            yield MagicMock()

        client = AsyncMock()
        client.receive_messages = silent_reader
        run = ClaudeActiveRun(client=client, request_id="req-1", _steer_timeout_s=0.2)
        reader = asyncio.create_task(anext(run.messages()))
        try:
            await asyncio.wait_for(entered.wait(), timeout=0.2)
            await asyncio.wait_for(run.steer("resume unfinished work"), timeout=0.2)
            client.interrupt.assert_awaited_once_with()
            client.query.assert_awaited_once()
            assert not reader.done()
        finally:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)

    @pytest.mark.anyio
    async def test_steer_failure_does_not_hide_next_error_result(self):
        client = AsyncMock()
        client.query.side_effect = RuntimeError("closed")
        run = ClaudeActiveRun(client=client, request_id="req-1")

        with pytest.raises(RuntimeError, match="closed"):
            await run.steer("change direction")

        client.disconnect.assert_awaited_once_with()
        interrupted = MagicMock(subtype="error_during_execution")
        assert run.consume_replaced_result(interrupted) is False

    @pytest.mark.anyio
    async def test_steer_returns_no_active_run_when_turn_finishes_during_interrupt(self):
        interrupt_started = asyncio.Event()
        release_interrupt = asyncio.Event()

        async def hanging_interrupt():
            interrupt_started.set()
            await release_interrupt.wait()

        client = AsyncMock()
        client.interrupt.side_effect = hanging_interrupt
        run = ClaudeActiveRun(client=client, request_id="req-1")

        steer_task = asyncio.create_task(run.steer("change direction"))
        await interrupt_started.wait()
        run.mark_completed()

        with pytest.raises(RuntimeError, match="no_active_run"):
            await steer_task

        client.query.assert_not_awaited()
        assert run.consume_replaced_result(MagicMock(subtype="success")) is False

    @pytest.mark.anyio
    async def test_steer_rejects_already_completed_run_without_interrupting(self):
        client = AsyncMock()
        run = ClaudeActiveRun(client=client, request_id="req-1")
        run.mark_completed()

        with pytest.raises(RuntimeError, match="no_active_run"):
            await run.steer("change direction")

        client.interrupt.assert_not_awaited()
        client.query.assert_not_awaited()

    @pytest.mark.anyio
    async def test_steer_bounds_unacknowledged_interrupt_before_rpc_deadline(self):
        interrupt_started = asyncio.Event()
        release_interrupt = asyncio.Event()

        async def hanging_interrupt():
            interrupt_started.set()
            await release_interrupt.wait()

        client = AsyncMock()
        client.interrupt.side_effect = hanging_interrupt
        run = ClaudeActiveRun(
            client=client,
            request_id="req-1",
            _steer_timeout_s=0.01,
        )

        with pytest.raises(RuntimeError, match="steer_interrupt_timeout"):
            await run.steer("change direction")

        assert interrupt_started.is_set()
        client.query.assert_not_awaited()
        assert run.consume_replaced_result(MagicMock(subtype="success")) is False

    @pytest.mark.anyio
    async def test_steer_bounds_replacement_query_before_rpc_deadline(self):
        query_started = asyncio.Event()
        release_query = asyncio.Event()

        async def hanging_query(_message):
            query_started.set()
            await release_query.wait()

        client = AsyncMock()
        client.query.side_effect = hanging_query
        run = ClaudeActiveRun(
            client=client,
            request_id="req-1",
            _steer_timeout_s=0.01,
        )

        with pytest.raises(RuntimeError, match="steer_query_timeout"):
            await run.steer("change direction")

        assert query_started.is_set()
        client.disconnect.assert_awaited_once_with()
        assert run.consume_replaced_result(MagicMock(subtype="success")) is False

    def test_success_result_in_interrupt_race_is_still_replaced(self):
        run = ClaudeActiveRun(client=AsyncMock(), request_id="req-1")
        run._expected_interrupt_results = 1

        assert run.consume_replaced_result(MagicMock(subtype="success")) is True
        assert run.consume_replaced_result(MagicMock(subtype="success")) is False


# ---------------------------------------------------------------------------
# EventIngestPipeline tests
# ---------------------------------------------------------------------------


class TestEventIngestPipeline:
    def setup_method(self):
        self.pipeline = EventIngestPipeline()

    def test_event_ingest_maps_delta(self):
        # Real gateway schema: event:"agent" + payload.stream:"assistant".
        raw = {
            "type": "event",
            "session_key": "main",
            "event_id": "evt_001",
            "event": "agent",
            "payload": {"stream": "assistant", "data": {"delta": "Hello "}},
        }
        event = self.pipeline.parse(raw, "main")
        assert event is not None
        assert event.kind == AgentEventKind.ASSISTANT_DELTA
        assert event.text == "Hello "
        assert event.event_id == "evt_001"
        assert event.session_key == "main"

    def test_event_ingest_maps_tool_start(self):
        # Real gateway schema: payload.stream:"tool", data.phase:"start".
        raw = {
            "type": "event",
            "session_key": "main",
            "event_id": "evt_002",
            "event": "agent",
            "payload": {
                "stream": "tool",
                "data": {
                    "phase": "start",
                    "name": "exec",
                    "arguments": {"command": "ls -la"},
                },
            },
        }
        event = self.pipeline.parse(raw, "main")
        assert event is not None
        assert event.kind == AgentEventKind.TOOL_START
        assert event.tool_name == "exec"
        assert event.tool_arguments == {"command": "ls -la"}

    def test_event_ingest_maps_tool_end(self):
        # Real gateway schema: payload.stream:"tool", data.phase:"end",
        # tool output carried as data.output.
        raw = {
            "type": "event",
            "session_key": "main",
            "event_id": "evt_003",
            "event": "agent",
            "payload": {
                "stream": "tool",
                "data": {
                    "phase": "end",
                    "name": "exec",
                    "output": "file1.txt\nfile2.txt",
                },
            },
        }
        event = self.pipeline.parse(raw, "main")
        assert event is not None
        assert event.kind == AgentEventKind.TOOL_END
        assert event.tool_name == "exec"
        assert event.tool_result == "file1.txt\nfile2.txt"

    def test_event_ingest_maps_run_started(self):
        # Real gateway schema: payload.stream:"lifecycle", data.phase:"start",
        # run id from payload.runId. Lifecycle frames carry NO model field, so
        # production leaves event.model as None.
        raw = {
            "type": "event",
            "session_key": "main",
            "event_id": "evt_004",
            "event": "agent",
            "payload": {"stream": "lifecycle", "data": {"phase": "start"}, "runId": "run_xyz"},
        }
        event = self.pipeline.parse(raw, "main")
        assert event is not None
        assert event.kind == AgentEventKind.RUN_STARTED
        assert event.run_id == "run_xyz"
        # Lifecycle start carries no model in the real contract.
        assert event.model is None

    def test_event_ingest_maps_run_completed(self):
        # Real gateway schema: payload.stream:"lifecycle", data.phase:"end".
        raw = {
            "type": "event",
            "session_key": "main",
            "event_id": "evt_005",
            "event": "agent",
            "payload": {"stream": "lifecycle", "data": {"phase": "end"}, "runId": "run_xyz"},
        }
        event = self.pipeline.parse(raw, "main")
        assert event is not None
        assert event.kind == AgentEventKind.RUN_COMPLETED
        assert event.run_id == "run_xyz"

    def test_event_ingest_drops_pings(self):
        for msg_type in ("ping", "pong", "heartbeat"):
            raw = {"type": msg_type}
            event = self.pipeline.parse(raw, "main")
            assert event is None, f"Expected None for {msg_type}"

    def test_event_ingest_drops_subscribe_acks(self):
        # CONTRACT: production does NOT drop subscribe acks. "subscribed" is not
        # in IngestFilterConfig.drop_msg_types ({"ping","pong","heartbeat"}), and
        # it isn't an "event"/"chat.sent"/"req"/"res" frame, so it falls through
        # to the unknown-frame fallback -> SYSTEM_NOTE (ingest.py ~line 244).
        # ALTERNATIVE: add "subscribed" to drop_msg_types so it returns None — a
        # one-line src change, deferred here because we may not edit production.
        raw = {"type": "subscribed", "session_keys": ["main"]}
        event = self.pipeline.parse(raw, "main")
        assert event is not None
        assert event.kind == AgentEventKind.SYSTEM_NOTE

    def test_event_ingest_unknown_event_passthrough(self):
        # An "event" frame whose event name is neither "agent" nor "chat" and is
        # not in drop_event_names falls through to SYSTEM_NOTE (ingest.py ~213).
        raw = {
            "type": "event",
            "session_key": "main",
            "event_id": "evt_006",
            "event": "some.unknown.type",
            "payload": {"foo": "bar"},
        }
        event = self.pipeline.parse(raw, "main")
        assert event is not None
        assert event.kind == AgentEventKind.SYSTEM_NOTE

    def test_event_ingest_chat_sent(self):
        raw = {
            "type": "chat.sent",
            "session_key": "main",
            "run_id": "run_abc",
            "message_id": "msg_001",
        }
        event = self.pipeline.parse(raw, "main")
        assert event is not None
        assert event.kind == AgentEventKind.USER_MESSAGE
        assert event.origin == "user"
        assert event.run_id == "run_abc"

    def test_event_ingest_error_event(self):
        # Real gateway schema: payload.stream:"error". Production sets
        # text = json.dumps(data), so assert the message via substring rather
        # than equality to the bare message string.
        raw = {
            "type": "event",
            "session_key": "main",
            "event_id": "evt_err",
            "event": "agent",
            "payload": {"stream": "error", "data": {"message": "something went wrong"}},
        }
        event = self.pipeline.parse(raw, "main")
        assert event is not None
        assert event.kind == AgentEventKind.ERROR
        assert "something went wrong" in event.text
        assert json.loads(event.text) == {"message": "something went wrong"}

    def test_event_ingest_chat_message_user(self):
        # Real contract: a user message arrives as a top-level "chat.sent" frame
        # (NOT an event:"chat" frame — those are state-keyed and map to
        # ASSISTANT_DONE/None by state, never by role). chat.sent -> USER_MESSAGE
        # with origin "user" (ingest.py ~224).
        raw = {
            "type": "chat.sent",
            "session_key": "main",
            "event_id": "evt_msg",
            "message": {"role": "user", "content": "hello"},
        }
        event = self.pipeline.parse(raw, "main")
        assert event is not None
        assert event.kind == AgentEventKind.USER_MESSAGE
        assert event.origin == "user"
        assert event.text == "hello"

    def test_event_ingest_chat_message_assistant(self):
        # Real contract: event:"chat" with state:"final" -> ASSISTANT_DONE,
        # text extracted from payload.message.content (ingest.py ~191).
        raw = {
            "type": "event",
            "session_key": "main",
            "event_id": "evt_msg2",
            "event": "chat",
            "payload": {"state": "final", "message": {"role": "assistant", "content": "Hi there!"}},
        }
        event = self.pipeline.parse(raw, "main")
        assert event is not None
        assert event.kind == AgentEventKind.ASSISTANT_DONE
        assert event.text == "Hi there!"

    def test_event_ingest_seq_preserved(self):
        # seq is read from payload.seq (int) first, falling back to raw.seq.
        raw = {
            "type": "event",
            "session_key": "main",
            "event_id": "evt_seq",
            "event": "agent",
            "payload": {"stream": "assistant", "seq": 42, "data": {"delta": "x"}},
        }
        event = self.pipeline.parse(raw, "main")
        assert event is not None
        assert event.seq == 42

    def test_event_ingest_tool_args_string_passthrough(self):
        """CONTRACT: tool `arguments` are passed through VERBATIM.

        Production (ingest.py ~151) does `data.get("arguments")` with no
        json.loads — so a string arg stays a string; auto-parse is NOT
        implemented. ALTERNATIVE: json.loads string args in the tool-start
        branch so consumers always see a dict; deferred (production change).
        """
        raw = {
            "type": "event",
            "session_key": "main",
            "event_id": "evt_tc",
            "event": "agent",
            "payload": {
                "stream": "tool",
                "data": {
                    "phase": "start",
                    "name": "search",
                    "arguments": '{"query": "test"}',
                },
            },
        }
        event = self.pipeline.parse(raw, "main")
        assert event is not None
        assert event.kind == AgentEventKind.TOOL_START
        # Verbatim pass-through: still the raw JSON string, NOT a parsed dict.
        assert event.tool_arguments == '{"query": "test"}'

    def test_synthesize_event_id_deterministic(self):
        id1 = synthesize_event_id("main", "delta", {"text": "hi"}, 1)
        id2 = synthesize_event_id("main", "delta", {"text": "hi"}, 1)
        assert id1 == id2

    def test_synthesize_event_id_differs_with_seq(self):
        id1 = synthesize_event_id("main", "delta", {"text": "hi"}, 1)
        id2 = synthesize_event_id("main", "delta", {"text": "hi"}, 2)
        assert id1 != id2


# ---------------------------------------------------------------------------
# EventStore tests (require PostgreSQL)
# ---------------------------------------------------------------------------

@pytest.fixture
def event_store(pg_repository):  # noqa: F811 (pytest injects imported fixture)
    """Give each test its own schema in the private, networkless PG cluster."""
    from sqlalchemy import create_engine, text
    from llm_bawt.integrations.agent_bridge_store import EventStore, create_agent_event_tables

    schema = f"event_test_{uuid.uuid4().hex}"
    with pg_repository.engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA {schema}"))
    url = pg_repository.engine.url.update_query_dict({"options": f"-csearch_path={schema}"})
    engine = create_engine(url)
    try:
        create_agent_event_tables(engine)
        yield EventStore(engine)
    finally:
        engine.dispose()


def _make_event(
    event_id: str = "evt_test",
    session_key: str = "main",
    kind: AgentEventKind = AgentEventKind.ASSISTANT_DELTA,
    text: str | None = "hello",
    run_id: str | None = "run_001",
    **kwargs,
) -> AgentEvent:
    return AgentEvent(
        event_id=event_id,
        session_key=session_key,
        run_id=run_id,
        kind=kind,
        origin=kwargs.get("origin", "system"),
        text=text,
        raw=kwargs.get("raw", {"test": True}),
        seq=kwargs.get("seq"),
    )


@pytest.mark.integration
class TestEventStore:
    def test_event_store_idempotent(self, event_store):
        """Storing same event_dedupe_key twice returns False second time."""
        event = _make_event(event_id="evt_idem_1")
        assert event_store.store(event) is True
        assert event.db_id is not None

        # Same dedupe key should return False
        event2 = _make_event(event_id="evt_idem_1", text="different")
        assert event_store.store(event2) is False

    def test_event_store_replay(self, event_store):
        """get_events(since_id=N) returns only events after N."""
        e1 = _make_event(event_id="evt_r1", text="first")
        e2 = _make_event(event_id="evt_r2", text="second")
        e3 = _make_event(event_id="evt_r3", text="third")

        event_store.store(e1)
        event_store.store(e2)
        event_store.store(e3)

        # Get all events after e1
        events = event_store.get_events("main", since_id=e1.db_id)
        assert len(events) == 2
        assert events[0].text == "second"
        assert events[1].text == "third"

    def test_event_store_assemble_run_text(self, event_store):
        """Deltas for a run reassemble into full text."""
        for i, chunk in enumerate(["Hello ", "world", "!"]):
            event_store.store(_make_event(
                event_id=f"evt_asm_{i}",
                run_id="run_assemble",
                kind=AgentEventKind.ASSISTANT_DELTA,
                text=chunk,
                seq=i,
            ))

        full = event_store.assemble_run_text("run_assemble")
        assert full == "Hello world!"

    def test_event_store_kind_filter(self, event_store):
        """get_events with kinds filter returns only matching kinds."""
        event_store.store(_make_event(event_id="evt_kf1", kind=AgentEventKind.ASSISTANT_DELTA))
        event_store.store(_make_event(event_id="evt_kf2", kind=AgentEventKind.RUN_STARTED, text=None))
        event_store.store(_make_event(event_id="evt_kf3", kind=AgentEventKind.ASSISTANT_DELTA))

        events = event_store.get_events(
            "main", kinds=[AgentEventKind.ASSISTANT_DELTA]
        )
        assert len(events) == 2
        assert all(e.kind == AgentEventKind.ASSISTANT_DELTA for e in events)

    def test_event_store_session_cursor(self, event_store):
        """Session cursor tracks last processed event."""
        assert event_store.get_session_cursor("main") is None

        event_store.update_session_cursor("main", 42)
        assert event_store.get_session_cursor("main") == 42

        event_store.update_session_cursor("main", 99)
        assert event_store.get_session_cursor("main") == 99

    def test_event_store_create_and_complete_run(self, event_store):
        """Run lifecycle: create → store deltas → complete."""
        event_store.create_run("run_lc", "main", model="gpt-5", origin="user")

        # Store some deltas
        for i, chunk in enumerate(["Hi ", "there"]):
            event_store.store(_make_event(
                event_id=f"evt_lc_{i}",
                run_id="run_lc",
                kind=AgentEventKind.ASSISTANT_DELTA,
                text=chunk,
                seq=i,
            ))

        full_text = event_store.assemble_run_text("run_lc")
        event_store.complete_run("run_lc", full_text, [{"name": "exec", "arguments": {}}])

        # Verify via direct query
        from sqlalchemy import text as sql_text
        with event_store._engine.connect() as conn:
            row = conn.execute(
                sql_text("SELECT status, full_text FROM agent_runs WHERE run_id = :rid"),
                {"rid": "run_lc"},
            ).fetchone()
        assert row is not None
        assert row[0] == "completed"
        assert row[1] == "Hi there"

    def test_event_store_ws_state(self, event_store):
        """WS connection state tracking."""
        event_store.update_session_ws_state("main", True)
        from sqlalchemy import text as sql_text
        with event_store._engine.connect() as conn:
            row = conn.execute(
                sql_text("SELECT ws_connected FROM agent_session_state WHERE session_key = :sk"),
                {"sk": "main"},
            ).fetchone()
        assert row is not None
        assert row[0] is True

        event_store.update_session_ws_state("main", False)
        with event_store._engine.connect() as conn:
            row = conn.execute(
                sql_text("SELECT ws_connected FROM agent_session_state WHERE session_key = :sk"),
                {"sk": "main"},
            ).fetchone()
        assert row[0] is False


# ---------------------------------------------------------------------------
# FanoutHub tests
# ---------------------------------------------------------------------------


class TestFanoutHub:
    def test_fanout_live_broadcast(self):
        """Subscriber receives events pushed via broadcast()."""
        from llm_bawt.integrations.agent_bridge_fanout import FanoutHub

        store = MagicMock()
        hub = FanoutHub(store)

        received = []

        async def run():
            sub = hub.subscribe("main")
            # Start consuming in background
            async def consume():
                async for event in sub:
                    received.append(event)
                    if len(received) >= 2:
                        break

            task = asyncio.create_task(consume())

            # Give subscriber time to register
            await asyncio.sleep(0.05)

            # Broadcast events
            e1 = _make_event(event_id="evt_f1", text="one")
            e2 = _make_event(event_id="evt_f2", text="two")
            hub.broadcast(e1)
            hub.broadcast(e2)

            await asyncio.wait_for(task, timeout=2.0)

        asyncio.run(run())
        assert len(received) == 2
        assert received[0].text == "one"
        assert received[1].text == "two"

    def test_fanout_replay_then_live(self):
        """Subscriber with since_event_id gets gap replay then live events."""
        from llm_bawt.integrations.agent_bridge_fanout import FanoutHub

        # Mock store that returns gap events
        store = MagicMock()
        gap_event = _make_event(event_id="evt_gap", text="from-db")
        store.get_events.return_value = [gap_event]

        hub = FanoutHub(store)
        received = []

        async def run():
            sub = hub.subscribe("main", since_event_id=1)

            async def consume():
                async for event in sub:
                    received.append(event)
                    if len(received) >= 2:
                        break

            task = asyncio.create_task(consume())
            await asyncio.sleep(0.05)

            # Broadcast a live event
            live = _make_event(event_id="evt_live", text="live-event")
            hub.broadcast(live)

            await asyncio.wait_for(task, timeout=2.0)

        asyncio.run(run())
        assert len(received) == 2
        assert received[0].text == "from-db"  # Gap replay
        assert received[1].text == "live-event"  # Live
        store.get_events.assert_called_once_with("main", since_id=1)


# ---------------------------------------------------------------------------
# SessionBridge tests
# ---------------------------------------------------------------------------


class TestSessionBridge:
    def test_bridge_full_flow(self):
        """Full flow: events arrive → run state tracked → fanout."""
        from llm_bawt.integrations.openclaw_bridge import SessionBridge
        from llm_bawt.integrations.agent_bridge_fanout import FanoutHub
        from llm_bawt.integrations.openclaw_ingest import EventIngestPipeline

        store = MagicMock()
        store.store.return_value = True
        store.assemble_run_text.return_value = "Hello world"

        # Track db_id assignment
        _next_id = [0]
        def mock_store(event):
            _next_id[0] += 1
            event.db_id = _next_id[0]
            return True
        store.store.side_effect = mock_store

        fanout = FanoutHub(store)
        ingest = EventIngestPipeline()
        ws_client = MagicMock()
        ws_client.subscribed_sessions = {"main"}

        bridge = SessionBridge(ws_client, ingest, store, fanout)

        broadcasted = []
        original_broadcast = fanout.broadcast
        def track_broadcast(event):
            broadcasted.append(event)
            original_broadcast(event)
        fanout.broadcast = track_broadcast

        async def run():
            # Simulate run lifecycle using the real gateway agent schema.
            await bridge._on_raw_event({
                "type": "event", "session_key": "main", "event_id": "evt_1",
                "event": "agent",
                "payload": {"stream": "lifecycle", "data": {"phase": "start"}, "runId": "run_1"},
            })
            await bridge._on_raw_event({
                "type": "event", "session_key": "main", "event_id": "evt_2",
                "event": "agent",
                "payload": {"stream": "assistant", "runId": "run_1", "data": {"delta": "Hello "}},
            })
            await bridge._on_raw_event({
                "type": "event", "session_key": "main", "event_id": "evt_3",
                "event": "agent",
                "payload": {"stream": "assistant", "runId": "run_1", "data": {"delta": "world"}},
            })
            await bridge._on_raw_event({
                "type": "event", "session_key": "main", "event_id": "evt_4",
                "event": "agent",
                "payload": {"stream": "lifecycle", "data": {"phase": "end"}, "runId": "run_1"},
            })

        asyncio.run(run())

        # Verify run was created and completed. Lifecycle start carries no model,
        # so production passes model=None; origin is "system".
        store.create_run.assert_called_once_with("run_1", "main", None, "system")
        store.complete_run.assert_called_once()
        call_args = store.complete_run.call_args
        assert call_args[0][0] == "run_1"
        assert call_args[0][1] == "Hello world"

        # Verify all events were broadcasted
        assert len(broadcasted) == 4

    def test_bridge_background_event_persisted(self):
        """Event with no matching user send still stored in EventStore."""
        from llm_bawt.integrations.openclaw_bridge import SessionBridge
        from llm_bawt.integrations.agent_bridge_fanout import FanoutHub
        from llm_bawt.integrations.openclaw_ingest import EventIngestPipeline

        store = MagicMock()
        _next_id = [0]
        def mock_store(event):
            _next_id[0] += 1
            event.db_id = _next_id[0]
            return True
        store.store.side_effect = mock_store

        fanout = FanoutHub(store)
        ingest = EventIngestPipeline()
        ws_client = MagicMock()
        ws_client.subscribed_sessions = {"main"}

        bridge = SessionBridge(ws_client, ingest, store, fanout)

        async def run():
            await bridge._on_raw_event({
                "type": "event", "session_key": "main", "event_id": "evt_bg",
                "event": "agent",
                "payload": {"stream": "assistant", "data": {"delta": "background message"}},
            })

        asyncio.run(run())

        # Event was stored
        store.store.assert_called_once()
        stored_event = store.store.call_args[0][0]
        assert stored_event.kind == AgentEventKind.ASSISTANT_DELTA
        assert stored_event.text == "background message"

    def test_bridge_duplicate_dropped(self):
        """Duplicate events (store returns False) are not broadcasted."""
        from llm_bawt.integrations.openclaw_bridge import SessionBridge
        from llm_bawt.integrations.agent_bridge_fanout import FanoutHub
        from llm_bawt.integrations.openclaw_ingest import EventIngestPipeline

        store = MagicMock()
        store.store.return_value = False  # duplicate

        fanout = FanoutHub(store)
        broadcasted = []
        fanout.broadcast = lambda e: broadcasted.append(e)

        ingest = EventIngestPipeline()
        ws_client = MagicMock()
        ws_client.subscribed_sessions = {"main"}

        bridge = SessionBridge(ws_client, ingest, store, fanout)

        async def run():
            await bridge._on_raw_event({
                "type": "event", "session_key": "main", "event_id": "evt_dup",
                "event": "agent",
                "payload": {"stream": "assistant", "data": {"delta": "dup"}},
            })

        asyncio.run(run())

        assert len(broadcasted) == 0  # not broadcasted

    def test_bridge_tool_calls_tracked(self):
        """Tool start/end events are tracked in run state."""
        from llm_bawt.integrations.openclaw_bridge import SessionBridge
        from llm_bawt.integrations.agent_bridge_fanout import FanoutHub
        from llm_bawt.integrations.openclaw_ingest import EventIngestPipeline

        store = MagicMock()
        _next_id = [0]
        def mock_store(event):
            _next_id[0] += 1
            event.db_id = _next_id[0]
            return True
        store.store.side_effect = mock_store
        store.assemble_run_text.return_value = ""

        fanout = FanoutHub(store)
        ingest = EventIngestPipeline()
        ws_client = MagicMock()
        ws_client.subscribed_sessions = {"main"}

        bridge = SessionBridge(ws_client, ingest, store, fanout)

        async def run():
            # Run started
            await bridge._on_raw_event({
                "type": "event", "session_key": "main", "event_id": "evt_t1",
                "event": "agent",
                "payload": {"stream": "lifecycle", "data": {"phase": "start"}, "runId": "run_tools"},
            })
            # Tool call (start). runId required so the bridge buckets the call
            # under the active run's tool-call list.
            await bridge._on_raw_event({
                "type": "event", "session_key": "main", "event_id": "evt_t2",
                "event": "agent",
                "payload": {
                    "stream": "tool", "runId": "run_tools",
                    "data": {"phase": "start", "name": "exec", "arguments": {"command": "ls"}},
                },
            })
            # Tool result (end)
            await bridge._on_raw_event({
                "type": "event", "session_key": "main", "event_id": "evt_t3",
                "event": "agent",
                "payload": {
                    "stream": "tool", "runId": "run_tools",
                    "data": {"phase": "end", "name": "exec", "output": "file.txt"},
                },
            })
            # Run completed
            await bridge._on_raw_event({
                "type": "event", "session_key": "main", "event_id": "evt_t4",
                "event": "agent",
                "payload": {"stream": "lifecycle", "data": {"phase": "end"}, "runId": "run_tools"},
            })

        asyncio.run(run())

        store.complete_run.assert_called_once()
        call_args = store.complete_run.call_args
        tool_calls = call_args[0][2]
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "exec"
        assert tool_calls[0]["result"] == "file.txt"
