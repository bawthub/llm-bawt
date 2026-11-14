"""Real private PG/Redis recovery across crashes and competing API followers."""
from __future__ import annotations

import asyncio
import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import redis.asyncio as aioredis

from agent_bridge.events import AgentEventKind
from agent_bridge.publisher import RUN_STREAM_PREFIX, UNIFIED_EVENTS_PREFIX
from llm_bawt.service import turn_terminal_recovery as recovery
from llm_bawt.service.turn_terminal_projection import terminal_message_rows
from llm_bawt.service.turn_terminal_protocol import TurnTerminalProtocol
from tests.session_lifecycle_pg_fixtures import rig  # noqa: F401
from tests.test_terminal_projection import history_rows
from tests.test_terminal_publish import private_redis  # noqa: F401
from tests.test_terminal_replay import done, event
from tests.test_turn_terminal_protocol import expire, insert_turn, read

pytestmark = [pytest.mark.integration, pytest.mark.offline_integration]


@pytest.fixture
def pipeline(rig, private_redis, monkeypatch):  # noqa: F811
    from llm_bawt.service.turn_logs import TurnLogStore
    from tests.pg_isolation import isolated_pg_store

    protocol = TurnTerminalProtocol(isolated_pg_store(TurnLogStore))
    from llm_bawt.approval_models import ToolApprovalRequest
    from llm_bawt.service.chat_pending_questions import PendingQuestion
    from llm_bawt.service.run_accepts import ensure_accept_schema

    PendingQuestion.__table__.create(protocol.engine, checkfirst=True)
    ToolApprovalRequest.__table__.create(protocol.engine, checkfirst=True)
    ensure_accept_schema(protocol.engine)
    rows = terminal_message_rows(protocol.engine, rig.bot)
    identity = {"request_id": uuid.uuid4().hex, "session_key": f"{rig.bot}:{rig.user}",
                "backend": "claude-code", "boot_id": "boot"}
    turn = insert_turn(protocol.engine, bot_id=rig.bot, user_id=rig.user,
                       history_session_id=rig.session_id, agent_request_id=identity["request_id"],
                       agent_session_key=identity["session_key"], agent_backend=identity["backend"],
                       response_text="durable prefix")
    effects = Mock()
    monkeypatch.setattr("llm_bawt.service.turn_terminal_effects.record_terminal_effects", effects)
    abort = AsyncMock(side_effect=AssertionError("complete retained run must not abort"))
    monkeypatch.setattr(recovery, "BridgeRunProbe", lambda *a, **kw: SimpleNamespace(abort=abort))
    stream = f"{UNIFIED_EVENTS_PREFIX}{rig.bot}:{rig.user}"

    def run(scenario):
        async def wrapper():
            client = aioredis.Redis(unix_socket_path=private_redis, decode_responses=True,
                                   socket_timeout=5, socket_connect_timeout=5)
            key = RUN_STREAM_PREFIX + identity["request_id"]
            try:
                await client.xadd(key, {"payload": json.dumps(event(
                    session_key=identity["session_key"], kind=AgentEventKind.ASSISTANT_DONE,
                    text="complete retained answer"))})
                await client.xadd(key, done(identity))
                subscriber = SimpleNamespace(_redis=client, _pub_redis=client)
                await scenario(client, subscriber, key)
            finally:
                await client.aclose()
        asyncio.run(wrapper())

    return SimpleNamespace(protocol=protocol, rows=rows, turn=turn, run=run,
                           effects=effects, abort=abort, stream=stream)


def test_observe_commit_crash_then_projection_without_observing_again(pipeline, monkeypatch):
    h = pipeline

    async def scenario(client, subscriber, key):
        source_events = await client.xrange(key)
        with monkeypatch.context() as patch:
            patch.setattr(recovery, "project_committed_terminal", AsyncMock(side_effect=ConnectionError("app died")))
            with pytest.raises(ConnectionError, match="app died"):
                await recovery.recover_terminal_turn(h.protocol, h.turn, subscriber)
        frozen = read(h.protocol.engine, h.turn).terminal_snapshot_json
        assert json.loads(frozen)["response_text"] == "complete retained answer"
        assert history_rows(h.rows) == [] and await client.xlen(h.stream) == 0
        h.effects.assert_called_once()
        with monkeypatch.context() as patch:
            forbidden = AsyncMock(side_effect=AssertionError("committed retry must not observe"))
            patch.setattr(recovery, "observe_bound_run", forbidden)
            await recovery.recover_terminal_turn(h.protocol, h.turn, subscriber)
            forbidden.assert_not_awaited()
        assert read(h.protocol.engine, h.turn).terminal_snapshot_json == frozen
        assert read(h.protocol.engine, h.turn).terminal_protocol_state == "published"
        assert len(history_rows(h.rows)) == 1
        assert history_rows(h.rows)[0]["content"] == "complete retained answer"
        events = await client.xrange(h.stream)
        assert len(events) == 1
        assert json.loads(events[0][1]["payload"]) == json.loads(frozen)["completion"]
        assert await client.xrange(key) == source_events
        h.effects.assert_called_once()
        h.abort.assert_not_awaited()

    h.run(scenario)


def test_competing_recovery_cannot_observe_or_commit_healthy_follower(pipeline, monkeypatch):
    h = pipeline

    async def scenario(client, subscriber, key):
        entered, release = asyncio.Event(), asyncio.Event()
        original = recovery.observe_bound_run

        async def paused(*args, **kwargs):
            entered.set()
            await release.wait()
            return await original(*args, **kwargs)

        observer = AsyncMock(side_effect=paused)
        monkeypatch.setattr(recovery, "observe_bound_run", observer)
        first = asyncio.create_task(recovery.recover_terminal_turn(h.protocol, h.turn, subscriber))
        try:
            await asyncio.wait_for(entered.wait(), timeout=10)
            assert await recovery.recover_terminal_turn(h.protocol, h.turn, subscriber) is None
            assert observer.await_count == 1
            h.effects.assert_not_called()
        finally:
            release.set()
            result = await first
        assert result.won and observer.await_count == 1
        h.effects.assert_called_once()
        assert len(history_rows(h.rows)) == 1
        assert await client.xlen(h.stream) == 1

    h.run(scenario)


def test_recovery_redis_ack_loss_is_retried_without_history_or_completion_duplicates(pipeline, monkeypatch):
    h = pipeline

    async def scenario(client, subscriber, key):
        with monkeypatch.context() as patch:
            patch.setattr(h.protocol, "record_publication", Mock(side_effect=ConnectionError("ACK lost")))
            with pytest.raises(ConnectionError, match="ACK lost"):
                await recovery.recover_terminal_turn(h.protocol, h.turn, subscriber)
        original_history = history_rows(h.rows)
        original_events = await client.xrange(h.stream)
        assert len(original_history) == len(original_events) == 1
        assert read(h.protocol.engine, h.turn).terminal_protocol_state == "projected"
        expire(h.protocol.engine, h.turn)
        await recovery.recover_terminal_turn(h.protocol, h.turn, subscriber)
        assert history_rows(h.rows) == original_history
        assert await client.xrange(h.stream) == original_events
        assert read(h.protocol.engine, h.turn).terminal_published_stream_id == original_events[0][0]
        h.effects.assert_called_once()

    h.run(scenario)
