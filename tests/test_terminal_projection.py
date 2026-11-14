"""Real private PG/Redis crash windows between durable terminal projections."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from unittest.mock import Mock

import pytest
import redis.asyncio as aioredis
from sqlalchemy import select, update

from agent_bridge.terminal_publish import TerminalPublishStatus
from agent_bridge.publisher import UNIFIED_EVENTS_PREFIX
from llm_bawt.service.turn_log_schema import TurnLog
from llm_bawt.service.turn_terminal_projection import project_committed_terminal, terminal_message_rows
from llm_bawt.service.turn_terminal_protocol import FollowerFenceLost, TurnTerminalProtocol
from tests.session_lifecycle_pg_fixtures import rig
from tests.test_terminal_publish import private_redis
from tests.test_turn_terminal_protocol import expire, insert_turn, read

__all__ = ["private_redis", "rig"]
pytestmark = [pytest.mark.integration, pytest.mark.offline_integration]


@pytest.fixture
def projection_case(rig, private_redis):
    from llm_bawt.service.turn_logs import TurnLogStore
    from tests.pg_isolation import isolated_pg_store

    protocol = TurnTerminalProtocol(isolated_pg_store(TurnLogStore))
    assert protocol.engine is rig.engine
    rows = terminal_message_rows(protocol.engine, rig.bot)
    turn = insert_turn(
        protocol.engine, bot_id=rig.bot, user_id=rig.user, history_session_id=rig.session_id,
    )
    lease = protocol.claim(turn)
    frozen = protocol.commit(
        lease, status="ok", end_reason="stop", response_text="immutable final answer",
        tool_context="[Read]\nimmutable tool result", reasoning="immutable reasoning",
        tts_tail={"delta": "final speech", "tts_offset": 10},
    ).snapshot
    completion = frozen["completion"]
    stream = f"{UNIFIED_EVENTS_PREFIX}{completion['bot_id']}:{completion['user_id']}"

    def run(scenario):
        async def wrapper():
            client = aioredis.Redis(
                unix_socket_path=private_redis, decode_responses=True,
                socket_timeout=5, socket_connect_timeout=5,
            )
            try:
                return await scenario(client)
            finally:
                await client.aclose()

        return asyncio.run(wrapper())

    # rig owns the real private partition/session and removes them at teardown.
    return protocol, rows, turn, frozen, stream, run


def history_rows(rows):
    """Projected rows only; preserve rig's existing user message unchanged."""
    with rows.engine.connect() as connection:
        return list(connection.execute(select(rows.table).where(
            rows.table.c.role.in_(("system", "assistant")),
        ).order_by(rows.table.c.timestamp)).mappings())


def test_committed_snapshot_projects_history_then_ordered_completion(projection_case):
    protocol, rows, turn, frozen, stream, run = projection_case

    async def scenario(client):
        assert history_rows(rows) == []
        assert await client.xlen(stream) == 0
        result = await project_committed_terminal(protocol, turn, client)
        assert result.status is TerminalPublishStatus.PUBLISHED
        events = await client.xrange(stream)
        assert [json.loads(fields["payload"])["_type"] for _, fields in events] == [
            "tts_delta", "turn_complete",
        ]
        assert json.loads(events[-1][1]["payload"]) == frozen["completion"]
        assert result.completion_stream_id == events[-1][0]
        assert await project_committed_terminal(protocol, turn, client, rows_for_bot=Mock()) is None

    run(scenario)
    stored = history_rows(rows)
    assert len(stored) == 2
    assert {item["session_id"] for item in stored} == {frozen["history_session_id"]}
    assert stored[1]["content"] == "immutable final answer"
    row = read(protocol.engine, turn)
    assert row.terminal_protocol_state == "published"
    assert row.terminal_history_projected_at and row.terminal_published_stream_id


def test_crash_after_history_before_db_receipt_retries_without_duplicate_rows(projection_case, monkeypatch):
    protocol, rows, turn, frozen, stream, run = projection_case

    async def scenario(client):
        with monkeypatch.context() as patch:
            patch.setattr(protocol, "mark_history_projected", Mock(side_effect=ConnectionError("lost DB write")))
            with pytest.raises(ConnectionError, match="lost DB write"):
                await project_committed_terminal(protocol, turn, client, rows_for_bot=lambda bot: rows)
        original = history_rows(rows)
        assert len(original) == 2
        assert read(protocol.engine, turn).terminal_history_projected_at is None
        assert await client.xlen(stream) == 0
        expire(protocol.engine, turn)
        await project_committed_terminal(protocol, turn, client, rows_for_bot=lambda bot: rows)
        assert history_rows(rows) == original
        assert await client.xlen(stream) == 2

    run(scenario)
    assert json.loads(read(protocol.engine, turn).terminal_snapshot_json) == frozen


def test_crash_after_redis_before_db_ack_reuses_original_receipt(projection_case, monkeypatch):
    protocol, rows, turn, frozen, stream, run = projection_case

    async def scenario(client):
        with monkeypatch.context() as patch:
            patch.setattr(protocol, "record_publication", Mock(side_effect=ConnectionError("lost ACK write")))
            with pytest.raises(ConnectionError, match="lost ACK write"):
                await project_committed_terminal(protocol, turn, client, rows_for_bot=lambda bot: rows)
        events = await client.xrange(stream)
        assert len(events) == 2
        original_history = history_rows(rows)
        assert read(protocol.engine, turn).terminal_protocol_state == "projected"
        expire(protocol.engine, turn)
        forbidden_factory = Mock(side_effect=AssertionError("history already durably projected"))
        result = await project_committed_terminal(protocol, turn, client, rows_for_bot=forbidden_factory)
        assert result.status is TerminalPublishStatus.ALREADY_PUBLISHED
        assert result.completion_stream_id == events[-1][0]
        assert await client.xrange(stream) == events
        assert history_rows(rows) == original_history
        forbidden_factory.assert_not_called()

    run(scenario)
    assert read(protocol.engine, turn).terminal_protocol_state == "published"


def test_unbound_setup_failure_projects_without_opening_any_history_store(projection_case):
    protocol, rows, _, frozen, stream, run = projection_case
    completion = frozen["completion"]
    turn = insert_turn(
        protocol.engine, unbound=True, bot_id=completion["bot_id"], user_id=completion["user_id"],
    )
    committed = protocol.commit(
        protocol.claim(turn), status="error", end_reason="setup_error", error_text="setup failed",
    ).snapshot

    async def scenario(client):
        forbidden_factory = Mock(side_effect=AssertionError("must not resolve a session/store"))
        result = await project_committed_terminal(
            protocol, turn, client, rows_for_bot=forbidden_factory,
        )
        assert result.status is TerminalPublishStatus.PUBLISHED
        assert json.loads((await client.xrange(stream))[0][1]["payload"]) == committed["completion"]
        forbidden_factory.assert_not_called()

    run(scenario)
    assert history_rows(rows) == []
    row = read(protocol.engine, turn)
    assert row.history_session_id is None
    assert row.terminal_history_projected_at
    assert row.terminal_protocol_state == "published"


def test_mismatched_history_engine_leaves_durable_retry_without_publication(projection_case):
    protocol, rows, turn, frozen, stream, run = projection_case

    async def scenario(client):
        wrong_rows = Mock(engine=object())
        with pytest.raises(ValueError, match="canonical engine"):
            await project_committed_terminal(protocol, turn, client, rows_for_bot=lambda bot: wrong_rows)
        wrong_rows.insert_terminal.assert_not_called()
        assert history_rows(rows) == []
        assert await client.xlen(stream) == 0
        assert read(protocol.engine, turn).terminal_history_projected_at is None
        expire(protocol.engine, turn)
        result = await project_committed_terminal(protocol, turn, client)
        assert result.status is TerminalPublishStatus.PUBLISHED

    run(scenario)
    assert len(history_rows(rows)) == 2


def test_publication_expiry_still_finishes_history_before_closing_work(projection_case):
    protocol, rows, turn, frozen, stream, run = projection_case
    with protocol.engine.begin() as connection:
        connection.execute(update(TurnLog).where(TurnLog.id == turn).values(
            terminal_publication_deadline=datetime(2000, 1, 1, tzinfo=timezone.utc),
        ))

    async def scenario(client):
        result = await project_committed_terminal(protocol, turn, client, rows_for_bot=lambda bot: rows)
        assert result.status is TerminalPublishStatus.EXPIRED
        assert result.completion_stream_id is None
        assert await client.xlen(stream) == 0
        assert await project_committed_terminal(protocol, turn, client, rows_for_bot=Mock()) is None

    run(scenario)
    assert len(history_rows(rows)) == 2
    row = read(protocol.engine, turn)
    assert row.terminal_protocol_state == "publication_expired"
    assert row.terminal_history_projected_at
    assert turn not in protocol.recoverable_ids()
    assert json.loads(row.terminal_snapshot_json) == frozen


def test_lease_loss_after_redis_ack_cannot_record_stale_owner_receipt(projection_case, monkeypatch):
    protocol, rows, turn, frozen, stream, run = projection_case
    original_record = protocol.record_publication

    def stale_record(lease, *, result):
        expire(protocol.engine, turn)
        return original_record(lease, result=result)

    async def scenario(client):
        with monkeypatch.context() as patch:
            patch.setattr(protocol, "record_publication", stale_record)
            with pytest.raises(FollowerFenceLost):
                await project_committed_terminal(protocol, turn, client, rows_for_bot=lambda bot: rows)
        original_events = await client.xrange(stream)
        assert len(original_events) == 2
        result = await project_committed_terminal(protocol, turn, client, rows_for_bot=Mock())
        assert result.status is TerminalPublishStatus.ALREADY_PUBLISHED
        assert await client.xrange(stream) == original_events

    run(scenario)
    assert read(protocol.engine, turn).terminal_protocol_state == "published"
