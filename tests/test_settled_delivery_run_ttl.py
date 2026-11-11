"""TASK-2191: settled deliveries cap their run streams; open ones keep the bridge TTL."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_bridge.subscriber import RedisSubscriber
from llm_bawt.service import inter_bot_delivery_events as events
from llm_bawt.service.inter_bot_dispatcher import (
    SETTLED_RUN_STREAM_TTL_S,
    InterBotDeliveryDispatcher,
    delivery_run_request_ids,
)


def _dispatcher(subscriber):
    dispatcher = object.__new__(InterBotDeliveryDispatcher)
    dispatcher.service = SimpleNamespace(
        _redis_subscriber=subscriber, config=SimpleNamespace(DEFAULT_USER="nick"),
    )
    dispatcher.store = SimpleNamespace(engine=None)
    return dispatcher


def _record(status):
    return SimpleNamespace(id="delivery-abc", status=status,
                           author_entity_type="bot", metadata={})


@pytest.fixture(autouse=True)
def _quiet_events(monkeypatch):
    monkeypatch.setattr(events, "lifecycle_event",
                        AsyncMock(return_value={"bot_id": "b", "user_id": "u"}))
    monkeypatch.setattr(events, "maintenance_steer_event", lambda record: None)


def test_request_ids_cover_first_run_and_recovery():
    assert delivery_run_request_ids("delivery-abc") == [
        "req_delivery_abc", "req_delivery_abc_recovery_1",
    ]


@pytest.mark.parametrize("status", ["DELIVERED", "FAILED", "CANCELLED"])
def test_settled_delivery_shortens_its_streams(status):
    subscriber = SimpleNamespace(shorten_run_streams=AsyncMock(), publish_tool_event=AsyncMock())
    record = _record(status)
    asyncio.run(_dispatcher(subscriber)._emit(record))
    subscriber.shorten_run_streams.assert_awaited_once_with(
        ["req_delivery_abc", "req_delivery_abc_recovery_1"], SETTLED_RUN_STREAM_TTL_S,
    )


@pytest.mark.parametrize("status", ["QUEUED", "DISPATCHING", "ACCEPTED"])
def test_open_delivery_keeps_its_streams(status):
    subscriber = SimpleNamespace(shorten_run_streams=AsyncMock(), publish_tool_event=AsyncMock())
    asyncio.run(_dispatcher(subscriber)._emit(_record(status)))
    subscriber.shorten_run_streams.assert_not_awaited()


def test_shorten_failure_still_emits_lifecycle():
    subscriber = SimpleNamespace(
        shorten_run_streams=AsyncMock(side_effect=ConnectionError("down")),
        publish_tool_event=AsyncMock(),
    )
    asyncio.run(_dispatcher(subscriber)._emit(_record("DELIVERED")))
    subscriber.publish_tool_event.assert_awaited()


def test_subscriber_only_ever_shortens():
    pipe = MagicMock()
    pipe.execute = AsyncMock()
    subscriber = object.__new__(RedisSubscriber)
    subscriber._pub_redis = SimpleNamespace(pipeline=lambda transaction: pipe)
    asyncio.run(subscriber.shorten_run_streams(["r1", "r2"], 86400))
    assert [c.args for c in pipe.expire.call_args_list] == [
        ("agent:run:r1", 86400), ("agent:run:r2", 86400),
    ]
    assert all(c.kwargs == {"lt": True} for c in pipe.expire.call_args_list)
