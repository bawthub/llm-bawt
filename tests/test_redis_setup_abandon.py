"""A Redis transport whose setup fails after connect() is unpublished and closed (TASK-1265).

Startup published the subscriber globally and on ``service`` before the rest
of setup ran. A later failure only dropped the local reference, so backends
kept using a subscriber that shutdown never closed.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from llm_bawt.agent_backends import agent_bridge
from llm_bawt.service import api


class _Subscriber:
    def __init__(self, *, close_raises: bool = False):
        self.closed = 0
        self.close_raises = close_raises

    async def close(self) -> None:
        self.closed += 1
        if self.close_raises:
            raise RuntimeError("close failed")


@pytest.fixture(autouse=True)
def _restore_global():
    before = agent_bridge.get_agent_subscriber()
    yield
    agent_bridge.set_agent_subscriber(before)


@pytest.mark.asyncio
async def test_failed_setup_unpublishes_stops_drain_and_closes():
    sub = _Subscriber()
    drain = asyncio.create_task(asyncio.sleep(3600))
    service = SimpleNamespace(_redis_subscriber=sub, _tool_persist_task=drain)
    agent_bridge.set_agent_subscriber(sub)

    await api.abandon_redis_subscriber(service, sub)

    assert agent_bridge.get_agent_subscriber() is None
    assert service._redis_subscriber is None
    assert service._tool_persist_task is None
    assert drain.cancelled()
    assert sub.closed == 1


@pytest.mark.asyncio
async def test_failure_before_publication_still_closes_and_leaves_others_alone():
    sub, other = _Subscriber(), _Subscriber()
    service = SimpleNamespace(_redis_subscriber=other)
    agent_bridge.set_agent_subscriber(other)

    await api.abandon_redis_subscriber(service, sub)

    assert agent_bridge.get_agent_subscriber() is other
    assert service._redis_subscriber is other
    assert sub.closed == 1 and other.closed == 0


@pytest.mark.asyncio
async def test_a_failing_close_is_logged_not_raised():
    sub = _Subscriber(close_raises=True)
    service = SimpleNamespace(_redis_subscriber=sub)
    agent_bridge.set_agent_subscriber(sub)

    await api.abandon_redis_subscriber(service, sub)

    assert agent_bridge.get_agent_subscriber() is None
    assert sub.closed == 1


@pytest.mark.asyncio
async def test_no_subscriber_is_a_no_op():
    await api.abandon_redis_subscriber(SimpleNamespace(), None)
