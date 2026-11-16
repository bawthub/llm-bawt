"""Provider-free inbox inventory, reconnect and wake-hint ownership."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from agent_bridge.publisher import INBOX_WAKE_CHANNEL
from agent_bridge.session_queue import SessionQueue
from claude_code_bridge.inbox_consumer import BotInbox, InboxConsumer


def listener_rig(monkeypatch):
    messages = asyncio.Queue()
    subscribed = asyncio.Event()

    async def subscribe(channel):
        assert channel == INBOX_WAKE_CHANNEL
        subscribed.set()

    async def listen():
        while True:
            yield await messages.get()

    pubsub = SimpleNamespace(subscribe=subscribe, listen=listen, aclose=AsyncMock())
    client = SimpleNamespace(pubsub=lambda: pubsub, aclose=AsyncMock())
    factory = Mock(return_value=client)
    monkeypatch.setattr("redis.asyncio.from_url", factory)
    consumer = InboxConsumer(AsyncMock(), SessionQueue(), owner="fixture")
    consumer._run = AsyncMock()
    return consumer, messages, subscribed, pubsub, client, factory


@pytest.mark.anyio
async def test_inventory_uses_keyset_pages_and_only_creates_runless_repair_actors():
    api = AsyncMock(side_effect=[
        {"bot_ids": ["first"], "after": "first"},
        {"bot_ids": ["last"], "after": None},
    ])
    consumer = InboxConsumer(api, SessionQueue(), owner="fixture")
    consumer._run = AsyncMock()
    try:
        consumer.wake("unknown-hint")
        assert not consumer.bots
        await consumer.resync()
        assert set(consumer.bots) == {"first", "last"}
        assert all(not inbox.runs for inbox in consumer.bots.values())
        assert [call.args[1]["after"] for call in api.await_args_list] == ["", "first"]
    finally:
        await consumer.stop()


@pytest.mark.anyio
async def test_stuck_inventory_cursor_fails_instead_of_hot_looping():
    api = AsyncMock(side_effect=[
        {"bot_ids": [], "after": "stuck"},
        {"bot_ids": [], "after": "stuck"},
    ])
    consumer = InboxConsumer(api, SessionQueue(), owner="fixture")
    with pytest.raises(ValueError, match="cursor did not advance"):
        await consumer.resync()
    assert api.await_count == 2


@pytest.mark.anyio
async def test_subscribe_ack_precedes_inventory_and_auto_resubscribe_resyncs(monkeypatch):
    consumer, messages, subscribed, pubsub, client, factory = listener_rig(monkeypatch)
    passes = asyncio.Queue()

    async def resync():
        assert subscribed.is_set()
        passes.put_nowait(True)

    consumer.resync = resync
    url = "rediss://fixture.invalid:6380/3"
    consumer.start(url)
    try:
        await subscribed.wait()
        assert passes.empty()
        messages.put_nowait({"type": "subscribe"})
        await passes.get()
        # A second acknowledgment on the same connection models redis-py's
        # transparent reconnect/resubscribe, not just an outer listener restart.
        messages.put_nowait({"type": "subscribe"})
        await passes.get()
        assert factory.call_count == 1
        assert factory.call_args.args == (url,)
        assert factory.call_args.kwargs["socket_timeout"] is None
    finally:
        await consumer.stop()
    assert consumer._listener_task.done() and consumer._resync_task.done()
    pubsub.aclose.assert_awaited_once()
    client.aclose.assert_awaited_once()


@pytest.mark.anyio
@pytest.mark.parametrize("cancel_during_retry", [False, True])
async def test_failed_inventory_retries_while_listener_keeps_handling_hints(monkeypatch, cancel_during_retry):
    consumer, messages, subscribed, _, _, factory = listener_rig(monkeypatch)
    retrying, release_retry, repaired, hinted = (asyncio.Event() for _ in range(4))
    consumer.bots["owned"] = BotInbox()
    wake = consumer.wake

    def observe_wake(bot_id, **kwargs):
        wake(bot_id, **kwargs)
        if bot_id == "owned":
            hinted.set()

    consumer.wake = observe_wake
    calls = 0

    async def inventory(path, body):
        nonlocal calls
        assert path == "/inventory"
        calls += 1
        if calls == 1:
            raise TimeoutError("fixture API outage")
        repaired.set()
        return {"bot_ids": ["orphan"], "after": None}

    async def backoff(seconds):
        assert seconds == 2
        retrying.set()
        await release_retry.wait()

    consumer.api = inventory
    monkeypatch.setattr("claude_code_bridge.inbox_consumer.asyncio.sleep", backoff)
    consumer.start("redis://fixture.invalid/0")
    try:
        await subscribed.wait()
        messages.put_nowait({"type": "subscribe"})
        await retrying.wait()
        hinted.clear()
        for data in ["not json", "[]",
                     json.dumps({"backend": "other", "bot_id": "owned"}),
                     json.dumps({"backend": "claude-code", "bot_id": "unknown"}),
                     json.dumps({"backend": "claude-code", "bot_id": "owned"})]:
            messages.put_nowait({"type": "message", "data": data})
        await hinted.wait()
        assert set(consumer.bots) == {"owned"}
        if not cancel_during_retry:
            release_retry.set()
            await repaired.wait()
            # resync runs through the returned page without an intervening await.
            assert set(consumer.bots) == {"owned", "orphan"}
            assert not consumer.bots["orphan"].runs
        assert factory.call_count == 1
    finally:
        await consumer.stop()
    assert calls == (1 if cancel_during_retry else 2)
    assert consumer._listener_task.done() and consumer._resync_task.done()
    assert all(inbox.task is None or inbox.task.done() for inbox in consumer.bots.values())


@pytest.mark.anyio
async def test_publisher_sends_only_backend_and_bot_hint():
    from agent_bridge.subscriber import RedisSubscriber

    subscriber = RedisSubscriber.__new__(RedisSubscriber)
    subscriber._pub_redis = SimpleNamespace(publish=AsyncMock(return_value=3))
    assert await subscriber.publish_inbox_wake("fixture", "claude-code") == 3
    channel, payload = subscriber._pub_redis.publish.call_args.args
    assert channel == INBOX_WAKE_CHANNEL
    assert json.loads(payload) == {"backend": "claude-code", "bot_id": "fixture"}
