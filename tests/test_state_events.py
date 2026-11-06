"""The state watcher publishes only topics whose fingerprint moved (TASK-2109)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from llm_bawt.service import state_events
from llm_bawt.service.state_events import StateWatcher, state_changed_event


class _Subscriber:
    def __init__(self, fail: bool = False):
        self.events: list[dict] = []
        self.fail = fail

    async def publish_global_config_event(self, event):
        if self.fail:
            return None
        self.events.append(event)
        return "1-0"


def _watcher(readings, subscriber):
    service = SimpleNamespace(_redis_subscriber=subscriber)
    watcher = StateWatcher(service, engine=None)
    seq = iter(readings)
    watcher._read = lambda: next(seq)
    return watcher


def test_event_is_closed_and_sorted():
    assert state_changed_event(["media", "nope", "gpu", "media"]) == {"_type": "state_changed",
                                                                     "topics": ["gpu", "media"]}
    assert state_changed_event(["nope"]) is None


def test_only_moved_topics_publish_and_new_topics_are_baseline():
    sub = _Subscriber()
    watcher = _watcher([{"media": (1,), "gpu": ("a",)},
                        {"media": (1,), "gpu": ("b",), "ops_jobs": (5,)},
                        {"media": (1,), "gpu": ("b",), "ops_jobs": (5,)}], sub)

    async def main():
        watcher._last = watcher._read()
        assert await watcher.check() == {"gpu"}  # ops_jobs only became readable: no baseline
        assert await watcher.check() == set()

    asyncio.run(main())
    assert sub.events == [{"_type": "state_changed", "topics": ["gpu"]}]


def test_marked_topic_publishes_without_a_fingerprint_change():
    sub = _Subscriber()
    watcher = _watcher([{"media": (1,)}, {"media": (1,)}], sub)

    async def main():
        watcher._last = watcher._read()
        watcher.mark("maintenance_jobs")
        assert await watcher.check() == {"maintenance_jobs"}

    asyncio.run(main())
    assert sub.events[-1]["topics"] == ["maintenance_jobs"]


def test_failed_publish_is_retried_on_the_next_pass():
    sub = _Subscriber(fail=True)
    watcher = _watcher([{"media": (1,)}, {"media": (2,)}, {"media": (2,)}], sub)

    async def main():
        watcher._last = watcher._read()
        await watcher.check()  # moved, but the stream was down
        sub.fail = False
        assert await watcher.check() == {"media"}

    asyncio.run(main())
    assert sub.events == [{"_type": "state_changed", "topics": ["media"]}]


def test_failed_read_keeps_the_baseline():
    sub = _Subscriber()
    calls = iter([{"media": (1,)}, RuntimeError("db down"), {"media": (2,)}])

    def read():
        value = next(calls)
        if isinstance(value, Exception):
            raise value
        return value

    watcher = StateWatcher(SimpleNamespace(_redis_subscriber=sub), engine=None)
    watcher._read = read

    async def main():
        watcher._last = watcher._read()
        assert await watcher.check() == set()
        assert await watcher.check() == {"media"}

    asyncio.run(main())


def test_run_coalesces_a_burst_of_wakes_into_one_pass():
    sub = _Subscriber()
    state = {"media": (1,)}

    async def main():
        hub = state_events.commit_wake.WakeHub()
        service = SimpleNamespace(_redis_subscriber=sub, _commit_wake_hub=hub)
        watcher = StateWatcher(service, engine=None, min_interval=0.2)
        watcher._read = lambda: dict(state)
        task = asyncio.create_task(watcher.run())
        while watcher._event is None:
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.25)  # past the first interval
        state["media"] = (2,)
        for _ in range(5):
            hub.fire()
        await asyncio.sleep(0.05)
        watcher.nudge()
        for _ in range(100):
            if sub.events:
                break
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.3)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return watcher.passes

    passes = asyncio.run(main())
    assert sub.events == [{"_type": "state_changed", "topics": ["media"]}]
    assert passes <= 2


def test_helpers_are_noops_without_a_watcher():
    service = SimpleNamespace()
    state_events.nudge(service)
    state_events.mark(service, "media")
    state_events.nudge(None)


class _BotSubscriber(_Subscriber):
    def __init__(self):
        super().__init__()
        self.tool_events: list[tuple] = []

    async def publish_tool_event(self, bot_id, user_id, event):
        self.tool_events.append((bot_id, user_id, event))


def test_fanout_rows_publish_deliveries_and_other_rows_do_not(monkeypatch):
    from llm_bawt.service import inter_bot_delivery_events
    from llm_bawt.service.inter_bot_dispatcher import InterBotDeliveryDispatcher

    async def lifecycle(engine, record, *, default_user):
        return {"_type": "inter_bot_delivery", "bot_id": record.sender_bot_id, "user_id": default_user}

    monkeypatch.setattr(inter_bot_delivery_events, "lifecycle_event", lifecycle)
    sub = _BotSubscriber()
    me = SimpleNamespace(service=SimpleNamespace(_redis_subscriber=sub, config=SimpleNamespace(DEFAULT_USER="u")),
                         store=SimpleNamespace(engine=None))

    def row(author, metadata):
        return SimpleNamespace(sender_bot_id="unknown", target_bot_id="al", author_entity_type=author,
                               author_entity_id="u", metadata=metadata, status="DELIVERED", user_message_id=None)

    asyncio.run(InterBotDeliveryDispatcher._emit(me, row("user", {"batch_id": "b1"})))
    asyncio.run(InterBotDeliveryDispatcher._emit(me, row("user", {})))
    asyncio.run(InterBotDeliveryDispatcher._emit(me, row("bot", {"batch_id": "b2"})))
    assert len(sub.tool_events) == 3
    assert sub.events == [{"_type": "state_changed", "topics": ["deliveries"]}]
