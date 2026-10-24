"""TASK-1396: POST /v1/events/task-changed relays BawtHub task writes to open UIs.

One narrow, content-free event on the existing global unified stream; every
other shape is refused before anything is published. The caller (BawtHub,
after its task transaction commits) drops a 503 rather than retrying.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_bridge.publisher import GLOBAL_CONFIG_STREAM
from agent_bridge.subscriber import RedisSubscriber
from llm_bawt.service.routes import all_routers, task_events

URL = "/v1/events/task-changed"
IDS = ["cmvmvxyf300150mlfzf90b7u6", "cmvmjoyo800gx0mnt925jw6gz"]


def _client(monkeypatch: pytest.MonkeyPatch, subscriber: object | None) -> TestClient:
    service = SimpleNamespace(_redis_subscriber=subscriber)
    monkeypatch.setattr(task_events, "get_service", lambda: service)
    app = FastAPI()
    app.include_router(task_events.router)
    return TestClient(app)


def _subscriber(stream_id: str | None = "1-0") -> SimpleNamespace:
    return SimpleNamespace(publish_global_config_event=AsyncMock(return_value=stream_id))


def test_route_is_registered_on_the_app() -> None:
    assert task_events.router in all_routers


@pytest.mark.parametrize("task_ids", [IDS, None, ["a" * 64], [f"t{i}" for i in range(task_events.MAX_TASK_IDS)]])
def test_task_ids_are_published_verbatim_on_the_global_stream(
    monkeypatch: pytest.MonkeyPatch, task_ids: list[str] | None
) -> None:
    subscriber = _subscriber()

    response = _client(monkeypatch, subscriber).post(URL, json={"_type": "task_changed", "task_ids": task_ids})

    assert (response.status_code, response.json()) == (202, {"stream_id": "1-0"})
    subscriber.publish_global_config_event.assert_awaited_once_with({"_type": "task_changed", "task_ids": task_ids})


def test_the_event_lands_on_the_stream_every_ui_window_reads() -> None:
    async def exercise() -> None:
        sub = object.__new__(RedisSubscriber)
        sub._pub_redis = SimpleNamespace(xadd=AsyncMock(return_value="9-0"))
        event = task_events.TaskChangedEvent.model_validate({"_type": "task_changed", "task_ids": IDS}).to_event()
        assert await sub.publish_global_config_event(event) == "9-0"
        assert sub._pub_redis.xadd.await_args.args[0] == GLOBAL_CONFIG_STREAM

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"task_ids": IDS},
        {"_type": "task_changed"},
        {"_type": "bot_profile_changed", "task_ids": IDS},
        {"_type": "TASK_CHANGED", "task_ids": IDS},
        {"type": "task_changed", "task_ids": IDS},
        {"_type": "task_changed", "task_ids": IDS, "status": "QUEUED"},
        {"_type": "task_changed", "task_ids": IDS, "title": "leaks content"},
        {"_type": "task_changed", "task_ids": []},
        {"_type": "task_changed", "task_ids": [f"t{i}" for i in range(task_events.MAX_TASK_IDS + 1)]},
        {"_type": "task_changed", "task_ids": [IDS[0], IDS[0]]},
        {"_type": "task_changed", "task_ids": ["TASK 7"]},
        {"_type": "task_changed", "task_ids": [""]},
        {"_type": "task_changed", "task_ids": ["a" * 65]},
        {"_type": "task_changed", "task_ids": [7]},
        {"_type": "task_changed", "task_ids": "cmvmvxyf300150mlfzf90b7u6"},
        ["task_changed"],
    ],
)
def test_any_other_shape_is_refused_and_nothing_is_published(monkeypatch: pytest.MonkeyPatch, body: object) -> None:
    subscriber = _subscriber()

    response = _client(monkeypatch, subscriber).post(URL, json=body)

    assert response.status_code == 422
    subscriber.publish_global_config_event.assert_not_awaited()


def test_there_is_no_generic_publish_route() -> None:
    paths = {route.path for router in all_routers for route in router.routes}
    assert {path for path in paths if path.startswith("/v1/events")} == {URL}


@pytest.mark.parametrize("subscriber", [None, _subscriber(stream_id=None)])
def test_an_unavailable_event_stream_is_a_503_the_caller_can_drop(
    monkeypatch: pytest.MonkeyPatch, subscriber: object | None
) -> None:
    response = _client(monkeypatch, subscriber).post(URL, json={"_type": "task_changed", "task_ids": IDS})

    assert response.status_code == 503
    assert response.json() == {"detail": "task event stream unavailable"}
