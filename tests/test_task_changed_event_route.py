"""TASK-1396/1402: POST /v1/events/{task,project}-changed relay BawtHub writes to open UIs.

Two narrow, content-free events on the existing global unified stream, one
closed route each; every other shape is refused before anything is
published. The caller (BawtHub, after its transaction commits) drops a 503
rather than retrying.
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

TASK_URL = "/v1/events/task-changed"
PROJECT_URL = "/v1/events/project-changed"
IDS = ["cmvmvxyf300150mlfzf90b7u6", "cmvmjoyo800gx0mnt925jw6gz"]

# (url, _type, id field, model, 503 noun)
ROUTES = [
    pytest.param(TASK_URL, "task_changed", "task_ids", task_events.TaskChangedEvent, "task", id="task"),
    pytest.param(PROJECT_URL, "project_changed", "project_ids", task_events.ProjectChangedEvent, "project", id="project"),
]
MAX = task_events.MAX_IDS


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


@pytest.mark.parametrize(("url", "kind", "field", "model", "noun"), ROUTES)
@pytest.mark.parametrize("ids", [IDS, None, ["a" * 64], [f"t{i}" for i in range(MAX)]])
def test_ids_are_published_verbatim_on_the_global_stream(
    monkeypatch: pytest.MonkeyPatch, url: str, kind: str, field: str, model: type, noun: str, ids: list[str] | None
) -> None:
    subscriber = _subscriber()

    response = _client(monkeypatch, subscriber).post(url, json={"_type": kind, field: ids})

    assert (response.status_code, response.json()) == (202, {"stream_id": "1-0"})
    subscriber.publish_global_config_event.assert_awaited_once_with({"_type": kind, field: ids})


@pytest.mark.parametrize(("url", "kind", "field", "model", "noun"), ROUTES)
def test_the_event_lands_on_the_stream_every_ui_window_reads(
    url: str, kind: str, field: str, model: type, noun: str
) -> None:
    async def exercise() -> None:
        sub = object.__new__(RedisSubscriber)
        sub._pub_redis = SimpleNamespace(xadd=AsyncMock(return_value="9-0"))
        event = model.model_validate({"_type": kind, field: IDS}).to_event()
        assert await sub.publish_global_config_event(event) == "9-0"
        assert sub._pub_redis.xadd.await_args.args[0] == GLOBAL_CONFIG_STREAM

    asyncio.run(exercise())


def _bad_bodies(kind: str, field: str, other_kind: str, other_field: str) -> list[object]:
    return [
        {},
        {field: IDS},
        {"_type": kind},
        {"_type": "bot_profile_changed", field: IDS},
        {"_type": kind.upper(), field: IDS},
        {"type": kind, field: IDS},
        {"_type": kind, field: IDS, "status": "QUEUED"},
        {"_type": kind, field: IDS, "title": "leaks content"},
        {"_type": kind, field: []},
        {"_type": kind, field: [f"t{i}" for i in range(MAX + 1)]},
        {"_type": kind, field: [IDS[0], IDS[0]]},
        {"_type": kind, field: ["TASK 7"]},
        {"_type": kind, field: [""]},
        {"_type": kind, field: ["a" * 65]},
        {"_type": kind, field: [7]},
        {"_type": kind, field: IDS[0]},
        [kind],
        # Each route accepts only its own kind (TASK-1402): no cross-posting.
        {"_type": other_kind, other_field: IDS},
        {"_type": kind, other_field: IDS},
        {"_type": other_kind, field: IDS},
        {"_type": kind, field: IDS, other_field: IDS},
    ]


BAD = [
    *[pytest.param(TASK_URL, body, id=f"task-{i}") for i, body in
      enumerate(_bad_bodies("task_changed", "task_ids", "project_changed", "project_ids"))],
    *[pytest.param(PROJECT_URL, body, id=f"project-{i}") for i, body in
      enumerate(_bad_bodies("project_changed", "project_ids", "task_changed", "task_ids"))],
]


@pytest.mark.parametrize(("url", "body"), BAD)
def test_any_other_shape_is_refused_and_nothing_is_published(
    monkeypatch: pytest.MonkeyPatch, url: str, body: object
) -> None:
    subscriber = _subscriber()

    response = _client(monkeypatch, subscriber).post(url, json=body)

    assert response.status_code == 422
    subscriber.publish_global_config_event.assert_not_awaited()


def test_there_is_no_generic_publish_route() -> None:
    paths = {route.path for router in all_routers for route in router.routes}
    assert {path for path in paths if path.startswith("/v1/events")} == {TASK_URL, PROJECT_URL}


@pytest.mark.parametrize(("url", "kind", "field", "model", "noun"), ROUTES)
@pytest.mark.parametrize("subscriber", [None, _subscriber(stream_id=None)])
def test_an_unavailable_event_stream_is_a_503_the_caller_can_drop(
    monkeypatch: pytest.MonkeyPatch, url: str, kind: str, field: str, model: type, noun: str, subscriber: object | None
) -> None:
    response = _client(monkeypatch, subscriber).post(url, json={"_type": kind, field: IDS})

    assert response.status_code == 503
    assert response.json() == {"detail": f"{noun} event stream unavailable"}
