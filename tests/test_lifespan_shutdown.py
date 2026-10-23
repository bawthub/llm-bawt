"""The app lifespan's shutdown phases are independent (TASK-1311).

One raising phase (the dispatcher's unlock on a lock session that died with
the nightly PostgreSQL restart) used to abort the rest of the teardown:
uvicorn logged "Application shutdown failed", and the task cancels, the Redis
close and ``service.shutdown()`` never ran. Each phase now fails on its own,
with a traceback, and the sequence always reaches its terminal log line.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest
from sqlalchemy.exc import OperationalError

from llm_bawt.service import api
from llm_bawt.service.lifespan_shutdown import cancel_tasks, run_shutdown_phases


def _phase(events: list[str], name: str, *, raises: Exception | None = None):
    async def run() -> None:
        events.append(name)
        if raises is not None:
            raise raises
    return name, run


@pytest.mark.asyncio
async def test_a_failing_phase_never_skips_the_later_ones(caplog):
    events: list[str] = []
    caplog.set_level(logging.WARNING)
    failed = await run_shutdown_phases([
        _phase(events, "first"),
        _phase(events, "dead lock session", raises=OperationalError("unlock", {}, Exception("gone"))),
        _phase(events, "cancelled", raises=asyncio.CancelledError()),
        _phase(events, "last"),
    ])
    assert events == ["first", "dead lock session", "cancelled", "last"]
    assert failed == ["dead lock session", "cancelled"]
    failures = [r for r in caplog.records if r.getMessage().startswith("lifespan shutdown: phase")]
    assert [r.levelno for r in failures] == [logging.ERROR, logging.WARNING]
    assert failures[0].exc_info is not None  # the traceback is kept


@pytest.mark.asyncio
async def test_no_failures_returns_an_empty_list():
    events: list[str] = []
    assert await run_shutdown_phases([_phase(events, "a"), _phase(events, "b")]) == []
    assert events == ["a", "b"]


@pytest.mark.asyncio
async def test_cancel_tasks_cancels_every_held_task_and_skips_missing_ones():
    gate = asyncio.Event()
    owner = SimpleNamespace(
        running=asyncio.create_task(gate.wait()),
        finished=None,
    )
    await cancel_tasks(owner, ("running", "finished", "absent"))
    assert owner.running.cancelled()


# --- what api.py wires -------------------------------------------------------


class _Service:
    def __init__(self, events: list[str], dispatcher):
        self.events = events
        self._inter_bot_dispatcher = dispatcher
        self._group_cleanup_task = None

    async def shutdown(self) -> None:
        self.events.append("service")


def _wiring(events: list[str], *, dispatcher_raises: Exception | None = None, with_optional=True):
    async def stop_named(name, raises=None):
        events.append(name)
        if raises is not None:
            raise raises

    dispatcher = SimpleNamespace(stop=lambda: stop_named("dispatcher", dispatcher_raises))
    service = _Service(events, dispatcher if with_optional else None)
    scheduler = SimpleNamespace(stop=lambda: stop_named("scheduler")) if with_optional else None
    redis = SimpleNamespace(close=lambda: stop_named("redis")) if with_optional else None
    home_audio = SimpleNamespace(stop=lambda: stop_named("home audio"))
    return service, scheduler, redis, home_audio


@pytest.mark.asyncio
async def test_a_dispatcher_unlock_failure_still_closes_redis_and_the_service(monkeypatch):
    events: list[str] = []
    cleared: list = []
    monkeypatch.setattr(
        "llm_bawt.agent_backends.agent_bridge.set_agent_subscriber", cleared.append
    )
    service, scheduler, redis, home_audio = _wiring(
        events, dispatcher_raises=OperationalError("unlock", {}, Exception("gone")),
    )
    failed = await run_shutdown_phases(api.shutdown_phases(
        service, scheduler=scheduler, redis_subscriber=redis, home_audio_worker=home_audio,
    ))
    assert events == ["home audio", "scheduler", "dispatcher", "redis", "service"]
    assert failed == ["inter-bot dispatcher"]
    assert cleared == [None]


def test_phases_omit_absent_optional_components():
    events: list[str] = []
    service, scheduler, redis, home_audio = _wiring(events, with_optional=False)
    names = [name for name, _ in api.shutdown_phases(
        service, scheduler=scheduler, redis_subscriber=redis, home_audio_worker=home_audio,
    )]
    assert names == ["home audio worker", "background tasks", "service"]


def test_every_background_task_attribute_is_a_task_the_lifespan_creates():
    """The attributes the cancel phase sweeps are the ones api.py assigns."""
    import inspect

    source = inspect.getsource(api)
    for attribute in api.BACKGROUND_TASK_ATTRIBUTES:
        assert f"service.{attribute} = " in source or f"{attribute} = " in source, attribute
