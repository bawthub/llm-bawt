"""TASK-1358: every background job runs whole on the bounded job pool.

A job handler used to be an ``async def`` that wrapped some of its blocking
calls in ``run_in_executor`` and ran the rest (LLM calls, embeddings, DB
writes) directly on the app event loop. Summary fact extraction stalled the
loop for ~7s on 2026-10-23, so the inter-bot dispatcher couldn't claim a
user's message. ``process_task`` now runs the whole handler on
``_bg_executor``; handlers are plain sync functions, so they can't block the
loop and never resubmit work to their own (bounded) pool.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from llm_bawt.service.background_service import BackgroundService
from llm_bawt.service.background_tasks import BackgroundTasksMixin
from llm_bawt.service.tasks import Task, TaskStatus, TaskType

HANDLED = [t for t in TaskType if t in BackgroundTasksMixin.TASK_HANDLERS]


def _service(workers: int = 1) -> BackgroundService:
    service = BackgroundService.__new__(BackgroundService)
    service._bg_executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bg-test")
    return service


def _task(task_type: TaskType, **payload) -> Task:
    return Task(task_type=task_type, payload=payload, bot_id="test", user_id="owner")


def test_every_task_type_except_consolidation_has_a_handler():
    # MEMORY_CONSOLIDATION has never had a handler (it is run through
    # MEMORY_MAINTENANCE); everything else must be routed.
    assert set(TaskType) - set(BackgroundTasksMixin.TASK_HANDLERS) == {TaskType.MEMORY_CONSOLIDATION}


@pytest.mark.parametrize("task_type", HANDLED, ids=lambda t: t.value)
def test_handlers_are_plain_sync_functions(task_type):
    handler = getattr(BackgroundTasksMixin, BackgroundTasksMixin.TASK_HANDLERS[task_type])
    assert not inspect.iscoroutinefunction(handler), (
        f"{handler.__name__} must be sync: process_task runs it on the job pool"
    )


@pytest.mark.parametrize("task_type", HANDLED, ids=lambda t: t.value)
def test_a_blocking_job_never_stalls_the_event_loop(task_type):
    service = _service()
    seen: dict[str, object] = {}

    def slow_handler(task):
        seen["thread"] = threading.current_thread()
        time.sleep(0.4)  # stands in for a blocking LLM/DB call
        return {"ok": True}

    setattr(service, BackgroundTasksMixin.TASK_HANDLERS[task_type], slow_handler)

    async def scenario():
        loop_thread = threading.current_thread()
        gaps: list[float] = []
        done = asyncio.Event()

        async def heartbeat():
            last = time.monotonic()
            while not done.is_set():
                await asyncio.sleep(0.01)
                now = time.monotonic()
                gaps.append(now - last)
                last = now

        beat = asyncio.create_task(heartbeat())
        result = await service.process_task(_task(task_type))
        done.set()
        await beat
        return loop_thread, gaps, result

    try:
        loop_thread, gaps, result = asyncio.run(scenario())
    finally:
        service._bg_executor.shutdown(wait=True)

    assert result.status is TaskStatus.COMPLETED and result.result == {"ok": True}
    assert seen["thread"] is not loop_thread
    assert seen["thread"].name.startswith("bg-test")
    assert len(gaps) >= 20 and max(gaps) < 0.15, f"loop stalled: max gap {max(gaps):.3f}s"


def test_handler_exception_becomes_a_failed_result():
    service = _service()

    def boom(task):
        raise RuntimeError("upstream down")

    service._process_tool_result_gc = boom
    try:
        result = asyncio.run(service.process_task(_task(TaskType.TOOL_RESULT_GC)))
    finally:
        service._bg_executor.shutdown(wait=True)
    assert result.status is TaskStatus.FAILED
    assert result.error == "upstream down"


def test_unhandled_task_type_still_fails_cleanly():
    service = _service()
    try:
        result = asyncio.run(service.process_task(_task(TaskType.MEMORY_CONSOLIDATION)))
    finally:
        service._bg_executor.shutdown(wait=True)
    assert result.status is TaskStatus.FAILED
    assert "Unknown task type" in result.error


def test_scheduler_wildcard_expansion_loads_bots_off_the_loop():
    """Expanding a ``bot_id="*"`` job builds a BotManager, which can reload
    every bot profile from the DB; that must not run on the event loop."""
    from types import SimpleNamespace

    from llm_bawt.bot_identities import WILDCARD_BOT_ID
    from llm_bawt.service.scheduler import JobScheduler, JobType, ScheduledJob

    seen: dict[str, object] = {}

    class SlowBotManager:
        def __init__(self, config):
            seen["thread"] = threading.current_thread()
            time.sleep(0.4)  # stands in for the DB reload after invalidation

        def list_bots(self):
            return [SimpleNamespace(slug="nova"), SimpleNamespace(slug="rook")]

    scheduler = JobScheduler(engine=None, task_processor=SimpleNamespace(config=object()))
    expanded: list[str] = []

    async def fake_single(job):
        expanded.append(job.bot_id)

    async def fake_timestamps(job):
        return None

    scheduler._execute_single_job = fake_single
    scheduler._update_job_timestamps = fake_timestamps
    job = ScheduledJob(job_type=JobType.HISTORY_SUMMARIZATION, bot_id=WILDCARD_BOT_ID, interval_minutes=60)

    async def scenario():
        loop_thread = threading.current_thread()
        gaps: list[float] = []
        done = asyncio.Event()

        async def heartbeat():
            last = time.monotonic()
            while not done.is_set():
                await asyncio.sleep(0.01)
                now = time.monotonic()
                gaps.append(now - last)
                last = now

        beat = asyncio.create_task(heartbeat())
        await scheduler._execute_job(job)
        done.set()
        await beat
        return loop_thread, gaps

    with patch("llm_bawt.bots.BotManager", SlowBotManager):
        loop_thread, gaps = asyncio.run(scenario())

    assert expanded == ["nova", "rook"]
    assert seen["thread"] is not loop_thread
    assert len(gaps) >= 20 and max(gaps) < 0.15, f"loop stalled: max gap {max(gaps):.3f}s"


def test_real_handler_completes_on_a_single_worker_pool():
    """A handler that resubmitted to its own pool would deadlock with one worker."""
    service = _service(workers=1)
    service.config = object()
    with patch("llm_bawt.service.jobs.tool_result_gc.run_tool_result_gc",
               return_value={"deleted": 2}) as gc:
        try:
            result = asyncio.run(asyncio.wait_for(
                service.process_task(_task(TaskType.TOOL_RESULT_GC, retention_days=3, dry_run=True)),
                timeout=5,
            ))
        finally:
            service._bg_executor.shutdown(wait=True)
    assert result.status is TaskStatus.COMPLETED and result.result == {"deleted": 2}
    gc.assert_called_once_with(service.config, retention_days=3, dry_run=True)
