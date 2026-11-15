"""TASK-1440: wildcard memory jobs skip bots tagged ``inactive``.

Only the inactive tag excludes a bot. A private bot keeps its memory
maintenance (private means invisible to other bots, not "no memory").
An unreadable tag snapshot processes nothing.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from llm_bawt.bot_identities import WILDCARD_BOT_ID
from llm_bawt.bot_tags.facade import TagSnapshot
from llm_bawt.service.scheduler import JobScheduler, JobType, ScheduledJob

MEMORY_JOBS = [JobType.MEMORY_EXTRACTION, JobType.MEMORY_CONSOLIDATION, JobType.MEMORY_DECAY,
               JobType.HISTORY_SUMMARIZATION]  # TASK-2333

ROWS = [
    {"slug": "nova", "include_in_global_search": True, "tags": []},
    {"slug": "hush", "include_in_global_search": True, "tags": ["private"]},
    {"slug": "sleepy", "include_in_global_search": True, "tags": ["inactive"]},
    {"slug": "ghost", "include_in_global_search": True, "tags": ["inactive", "private"]},
]


def _run(job_type: JobType, snapshot: TagSnapshot) -> list[str]:
    scheduler = JobScheduler(engine=None, task_processor=SimpleNamespace(config=object()))
    expanded: list[str] = []
    stamped: list[str] = []

    async def fake_single(job):
        expanded.append(job.bot_id)

    async def fake_timestamps(job):
        stamped.append(job.bot_id)

    scheduler._execute_single_job = fake_single
    scheduler._update_job_timestamps = fake_timestamps

    class FakeBotManager:
        def __init__(self, config):
            pass

        def list_bots(self):
            return [SimpleNamespace(slug=r["slug"]) for r in ROWS]

    class FakeBotTags:
        def __init__(self, engine=None):
            pass

        def snapshot(self, conn=None):
            return snapshot

    job = ScheduledJob(job_type=job_type, bot_id=WILDCARD_BOT_ID, interval_minutes=60)
    with patch("llm_bawt.bots.BotManager", FakeBotManager), \
            patch("llm_bawt.service.scheduler.BotTags", FakeBotTags):
        asyncio.run(scheduler._execute_job(job))
    # With nothing to run the job is left unstamped, as for an empty bot list,
    # so the next tick retries once tags are readable again.
    assert stamped == ([WILDCARD_BOT_ID] if expanded else [])
    return expanded


@pytest.mark.parametrize("job_type", MEMORY_JOBS)
def test_inactive_bots_are_skipped_and_private_active_bots_still_run(job_type):
    assert _run(job_type, TagSnapshot.from_rows(ROWS)) == ["nova", "hush"]


@pytest.mark.parametrize("job_type", MEMORY_JOBS)
def test_unavailable_tag_snapshot_processes_nothing(job_type):
    assert _run(job_type, TagSnapshot.unavailable()) == []
