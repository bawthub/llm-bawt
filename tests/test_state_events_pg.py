"""State fingerprints and writer commit notifies on real PostgreSQL (TASK-2109)."""

from __future__ import annotations

import asyncio
import threading
import uuid
from types import SimpleNamespace

import pytest

from llm_bawt.approval_policy_store import ToolApprovalPolicyStore
from llm_bawt.media.db import MediaGenerationStore
from llm_bawt.media.gpu_handoff_store import GpuHandoffStore
from llm_bawt.ops.reconciler import JobCreatedWaiter
from llm_bawt.ops.release_store import ReleaseStore
from llm_bawt.ops.store import OpsStore
from llm_bawt.service.commit_wake import WAKE_CHANNEL
from llm_bawt.service.state_events import StateWatcher
from llm_bawt.service.user_stop_hold import open_or_extend, resume

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


class _Subscriber:
    def __init__(self):
        self.events: list[dict] = []

    async def publish_global_config_event(self, event):
        self.events.append(event)
        return "1-0"


@pytest.fixture
def stores(private_app_pg):
    from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
    from llm_bawt.runtime_settings import BotProfileStore

    engine = OpsStore(private_app_pg).engine
    InterBotDeliveryStore(private_app_pg)  # provisions the gate schema
    bot = f"test-state-{uuid.uuid4().hex[:12]}"
    BotProfileStore(private_app_pg).upsert({"slug": bot, "name": bot, "bot_type": "agent",
                                            "agent_backend": "claude-code"})
    return SimpleNamespace(
        bot=bot,
        engine=engine,
        media=MediaGenerationStore(private_app_pg),
        gpu=GpuHandoffStore(engine),
        releases=ReleaseStore(private_app_pg, engine=engine),
        policies=ToolApprovalPolicyStore(private_app_pg, engine=engine),
    )


def _notifies(engine, write) -> bool:
    waiter = JobCreatedWaiter(engine, channel=WAKE_CHANNEL)
    stop = threading.Event()
    try:
        waiter.wait(stop, 0.05)  # connect and LISTEN
        write()
        return waiter.wait(stop, 2)
    finally:
        waiter.close()


def test_each_writer_moves_only_its_topic_and_notifies(stores):
    sub = _Subscriber()
    watcher = StateWatcher(SimpleNamespace(_redis_subscriber=sub), stores.engine)

    def check():
        return asyncio.run(watcher.check())

    watcher._last = watcher._read()
    for topic in ("media", "gpu", "ops_releases", "approval_policies", "ops_jobs", "queue_hold"):
        assert topic in watcher._last, topic
    assert check() == set()

    gen_id = uuid.uuid4().hex
    assert _notifies(stores.engine, lambda: stores.media.insert(
        {"id": gen_id, "status": "processing", "media_type": "image", "prompt": "p", "provider": "x"}))
    assert check() == {"media"}
    assert _notifies(stores.engine, lambda: stores.media.update(gen_id, {"progress": 50}))
    assert check() == {"media"}

    state = stores.gpu.status()
    assert _notifies(stores.engine, lambda: stores.gpu.offer(
        user="nick", target="video", actions=("stop_moshi_tts",), expected_generation=state.generation))
    assert check() == {"gpu"}

    assert _notifies(stores.engine, lambda: stores.releases.create(
        parent_job_id=uuid.uuid4().hex, release_task="TASK-0", bump="patch", github_repository="o/r",
        workflow_path=".github/workflows/r.yml", canonical_branch="main"))
    assert check() == {"ops_releases"}

    assert _notifies(stores.engine, stores.policies.seed_defaults)
    assert check() == {"approval_policies"}

    with stores.engine.begin() as conn:
        open_or_extend(conn, stores.bot, 300)
    assert check() == {"queue_hold"}
    resume(stores.engine, stores.bot)
    assert check() == {"queue_hold"}

    assert check() == set()
    assert [event["topics"] for event in sub.events] == [
        ["media"], ["media"], ["gpu"], ["ops_releases"], ["approval_policies"], ["queue_hold"], ["queue_hold"]]


def test_rolled_back_write_does_not_notify(stores):
    def write():
        try:
            with stores.engine.connect() as conn:
                from llm_bawt.service.commit_wake import notify_in_transaction
                notify_in_transaction(conn)
                conn.rollback()
        except Exception:
            pass

    assert _notifies(stores.engine, write) is False
