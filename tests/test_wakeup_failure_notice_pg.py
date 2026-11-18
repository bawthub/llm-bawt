"""Wake-up failure notices on real PostgreSQL locks (TASK-2436).

Enqueue takes the recipient's transaction-scoped admission lock. These tests
run the real delivery store against a private database, so the lock graph is
PostgreSQL's, not a SQLite shim's.
"""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
import uuid

import pytest
from sqlalchemy import text
from sqlmodel import Session

from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
from llm_bawt.runtime_settings import BotProfileStore
from llm_bawt.service.prompt_delivery import PromptDeliveryWorker, _notifies_after_commit
from llm_bawt.service.turn_admission import lock_target
from tests.pg_isolation import isolated_pg_store

pytest_plugins = ("tests.private_postgres",)
pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.fixture
def rig(monkeypatch):
    monkeypatch.setenv("LLM_BAWT_DEFAULT_USER", "operator")
    store = isolated_pg_store(InterBotDeliveryStore)
    suffix = uuid.uuid4().hex[:12]
    bots = [f"notice-a-{suffix}", f"notice-b-{suffix}"]
    profiles = BotProfileStore(store.config)
    for bot in bots:
        profiles.upsert({"slug": bot, "name": "Disposable notice test bot"})
    monkeypatch.setattr("llm_bawt.service.prompt_delivery.failed_notice",
                        lambda *args: "Disposable failure notice, not a retry")
    return store, bots


def snapshot(target, creator):
    return {"owner_user_id": "operator", "bot_id": target,
            "creator_bot_id": creator, "name": "Disposable wake-up"}


class Worker(PromptDeliveryWorker):
    """The production notice path around a stand-in occurrence transaction.

    ``fail`` holds the failing wake-up's own admission lock, as the enqueue
    path does, then queues the failure exactly as ``_finish`` does.
    """

    def __init__(self, store):
        self.engine = store.engine
        self.service = SimpleNamespace(_inter_bot_dispatcher=SimpleNamespace(store=store))
        self._notices = []
        self._failed_pending = []

    @_notifies_after_commit
    def fail(self, run, target, creator, *, barrier=None, explode=False):
        with Session(self.engine) as session, session.begin():
            session.connection().execute(text("SET LOCAL lock_timeout='5s'"))
            lock_target(session.connection(), target)
            self._failed_pending.append((run, snapshot(target, creator), "failed", "test failure"))
            if barrier is not None:
                barrier.wait(timeout=10)
            if explode:
                raise RuntimeError("occurrence transaction rolled back")


def notice_targets(store, *runs):
    with store.engine.connect() as conn:
        return sorted(conn.execute(text("""
            SELECT target_bot_id FROM inter_bot_deliveries
            WHERE split_part(idempotency_key, ':', 2) = ANY(:runs)
              AND idempotency_key LIKE 'wakeup-failed:%'
        """), {"runs": list(runs)}).scalars().all())


def test_reciprocal_failures_deliver_every_notice(rig):
    """Wake-up a (created by b) and b (created by a) fail at the same moment.

    Each failure holds its own bot's lock until it commits. Sending both
    notices inside it would collect a then b against b then a and deadlock;
    PostgreSQL would abort one and that notice would be lost for good.
    """
    store, (a, b) = rig
    runs = [uuid.uuid4().hex, uuid.uuid4().hex]
    barrier = Barrier(2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(Worker(store).fail, runs[0], a, b, barrier=barrier),
                   pool.submit(Worker(store).fail, runs[1], b, a, barrier=barrier)]
        for future in futures:
            future.result(timeout=30)
    assert notice_targets(store, runs[0]) == sorted([a, b])
    assert notice_targets(store, runs[1]) == sorted([a, b])


def test_rolled_back_failure_sends_nothing(rig):
    store, (a, b) = rig
    run = uuid.uuid4().hex
    worker = Worker(store)
    with pytest.raises(RuntimeError):
        worker.fail(run, a, b, explode=True)
    assert notice_targets(store, run) == []
    assert worker._failed_pending == []


def test_one_recipient_fault_keeps_the_others(rig, monkeypatch):
    store, (a, b) = rig
    run = uuid.uuid4().hex
    original = store.enqueue

    def enqueue(**kwargs):
        assert kwargs.get("connection") is None, "notice must not join another transaction"
        if kwargs["target_bot_id"] == a:
            with store.engine.begin() as conn:
                conn.execute(text("SELECT 1/0"))
        return original(**kwargs)

    monkeypatch.setattr(store, "enqueue", enqueue)
    Worker(store).fail(run, a, b)
    assert notice_targets(store, run) == [b]
    # A replay neither duplicates the sent notice nor fails.
    monkeypatch.setattr(store, "enqueue", original)
    Worker(store).fail(run, a, b)
    assert notice_targets(store, run) == sorted([a, b])
