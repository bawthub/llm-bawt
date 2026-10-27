"""notify_user admission on PostgreSQL (TASK-1531): one lock serializes the caps."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import text

from llm_bawt.integrations.notify_store import _ADMIT_LOCK, Limits, NotifyStore
from llm_bawt.utils.db import get_shared_engine
from tests.pg_isolation import require_test_db_config
from tests.pg_lock_wait import wait_blocked_by

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]

LIMITS = Limits(per_bot_interval_seconds=600, global_per_hour=1, dedupe_window_seconds=3600)


@pytest.fixture
def store():
    store = NotifyStore(get_shared_engine(require_test_db_config()))
    with store.engine.begin() as conn:
        conn.execute(text("DELETE FROM notify_log"))
    yield store
    with store.engine.begin() as conn:
        conn.execute(text("DELETE FROM notify_log"))


def test_admission_waits_for_a_concurrent_admitter_and_sees_its_reservation(store):
    now = time.time()
    with store.engine.connect() as holder:
        transaction = holder.begin()
        holder.execute(text(_ADMIT_LOCK))
        pid = holder.execute(text("SELECT pg_backend_pid()")).scalar_one()
        with ThreadPoolExecutor(1) as pool:
            waiter = pool.submit(store.admit, "bot-b", "normal", "second", None, LIMITS, now)
            wait_blocked_by(store.engine, pid, waiter)
            # The holder reserves the last global slot while the waiter is blocked.
            holder.execute(text(
                "INSERT INTO notify_log (id, bot_id, created_at, updated_at, urgency, "
                "message_hash, result) VALUES ('held', 'bot-a', :now, :now, 'normal', "
                "'x', 'sending')"), {"now": now})
            transaction.commit()
            _, result = waiter.result(10)
    assert result == "suppressed:rate_limit"


def test_admit_then_finish_on_postgres(store):
    row, result = store.admit("bot-a", "high", "message", "key", LIMITS)
    assert result == "sending"
    store.finish(row, "sent")
    assert store.get(row)["result"] == "sent"
    # finish only moves a reservation, never rewrites a terminal row.
    store.finish(row, "ha_error:500")
    assert store.get(row)["result"] == "sent"
    assert store.admit("bot-a", "normal", "again", "key", LIMITS)[1] == "suppressed:dedupe"
