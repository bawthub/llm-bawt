"""Overflow recovery must obey target-before-delivery locking."""
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import text

from llm_bawt.service.turn_admission import lock_target
from tests.test_inter_bot_delivery import delivery_store, _enqueue  # noqa: F401

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


def test_overflow_waits_for_target_without_holding_delivery_row(delivery_store):
    store, sender, target = delivery_store
    record, _ = _enqueue(store, sender, target, "overflow lock-order fixture")
    claimed = store.claim_next(target, claim_owner="review-owner", steer_capable=False)
    with ThreadPoolExecutor(max_workers=1) as workers:
        with store.engine.begin() as holder:
            lock_target(holder, target)
            holder_pid = holder.execute(text("SELECT pg_backend_pid()")).scalar_one()
            recovery = workers.submit(
                store.recover_context_overflow, record.id, claimed.claim_token,
                backend="claude-code", error="maximum context length exceeded",
            )
            deadline = time.monotonic() + 5
            with store.engine.connect() as observer:
                while time.monotonic() < deadline:
                    blocked = observer.execute(text("""
                        SELECT EXISTS (
                            SELECT 1 FROM pg_stat_activity
                            WHERE datname=current_database() AND wait_event_type='Lock'
                              AND :holder = ANY(pg_blocking_pids(pid))
                        )
                    """), {"holder": holder_pid}).scalar_one()
                    if blocked:
                        break
                    if recovery.done():
                        pytest.fail(f"recovery did not wait for target: {recovery.result()}")
                    time.sleep(0.02)
                else:
                    pytest.fail("did not observe overflow recovery waiting for target lock")
            # claim_next owns target, then locks the delivery. Recovery must not
            # hold that row while waiting for target, or these two paths deadlock.
            assert holder.execute(text("""
                SELECT id FROM inter_bot_deliveries WHERE id=:id FOR UPDATE NOWAIT
            """), {"id": record.id}).scalar_one() == record.id
        assert recovery.result(timeout=10) is not None
