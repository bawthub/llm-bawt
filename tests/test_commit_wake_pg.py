"""Commit wakes on real PostgreSQL: only commits notify, and the listener wakes the hub (TASK-2108)."""

from __future__ import annotations

import asyncio
import hashlib
import threading
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from llm_bawt.ops.reconciler import JobCreatedWaiter
from llm_bawt.ops.store import OpsStore
from llm_bawt.service.commit_wake import WAKE_CHANNEL, WakeHub, run_commit_wake_listener, wait_for_wake
from llm_bawt.system_events import CommitStagedAppends

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


def _stage_noop(conn):
    CommitStagedAppends.stage(conn, lambda c: c.execute(text("SELECT 1")))


def test_only_a_committed_staged_append_notifies(private_app_pg):
    engine = OpsStore(private_app_pg).engine
    waiter = JobCreatedWaiter(engine, channel=WAKE_CHANNEL)
    stop = threading.Event()
    try:
        assert waiter.wait(stop, 0.05) is False  # connects and LISTENs
        try:
            with engine.begin() as conn:
                _stage_noop(conn)
                raise RuntimeError("roll back")
        except RuntimeError:
            pass
        assert waiter.wait(stop, 0.3) is False
        with engine.begin() as conn:
            _stage_noop(conn)
        assert waiter.wait(stop, 2) is True
    finally:
        waiter.close()


def test_listener_wakes_subscribers_and_recovers_a_dropped_connection(private_app_pg):
    engine = OpsStore(private_app_pg).engine

    async def main():
        dispatcher = SimpleNamespace(wakes=0)
        dispatcher.wake = lambda: setattr(dispatcher, "wakes", dispatcher.wakes + 1)
        service = SimpleNamespace(_commit_wake_hub=WakeHub(), _inter_bot_dispatcher=dispatcher)
        wake = service._commit_wake_hub.subscribe()
        task = asyncio.create_task(run_commit_wake_listener(service, engine))
        assert await wait_for_wake(wake, 5)  # the connect wake
        await asyncio.to_thread(_commit)
        assert await wait_for_wake(wake, 1)  # prompt: the wait's own timeout result
        # Drop the listener's connection: it reconnects and wakes everyone once.
        service._commit_wake_waiter._conn.close()
        assert await wait_for_wake(wake, 15)
        await asyncio.to_thread(_commit)
        assert await wait_for_wake(wake, 5)
        assert dispatcher.wakes >= 4
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    def _commit():
        with engine.begin() as conn:
            _stage_noop(conn)

    asyncio.run(main())


def _ops_waiter(engine):
    waiter = JobCreatedWaiter(engine)  # the reconciler's own channel
    assert waiter.wait(threading.Event(), 0.05) is False  # connects and LISTENs
    return waiter


def test_approving_an_ops_run_wakes_the_reconciler(private_app_pg):
    from llm_bawt.approval_policies import REQ_APPROVED, REQ_DENIED, ToolApprovalPolicyStore
    from tests.test_approval_mcp_state import _mcp_req

    store = ToolApprovalPolicyStore(None, engine=OpsStore(private_app_pg).engine)
    waiter = _ops_waiter(store.engine)
    try:
        for rid, status, woke in (("t2108-deny", REQ_DENIED, False), ("t2108-ok", REQ_APPROVED, True)):
            store.record_mcp_request(**_mcp_req(request_id=rid, tool_use_id=f"toolu_{rid}",
                                                invocation_hash=hashlib.sha256(rid.encode()).hexdigest()))
            waiter.received.clear()
            store.resolve_request(rid, status=status, resolved_by="nick")
            assert waiter.wait(threading.Event(), 1 if woke else 0.3) is woke
            if woke:
                assert f"approval:{rid}" in waiter.received
    finally:
        waiter.close()
