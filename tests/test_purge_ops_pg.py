"""TASK-1232 B1/S1: operator purge of in-flight ops actions, against real PostgreSQL.

Each barrier test commits the purge between a writer's read and its write, the
window the old read-then-write writers lost. Every writer must leave a purged
request cancelled with its continuation suppressed, never back in the outbox.
"""
from __future__ import annotations

import json
import threading
import uuid

import pytest
from sqlalchemy import text as sa_text
from sqlmodel import Session

from llm_bawt.approval_models import (
    ApprovalPersistError,
    CONT_DELIVERED,
    CONT_DISPATCHING,
    CONT_NOT_NEEDED,
    CONT_SUPPRESSED,
    EXEC_SKIPPED,
    REQ_APPROVED,
    REQ_CANCELLED,
    ToolApprovalRequest,
)
from llm_bawt.service.recovery.audit import RunStamp
from llm_bawt.service.recovery.purge_ops import PURGED_BY_OPERATOR, PURGED_RESULT, purge_ops
from tests.test_approval_mcp_state import _mcp_req

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.fixture
def stores(private_app_pg):
    from llm_bawt.approval_policies import ToolApprovalPolicyStore
    from llm_bawt.ops.store import OpsStore

    approvals = ToolApprovalPolicyStore(private_app_pg)
    OpsStore(private_app_pg)  # app-owned ops schema bootstrap
    assert approvals.engine is not None
    yield approvals
    with approvals.engine.begin() as conn:
        conn.execute(sa_text("DELETE FROM ops_jobs"))
        conn.execute(sa_text("DELETE FROM tool_approval_requests"))


def _ops_run(store, *, approved=True) -> str:
    request_id = f"req-{uuid.uuid4().hex}"
    store.record_mcp_request(**_mcp_req(
        request_id=request_id, tool_use_id=f"toolu_{uuid.uuid4().hex}", turn_id=f"turn-{uuid.uuid4().hex}",
        invocation_hash=uuid.uuid4().hex * 2,
    ))
    if approved:
        store.resolve_request(request_id, status=REQ_APPROVED, resolved_by="nick", continuation_owner="server")
    return request_id


def _row(store, request_id) -> ToolApprovalRequest:
    with Session(store.engine) as session:
        return session.get(ToolApprovalRequest, request_id)


def _dispatching(store) -> tuple[str, str]:
    request_id = _ops_run(store)
    claim = store.claim_mcp_execution(request_id)
    store.complete_mcp_execution(request_id, result_json='{"ok": true}', is_error=False,
                                 claim_token=claim.execution_claim_token)
    continuation = store.claim_continuation(request_id)
    assert continuation is not None and continuation.continuation_state == CONT_DISPATCHING
    return request_id, continuation.continuation_claim_token


def _insert_job(store, *, state, approval_request_id=None) -> str:
    from llm_bawt.ops.models import OpsJob

    job_id = uuid.uuid4().hex
    with Session(store.engine) as session:
        session.add(OpsJob(id=job_id, operation_slug="llm-bawt.restart-app", idempotency_key=job_id,
                           state=state, approval_request_id=approval_request_id))
        session.commit()
    return job_id


def _purge_between_read_and_write(monkeypatch, store) -> list:
    """Commit a purge on another connection right after the writer's first row read."""
    from llm_bawt import approval_mcp_store

    fired: list = []

    class PurgeAfterRead(approval_mcp_store.Session):
        def get(self, *args, **kwargs):
            row = super().get(*args, **kwargs)
            if not fired:
                fired.append(True)
                worker = threading.Thread(target=lambda: fired.append(
                    purge_ops(store.engine, stamp=RunStamp.new("snark"))))
                worker.start()
                worker.join(timeout=30)
                assert not worker.is_alive(), "purge blocked behind the writer's read"
            return row

    monkeypatch.setattr(approval_mcp_store, "Session", PurgeAfterRead)
    return fired


def _assert_purged(store, request_id) -> ToolApprovalRequest:
    row = _row(store, request_id)
    assert row.status == REQ_CANCELLED
    assert row.continuation_state == CONT_SUPPRESSED and row.continuation_claim_token is None
    assert store.claim_continuation(request_id) is None
    assert request_id not in {r.id for r in store.find_pending_continuations(include_dispatching_older_than_s=0)}
    return row


def test_enqueue_cannot_reenqueue_a_request_purged_after_its_read(stores, monkeypatch):
    request_id = _ops_run(stores)
    claim = stores.claim_mcp_execution(request_id)
    # A committed result whose enqueue never ran (the stranded-result shape).
    with stores.engine.begin() as conn:
        conn.execute(sa_text("""
            UPDATE tool_approval_requests SET result_json='{}', execution_state='succeeded',
                execution_claim_token=NULL, continuation_state=:not_needed WHERE id=:id
        """), {"id": request_id, "not_needed": CONT_NOT_NEEDED})
    assert claim is not None
    fired = _purge_between_read_and_write(monkeypatch, stores)
    stores.enqueue_continuation(request_id)
    assert fired, "barrier never ran"
    _assert_purged(stores, request_id)


def test_completion_snapshot_taken_before_a_purge_cannot_enqueue(stores, monkeypatch):
    request_id = _ops_run(stores)
    claim = stores.claim_mcp_execution(request_id)
    fired = _purge_between_read_and_write(monkeypatch, stores)
    with pytest.raises(ApprovalPersistError):
        stores.complete_mcp_execution(request_id, result_json='{"ok": true}', is_error=False,
                                      claim_token=claim.execution_claim_token)
    assert fired, "barrier never ran"
    row = _assert_purged(stores, request_id)
    assert (json.loads(row.result_json), row.execution_state) == (PURGED_RESULT, EXEC_SKIPPED)
    assert row.execution_error.startswith(PURGED_BY_OPERATOR)


def test_approved_claimant_past_its_claim_cannot_start_work_after_a_purge(stores):
    request_id = _ops_run(stores)
    claim = stores.claim_mcp_execution(request_id)
    fence = stores.execution_fence(request_id, claim.execution_claim_token)
    purge_ops(stores.engine, stamp=RunStamp.new("snark"))
    with Session(stores.engine) as session, pytest.raises(ApprovalPersistError):
        fence(session)
    _assert_purged(stores, request_id)


def test_work_the_fence_admitted_before_a_purge_is_cancelled(stores):
    request_id = _ops_run(stores)
    claim = stores.claim_mcp_execution(request_id)
    fence = stores.execution_fence(request_id, claim.execution_claim_token)
    with Session(stores.engine) as session:
        fence(session)
        session.commit()
    job_id = _insert_job(stores, state="running", approval_request_id=request_id)
    finished = _insert_job(stores, state="succeeded")
    report = purge_ops(stores.engine, stamp=RunStamp.new("snark"))
    assert {job["id"] for job in report["jobs"]} == {job_id}
    with stores.engine.connect() as conn:
        states = dict(conn.execute(sa_text("SELECT id, state FROM ops_jobs")).all())
        error = conn.execute(sa_text("SELECT error_text FROM ops_jobs WHERE id=:id"), {"id": job_id}).scalar_one()
    assert states == {job_id: "cancelled", finished: "succeeded"} and error.startswith(PURGED_BY_OPERATOR)
    _assert_purged(stores, request_id)


@pytest.mark.parametrize("writer", ["failed_retry", "defer"])
def test_dispatching_writers_suppress_a_purged_continuation(stores, writer):
    """Vex S1: a dispatching claim can't be recalled, so purge only cancels the
    request; the claimant's own dispatching→pending write must suppress it."""
    request_id, token = _dispatching(stores)
    report = purge_ops(stores.engine, stamp=RunStamp.new("snark"))
    assert [entry["id"] for entry in report["unrecallable"]] == [request_id]
    assert (_row(stores, request_id).status, _row(stores, request_id).continuation_state) == (REQ_CANCELLED, CONT_DISPATCHING)
    if writer == "failed_retry":
        stores.mark_continuation_failed(request_id, error="target busy", claim_token=token)
    else:
        assert stores.defer_continuation_claim(request_id, claim_token=token, reason="busy", delay_seconds=0)
    _assert_purged(stores, request_id)


def test_a_dispatching_continuation_that_lands_after_a_purge_stays_delivered(stores):
    request_id, token = _dispatching(stores)
    purge_ops(stores.engine, stamp=RunStamp.new("snark"))
    stores.mark_continuation_delivered(request_id, claim_token=token)
    assert _row(stores, request_id).continuation_state == CONT_DELIVERED


def test_an_undecided_card_is_cancelled_not_denied_and_stays_cancelled(stores):
    request_id = _ops_run(stores, approved=False)
    report = purge_ops(stores.engine, stamp=RunStamp.new("snark"))
    assert [entry["id"] for entry in report["approvals"]] == [request_id]
    row = _assert_purged(stores, request_id)
    assert row.resolved_by == "snark" and row.resolution_message.startswith(PURGED_BY_OPERATOR)
    # First decision wins: a late approve can't revive it.
    stores.resolve_request(request_id, status=REQ_APPROVED, resolved_by="nick", continuation_owner="server")
    assert _row(stores, request_id).status == REQ_CANCELLED
    assert stores.claim_mcp_execution(request_id) is None


def test_dry_run_reports_and_changes_nothing(stores):
    pending = _ops_run(stores, approved=False)
    dispatching, _ = _dispatching(stores)
    job_id = _insert_job(stores, state="accepted")
    before = {rid: _row(stores, rid).model_dump() for rid in (pending, dispatching)}
    report = purge_ops(stores.engine, stamp=RunStamp.new("snark"), dry_run=True)
    assert report["dry_run"] is True
    assert [e["id"] for e in report["approvals"]] == [pending]
    assert [e["id"] for e in report["unrecallable"]] == [dispatching]
    assert [e["id"] for e in report["jobs"]] == [job_id]
    assert {rid: _row(stores, rid).model_dump() for rid in (pending, dispatching)} == before
    with stores.engine.connect() as conn:
        assert conn.execute(sa_text("SELECT state FROM ops_jobs WHERE id=:id"), {"id": job_id}).scalar_one() == "accepted"
    json.dumps(report)  # the admin API returns it as-is


def test_a_purged_cards_stored_resolution_and_tool_card_carry_the_cancelled_result(stores):
    """Vex C1: without a stored result, an undecided card stayed actionable and
    an approved ops_run awaiting admission showed "awaiting result" forever."""
    from llm_bawt.service.tool_call_store import ToolCallStore

    undecided = _ops_run(stores, approved=False)
    awaiting = _ops_run(stores)
    cards = ToolCallStore(stores.engine)
    for request_id in (undecided, awaiting):
        row = _row(stores, request_id)
        with stores.engine.begin() as conn:  # tool_call_records has a turn foreign key
            conn.execute(sa_text("""
                INSERT INTO turn_logs (id, created_at, path, stream, bot_id, user_id, status, user_prompt,
                                       response_text, ended_at)
                VALUES (:id, now(), '/test', true, :bot, 'nick', 'ok', '', '', now())
            """), {"id": row.turn_id, "bot": row.bot_id})
        cards.save_start(turn_id=row.turn_id, bot_id=row.bot_id, user_id=row.user_id, call_id=row.tool_use_id,
                         tool_name="ops_run", tool_use_id=row.tool_use_id)
    purge_ops(stores.engine, stamp=RunStamp.new("snark"))
    for request_id in (undecided, awaiting):
        row = _assert_purged(stores, request_id)
        assert (json.loads(row.result_json), row.result_is_error) == (PURGED_RESULT, False)
        with stores.engine.connect() as conn:
            result_text, status, approval_id = conn.execute(sa_text(
                "SELECT result_text, approval_status, approval_request_id FROM tool_call_records WHERE tool_use_id=:id"
            ), {"id": row.tool_use_id}).one()
        assert (status, approval_id) == ("cancelled", request_id) and PURGED_BY_OPERATOR in result_text


def test_one_query_by_run_id_rebuilds_the_purge_report(stores, private_app_pg):
    """Snark ruling (a): every row a purge writes carries its actor and run id."""
    from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
    from llm_bawt.service.recovery.audit import rows_for_run

    InterBotDeliveryStore(private_app_pg)  # rows_for_run reads every stamped table

    pending = _ops_run(stores, approved=False)
    dispatching, _ = _dispatching(stores)
    job_id = _insert_job(stores, state="accepted")
    untouched = _insert_job(stores, state="succeeded")
    stamp = RunStamp.new("snark")
    report = purge_ops(stores.engine, stamp=stamp)
    assert (report["actor"], report["sweep_run_id"]) == ("snark", stamp.run_id)
    rows = rows_for_run(stores.engine, stamp.run_id)
    assert sorted(row["id"] for row in rows["approvals"]) == sorted([pending, dispatching])
    assert [row["id"] for row in rows["jobs"]] == [job_id] != [untouched]
    assert rows["turns"] == [] and rows["deliveries"] == []
    assert rows_for_run(stores.engine, RunStamp.new("snark").run_id)["approvals"] == []


class _Redis:
    def __init__(self, fail=False):
        self.fail, self.events = fail, []

    async def xadd(self, stream, fields, **kwargs):
        if self.fail:
            raise ConnectionError("redis down")
        self.events.append((stream, json.loads(fields["payload"])))


def _run_purge_step(engine, monkeypatch, redis, *, dry_run=False):
    import asyncio
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from llm_bawt.service.recovery import purge_ops as module
    from llm_bawt.service.recovery.base import SweepScope
    from llm_bawt.service.recovery.sweep import PurgeOpsStep

    monkeypatch.setattr(module, "get_agent_subscriber", lambda: SimpleNamespace(_redis=redis))
    scope = SweepScope(engine=engine, now=datetime.now(timezone.utc), dry_run=dry_run, stamp=RunStamp.new("snark"))
    return asyncio.run(PurgeOpsStep().run(scope))


def test_a_purge_publishes_each_cancelled_cards_resolution_as_the_cancel_route_does(stores, monkeypatch):
    """Snark's fan-out grant: an open client finalizes a purged card live, from
    the same ``approval_resolved`` payload the decide route publishes."""
    from llm_bawt.service.routes import approval_policies as routes

    undecided = _ops_run(stores, approved=False)
    dispatching, _ = _dispatching(stores)  # unrecallable, keeps its own result
    redis = _Redis()
    _run_purge_step(stores.engine, monkeypatch, redis)

    published = {event["request_id"]: (stream, event) for stream, event in redis.events}
    assert sorted(published) == sorted([undecided, dispatching])
    for request_id, (stream, event) in published.items():
        row = _assert_purged(stores, request_id) if request_id == undecided else _row(stores, request_id)
        route = routes._stored_mcp_resolution(row)  # what resolve_approval publishes for MCP
        assert stream == f"events:{row.bot_id}:{row.user_id}"
        assert {k: v for k, v in event.items() if k != "ts"} == {
            "_type": "approval_resolved", "bot_id": row.bot_id, "user_id": row.user_id,
            "request_id": request_id, "turn_id": row.turn_id, "status": REQ_CANCELLED,
            "tool_use_id": row.tool_use_id, "result": route["result"],
            "is_error": route["result_is_error"], "execution_state": route["execution_state"],
        }
    assert published[undecided][1]["result"] == PURGED_RESULT
    assert published[dispatching][1]["result"] == {"ok": True}


@pytest.mark.parametrize("failure", ["publish", "read_back"])
def test_a_failed_publish_never_fails_the_purge(stores, monkeypatch, failure):
    from llm_bawt.service.recovery import purge_ops as module

    request_id = _ops_run(stores, approved=False)
    if failure == "read_back":
        def broken(*args, **kwargs):
            raise RuntimeError("database gone")
        monkeypatch.setattr(module, "purged_resolutions", broken)
    report = _run_purge_step(stores.engine, monkeypatch, _Redis(fail=failure == "publish"))
    assert [entry["id"] for entry in report.done] == [request_id]
    _assert_purged(stores, request_id)


def test_a_dry_run_purge_publishes_nothing(stores, monkeypatch):
    _ops_run(stores, approved=False)
    redis = _Redis()
    report = _run_purge_step(stores.engine, monkeypatch, redis, dry_run=True)
    assert len(report.planned) == 1 and redis.events == []
