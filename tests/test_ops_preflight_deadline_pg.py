"""TASK-1169 S1e: a queued job whose preflight keeps failing fails at a deadline.

PostgreSQL only: the expiry is one CAS on persisted state against the DB clock.
"""

from __future__ import annotations

import json
import threading
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from llm_bawt.approval_models import OPS_RUN_TOOL
from llm_bawt.approval_policies import KIND_MCP
from llm_bawt.ops import JOB_DISPATCHING, JOB_FAILED, JOB_QUEUED, OpsDispatchError, OpsService
from llm_bawt.ops.service import PREFLIGHT_DEADLINE_ENV
from llm_bawt.service.approval_continuations import _terminal_ops_result
from tests.test_ops_service import FakeExecutor, _op_data

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.fixture
def rig(private_app_pg, monkeypatch):
    from llm_bawt.ops.store import OpsStore

    monkeypatch.setenv(PREFLIGHT_DEADLINE_ENV, "300")
    store = OpsStore(private_app_pg)
    slugs = []
    executor = FakeExecutor(available=False)  # Every preflight fails.
    service = OpsService(store, executor=executor)

    def operation(**over):
        slug = f"test.preflight-{uuid.uuid4().hex[:12]}"
        store.create_operation(_op_data(slug=slug, **over))
        slugs.append(slug)
        return slug

    def job(slug):
        try:
            return service.dispatch_job(operation_slug=slug, args={"target": "a"},
                                        idempotency_key=uuid.uuid4().hex)["id"]
        except OpsDispatchError:
            return store.list_jobs(operation_slug=slug)[0].id

    def age(job_id, seconds):
        with store.engine.begin() as conn:
            conn.execute(text("UPDATE ops_jobs SET submitted_at = now() - make_interval(secs => :s)"
                              " WHERE id = :id"), {"s": seconds, "id": job_id})

    def tick(job_id):
        service._dispatch_queued(store.get_job(job_id))
        return store.get_job(job_id)

    yield SimpleNamespace(store=store, service=service, executor=executor, operation=operation,
                          job=job, age=age, tick=tick)
    with store.engine.begin() as conn:
        for slug in slugs:
            conn.execute(text("DELETE FROM ops_jobs WHERE operation_slug = :s"), {"s": slug})
            conn.execute(text("DELETE FROM ops_operation_revisions WHERE operation_slug = :s"), {"s": slug})
            conn.execute(text("DELETE FROM ops_operations WHERE slug = :s"), {"s": slug})


def test_preflight_failing_past_the_deadline_fails_with_the_code(rig):
    job_id = rig.job(rig.operation())
    assert rig.tick(job_id).state == JOB_QUEUED  # Fresh: stays queued with its reason.
    rig.age(job_id, 301)
    job = rig.tick(job_id)
    assert job.state == JOB_FAILED
    assert job.error_text == "preflight_failed: executor not available"
    assert job.finished_at is not None and job.dispatched_at is None
    assert rig.executor.dispatch_calls == []


def test_deadline_counts_the_start_delay(rig):
    job_id = rig.job(rig.operation(start_delay_seconds=600))
    rig.age(job_id, 800)
    assert rig.tick(job_id).state == JOB_QUEUED  # 800 < 300 + 600.
    rig.age(job_id, 901)
    assert rig.tick(job_id).state == JOB_FAILED


def test_job_without_a_noted_preflight_error_never_expires(rig):
    # A job waiting on a concurrency slot has no error_text: claim_job just
    # returned False. Past any deadline it stays queued.
    job_id = rig.job(rig.operation())
    with rig.store.engine.begin() as conn:
        conn.execute(text("UPDATE ops_jobs SET error_text = NULL WHERE id = :id"), {"id": job_id})
    rig.age(job_id, 10_000)
    assert rig.store.fail_stale_queued(job_id, deadline_seconds=300) is False
    assert rig.store.get_job(job_id).state == JOB_QUEUED


def test_slot_capped_job_past_the_deadline_stays_queued(rig):
    slug = rig.operation(max_concurrent=1)
    rig.executor._available = True
    holder = rig.job(slug)  # Claims the only slot and stays active.
    waiting = rig.job(slug)
    assert rig.store.get_job(holder).state != JOB_QUEUED
    # An earlier preflight failure left its reason; preflight now passes and
    # the claim is refused for the slot, so the deadline never applies.
    with rig.store.engine.begin() as conn:
        conn.execute(text("UPDATE ops_jobs SET error_text = 'executor not available' WHERE id = :id"),
                     {"id": waiting})
    rig.age(waiting, 10_000)
    assert rig.tick(waiting).state == JOB_QUEUED


def test_claim_that_commits_first_wins_the_race(rig):
    job_id = rig.job(rig.operation())
    rig.age(job_id, 10_000)
    outcome = {}
    claim = rig.store.engine.connect()
    tx = claim.begin()
    # The claim's QUEUED CAS holds the row lock, uncommitted.
    claim.execute(text("UPDATE ops_jobs SET state = :d, dispatched_at = now(), error_text = NULL"
                       " WHERE id = :id AND state = :q"), {"d": JOB_DISPATCHING, "q": JOB_QUEUED, "id": job_id})
    racer = threading.Thread(target=lambda: outcome.setdefault(
        "failed", rig.store.fail_stale_queued(job_id, deadline_seconds=300)))
    racer.start()
    racer.join(1.0)
    assert racer.is_alive()  # Blocked on the claim's row lock.
    tx.commit()
    claim.close()
    racer.join(10)
    assert outcome == {"failed": False}
    assert rig.store.get_job(job_id).state == JOB_DISPATCHING


def test_expiry_that_commits_first_makes_the_claim_lose(rig):
    job_id = rig.job(rig.operation())
    rig.age(job_id, 10_000)
    assert rig.store.fail_stale_queued(job_id, deadline_seconds=300) is True
    rig.executor._available = True
    assert rig.store.claim_job(job_id) is False
    assert rig.store.get_job(job_id).state == JOB_FAILED


@pytest.mark.parametrize("state", [JOB_FAILED, JOB_DISPATCHING])
def test_only_a_queued_job_can_expire(rig, state):
    # A terminal or claimed row can still carry error_text and an old
    # submitted_at; it must never be (re)failed or re-prefixed.
    job_id = rig.job(rig.operation())
    with rig.store.engine.begin() as conn:
        conn.execute(text("UPDATE ops_jobs SET state = :s, error_text = 'boom' WHERE id = :id"),
                     {"s": state, "id": job_id})
    rig.age(job_id, 10_000)
    assert rig.store.fail_stale_queued(job_id, deadline_seconds=300) is False
    job = rig.store.get_job(job_id)
    assert (job.state, job.error_text) == (state, "boom")


def test_failed_job_completes_the_approval_continuation(rig):
    job_id = rig.job(rig.operation())
    row = SimpleNamespace(request_kind=KIND_MCP, tool_name=OPS_RUN_TOOL,
                          result_json=json.dumps({"job_id": job_id, "state": "queued"}))
    host = SimpleNamespace(_ops_service=rig.service)
    rig.age(job_id, 301)
    rig.tick(job_id)
    result = _terminal_ops_result(host, row)
    assert result["terminal"] is True and result["state"] == JOB_FAILED
    assert result["error_text"].startswith("preflight_failed: ")


@pytest.mark.parametrize("raw,expected", [("", 300), ("abc", 300), ("0", 300), ("-5", 300), ("45", 45)])
def test_deadline_setting(monkeypatch, raw, expected):
    from llm_bawt.ops.service import preflight_deadline_seconds

    monkeypatch.setenv(PREFLIGHT_DEADLINE_ENV, raw)
    assert preflight_deadline_seconds() == expected
