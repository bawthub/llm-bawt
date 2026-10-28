"""TASK-1169 S1d: operator and agent job-status reads never write job state.

The ops reconciler is the one writer. A status read only reports, with
``observed_at`` so a caller can compare it to ``last_reconcile_at``.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from llm_bawt.mcp_server import ops_tools
from llm_bawt.ops import JOB_QUEUED, OpsDispatchError
from llm_bawt.service.routes import ops as ops_routes
from tests.test_ops_service import _make, _op_data, _store


@pytest.fixture
def queued():
    """A job left QUEUED by a failed preflight; its executor is then healthy,
    so any reconciling read would dispatch it."""
    store = _store()
    store.create_operation(_op_data())
    service, executor = _make(store)
    executor._available = False
    try:
        service.dispatch_job(operation_slug="test.restart-thing", args={"target": "a"}, idempotency_key="k1")
    except OpsDispatchError:
        pass
    [job] = store.list_jobs(operation_slug="test.restart-thing")
    assert job.state == JOB_QUEUED
    executor._available = True
    executor.dispatch_calls.clear()
    return store, service, executor, job.id


def _unchanged(store, executor, job_id):
    job = store.get_job(job_id)
    assert job.state == JOB_QUEUED and job.dispatched_at is None
    assert job.last_reconcile_at is None
    assert executor.dispatch_calls == []


def test_http_job_status_is_a_read_only(queued, monkeypatch):
    store, service, executor, job_id = queued
    monkeypatch.setattr(ops_routes, "_service", lambda: service)
    result = ops_routes.get_job(job_id)
    assert result["state"] == JOB_QUEUED
    assert datetime.fromisoformat(result["observed_at"]).tzinfo is not None
    _unchanged(store, executor, job_id)


def test_mcp_job_status_is_a_read_only(queued, monkeypatch):
    store, service, executor, job_id = queued
    monkeypatch.setattr(ops_tools, "_get_ops_service", lambda: service)
    monkeypatch.setattr(ops_tools, "_caller_provenance", lambda: {})
    monkeypatch.setattr(ops_tools, "_caller_instruction", lambda *_: None)
    tool = getattr(ops_tools.ops_job_status, "fn", ops_tools.ops_job_status)
    result = asyncio.run(tool(job_id))
    assert result["state"] == JOB_QUEUED and "observed_at" in result
    _unchanged(store, executor, job_id)


def test_the_reconciler_still_advances_the_job(queued):
    store, service, executor, job_id = queued
    service.reconcile_active_jobs()
    assert store.get_job(job_id).state != JOB_QUEUED
    assert len(executor.dispatch_calls) == 1
