"""TASK-1169 S4b: the worker slot bound on process-mode children (hermetic).

The PostgreSQL race and quiesce cases live in ``test_ops_competing_dispatch_pg``
and ``test_ops_quiesce_pg``.
"""
from __future__ import annotations

import uuid

import pytest

from llm_bawt.ops import JOB_ACCEPTED, JOB_QUEUED, JOB_SUCCEEDED
from llm_bawt.ops.worker_slots import (DEFAULT_WORKER_CONCURRENCY, WORKER_CONCURRENCY_ENV, needs_worker_slot,
                                       worker_concurrency)
from tests.test_ops_service import _make, _op_data, _store


@pytest.fixture
def bound(monkeypatch):
    def set_bound(value):
        monkeypatch.setenv(WORKER_CONCURRENCY_ENV, str(value))
    return set_bound


def _dispatch(service, slug):
    return service.dispatch_job(operation_slug=slug, args={"target": "a"}, idempotency_key=uuid.uuid4().hex)["id"]


def test_process_jobs_past_the_bound_stay_queued_without_an_error_until_a_slot_frees(bound):
    bound(2)
    store = _store()
    store.create_operation(_op_data(slug="test.proc", execution_mode="process"))
    service, executor = _make(store)
    jobs = [_dispatch(service, "test.proc") for _ in range(3)]
    rows = [store.get_job(job_id) for job_id in jobs]
    assert [row.state for row in rows] == [JOB_ACCEPTED, JOB_ACCEPTED, JOB_QUEUED]
    assert rows[0].worker_slot_at is not None and rows[1].worker_slot_at is not None
    # Waiting on a slot is not a preflight failure: the S1e deadline never starts.
    assert rows[2].worker_slot_at is None and rows[2].error_text is None
    assert len(executor.dispatch_calls) == 2

    service.get_job_status(jobs[2])
    assert store.get_job(jobs[2]).state == JOB_QUEUED
    store.mark_terminal(jobs[0], state=JOB_SUCCEEDED, exit_code=0)
    service.get_job_status(jobs[2])
    assert store.get_job(jobs[2]).state == JOB_ACCEPTED
    assert store.get_job(jobs[2]).worker_slot_at is not None


def test_isolated_jobs_neither_take_nor_wait_for_a_worker_slot(bound):
    bound(1)
    store = _store()
    store.create_operation(_op_data(slug="test.proc", execution_mode="process"))
    store.create_operation(_op_data(slug="test.iso", execution_mode="isolated"))
    service, _ = _make(store)
    held = _dispatch(service, "test.proc")
    isolated = [_dispatch(service, "test.iso") for _ in range(2)]
    assert store.get_job(held).state == JOB_ACCEPTED
    assert [store.get_job(job_id).state for job_id in isolated] == [JOB_ACCEPTED, JOB_ACCEPTED]
    assert [store.get_job(job_id).worker_slot_at for job_id in isolated] == [None, None]
    assert store.get_job(_dispatch(service, "test.proc")).state == JOB_QUEUED


def test_the_per_operation_cap_still_applies_in_process_mode(bound):
    bound(4)
    store = _store()
    store.create_operation(_op_data(slug="test.proc", execution_mode="process", max_concurrent=1))
    service, _ = _make(store)
    jobs = [_dispatch(service, "test.proc") for _ in range(2)]
    assert [store.get_job(job_id).state for job_id in jobs] == [JOB_ACCEPTED, JOB_QUEUED]


def _snapshot(kind, mode=None):
    execution, operation = {"executor_kind": kind}, {}
    if mode is not None:
        execution["execution_mode"] = operation["execution_mode"] = mode
    return {"execution": execution, "operation": operation}


@pytest.mark.parametrize("snapshot, expected", [
    (_snapshot("docker", "process"), True),
    (_snapshot("docker", "isolated"), False),
    (_snapshot("docker"), False),  # Legacy: approved as an isolated container.
    (_snapshot("release", "process"), False),  # Release never starts a worker.
])
def test_only_a_docker_process_mode_snapshot_needs_a_worker_slot(snapshot, expected):
    assert needs_worker_slot(snapshot) is expected


@pytest.mark.parametrize("raw, expected", [
    (None, DEFAULT_WORKER_CONCURRENCY), ("", DEFAULT_WORKER_CONCURRENCY), ("7", 7),
    ("0", DEFAULT_WORKER_CONCURRENCY), ("-2", DEFAULT_WORKER_CONCURRENCY), ("lots", DEFAULT_WORKER_CONCURRENCY),
])
def test_the_bound_comes_from_the_environment(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv(WORKER_CONCURRENCY_ENV, raising=False)
    else:
        monkeypatch.setenv(WORKER_CONCURRENCY_ENV, raw)
    assert worker_concurrency() == expected
