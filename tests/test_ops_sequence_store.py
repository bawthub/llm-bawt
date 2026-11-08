"""TASK-1263: one sequence job fences every pinned catalog operation.

These tests exercise the store boundary directly, not sequence admission or the
runner. Step snapshots are ordinary single-operation approval snapshots.
"""
from __future__ import annotations

import copy
import json
import uuid

import pytest
from sqlalchemy import event, text

from llm_bawt.ops import JOB_ACCEPTED, JOB_DISPATCHING, JOB_FAILED, JOB_RUNNING, JOB_SUCCEEDED
from llm_bawt.ops.models import JOB_QUIESCING
from llm_bawt.ops.operation_revision import OperationChanged
from llm_bawt.ops.service import OpsService
from tests.test_ops_service import FakeExecutor
from tests.test_ops_store import _op_data, _store


def operation(store, name):
    return store.create_operation(_op_data(slug=f"test.sequence-{name}-{uuid.uuid4().hex}",
                                           execution_mode="isolated", max_concurrent=2))


def single_snapshot(store, op):
    return OpsService(store, executor=FakeExecutor()).prepare_invocation(op.slug, {})


def sequence_snapshot(store, parent, steps):
    snapshot = single_snapshot(store, parent)
    # The store consumes an already-admitted snapshot; admission/spec validation
    # is covered separately. No child OpsJob is needed to reserve a step.
    snapshot["spec"] = {"action": "operation_sequence"}
    snapshot["sequence"] = {"steps": [
        {"snapshot": single_snapshot(store, op),
         "target": {"id": f"{index + 1:064x}", "name": f"container-{index}"},
         "display_args": {}}
        for index, op in enumerate(steps)
    ]}
    return snapshot


def create_job(store, op, snapshot, *, quiesce=False):
    snapshot = copy.deepcopy(snapshot)
    if quiesce:
        snapshot["quiesce"] = {"enabled": True, "timeout_seconds": 300,
                               "scope": {"bots": [f"test-sequence-{op.id}"]}}
    return store.create_job(operation=op, args_json="{}", display_args_json="{}",
                            idempotency_key=uuid.uuid4().hex,
                            invocation_snapshot_json=json.dumps(snapshot),
                            state=JOB_QUIESCING if quiesce else "queued")


@pytest.mark.parametrize("edit, expected", [
    ({"title": "Edited step metadata"}, r"\[operation_changed\]"),
    ({"enabled": False}, "disabled"),
])
def test_create_refuses_changed_or_disabled_step_without_a_job(edit, expected):
    store = _store()
    parent, step = operation(store, "parent"), operation(store, "step")
    snapshot = sequence_snapshot(store, parent, [step])
    store.update_operation(step.slug, edit)

    with pytest.raises(OperationChanged, match=expected):
        create_job(store, parent, snapshot)
    assert store.count_jobs() == 0


@pytest.mark.parametrize("edit", [
    {"title": "Edited step metadata"},
    {"enabled": False},
    {"command_script": '{"action":"start","container_name":"test"}'},
])
def test_claim_refuses_changed_or_disabled_step(edit):
    store = _store()
    parent, step = operation(store, "parent"), operation(store, "step")
    job = create_job(store, parent, sequence_snapshot(store, parent, [step]))
    store.update_operation(step.slug, edit)

    assert store.claim_job(job.id) is False
    assert store.get_job(job.id).state == "queued"


@pytest.mark.parametrize("field, value", [("id", "wrong-id"), ("version", 999), ("script_hash", "wrong-hash")])
def test_create_checks_every_step_catalog_identity(field, value):
    store = _store()
    parent, first, last = (operation(store, name) for name in ("parent", "first", "last"))
    snapshot = sequence_snapshot(store, parent, [first, last])
    snapshot["sequence"]["steps"][1]["snapshot"]["operation"][field] = value
    with pytest.raises(OperationChanged, match=r"\[operation_changed\]"):
        create_job(store, parent, snapshot)
    assert store.count_jobs() == 0


@pytest.mark.parametrize("sequence_first", [False, True])
@pytest.mark.parametrize("active_state", [JOB_DISPATCHING, JOB_ACCEPTED, JOB_RUNNING])
def test_sequence_and_standalone_claims_serialize_in_both_orders(sequence_first, active_state):
    store = _store()
    parent, step = operation(store, "parent"), operation(store, "step")
    sequence = create_job(store, parent, sequence_snapshot(store, parent, [step]))
    standalone = create_job(store, step, single_snapshot(store, step))
    first, second = (sequence, standalone) if sequence_first else (standalone, sequence)

    assert store.claim_job(first.id) is True
    if active_state != JOB_DISPATCHING:
        store.mark_accepted(first.id)
    if active_state == JOB_RUNNING:
        store.mark_running(first.id)
    assert store.get_job(first.id).state == active_state
    assert store.claim_job(second.id) is False
    store.mark_terminal(first.id, state=JOB_SUCCEEDED)
    assert store.claim_job(second.id) is True
    assert store.count_jobs() == 2  # no child job rows


def test_sequences_with_different_parents_and_a_shared_step_serialize():
    store = _store()
    left, right, shared = (operation(store, name) for name in ("left", "right", "shared"))
    first = create_job(store, left, sequence_snapshot(store, left, [shared]))
    second = create_job(store, right, sequence_snapshot(store, right, [shared]))
    assert store.claim_job(first.id) is True
    assert store.claim_job(second.id) is False
    store.mark_terminal(first.id, state=JOB_FAILED)
    assert store.claim_job(second.id) is True


def test_disjoint_sequences_and_an_independent_single_operation_do_not_conflict():
    store = _store()
    left, right, a, b, independent = (operation(store, name)
                                      for name in ("left", "right", "a", "b", "independent"))
    jobs = [
        create_job(store, left, sequence_snapshot(store, left, [a])),
        create_job(store, right, sequence_snapshot(store, right, [b])),
        create_job(store, independent, single_snapshot(store, independent)),
    ]
    assert all(store.claim_job(job.id) for job in jobs)


def test_ordinary_single_operation_max_concurrent_is_unchanged():
    store = _store()
    op = operation(store, "single")
    jobs = [create_job(store, op, single_snapshot(store, op)) for _ in range(3)]
    assert store.claim_job(jobs[0].id, max_concurrent=2) is True
    assert store.claim_job(jobs[1].id, max_concurrent=2) is True
    assert store.claim_job(jobs[2].id, max_concurrent=2) is False


@pytest.mark.parametrize("quiescing_reservation", [False, True])
def test_a_sequence_reservation_does_not_block_a_step_claim(quiescing_reservation):
    store = _store()
    parent, step = operation(store, "parent"), operation(store, "step")
    reserved = create_job(store, parent, sequence_snapshot(store, parent, [step]),
                          quiesce=quiescing_reservation)
    if quiescing_reservation:
        # Even an opened gate is not active execution of the shared step.
        with store.engine.begin() as conn:
            conn.execute(text("UPDATE ops_jobs SET quiesce_gate_id=123 WHERE id=:id"), {"id": reserved.id})
    single = create_job(store, step, single_snapshot(store, step))
    assert store.claim_job(single.id) is True


def test_create_and_claim_lock_all_catalog_rows_in_sorted_slug_order():
    store = _store()
    parent, a, b = (operation(store, name) for name in ("z-parent", "a", "b"))
    snapshot = sequence_snapshot(store, parent, [b, a, b])
    locked = []

    def record(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("UPDATE ops_operations SET version=version"):
            locked.append(parameters[0])

    event.listen(store.engine, "before_cursor_execute", record)
    try:
        job = create_job(store, parent, snapshot)
        assert locked == sorted({parent.slug, a.slug, b.slug})
        locked.clear()
        assert store.claim_job(job.id) is True
        assert locked == sorted({parent.slug, a.slug, b.slug})
    finally:
        event.remove(store.engine, "before_cursor_execute", record)
