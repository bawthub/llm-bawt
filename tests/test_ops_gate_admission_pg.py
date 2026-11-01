"""Trusted admission against disposable PG; no runner or live identities."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from llm_bawt.ops.gate_admission import create_gate_job
from llm_bawt.ops.gate_plan import GatePlan
from llm_bawt.ops.service import OpsService
from llm_bawt.ops.store import IdempotencyConflict
from tests import test_ops_gate_claim_pg as claim_helpers
from tests.test_ops_service import FakeExecutor

rig = claim_helpers.rig
HANG_GUARD = claim_helpers.HANG_GUARD
pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


def _create(rig, *, store=None, **changes):
    return create_gate_job(store or rig.store, **{**dict(operation_slug=rig.op.slug,
        plan=rig.plan.to_dict(), turn_token=rig.token, idempotency_key="fixture-admission"), **changes})


def test_admitted_job_retains_generic_dispatch_exclusion_and_separate_gate_authorization(rig):
    job = _create(rig)
    executor = FakeExecutor()
    service = OpsService(rig.store, executor=executor)
    assert service.get_job_status(job.id)["state"] == "queued"
    assert rig.store.claim_job(job.id) is False
    assert executor.dispatch_calls == executor.reconcile_calls == []
    assert job.receipt_callback_json is None and job.worker_slot_at is None
    # Only an explicit, separately authorized gate-only claim can transition it.
    assert rig.claim(job.id)
    before = rig.store.get_job(job.id).model_dump()
    events = rig.events(job.id)
    assert _create(rig).model_dump() == before
    assert rig.events(job.id) == events


def test_revocation_after_admission_still_denies_claim_without_dispatch(rig):
    job = _create(rig)
    rig.sql("DELETE FROM bot_profile_tags WHERE bot_slug=:b AND tag_slug=:t", b=rig.bot, t=rig.role)
    assert rig.claim(job.id) is False
    assert rig.store.get_job(job.id).model_dump() == job.model_dump()
    assert job.dispatched_at is None


@pytest.mark.parametrize("conflict", [False, True])
def test_concurrent_key_admission_uses_existing_unique_ledger_fence(rig, monkeypatch, conflict):
    barrier = Barrier(2)
    # Both admissions observe a missing key before either insert. Conflict
    # recovery uses the normal ledger reader, without re-entering the barrier.
    for store in (rig.store, rig.other):
        original = store.get_job_by_key

        def reader(original=original):
            first = True

            def get(key):
                nonlocal first
                result = original(key)
                if first:
                    first = False
                    assert result is None
                    barrier.wait(timeout=HANG_GUARD)
                return result

            return get

        monkeypatch.setattr(store, "get_job_by_key", reader())

    other_plan = GatePlan("llm-bawt", "b" * 40).to_dict() if conflict else rig.plan.to_dict()
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(_create, rig, store=store, plan=plan) for store, plan in
            ((rig.store, rig.plan.to_dict()), (rig.other, other_plan))]
        results, errors = [], []
        for future in futures:
            try:
                results.append(future.result(timeout=HANG_GUARD))
            except IdempotencyConflict as exc:
                errors.append(exc)
    assert len(errors) == int(conflict)
    assert len({job.id for job in results}) == 1
    job = results[0]
    assert rig.store.count_jobs(operation_slug=rig.op.slug) == 1
    assert len(rig.events(job.id)[1]) == 1
    assert job.state == "queued" and job.dispatched_at is None
    assert job.receipt_callback_json is None and job.worker_slot_at is None
