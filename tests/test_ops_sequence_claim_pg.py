"""TASK-1263: real row-lock races across normal and quiesce sequence claims."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from threading import Event
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from llm_bawt.ops import JOB_DISPATCHING, JOB_FAILED, JOB_SUCCEEDED
from llm_bawt.ops import quiesce as quiesce_module
from llm_bawt.ops import store as store_module
from llm_bawt.ops.quiesce import QuiesceDriver
from llm_bawt.ops.store import OpsStore
from llm_bawt.runtime_settings import BotProfileStore
from tests.pg_lock_wait import wait_blocked_by
from tests.test_ops_sequence_store import create_job, operation, sequence_snapshot, single_snapshot

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.fixture
def rig(private_app_pg):
    store = OpsStore(private_app_pg)
    BotProfileStore(private_app_pg)
    parent, step, independent = (operation(store, name) for name in ("parent", "step", "independent"))
    driver = QuiesceDriver(store.engine)
    ctx = SimpleNamespace(store=store, engine=store.engine, driver=driver,
                          parent=parent, step=step, independent=independent)
    yield ctx
    with store.engine.begin() as conn:
        slugs = [parent.slug, step.slug, independent.slug]
        ids = list(conn.execute(text("SELECT id FROM ops_jobs WHERE operation_slug=ANY(:slugs)"),
                                {"slugs": slugs}).scalars())
        conn.execute(text("DELETE FROM gate_members WHERE gate_id IN"
                          " (SELECT id FROM system_gates WHERE owner_id=ANY(:ids))"), {"ids": ids})
        conn.execute(text("DELETE FROM system_gates WHERE owner_id=ANY(:ids)"), {"ids": ids})
        conn.execute(text("DELETE FROM ops_jobs WHERE operation_slug=ANY(:slugs)"), {"slugs": slugs})
        conn.execute(text("DELETE FROM ops_operation_revisions WHERE operation_slug=ANY(:slugs)"), {"slugs": slugs})
        conn.execute(text("DELETE FROM ops_operations WHERE slug=ANY(:slugs)"), {"slugs": slugs})


def hold_first_fence(monkeypatch, first_job_id):
    """Pause the winner with all its catalog locks held, before its claim CAS."""
    locked, release = Event(), Event()
    observed = {}
    original = store_module.sequence_conflict

    def fence(conn, snapshot, **kwargs):
        result = original(conn, snapshot, **kwargs)
        if kwargs["exclude_job_id"] == first_job_id:
            observed["pid"] = conn.execute(text("SELECT pg_backend_pid()")).scalar_one()
            locked.set()
            assert release.wait(10), "test did not release the first operation-row claimant"
        return result

    monkeypatch.setattr(store_module, "sequence_conflict", fence)
    monkeypatch.setattr(quiesce_module, "sequence_conflict", fence)
    return locked, release, observed


@pytest.mark.parametrize("sequence_first", [False, True])
def test_normal_sequence_and_single_claim_race_has_one_winner(rig, monkeypatch, sequence_first):
    sequence = create_job(rig.store, rig.parent, sequence_snapshot(rig.store, rig.parent, [rig.step]))
    standalone = create_job(rig.store, rig.step, single_snapshot(rig.store, rig.step))
    first, second = (sequence, standalone) if sequence_first else (standalone, sequence)
    locked, release, observed = hold_first_fence(monkeypatch, first.id)
    with ThreadPoolExecutor(max_workers=2) as pool:
        winner = pool.submit(rig.store.claim_job, first.id)
        try:
            assert locked.wait(5)
            loser = pool.submit(rig.store.claim_job, second.id)
            wait_blocked_by(rig.engine, observed["pid"], loser)
        finally:
            release.set()
        assert winner.result(10) is True
        assert loser.result(10) is False
    assert rig.store.get_job(first.id).state == JOB_DISPATCHING
    assert rig.store.get_job(second.id).state == "queued"


@pytest.mark.parametrize("quiescing_sequence", [False, True])
@pytest.mark.parametrize("quiesce_first", [False, True])
def test_quiesce_dispatch_and_normal_claim_share_the_same_fence(
    rig, monkeypatch, quiescing_sequence, quiesce_first,
):
    sequence_snapshot_ = sequence_snapshot(rig.store, rig.parent, [rig.step])
    single_snapshot_ = single_snapshot(rig.store, rig.step)
    sequence = create_job(rig.store, rig.parent, sequence_snapshot_, quiesce=quiescing_sequence)
    standalone = create_job(rig.store, rig.step, single_snapshot_, quiesce=not quiescing_sequence)
    quiescing, normal = (sequence, standalone) if quiescing_sequence else (standalone, sequence)
    assert rig.driver.step(quiescing.id).outcome == "opened"

    first, second = (quiescing, normal) if quiesce_first else (normal, quiescing)
    first_call = rig.driver.step if quiesce_first else rig.store.claim_job
    second_call = rig.store.claim_job if quiesce_first else rig.driver.step
    locked, release, observed = hold_first_fence(monkeypatch, first.id)
    with ThreadPoolExecutor(max_workers=2) as pool:
        winner = pool.submit(first_call, first.id)
        try:
            assert locked.wait(5)
            loser = pool.submit(second_call, second.id)
            wait_blocked_by(rig.engine, observed["pid"], loser)
        finally:
            release.set()
        won, lost = winner.result(10), loser.result(10)
    assert (won.outcome == "dispatched") if quiesce_first else (won is True)
    assert (lost is False) if quiesce_first else (lost.outcome == "waiting_slot")
    assert rig.store.get_job(first.id).state == JOB_DISPATCHING
    assert rig.store.get_job(second.id).state != JOB_DISPATCHING

    # Neither a rejected claim nor an opened quiesce gate reserves the shared
    # operation indefinitely: the waiting claimant can run once execution ends.
    rig.store.mark_terminal(first.id, state=JOB_SUCCEEDED)
    retry = second_call(second.id)
    assert (retry is True) if quiesce_first else (retry.outcome == "dispatched")


@pytest.mark.parametrize("opened", [False, True])
@pytest.mark.parametrize("edit, expected", [
    ({"title": "Edited step"}, "[operation_changed]"),
    ({"enabled": False}, "disabled"),
])
def test_quiesce_refuses_changed_or_disabled_step_and_releases_its_gate(rig, opened, edit, expected):
    job = create_job(rig.store, rig.parent, sequence_snapshot(rig.store, rig.parent, [rig.step]),
                     quiesce=True)
    if opened:
        assert rig.driver.step(job.id).outcome == "opened"
    rig.store.update_operation(rig.step.slug, edit)
    assert rig.driver.step(job.id).outcome == "failed"
    fresh = rig.store.get_job(job.id)
    assert fresh.state == JOB_FAILED
    assert expected in fresh.error_text
    assert fresh.dispatched_at is None
    assert [s["state"] for s in json.loads(fresh.sequence_json)["steps"]] == ["skipped"]
    with rig.engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM system_gates WHERE owner_id=:id"
                                 " AND released_at IS NULL"), {"id": job.id}).scalar_one() == 0


def test_independent_operation_claim_does_not_wait_on_sequence_catalog_locks(rig, monkeypatch):
    sequence = create_job(rig.store, rig.parent, sequence_snapshot(rig.store, rig.parent, [rig.step]))
    independent = create_job(rig.store, rig.independent, single_snapshot(rig.store, rig.independent))
    locked, release, observed = hold_first_fence(monkeypatch, sequence.id)
    with ThreadPoolExecutor(max_workers=2) as pool:
        winner = pool.submit(rig.store.claim_job, sequence.id)
        try:
            assert locked.wait(5)
            independent_claim = pool.submit(rig.store.claim_job, independent.id)
            assert independent_claim.result(5) is True
        finally:
            release.set()
        assert winner.result(10) is True
