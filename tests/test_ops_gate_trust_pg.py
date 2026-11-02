"""Trusted snapshot races in disposable PG; no runtime/containers or live keys."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from threading import Barrier, Event
from types import SimpleNamespace

import pytest
from sqlalchemy import text
from sqlmodel import Session

from llm_bawt.ops import gate_trust as trust
from llm_bawt.ops.store import IdempotencyConflict
from llm_bawt.ops.validation import canonical_json
from llm_bawt.runtime_settings import RuntimeSetting
from tests import test_ops_gate_claim_pg as claim_helpers
from tests.pg_lock_wait import wait_blocked_by
from tests.test_ops_gate_runner import evidence

rig = claim_helpers.rig
HANG_GUARD = claim_helpers.HANG_GUARD
pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.fixture
def configured(rig, monkeypatch):
    monkeypatch.setattr(trust.crypto, "derive_key", lambda label: b"fixture-gate-key".ljust(32, b"!"))
    RuntimeSetting.__table__.create(rig.engine, checkfirst=True)
    raw = canonical_json(evidence(SimpleNamespace(source_sha=rig.plan.source_sha, tree_sha="b" * 40)))
    config = {"schema": trust._CONFIG_SCHEMA, "evidence_json": raw,
        "evidence_sha256": hashlib.sha256(raw.encode()).hexdigest()}
    key = trust.environment_setting_key(rig.plan)
    with Session(rig.engine) as session:
        session.add(RuntimeSetting(scope_type="global", scope_id="*", key=key,
            value_json=canonical_json(config)))
        session.commit()
    yield SimpleNamespace(rig=rig, config=config, key=key)
    rig.sql("DELETE FROM runtime_settings WHERE scope_type='global' AND scope_id='*' AND key=:key", key=key)


def _create(configured, *, store=None, **changes):
    rig = configured.rig
    return trust.create_trusted_gate_job(store or rig.store, **{**dict(operation_slug=rig.op.slug,
        plan=rig.plan.to_dict(), turn_token=rig.token, idempotency_key="fixture-trusted-admission"), **changes})


def test_new_proof_does_not_enable_either_existing_claim(configured):
    rig = configured.rig
    job = _create(configured)
    assert rig.store.claim_job(job.id) is False
    assert rig.claim(job.id) is False  # Strict old snapshot shape, even with a valid signed token.
    assert rig.store.get_job(job.id).model_dump() == job.model_dump()
    assert trust.read_trusted_gate_request(job).caller.bot_id == rig.bot
    assert job.receipt_callback_json is None and job.dispatched_at is None


@pytest.mark.parametrize("conflict", [False, True])
def test_concurrent_admission_uses_deterministic_proof_and_unique_ledger_key(configured, monkeypatch, conflict):
    rig = configured.rig
    barrier = Barrier(2)
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
    other_token = rig.peer() if conflict else rig.token
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(_create, configured, store=store, turn_token=token)
            for store, token in ((rig.store, rig.token), (rig.other, other_token))]
        jobs, errors = [], []
        for future in futures:
            try:
                jobs.append(future.result(timeout=HANG_GUARD))
            except IdempotencyConflict as exc:
                errors.append(exc)
    assert len(errors) == int(conflict)
    assert len({job.id for job in jobs}) == 1
    assert rig.store.count_jobs(operation_slug=rig.op.slug) == 1
    assert len(rig.events(jobs[0].id)[1]) == 1
    assert trust.read_trusted_gate_request(jobs[0])


def test_config_row_is_frozen_through_request_insert_not_a_stale_preread(configured, monkeypatch):
    rig = configured.rig
    locked, release = Event(), Event()
    original = trust._configured_environment
    holder = []

    def hold(conn, plan, *, lock=False):
        result = original(conn, plan, lock=lock)
        if lock:
            holder.append(conn.execute(text("SELECT pg_backend_pid()")).scalar_one())
            locked.set()
            assert release.wait(timeout=HANG_GUARD)
        return result

    monkeypatch.setattr(trust, "_configured_environment", hold)

    def edit():
        with rig.other.engine.begin() as conn:
            conn.execute(text("SET LOCAL lock_timeout = '20s'"))
            conn.execute(text("DELETE FROM runtime_settings WHERE scope_type='global' "
                "AND scope_id='*' AND key=:key"), {"key": configured.key})

    with ThreadPoolExecutor(2) as pool:
        creating = pool.submit(_create, configured)
        try:
            assert locked.wait(timeout=HANG_GUARD)
            editing = pool.submit(edit)
            wait_blocked_by(rig.engine, holder[0], editing)
        finally:
            release.set()
        job = creating.result(timeout=HANG_GUARD)
        editing.result(timeout=HANG_GUARD)
    assert trust.read_trusted_gate_request(job).environment_json == configured.config["evidence_json"].encode()
    assert _create(configured).id == job.id  # Replay remains read-only after config removal.
    assert rig.claim(job.id) is False  # Durable provenance is still not dispatch authority.
