"""Dormant quarantine policy: classify UNKNOWN, never infer mutation completion."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import inspect, text

from llm_bawt.ops.models import JOB_FAILED, JOB_LOST, JOB_SUCCEEDED, JOB_TERMINAL_STATES
from llm_bawt.ops.process_spawn_fence import process_spawn_guard
from llm_bawt.ops.executor_types import SpawnDeferred
from llm_bawt.ops.spawn_hold_recovery import VerifiedHoldOutcome
from llm_bawt.ops.store import OpsStore
from tests.test_ops_service import _store
from tests.test_ops_spawn_hold import _admit, _hold

TARGET = {"container_id": "a" * 64, "compose_project": "test", "compose_service": "ops"}


def _deadline():
    return datetime.now(timezone.utc) + timedelta(minutes=5)


def _prepared(store, root=None):
    job_id = _admit(store)
    # FakeExecutor has no receipt mount settings. Bind this isolated fixture's
    # snapshot before preparing the policy; production snapshots pin this root.
    snapshot = json.loads(store.get_job(job_id).invocation_snapshot_json)
    snapshot["execution"]["receipt_root"] = str(root or "/unused-test-receipts")
    with store.engine.begin() as conn:
        conn.execute(text("UPDATE ops_jobs SET invocation_snapshot_json=:raw WHERE id=:id"),
                     {"raw": json.dumps(snapshot), "id": job_id})
    assert store.prepare_process_spawn_hold(job_id, target_identity=TARGET, deadline_at=_deadline())
    return job_id


def _expire(store, job_id):
    # Deterministic passage of persisted deadline, no sleeping/test wall clock race.
    with store.engine.begin() as conn:
        conn.execute(text("UPDATE ops_jobs SET process_spawn_hold_deadline_at=:past WHERE id=:id"),
                     {"past": datetime(2000, 1, 1, tzinfo=timezone.utc), "id": job_id})


def _evidence(store, job_id, *, authority_id=None, outcome="verified_success"):
    authority = store.get_job(authority_id or job_id)
    return VerifiedHoldOutcome(job_id, TARGET, authority.id,
                               hashlib.sha256(authority.invocation_snapshot_json.encode()).hexdigest(), outcome)


def _release(store, job_id, root, evidence):
    return store.release_process_spawn_hold(job_id, receipt_root=root, acquire_deadline=_deadline(),
                                             verifier=lambda _job: evidence)


def test_preparation_pins_target_and_budget_before_retries_without_taking_hold():
    store = _store()
    job_id = _prepared(store)
    before = store.get_job(job_id)
    assert not store.process_spawns_held() and before.process_spawn_hold_at is None
    assert not store.prepare_process_spawn_hold(job_id, target_identity={**TARGET, "container_id": "b" * 64},
                                                deadline_at=_deadline() + timedelta(days=1))
    after = store.get_job(job_id)
    assert (after.process_spawn_hold_target_json, after.process_spawn_hold_deadline_at) == (
        before.process_spawn_hold_target_json, before.process_spawn_hold_deadline_at)
    assert not store.claim_job(job_id)  # No accidental ordinary-dispatch bypass.
    _expire(store, job_id)
    assert not _hold(store, job_id)  # No hold or side effect after acquisition budget.
    assert not store.process_spawns_held()


@pytest.mark.parametrize("state", sorted(JOB_TERMINAL_STATES))
def test_prepared_hold_survives_every_terminal_state_and_store_restart(state, tmp_path):
    store = _store()
    job_id = _prepared(store)
    assert _hold(store, job_id)
    store.mark_terminal(job_id, state=state)
    restarted = object.__new__(OpsStore)
    restarted.engine = store.engine
    restarted._ensure_tables_exist()
    assert restarted.process_spawns_held()
    assert restarted.classify_process_spawn_hold(job_id) == "quarantined"
    assert restarted.get_job(job_id).state == state  # Never rewrite original terminal history.
    assert restarted.get_job(job_id).to_api()["process_spawn_hold"]["quarantined"] is True
    with pytest.raises(SpawnDeferred):
        with process_spawn_guard(tmp_path, restarted.process_spawns_held):
            pytest.fail("terminalization must not admit a process spawn")
    assert not restarted.claim_job(_admit(restarted, mode="process"))
    assert restarted.claim_job(_admit(restarted))  # Unrelated isolated ops/recovery remains possible.
    assert not _hold(restarted, _admit(restarted))  # Another holder cannot bypass the quarantine.


def test_deadline_terminalizes_open_holder_without_claiming_death_or_clearing_quarantine():
    store = _store()
    job_id = _prepared(store)
    assert _hold(store, job_id)
    store.mark_accepted(job_id)
    assert store.classify_process_spawn_hold(job_id) == "waiting"
    _expire(store, job_id)
    assert store.classify_process_spawn_hold(job_id) == "quarantined"
    before = store.get_job(job_id)
    assert before.state == JOB_LOST and before.process_spawn_hold_quarantine_at is not None
    assert "inspection required" in before.error_text and store.process_spawns_held()
    assert store.classify_process_spawn_hold(job_id) == "quarantined"
    assert store.get_job(job_id).process_spawn_hold_quarantine_at == before.process_spawn_hold_quarantine_at
    # Late receipt import cannot overwrite LOST or release the independent hold.
    store.mark_terminal(job_id, state=JOB_SUCCEEDED, exit_code=0)
    assert store.get_job(job_id).state == JOB_LOST and store.process_spawns_held()


@pytest.mark.parametrize("state,outcome", [(JOB_SUCCEEDED, "verified_success"), (JOB_FAILED, "verified_restore")])
def test_normal_known_safe_outcomes_require_explicit_verified_release_and_retain_audit(state, outcome, tmp_path):
    store = _store()
    job_id = _prepared(store, tmp_path)
    assert _hold(store, job_id)
    store.mark_terminal(job_id, state=state)
    assert store.process_spawns_held()  # A generic success/failed state alone is not safety evidence.
    evidence = _evidence(store, job_id, outcome=outcome)

    def verified(job):
        assert job.id == job_id
        # Verification is outside a DB transaction, inside the retained EX.
        with pytest.raises(SpawnDeferred):
            with process_spawn_guard(tmp_path, lambda: False):
                pytest.fail("verifier requires exclusive authority fence")
        return evidence

    assert store.release_process_spawn_hold(job_id, receipt_root=tmp_path,
                                            acquire_deadline=_deadline(), verifier=verified)
    assert not store.process_spawns_held()
    assert store.classify_process_spawn_hold(job_id) == "released"
    row = store.get_job(job_id)
    assert row.state == state and row.process_spawn_hold_at is not None
    assert json.loads(row.process_spawn_hold_release_json)["authority_snapshot_hash"] == evidence.authority_snapshot_hash
    assert not _release(store, job_id, tmp_path, evidence)  # Immutable release evidence, no duplicate verifier.
    assert store.claim_job(_admit(store, mode="process"))


@pytest.mark.parametrize("change", ["wrong_target", "wrong_labels", "wrong_job", "unknown_outcome",
                                     "wrong_snapshot", "missing_authority", "unknown_authority"])
def test_wrong_or_unknown_release_evidence_never_reopens_spawn_admission(change, tmp_path):
    store = _store()
    job_id = _prepared(store, tmp_path)
    assert _hold(store, job_id)
    store.mark_terminal(job_id, state=JOB_SUCCEEDED)
    evidence = _evidence(store, job_id)
    if change == "wrong_target":
        evidence = replace(evidence, target_identity={**TARGET, "container_id": "b" * 64})
    elif change == "wrong_labels":
        evidence = replace(evidence, target_identity={**TARGET, "compose_service": "other"})
    elif change == "wrong_job":
        evidence = replace(evidence, job_id="different")
    elif change == "unknown_outcome":
        evidence = replace(evidence, outcome="pid_absent")
    elif change == "wrong_snapshot":
        evidence = replace(evidence, authority_snapshot_hash="0" * 64)
    elif change == "missing_authority":
        evidence = replace(evidence, authority_job_id="missing")
    else:
        authority = _admit(store)
        store.mark_terminal(authority, state=JOB_LOST)
        evidence = _evidence(store, job_id, authority_id=authority)
    assert not _release(store, job_id, tmp_path, evidence)
    assert store.process_spawns_held() and store.get_job(job_id).process_spawn_hold_release_json is None


def test_unknown_original_requires_a_separately_verified_existing_recovery_action(tmp_path):
    store = _store()
    job_id = _prepared(store, tmp_path)
    assert _hold(store, job_id)
    _expire(store, job_id)
    assert store.classify_process_spawn_hold(job_id) == "quarantined"
    assert not _release(store, job_id, tmp_path, _evidence(store, job_id))
    authority = _admit(store)  # Existing admitted exact-action job, not a new approval role.
    store.mark_terminal(authority, state=JOB_SUCCEEDED)
    assert _release(store, job_id, tmp_path, _evidence(store, job_id, authority_id=authority))
    assert store.get_job(job_id).state == JOB_LOST and not store.process_spawns_held()
    assert not _hold(store, job_id)  # Original job remains no-replay.


@pytest.mark.parametrize("failure", ["missing_receipt", "unreadable_receipt", "restore_unknown", "authority_denied"])
def test_verifier_refusal_retains_quarantine_and_releases_only_the_flock(failure, tmp_path):
    store = _store()
    job_id = _prepared(store, tmp_path)
    assert _hold(store, job_id)
    store.mark_terminal(job_id, state=JOB_LOST)

    def refuse(_job):
        raise RuntimeError(failure)

    with pytest.raises(RuntimeError, match=failure):
        store.release_process_spawn_hold(job_id, receipt_root=tmp_path, acquire_deadline=_deadline(), verifier=refuse)
    assert store.process_spawns_held() and store.get_job(job_id).process_spawn_hold_release_json is None
    # A later independent verifier can still acquire the lock, not the durable authority.
    with pytest.raises(RuntimeError, match=failure):
        store.release_process_spawn_hold(job_id, receipt_root=tmp_path, acquire_deadline=_deadline(), verifier=refuse)


def test_wrong_receipt_root_cannot_substitute_an_unrelated_empty_fence(tmp_path):
    store = _store()
    job_id = _prepared(store, tmp_path)
    assert _hold(store, job_id)
    store.mark_terminal(job_id, state=JOB_SUCCEEDED)
    wrong = tmp_path / "unrelated"
    wrong.mkdir()
    assert not store.release_process_spawn_hold(
        job_id, receipt_root=wrong, acquire_deadline=_deadline(),
        verifier=lambda _job: pytest.fail("wrong fence must not reach verification"))
    assert not list(wrong.iterdir()) and store.process_spawns_held()


def test_a_running_owner_cannot_be_released_even_with_an_empty_flock(tmp_path):
    store = _store()
    job_id = _prepared(store, tmp_path)
    assert _hold(store, job_id)
    assert not store.release_process_spawn_hold(
        job_id, receipt_root=tmp_path, acquire_deadline=_deadline(),
        verifier=lambda _job: pytest.fail("active owner cannot be released"))
    assert store.process_spawns_held()


def test_recovery_columns_migrate_additively_without_arming_legacy_holds():
    store = _store()
    job_id = _admit(store)
    columns = ("process_spawn_hold_deadline_at", "process_spawn_hold_target_json",
               "process_spawn_hold_quarantine_at", "process_spawn_hold_release_json")
    with store.engine.begin() as conn:
        for column in columns:
            conn.execute(text(f"ALTER TABLE ops_jobs DROP COLUMN {column}"))
        store._migrate_add_columns(conn)
        store._migrate_add_columns(conn)
        assert set(columns) <= {col["name"] for col in inspect(conn).get_columns("ops_jobs")}
    assert _hold(store, job_id)
    store.mark_terminal(job_id, state=JOB_LOST)
    assert not store.process_spawns_held()  # Prior dormant contract, no global terminal rewrite.


@pytest.mark.parametrize("target", [{**TARGET, "container_id": "short"}, {**TARGET, "compose_project": ""}, {}])
def test_unpinned_or_incomplete_target_identity_is_rejected(target):
    store = _store()
    job_id = _admit(store)
    with pytest.raises(ValueError):
        store.prepare_process_spawn_hold(job_id, target_identity=target, deadline_at=_deadline())
    assert store.get_job(job_id).process_spawn_hold_target_json is None
