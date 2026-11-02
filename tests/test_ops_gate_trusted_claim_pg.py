"""Live-authority claim races on disposable PG, never real dispatch/runner."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json

import pytest
from sqlalchemy import text

from llm_bawt.ops import gate_claim as legacy_claim
from llm_bawt.bot_tags import catalog
from llm_bawt.bot_tags.store import BotTagStore
from llm_bawt.ops import gate_trust as trust
from llm_bawt.ops import gate_trusted_claim as claim
from llm_bawt.ops.models import JOB_TERMINAL_STATES
from llm_bawt.ops.gate_plan import GatePlan
from llm_bawt.ops.validation import canonical_json
from tests.pg_lock_wait import wait_blocked_by
from tests import test_ops_gate_trust_pg as helpers

rig = helpers.rig
configured = helpers.configured
HANG_GUARD = helpers.HANG_GUARD
pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


def _create(configured, **changes):
    return helpers._create(configured, **changes)


def _unchanged(configured, job):
    rig = configured.rig
    before = rig.store.get_job(job.id).model_dump()
    events = rig.events(job.id)
    assert claim.claim_trusted_gate_job(rig.store, job.id) is False
    assert rig.store.get_job(job.id).model_dump() == before
    assert rig.events(job.id) == events


def _change_config(configured, kind):
    rig = configured.rig
    if kind == "missing":
        rig.sql("DELETE FROM runtime_settings WHERE key=:key", key=configured.key)
        return
    value = configured.config.copy()
    if kind == "changed":
        data = json.loads(value["evidence_json"])
        data["runner_image_id"] = "sha256:" + "9" * 64
        value["evidence_json"] = canonical_json(data)
        value["evidence_sha256"] = hashlib.sha256(value["evidence_json"].encode()).hexdigest()
    elif kind == "invalid":
        value["evidence_sha256"] = "0" * 64
    elif kind == "oversize":
        value = {"padding": "x" * (trust._MAX_CONFIG + 1)}
    if kind == "duplicate":
        rig.sql("INSERT INTO runtime_settings (scope_type,scope_id,key,value_json,updated_at) "
            "SELECT scope_type,scope_id,key,value_json,updated_at FROM runtime_settings WHERE key=:key",
            key=configured.key)
    elif kind == "bot-override":
        rig.sql("UPDATE runtime_settings SET scope_type='bot',scope_id=:bot WHERE key=:key",
            bot=rig.bot, key=configured.key)
    else:
        rig.sql("UPDATE runtime_settings SET value_json=:raw WHERE key=:key",
            raw=canonical_json(value), key=configured.key)


def test_first_claim_revalidates_durable_identity_without_expired_token_and_is_not_dispatch(configured, monkeypatch):
    rig = configured.rig
    job = _create(configured)
    # The public generic/legacy entry points have not been widened.
    assert rig.store.claim_job(job.id) is False
    assert rig.claim(job.id) is False

    def forbidden(*args, **kwargs):
        pytest.fail("durable claim reopened an expired capability or tried a generic claim")

    monkeypatch.setattr(legacy_claim, "open_task_turn_context", forbidden)
    monkeypatch.setattr(trust, "open_task_turn_context", forbidden)
    monkeypatch.setattr(rig.store, "claim_job", forbidden)
    assert claim.claim_trusted_gate_job(rig.store, job.id)
    current = rig.store.get_job(job.id)
    assert current.state == "dispatching" and current.dispatched_at is not None
    assert current.worker_slot_at is None and current.host_unit_name is None
    assert current.receipt_callback_json is None and current.exit_code is None
    assert current.status_file_path is None and current.finished_at is None
    _unchanged(configured, job)
    assert sum("dispatching" in row[0] for row in rig.events(job.id)[1]) == 1


@pytest.mark.parametrize("kind", ["missing", "changed", "invalid", "duplicate", "oversize", "bot-override"])
def test_frozen_config_is_not_current_permission(configured, kind):
    job = _create(configured)
    _change_config(configured, kind)
    assert trust.read_trusted_gate_request(configured.rig.store.get_job(job.id))
    _unchanged(configured, job)


@pytest.mark.parametrize("mutation", ["missing_profile", "missing_assignment", "unknown_tag", "private", "inactive"])
def test_current_caller_grants_and_profile_are_required(configured, mutation):
    rig = configured.rig
    job = _create(configured)
    if mutation == "missing_profile":
        rig.sql("DELETE FROM bot_profiles WHERE slug=:bot", bot=rig.bot)
    elif mutation == "missing_assignment":
        BotTagStore(rig.engine).unassign(rig.role, rig.bot)
    elif mutation == "unknown_tag":
        with rig.engine.begin() as conn:
            catalog.lock_bot(conn, rig.bot)
            conn.execute(text("DELETE FROM bot_profile_tags WHERE tag_slug=:tag"), {"tag": rig.role})
            conn.execute(text("DELETE FROM bot_tag_catalog WHERE slug=:tag"), {"tag": rig.role})
    else:
        with rig.engine.begin() as conn:
            catalog.lock_bot(conn, rig.bot)
            catalog.assign(conn, rig.bot, [mutation], "fixture")
    _unchanged(configured, job)


@pytest.mark.parametrize("change", [{"enabled": False}, {"execution_tag": None},
    {"execution_tag": "fixture_unknown"}, {"title": "new-title"}, {"max_concurrent": 1},
    {"quiesce_enabled": True}, {"timeout_seconds": 301}])
def test_full_live_catalog_and_execution_limits_must_match(configured, change):
    rig = configured.rig
    job = _create(configured)
    rig.store.update_operation(rig.op.id, change)
    _unchanged(configured, job)


def test_unversioned_grant_script_and_soft_delete_edits_refuse(configured):
    rig = configured.rig
    job = _create(configured)
    for column, value in (("execution_tag", "fixture_unknown"), ("command_script", "tampered"),
            ("soft_deleted_at", "2025-01-01T00:00:00Z")):
        # Test each edit against the original otherwise-identical catalog.
        with rig.engine.connect() as conn:
            conn.begin()
            conn.execute(text(f"UPDATE ops_operations SET {column}=:value WHERE id=:id"),
                {"value": value, "id": rig.op.id})
            assert claim._claim_trusted(conn, job.id, events=rig.store.events) is False
            conn.rollback()
    assert claim.claim_trusted_gate_job(rig.store, job.id)


@pytest.mark.parametrize("failure", ["forged", "rehash", "key-rotated", "key-unavailable", "legacy"])
def test_provenance_must_authenticate_before_any_transition(configured, monkeypatch, failure):
    rig = configured.rig
    job = _create(configured)
    if failure in {"forged", "rehash", "legacy"}:
        snapshot = json.loads(job.invocation_snapshot_json)
        if failure == "legacy":
            snapshot = trust.gate_claim_snapshot(rig.op, rig.plan, turn_token=rig.token)
        else:
            snapshot["caller"]["session_id"] = "00000000-0000-0000-0000-000000000001"
            if failure == "rehash":
                snapshot.pop("snapshot_hash")
                snapshot["snapshot_hash"] = hashlib.sha256(canonical_json(snapshot).encode()).hexdigest()
        rig.sql("UPDATE ops_jobs SET invocation_snapshot_json=:raw WHERE id=:id",
            raw=canonical_json(snapshot), id=job.id)
    else:
        def derived(label):
            if failure == "key-rotated":
                return b"different-fixture-key".ljust(32, b"!")
            raise RuntimeError("fixture unavailable")

        monkeypatch.setattr(trust.crypto, "derive_key", derived)
    _unchanged(configured, job)


@pytest.mark.parametrize("state", sorted(JOB_TERMINAL_STATES))
def test_terminal_is_not_positive_retirement_or_attempt_two(configured, state):
    rig = configured.rig
    first = _create(configured)
    assert claim.claim_trusted_gate_job(rig.store, first.id)
    rig.store.mark_terminal(first.id, state=state)
    _unchanged(configured, first)
    second = _create(configured, idempotency_key="fixture-no-attempt-two")
    _unchanged(configured, second)


@pytest.mark.parametrize("state", ["lost", "timed_out", "cancelled"])
def test_uncertain_terminal_even_without_dispatch_fences_new_keys(configured, state):
    first = _create(configured)
    configured.rig.store.mark_terminal(first.id, state=state)
    _unchanged(configured, _create(configured, idempotency_key="fixture-no-retirement"))


def test_rollback_leaves_no_claim_event_or_attempt(configured):
    rig = configured.rig
    job = _create(configured)
    before, events = job.model_dump(), rig.events(job.id)
    with rig.engine.connect() as conn:
        conn.begin()
        assert claim._claim_trusted(conn, job.id, events=rig.store.events)
        conn.rollback()
    assert rig.store.get_job(job.id).model_dump() == before
    assert rig.events(job.id) == events
    assert claim.claim_trusted_gate_job(rig.store, job.id)


@pytest.mark.parametrize("change", ["config", "revoke", "catalog"])
def test_change_committing_before_claim_denies_after_observed_lock_wait(configured, change):
    rig = configured.rig
    job = _create(configured)
    with ThreadPoolExecutor(1) as pool, rig.engine.connect() as holder:
        holder.begin()
        pid = holder.execute(text("SELECT pg_backend_pid()")).scalar_one()
        if change == "config":
            holder.execute(text("DELETE FROM runtime_settings WHERE key=:key"), {"key": configured.key})
        elif change == "revoke":
            catalog.lock_bot(holder, rig.bot)
            holder.execute(text("DELETE FROM bot_profile_tags WHERE bot_slug=:bot AND tag_slug=:tag"),
                {"bot": rig.bot, "tag": rig.role})
        else:
            holder.execute(text("UPDATE ops_operations SET execution_tag=NULL WHERE id=:id"), {"id": rig.op.id})
        future = pool.submit(claim.claim_trusted_gate_job, rig.other, job.id)
        try:
            wait_blocked_by(rig.engine, pid, future)
        finally:
            holder.commit()
        assert future.result(timeout=HANG_GUARD) is False
    _unchanged(configured, job)


@pytest.mark.parametrize("change", ["config", "revoke", "catalog"])
def test_changes_after_winning_claim_wait_until_commit(configured, change):
    rig = configured.rig
    job = _create(configured)

    def edit():
        if change == "config":
            _change_config(configured, "missing")
        elif change == "revoke":
            BotTagStore(rig.engine).unassign(rig.role, rig.bot)
        else:
            rig.store.update_operation(rig.op.id, {"execution_tag": None})

    with ThreadPoolExecutor(1) as pool, rig.engine.connect() as holder:
        holder.begin()
        pid = holder.execute(text("SELECT pg_backend_pid()")).scalar_one()
        assert claim._claim_trusted(holder, job.id, events=rig.store.events)
        future = pool.submit(edit)
        try:
            wait_blocked_by(rig.engine, pid, future)
        finally:
            holder.commit()
        future.result(timeout=HANG_GUARD)
    assert rig.store.get_job(job.id).state == "dispatching"
    _unchanged(configured, job)


def test_same_job_concurrent_claim_has_one_transition(configured):
    rig = configured.rig
    job = _create(configured)
    with ThreadPoolExecutor(2) as pool, rig.engine.connect() as holder:
        holder.begin()
        pid = holder.execute(text("SELECT pg_backend_pid()")).scalar_one()
        holder.execute(text("SELECT 1 FROM ops_operations WHERE id=:id FOR UPDATE"), {"id": rig.op.id})
        futures = [pool.submit(claim.claim_trusted_gate_job, store, job.id) for store in (rig.store, rig.other)]
        try:
            for future in futures:
                wait_blocked_by(rig.engine, pid, future)
        finally:
            holder.commit()
        assert sorted(future.result(timeout=HANG_GUARD) for future in futures) == [False, True]
    assert sum("dispatching" in row[0] for row in rig.events(job.id)[1]) == 1


def test_same_sha_across_operations_and_authenticated_callers_has_one_first_attempt(configured):
    rig = configured.rig
    first = _create(configured)
    peer = rig.peer()
    other = _create(configured, operation_slug=rig.operation().slug, turn_token=peer,
        idempotency_key="fixture-other-authenticated-request")
    with ThreadPoolExecutor(1) as pool, rig.engine.connect() as holder:
        holder.begin()
        pid = holder.execute(text("SELECT pg_backend_pid()")).scalar_one()
        assert claim._claim_trusted(holder, first.id, events=rig.store.events)
        future = pool.submit(claim.claim_trusted_gate_job, rig.other, other.id)
        try:
            wait_blocked_by(rig.engine, pid, future)
        finally:
            holder.commit()
        assert future.result(timeout=HANG_GUARD) is False
    _unchanged(configured, other)


def test_trusted_and_legacy_claim_share_same_attempt_fence(configured):
    rig = configured.rig
    trusted = _create(configured)
    peer = rig.peer()
    legacy = rig.job(op=rig.operation(), caller_token=peer)
    assert claim.claim_trusted_gate_job(rig.store, trusted.id)
    assert rig.store.claim_gate_job(legacy.id, turn_token=peer) is False
    assert rig.store.get_job(legacy.id).state == "queued"


def test_same_operation_cap_counts_legacy_holder_for_a_different_sha(configured):
    rig = configured.rig
    rig.op = rig.store.update_operation(rig.op.id, {"max_concurrent": 1})
    job = _create(configured)
    other_sha = "f" * 40 if rig.plan.source_sha != "f" * 40 else "e" * 40
    busy = rig.job(op=rig.op, use_plan=GatePlan("llm-bawt", other_sha))
    assert rig.claim(busy.id)
    _unchanged(configured, job)
    rig.store.mark_terminal(busy.id, state="succeeded")
    assert claim.claim_trusted_gate_job(rig.store, job.id)


def test_nonqueued_wrong_plan_key_refuses_and_missing_job_refuses(configured):
    rig = configured.rig
    job = _create(configured)
    rig.sql("UPDATE ops_jobs SET gate_plan_key='forged' WHERE id=:id", id=job.id)
    _unchanged(configured, job)
    assert claim.claim_trusted_gate_job(rig.store, "missing") is False
