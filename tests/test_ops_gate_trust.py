"""Dormant trusted snapshots on scratch SQLite; never dispatch or use live keys."""
from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
from sqlalchemy import text
from sqlmodel import Session

from llm_bawt import task_turn_context as codec
from llm_bawt.ops import gate_trust as trust
from llm_bawt.ops.gate_environment import validate_environment
from llm_bawt.ops.gate_plan import GatePlan
from llm_bawt.ops.models import JOB_TERMINAL_STATES
from llm_bawt.ops.store import IdempotencyConflict
from llm_bawt.ops.validation import canonical_json
from llm_bawt.runtime_settings import RuntimeSetting
from tests import test_ops_gate_admission as admission_helpers
from tests.test_ops_gate_runner import evidence

admission = admission_helpers.admission


@pytest.fixture
def trusted(admission, monkeypatch):
    labels = []

    def derived(label):
        labels.append(label)
        return b"fixture-provenance-key".ljust(32, b"!")

    monkeypatch.setattr(trust.crypto, "derive_key", derived)
    RuntimeSetting.__table__.create(admission.store.engine, checkfirst=True)
    raw = canonical_json(evidence(SimpleNamespace(source_sha=admission.plan.source_sha, tree_sha="b" * 40)))
    config = {"schema": trust._CONFIG_SCHEMA, "evidence_json": raw,
        "evidence_sha256": hashlib.sha256(raw.encode()).hexdigest()}

    def configure(value=config, *, scope_type="global", scope_id="*", key=None):
        key = key or trust.environment_setting_key(admission.plan)
        with Session(admission.store.engine) as session:
            session.add(RuntimeSetting(scope_type=scope_type, scope_id=scope_id, key=key,
                value_json=canonical_json(value)))
            session.commit()

    def create(**changes):
        return trust.create_trusted_gate_job(admission.store, **{**admission.defaults, **changes})

    configure()
    return SimpleNamespace(base=admission, create=create, configure=configure, config=config,
        labels=labels, store=admission.store)


def _delete_config(trusted):
    with trusted.store.engine.begin() as conn:
        conn.execute(text("DELETE FROM runtime_settings"))


def test_freezes_exact_config_and_signed_provenance_without_execution(trusted, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("snapshot admission cannot claim or execute")

    monkeypatch.setattr(trusted.store, "claim_job", forbidden)
    monkeypatch.setattr(trusted.store, "claim_gate_job", forbidden)
    job = trusted.create()
    result = trust.read_trusted_gate_request(job)
    assert result.plan == trusted.base.plan
    assert vars(result.caller) == trusted.base.values
    assert result.environment_json == trusted.config["evidence_json"].encode()
    assert result.environment_sha256 == trusted.config["evidence_sha256"]
    assert job.state == "queued" and job.dispatched_at is None and job.worker_slot_at is None
    assert job.receipt_callback_json is None and job.approval_request_id is None
    assert set(trusted.labels) == {trust._KEY_LABEL}
    snapshot = json.loads(job.invocation_snapshot_json)
    assert snapshot["gate_trust"]["environment"] == trusted.config
    assert trusted.base.defaults["turn_token"] not in job.invocation_snapshot_json
    assert "issued_at" not in job.invocation_snapshot_json
    assert "spec" not in snapshot


def test_enriched_snapshot_is_excluded_from_existing_generic_claim(trusted):
    job = trusted.create()
    assert trusted.store.claim_job(job.id) is False
    assert trusted.store.get_job(job.id).model_dump() == job.model_dump()


@pytest.mark.parametrize("state", ["queued", *sorted(JOB_TERMINAL_STATES)])
def test_replay_keeps_frozen_config_after_removal_and_terminalization(trusted, monkeypatch, state):
    job = trusted.create()
    if state != "queued":
        trusted.store.mark_terminal(job.id, state=state)
    _delete_config(trusted)
    trusted.store.update_operation(trusted.base.operation.id, {"enabled": False})
    before = trusted.store.get_job(job.id).model_dump()

    def forbidden(*args, **kwargs):
        pytest.fail("replay may not rebuild, read configuration or insert")

    monkeypatch.setattr(trust, "_configured_environment", forbidden)
    monkeypatch.setattr(trusted.store, "create_job", forbidden)
    token = codec.mint_task_turn_context(**trusted.base.values, issued_at=1)
    assert trusted.create(turn_token=token).model_dump() == before
    assert trust.read_trusted_gate_request(trusted.store.get_job(job.id)).environment_json == (
        trusted.config["evidence_json"].encode())


def test_durable_reader_does_not_reopen_current_turn_capability(trusted, monkeypatch):
    job = trusted.create()

    def expired(*args, **kwargs):
        raise codec.TaskTurnContextError("fixture original capability expired")

    monkeypatch.setattr(trust, "open_task_turn_context", expired)
    assert trust.read_trusted_gate_request(job).caller.turn_id == trusted.base.values["turn_id"]
    with pytest.raises(codec.TaskTurnContextError):
        trusted.create()


def test_legacy_unsigned_rows_are_not_upgraded_in_place(trusted):
    job = trusted.base.create()
    with pytest.raises(trust.GateTrustError):
        trusted.create()
    assert trusted.store.get_job(job.id).model_dump() == job.model_dump()


@pytest.mark.parametrize("token", [None, "", "forged"])
def test_invalid_identity_does_not_read_ledger_or_config(trusted, monkeypatch, token):
    def forbidden(*args, **kwargs):
        pytest.fail("unauthenticated request reached ledger")

    monkeypatch.setattr(trusted.store, "get_job_by_key", forbidden)
    with pytest.raises(codec.TaskTurnContextError):
        trusted.create(turn_token=token)
    assert trusted.store.count_jobs() == 0


@pytest.mark.parametrize("kind", ["missing", "bot", "other-sha", "duplicate", "oversize"])
def test_config_has_one_bounded_exact_source_global_value_only(trusted, kind):
    _delete_config(trusted)
    if kind == "bot":
        trusted.configure(scope_type="bot", scope_id=trusted.base.values["bot_id"])
    elif kind == "other-sha":
        trusted.configure(key=trust.environment_setting_key(GatePlan("llm-bawt", "c" * 40)))
    elif kind == "duplicate":
        trusted.configure()
        trusted.configure()
    elif kind == "oversize":
        trusted.configure({"padding": "x" * (trust._MAX_CONFIG + 1)})
    with pytest.raises(trust.GateTrustError):
        trusted.create()
    assert trusted.store.count_jobs() == 0


@pytest.mark.parametrize("change", [{"schema": "wrong"}, {"extra": True},
    {"evidence_sha256": "f" * 64}, {"evidence_json": "{}"}, {"evidence_json": 7}])
def test_malformed_or_unpinned_config_refuses(trusted, change):
    _delete_config(trusted)
    trusted.configure({**trusted.config, **change})
    with pytest.raises(trust.GateTrustError):
        trusted.create()
    assert trusted.store.count_jobs() == 0


def test_config_changed_between_read_and_insert_refuses(trusted, monkeypatch):
    original = trusted.store.create_job

    def changing(**kwargs):
        _delete_config(trusted)
        changed = json.loads(trusted.config["evidence_json"])
        changed["runner_image_id"] = "sha256:" + "9" * 64
        raw = canonical_json(changed)
        trusted.configure({**trusted.config, "evidence_json": raw,
            "evidence_sha256": hashlib.sha256(raw.encode()).hexdigest()})
        return original(**kwargs)

    monkeypatch.setattr(trusted.store, "create_job", changing)
    with pytest.raises(IdempotencyConflict, match="environment changed"):
        trusted.create()
    assert trusted.store.count_jobs() == 0


def test_signed_snapshot_is_deterministic_for_same_turn_reissue(trusted, monkeypatch):
    first = trusted.create()
    # Simulate a concurrent insert winner: caller observes no row, create_job's
    # existing IntegrityError path compares the same payload and exact snapshot.
    original = trusted.store.get_job_by_key
    calls = 0

    def missing_once(key):
        nonlocal calls
        calls += 1
        return None if calls == 1 else original(key)

    monkeypatch.setattr(trusted.store, "get_job_by_key", missing_once)
    token = codec.mint_task_turn_context(**trusted.base.values, issued_at=1)
    assert trusted.create(turn_token=token).model_dump() == first.model_dump()
    assert trusted.store.count_jobs() == 1


@pytest.mark.parametrize("field,value", [("caller_bot_id", "fixture-peer"),
    ("caller_user_id", "fixture-peer"), ("caller_turn_id", "turn-" + "9" * 32),
    ("caller_actor", "bot:fixture-peer"), ("idempotency_key", "other-key"),
    ("operation_slug", "other.operation"), ("operation_version", 999),
    ("operation_script_hash", "9" * 64), ("gate_plan_key", "other"),
    ("args_json", canonical_json(GatePlan("llm-bawt", "c" * 40).to_dict())),
    ("request_payload_json", "{}")])
def test_provenance_is_bound_to_immutable_ledger_columns(trusted, field, value):
    job = trusted.create().model_copy(update={field: value})
    with pytest.raises(trust.GateTrustError):
        trust.read_trusted_gate_request(job)


@pytest.mark.parametrize("location", ["caller", "environment", "operation", "request", "mac"])
def test_rehashing_tampered_snapshot_is_not_authentication(trusted, location):
    job = trusted.create()
    snapshot = json.loads(job.invocation_snapshot_json)
    if location == "caller":
        snapshot["caller"]["session_id"] = "00000000-0000-0000-0000-000000000001"
    elif location == "environment":
        snapshot["gate_trust"]["environment"]["evidence_sha256"] = "9" * 64
    elif location == "operation":
        snapshot["operation"]["execution_tag"] = "fixture_peer"
    elif location == "request":
        snapshot["gate_trust"]["request"]["caller"]["trigger_message_id"] = "other"
    else:
        snapshot["provenance_hmac_sha256"] = "9" * 64
    snapshot.pop("snapshot_hash")
    snapshot["snapshot_hash"] = hashlib.sha256(canonical_json(snapshot).encode()).hexdigest()
    altered = job.model_copy(update={"invocation_snapshot_json": canonical_json(snapshot)})
    with pytest.raises(trust.GateTrustError, match="provenance_invalid"):
        trust.read_trusted_gate_request(altered)


@pytest.mark.parametrize("failure", ["rotation", "unavailable"])
def test_missing_or_rotated_persistent_key_refuses(trusted, monkeypatch, failure):
    job = trusted.create()

    def derived(label):
        if failure == "rotation":
            return b"rotated-fixture-key".ljust(32, b"!")
        raise RuntimeError("fixture sensitive key detail")

    monkeypatch.setattr(trust.crypto, "derive_key", derived)
    with pytest.raises(trust.GateTrustError) as caught:
        trust.read_trusted_gate_request(job)
    assert "sensitive" not in str(caught.value)
    with pytest.raises(trust.GateTrustError):
        trusted.create()
    assert trusted.store.count_jobs() == 1


def test_shared_environment_validator_still_requires_independent_staged_tree(trusted):
    raw = trusted.config["evidence_json"].encode()
    digest = trusted.config["evidence_sha256"]
    assert validate_environment(raw, digest, trusted.base.plan, tree_sha="b" * 40)
    with pytest.raises(ValueError, match="incompatible"):
        validate_environment(raw, digest, trusted.base.plan, tree_sha="c" * 40)


def test_duplicate_json_keys_are_not_canonicalized_into_acceptance(trusted):
    _delete_config(trusted)
    raw = canonical_json(trusted.config)
    with trusted.store.engine.begin() as conn:
        conn.execute(text("INSERT INTO runtime_settings (scope_type, scope_id, key, value_json, updated_at) "
            "VALUES ('global', '*', :key, :raw, CURRENT_TIMESTAMP)"),
            {"key": trust.environment_setting_key(trusted.base.plan), "raw": raw[:-1] + ',"schema":"wrong"}'})
    with pytest.raises(trust.GateTrustError):
        trusted.create()


def test_signature_cannot_be_transplanted_between_different_requests(trusted):
    first = trusted.create()
    second = trusted.create(idempotency_key="other-key")
    snapshot = copy.deepcopy(json.loads(first.invocation_snapshot_json))
    transplanted = second.model_copy(update={"invocation_snapshot_json": canonical_json(snapshot)})
    with pytest.raises(trust.GateTrustError):
        trust.read_trusted_gate_request(transplanted)
