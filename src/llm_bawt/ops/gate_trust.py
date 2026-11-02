"""Unused admission of configured evidence and authenticated durable provenance.

No claim, route, callback, worker, runner, migration or dispatch calls this.
A verified snapshot is historical authenticity/integrity, NOT execution authority.
The existing gate-only claim deliberately refuses this enriched snapshot.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import re

from sqlalchemy import text

from ..service.providers import crypto
from ..task_turn_context import open_task_turn_context
from .gate_admission import GATE_REQUEST_SCHEMA
from .gate_claim import GATE_CLAIM_SCHEMA, gate_claim_snapshot
from .gate_environment import no_duplicates, validate_environment
from .gate_history import plan_key
from .gate_plan import GatePlan
from .models import OpsJob
from .store import IdempotencyConflict, OpsStore
from .validation import canonical_json

_CONFIG_SCHEMA = "llm-bawt.ops.gate-environment-config/v1"
_TRUST_SCHEMA = "llm-bawt.ops.gate-trust/v1"
_KEY_LABEL = "llm-bawt.ops.gate-request-provenance/v1"
_CALLER_FIELDS = ("bot_id", "user_id", "turn_id", "session_id", "trigger_message_id")
_MAX_CONFIG = 16384
_MAX_SNAPSHOT = 65536


class GateTrustError(ValueError):
    """Closed refusal; do not expose raw settings, snapshots or crypto errors."""


@dataclass(frozen=True)
class GateProvenance:
    bot_id: str
    user_id: str
    turn_id: str
    session_id: str
    trigger_message_id: str


@dataclass(frozen=True)
class TrustedGateRequest:
    plan: GatePlan
    caller: GateProvenance
    environment_json: bytes
    environment_sha256: str


def environment_setting_key(plan: GatePlan) -> str:
    """One exact-source global setting, never a request/bot/env override."""
    plan = GatePlan.from_dict(plan.to_dict())
    return f"ops_gate_environment:{plan.repository}:{plan.source_sha}"


def _json(raw, limit):
    if not isinstance(raw, str) or len(raw) > limit:
        raise GateTrustError("gate_trust_invalid")
    try:
        if len(raw.encode("utf-8")) > limit:
            raise ValueError("byte bound")
        value = json.loads(raw, object_pairs_hook=no_duplicates)
        canonical_json(value).encode("utf-8")  # Refuse NaN and lone surrogates as well.
        return value
    except (ValueError, UnicodeError, RecursionError):
        raise GateTrustError("gate_trust_invalid") from None


def _environment(value, plan):
    if (not isinstance(value, dict) or value.keys() != {"schema", "evidence_json", "evidence_sha256"}
            or value["schema"] != _CONFIG_SCHEMA or not isinstance(value["evidence_json"], str)):
        raise GateTrustError("gate_environment_unconfigured")
    try:
        raw = value["evidence_json"].encode("utf-8")
        validate_environment(raw, value["evidence_sha256"], plan)
    except (ValueError, UnicodeError):
        raise GateTrustError("gate_environment_invalid") from None
    return raw, value["evidence_sha256"]


def _configured_environment(conn, plan, *, lock=False):
    # Exact DB read, no cached layered resolver and no schema/bootstrap side effects.
    # Fetch at most two bounded rows; duplicates refuse even on a legacy schema.
    rows = conn.execute(text("SELECT CASE WHEN length(value_json) <= :bound THEN value_json "
        "ELSE NULL END FROM runtime_settings WHERE "
        "scope_type='global' AND scope_id='*' AND key=:key LIMIT 2" +
        (" FOR SHARE" if lock and conn.dialect.name == "postgresql" else "")),
        {"key": environment_setting_key(plan), "bound": _MAX_CONFIG}).scalars().all()
    if len(rows) != 1:
        raise GateTrustError("gate_environment_unconfigured")
    value = _json(rows[0], _MAX_CONFIG)
    _environment(value, plan)
    return value


def _mac(body):
    try:
        key = crypto.derive_key(_KEY_LABEL)
    except Exception:
        raise GateTrustError("gate_provenance_key_unavailable") from None
    return hmac.new(key, canonical_json(body).encode(), hashlib.sha256).hexdigest()


def _seal(snapshot):
    body = {k: v for k, v in snapshot.items() if k != "snapshot_hash"}
    body["provenance_hmac_sha256"] = _mac(body)
    body["snapshot_hash"] = hashlib.sha256(canonical_json(body).encode()).hexdigest()
    raw = canonical_json(body)
    if len(raw) > _MAX_SNAPSHOT:
        raise GateTrustError("gate_snapshot_too_large")
    return raw


def read_trusted_gate_request(job: OpsJob) -> TrustedGateRequest:
    """Verify a durable ledger-bound record without reopening the expired token.

    Uses only the stored bytes and persistent, domain-derived server key. Does
    not consult live grants/configuration, change state, or authorize execution.
    A future claim MUST recheck live grants/catalog/environment under its locks.
    Missing legacy proof, key rotation/unavailability and any drift fail closed.
    """
    snapshot = _json(job.invocation_snapshot_json, _MAX_SNAPSHOT)
    if (not isinstance(snapshot, dict) or snapshot.keys() != {"gate_claim_schema", "operation",
            "gate_plan", "caller", "execution", "gate_trust", "provenance_hmac_sha256", "snapshot_hash"}
            or snapshot["gate_claim_schema"] != GATE_CLAIM_SCHEMA):
        raise GateTrustError("gate_provenance_invalid")
    body = {k: v for k, v in snapshot.items() if k != "snapshot_hash"}
    digest = hashlib.sha256(canonical_json(body).encode()).hexdigest()
    mac = body.pop("provenance_hmac_sha256")
    if (not isinstance(mac, str) or not re.fullmatch(r"[0-9a-f]{64}", mac)
            or snapshot["snapshot_hash"] != digest or not hmac.compare_digest(mac, _mac(body))):
        raise GateTrustError("gate_provenance_invalid")
    trust = snapshot["gate_trust"]
    if (not isinstance(trust, dict) or trust.keys() != {"schema", "idempotency_key", "request", "environment"}
            or trust["schema"] != _TRUST_SCHEMA or trust["idempotency_key"] != job.idempotency_key):
        raise GateTrustError("gate_provenance_invalid")
    request = _json(job.request_payload_json, _MAX_SNAPSHOT)
    if (not isinstance(request, dict) or request.keys() != {"schema", "operation", "gate_plan", "caller"}
            or request != trust["request"] or request["schema"] != GATE_REQUEST_SCHEMA
            or not isinstance(request["caller"], dict) or request["caller"].keys() != set(_CALLER_FIELDS)):
        raise GateTrustError("gate_request_drift")
    caller = GateProvenance(**request["caller"])
    plan = GatePlan.from_dict(request["gate_plan"])
    operation = snapshot["operation"]
    if (not isinstance(operation, dict) or operation.get("slug") != job.operation_slug
            or operation.get("version") != job.operation_version
            or operation.get("script_hash") != job.operation_script_hash
            or request["operation"] != job.operation_slug or snapshot["gate_plan"] != plan.to_dict()
            or snapshot["caller"] != {k: request["caller"][k] for k in _CALLER_FIELDS[:-1]}
            or GatePlan.from_dict(_json(job.args_json, _MAX_SNAPSHOT)) != plan
            or job.gate_plan_key != plan_key(plan)
            or (job.caller_actor, job.caller_bot_id, job.caller_user_id, job.caller_turn_id) !=
                (f"bot:{caller.bot_id}", caller.bot_id, caller.user_id, caller.turn_id)):
        raise GateTrustError("gate_request_drift")
    raw, digest = _environment(trust["environment"], plan)
    return TrustedGateRequest(plan, caller, raw, digest)


def create_trusted_gate_job(store: OpsStore, *, operation_slug: str, plan: dict,
                            turn_token: str, idempotency_key: str) -> OpsJob:
    """Freeze configured evidence and authenticated identity in one QUEUED insert.

    The only caller identity is opened from the valid authenticated capability;
    neither token nor issue time is persisted. No evidence/digest/identity/grant
    argument exists. Identical replay is read-only, including after configuration
    removal/terminalization. Concurrent same-request snapshots are deterministic;
    the existing unique key/create_job fence handles insertion, not another ledger.
    """
    workload = GatePlan.from_dict(plan)
    caller = open_task_turn_context(turn_token)
    if (not isinstance(idempotency_key, str) or not idempotency_key.strip()
            or len(idempotency_key) > 128):
        raise ValueError("idempotency_key must be 1..128 nonblank characters")
    request = {"schema": GATE_REQUEST_SCHEMA, "operation": operation_slug,
        "gate_plan": workload.to_dict(), "caller": {k: getattr(caller, k) for k in _CALLER_FIELDS}}
    payload = canonical_json(request)
    existing = store.get_job_by_key(idempotency_key)
    if existing is not None:
        store.verify_payload(existing, payload)
        read_trusted_gate_request(existing)  # No upgrading legacy rows or repairing drift.
        return existing
    operation = store.get_operation_by_slug(operation_slug)
    if operation is None:
        raise ValueError("gate operation is not configured")
    if (not operation.enabled or operation.soft_deleted_at is not None
            or operation.quiesce_enabled or operation.execution_tag is None):
        raise ValueError("gate operation is disabled, quiescing or has no configured execution grant")
    with store.engine.connect() as conn:
        environment = _configured_environment(conn, workload)
    snapshot = gate_claim_snapshot(operation, workload, turn_token=turn_token)
    snapshot["gate_trust"] = {"schema": _TRUST_SCHEMA, "idempotency_key": idempotency_key,
        "request": request, "environment": environment}
    sealed = _seal(snapshot)

    def fence(session):
        # Freeze exactly what was read, under a shared row lock through insert.
        # A config edit committed before this transaction is a conflict, not an
        # implicit rebind. Edits waiting on the lock are prospective; a future
        # claim must separately compare the frozen evidence with live config.
        if _configured_environment(session.connection(), workload, lock=True) != environment:
            raise IdempotencyConflict("gate environment changed before request insertion")

    job = store.create_job(operation=operation, args_json=canonical_json(workload.to_dict()),
        display_args_json=canonical_json(workload.to_dict()), idempotency_key=idempotency_key,
        invocation_snapshot_json=sealed, request_payload_json=payload, fence=fence,
        caller_actor=f"bot:{caller.bot_id}", caller_bot_id=caller.bot_id,
        caller_user_id=caller.user_id, caller_turn_id=caller.turn_id)
    read_trusted_gate_request(job)
    return job
