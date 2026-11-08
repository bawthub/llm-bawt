"""Canonical catalog/history and job ledger with transactional claims and CAS.

No side effects run in a transaction. Operation-row write locks serialize edits
and concurrency-slot claims in PostgreSQL AND SQLite. Conditional SQL updates
ensure a stale reconciler cannot overwrite a terminal result.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, inspect, or_, text, update
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, SQLModel, select

from ..system_events import SystemEventEmitter
from ..utils.config import Config, has_database_credentials
from ..utils.schema import SchemaBootstrapGuard
from .events import append_job_transition, append_job_transition_by_id
from .container_remove import ACTION as REMOVE_ACTION, validate_remove_schema  # TASK-1439 T2c
from .executor import validate_spec
from .release_spec import is_release_spec
from .execution_mode import (DEFAULT_EXECUTION_MODE, EXECUTION_ISOLATED, legacy_execution_mode,
                             snapshot_execution_mode, validate_execution_mode)
from .models import (EXECUTOR_DOCKER, EXECUTOR_RELEASE, JOB_ACCEPTED, JOB_DISPATCHING, JOB_FAILED, JOB_QUEUED,
                     JOB_QUIESCING, JOB_RUNNING, JOB_SUCCEEDED, JOB_TERMINAL_STATES, OpsJob, OpsOperation,
                     OpsOperationRevision)
from .quiesce_settings import validate_quiesce_settings
from .operation_revision import OperationChanged, changed_reason
from .sequence import initial_ledger, interrupted, read_ledger
from .worker_slots import UNRETIRED_GATE_SLOT
from .gate_history import gate_plan_key, migrate_gate_history
from .validation import canonical_json, validate_catalog
from .worker_slots import process_spawn_held, process_spawn_hold_available, worker_slot_free
from .wake_on_lan import ACTION as WAKE_ACTION, validate_budget as validate_wake_budget

logger = logging.getLogger(__name__)

# error_text prefix of a job failed for preflight past its deadline (TASK-1169 S1e).
PREFLIGHT_FAILED = "preflight_failed"
# Notified in a job's insert transaction, so the ops reconciler dispatches it
# without waiting out its interval (TASK-1169 S3b). A wake-up only: the
# reconciler still scans the ledger, so a lost notify costs one interval.
JOB_CREATED_CHANNEL = "ops_job_created"
#: TASK-2105: operation-catalog writes wake the ops health catalog reader.
CATALOG_CHANGED_CHANNEL = "ops_catalog_changed"


def _utcnow():
    return datetime.now(timezone.utc)


def _script_hash(script):
    return hashlib.sha256(script.encode()).hexdigest()


_OP_WRITABLE = {"slug", "title", "description", "enabled", "executor_kind", "target_host",
                "run_as_user", "working_directory", "command_script", "args_schema_json",
                "args_defaults_json", "timeout_seconds", "start_delay_seconds", "max_output_bytes",
                "max_concurrent", "risk_level", "category", "approval_prompt_prefix",
                "quiesce_enabled", "quiesce_timeout_seconds", "quiesce_scope_json",
                "execution_mode", "execution_tag"}
_ACTIVE_CLAIMED = (JOB_DISPATCHING, JOB_ACCEPTED, JOB_RUNNING)
# Non-terminal states a terminal write may leave (QUIESCING: TASK-1233).
_OPEN_STATES = (JOB_QUEUED, JOB_QUIESCING, *_ACTIVE_CLAIMED)

_JOB_QUIESCE_COLUMNS = (
    ("quiesce_gate_id", "BIGINT"), ("quiesce_driver_id", "VARCHAR(64)"),
    ("quiesce_driver_lease_expires_at", "TIMESTAMP WITH TIME ZONE"),
    ("quiesce_deadline_at", "TIMESTAMP WITH TIME ZONE"),
    ("quiesce_fence_json", "TEXT"), ("quiesce_json", "TEXT"),
    ("receipt_callback_json", "TEXT"), ("receipt_callback_delivery_id", "VARCHAR(128)"),
)
_JOB_WORKER_SLOT_COLUMNS = (  # TASK-1169 process admission and self-config drain ledger.
    ("worker_slot_at", "TIMESTAMP WITH TIME ZONE"),
    ("process_spawn_hold_at", "TIMESTAMP WITH TIME ZONE"),
    ("process_spawn_hold_deadline_at", "TIMESTAMP WITH TIME ZONE"),
    ("process_spawn_hold_target_json", "TEXT"),
    ("process_spawn_hold_quarantine_at", "TIMESTAMP WITH TIME ZONE"),
    ("process_spawn_hold_release_json", "TEXT"),
    ("gate_retired_at", "TIMESTAMP WITH TIME ZONE"),  # TASK-1865 positive gate retirement.
)
_OP_QUIESCE_COLUMNS = (
    ("quiesce_enabled", "BOOLEAN NOT NULL DEFAULT false"),
    ("quiesce_timeout_seconds", "INTEGER NOT NULL DEFAULT 600"),
    ("quiesce_scope_json", "TEXT"),
)


def slot_holders(conn, operation_slug, *, exclude_job_id=None):
    """Snapshots of the jobs holding one of ``operation_slug``'s concurrency slots.

    A claimed job holds a slot, and so does a quiescing job once its gate is
    open (TASK-1233 C5): its hold is already in effect. Run under the
    operation-row lock, like every slot claim. A reserved gate keeps its
    slot in any state until positively retired (TASK-1865); a terminal row
    alone never proves its runner stopped. Each row is listed once.
    """
    rows = conn.execute(text(
        "SELECT id, invocation_snapshot_json FROM ops_jobs WHERE operation_slug=:slug"
        " AND (state IN (:dispatching, :accepted, :running)"
        "      OR (state=:quiescing AND quiesce_gate_id IS NOT NULL)"
        f"      OR {UNRETIRED_GATE_SLOT})"
    ), {"slug": operation_slug, "dispatching": JOB_DISPATCHING, "accepted": JOB_ACCEPTED,
        "running": JOB_RUNNING, "quiescing": JOB_QUIESCING}).all()
    return [row[1] for row in rows if row[0] != exclude_job_id]


def slot_cap_reached(holders, own_cap):
    """Is the strictest cap among this job's and every holder's snapshot reached?"""
    limits = [own_cap] if own_cap is not None else []
    for raw in holders:
        if raw:
            cap = json.loads(raw)["execution"].get("max_concurrent")
            if cap is not None:
                limits.append(cap)
    return bool(limits) and len(holders) >= min(limits)


def is_sequence_snapshot(snapshot):
    return (snapshot.get("spec") or {}).get("action") == "operation_sequence"


def snapshot_operations(snapshot):
    """Approved catalog identities, including every sequence step (not child jobs)."""
    operations = [snapshot["operation"]] if snapshot.get("operation") else []
    if is_sequence_snapshot(snapshot):
        operations.extend(step["snapshot"]["operation"] for step in snapshot["sequence"]["steps"])
    return operations


def lock_snapshot_operations(conn, snapshot, *, operation_slug):
    """Lock the parent and all step rows in one stable, shared claim order.

    The no-op UPDATE is a write lock on both PostgreSQL and SQLite, just like
    ``OpsStore._lock_operation``. Take these locks before worker/job/gate locks.
    Snapshots are immutable after insertion, so callers may read them first.
    """
    slugs = {operation_slug, *(op["slug"] for op in snapshot_operations(snapshot))}
    operations = {}
    for slug in sorted(slugs):
        conn.execute(text("UPDATE ops_operations SET version=version WHERE slug=:slug"), {"slug": slug})
        operations[slug] = conn.execute(text(
            "SELECT id, slug, version, script_hash, enabled, soft_deleted_at"
            " FROM ops_operations WHERE slug=:slug"
        ), {"slug": slug}).mappings().first()
    return operations


def snapshot_changed_reason(snapshot, operations, *, require_enabled=True):
    """Validate every pinned identity while its catalog row remains locked."""
    for approved in snapshot_operations(snapshot):
        current = operations.get(approved["slug"])
        if require_enabled and (current is None or not current["enabled"] or current["soft_deleted_at"] is not None):
            return "operation disabled before dispatch; no action submitted"
        if reason := changed_reason(approved, current):
            return reason
    return None


def sequence_conflict(conn, snapshot, *, operation_slug, exclude_job_id=None):
    """Sequences exclusively use all their operations for active execution only.

    An ordinary job still uses the existing max_concurrent policy against other
    ordinary jobs. Queued/quiescing reservations are deliberately not sequence
    conflicts: reserving shared step operations before dispatch can deadlock.
    Call only with all this invocation's operation rows locked.
    """
    own_sequence = is_sequence_snapshot(snapshot)
    own_slugs = {operation_slug, *(op["slug"] for op in snapshot_operations(snapshot))}
    rows = conn.execute(text(
        "SELECT id, operation_slug, invocation_snapshot_json FROM ops_jobs"
        " WHERE state IN (:dispatching, :accepted, :running)"
    ), {"dispatching": JOB_DISPATCHING, "accepted": JOB_ACCEPTED, "running": JOB_RUNNING}).all()
    for job_id, slug, raw in rows:
        if job_id == exclude_job_id:
            continue
        other = json.loads(raw) if raw else {}
        if not own_sequence and not is_sequence_snapshot(other):
            continue
        other_slugs = {slug, *(op["slug"] for op in snapshot_operations(other))}
        if own_slugs & other_slugs:
            return True
    return False


def is_quiesce_snapshot(raw):
    """Does this persisted invocation snapshot require a quiesce (TASK-1233)?"""
    return bool(raw) and bool((json.loads(raw).get("quiesce") or {}).get("enabled"))


class OpsStoreUnavailable(RuntimeError):
    pass


class IdempotencyConflict(ValueError):
    pass


class OpsStore:
    _schema_guard = SchemaBootstrapGuard()

    def __init__(self, config: Config, engine: Any = None):
        self.config = config
        self.engine = engine
        self.events: SystemEventEmitter | None = None
        if engine is not None:
            self._ensure_tables_exist()
            self._ensure_event_ledger()
        elif has_database_credentials(config):
            try:
                from ..utils.db import get_shared_engine
                self.engine = get_shared_engine(config)
                self._ensure_tables_exist()
                self._ensure_event_ledger()
            except Exception as exc:
                self.engine = None
                logger.warning("Ops store DB unavailable: %s", exc)

    def _ensure_event_ledger(self):
        if self.engine is None:
            return
        try:
            self.events = SystemEventEmitter(self.engine)
        except Exception:
            # Event projection must not prevent reconciler claims or job creation.
            logger.exception("Ops event ledger unavailable; domain jobs continue")

    def _ensure_tables_exist(self):
        if self.engine is None:
            return
        def bootstrap(conn):
            SQLModel.metadata.create_all(bind=conn, tables=[OpsOperation.__table__, OpsJob.__table__, OpsOperationRevision.__table__])
            self._migrate_add_columns(conn)
            # Pre-existing rows have only their current revision available.
            # Never invent historical revisions that were not recorded.
            with Session(bind=conn) as session:
                for op in session.exec(select(OpsOperation)).all():
                    existing = session.exec(select(OpsOperationRevision.id).where(
                        OpsOperationRevision.operation_id == op.id, OpsOperationRevision.version == op.version)).first()
                    if existing is None:
                        self._record_revision(session, op)
                session.flush()
                session.commit()  # join the bootstrap connection transaction
        self._schema_guard.run(self.engine, "ops-store-task861-v1", bootstrap)

    def _migrate_add_columns(self, conn):
        clause = " IF NOT EXISTS" if conn.dialect.name == "postgresql" else ""
        for table, columns in (
            ("ops_jobs", (("invocation_snapshot_json", "TEXT"), ("request_payload_json", "TEXT"),
                          ("caller_actor", "VARCHAR(128)"), *_JOB_QUIESCE_COLUMNS,
                          *_JOB_WORKER_SLOT_COLUMNS, ("sequence_json", "TEXT"))),
            ("ops_operations", (*_OP_QUIESCE_COLUMNS, ("execution_tag", "VARCHAR(64)"))),
        ):
            existing = {c["name"] for c in inspect(conn).get_columns(table)}
            for name, kind in columns:
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN{clause} {name} {kind}"))
        self._migrate_execution_mode(conn, clause)
        migrate_gate_history(conn)

    @staticmethod
    def _migrate_execution_mode(conn, clause):
        """TASK-1169 S4: backfill existing rows to what they already do.

        Existing Docker operations ran in an isolated worker container, so they
        get ``isolated``. Release rows never start a worker; they get the inert
        default. New rows default to ``process``. Runs only when the column is
        added, so an admin's later choice is never rewritten.
        """
        if "execution_mode" in {c["name"] for c in inspect(conn).get_columns("ops_operations")}:
            return
        conn.execute(text(f"ALTER TABLE ops_operations ADD COLUMN{clause} execution_mode"
                          f" VARCHAR(16) NOT NULL DEFAULT '{legacy_execution_mode(EXECUTOR_DOCKER)}'"))
        conn.execute(text("UPDATE ops_operations SET execution_mode=:mode WHERE executor_kind=:release"),
                     {"mode": legacy_execution_mode(EXECUTOR_RELEASE), "release": EXECUTOR_RELEASE})
        if conn.dialect.name == "postgresql":
            conn.execute(text(f"ALTER TABLE ops_operations ALTER COLUMN execution_mode"
                              f" SET DEFAULT '{DEFAULT_EXECUTION_MODE}'"))

    def _require(self):
        if self.engine is None:
            raise OpsStoreUnavailable("ops store has no DB engine")

    @staticmethod
    def _op_filter(include_disabled=False, include_soft_deleted=False):
        filters = []
        if not include_disabled:
            filters.append(OpsOperation.enabled.is_(True))
        if not include_soft_deleted:
            filters.append(OpsOperation.soft_deleted_at.is_(None))
        return filters

    def list_operations(self, *, include_disabled=False, include_soft_deleted=False, limit=None, offset=0):
        if self.engine is None:
            return []
        with Session(self.engine) as session:
            stmt = select(OpsOperation).where(*self._op_filter(include_disabled, include_soft_deleted)).order_by(OpsOperation.category, OpsOperation.slug).offset(offset)
            if limit is not None:
                stmt = stmt.limit(limit)
            return list(session.exec(stmt).all())

    def count_operations(self, *, include_disabled=False, include_soft_deleted=False):
        if self.engine is None:
            return 0
        with Session(self.engine) as session:
            return session.exec(select(func.count()).select_from(OpsOperation).where(*self._op_filter(include_disabled, include_soft_deleted))).one()

    def get_operation(self, slug_or_id):
        if self.engine is None:
            return None
        with Session(self.engine) as session:
            return session.exec(select(OpsOperation).where((OpsOperation.id == slug_or_id) | (OpsOperation.slug == slug_or_id))).first()

    def get_operation_by_slug(self, slug):
        if self.engine is None:
            return None
        with Session(self.engine) as session:
            return session.exec(select(OpsOperation).where(OpsOperation.slug == slug)).first()

    def _clean_op(self, data):
        out = {k: v for k, v in data.items() if k in _OP_WRITABLE}
        for field in ("args_schema_json", "args_defaults_json"):
            if field in out and isinstance(out[field], dict):
                out[field] = canonical_json(out[field])
        return out

    @staticmethod
    def _validate_op(op):
        if not isinstance(op.slug, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", op.slug):
            raise ValueError("slug must be 1..128 letters, digits, dots, underscores or hyphens")
        if type(op.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        from ..bot_tags.vocabulary import TAG_SLUG
        if op.execution_tag is not None and (
                not isinstance(op.execution_tag, str) or not TAG_SLUG.fullmatch(op.execution_tag)):
            raise ValueError("execution_tag must be a canonical tag slug or null")
        for field, low, high in (("timeout_seconds", 1, 86400), ("start_delay_seconds", 0, 86400),
                                 ("max_output_bytes", 1, 1048576), ("max_concurrent", 1, 1000)):
            value = getattr(op, field)
            if value is None and field == "max_concurrent":
                continue
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{field} must be an integer in {low}..{high}")
        schema, _ = validate_catalog(op.args_schema_json, op.args_defaults_json)
        if op.executor_kind not in (EXECUTOR_DOCKER, EXECUTOR_RELEASE):
            raise ValueError("executor_kind must be 'docker' or 'release'")
        validate_execution_mode(op.execution_mode, op.executor_kind)
        if op.target_host or op.run_as_user or op.working_directory:
            raise ValueError("Operations do not support target_host/run_as_user/working_directory; no SSH or shell runner")
        spec = validate_spec(op.command_script)
        if spec["action"] == WAKE_ACTION:
            if schema.get("properties") or json.loads(op.args_defaults_json):
                raise ValueError("wake_on_lan accepts no caller arguments")
            validate_wake_budget(spec, op.timeout_seconds)
        # TASK-1030: the release coordinator runs ONLY the pinned release spec,
        # and that spec never reaches the Docker executor.
        if (op.executor_kind == EXECUTOR_RELEASE) != is_release_spec(spec):
            raise ValueError("release_orchestrate specs require executor_kind 'release' and vice versa")
        for key in ("container_name_from_arg", "compose_service_from_arg"):
            if key in spec and schema.get("properties", {}).get(spec[key], {}).get("type") != "string":
                raise ValueError(f"{key} must reference a declared string argument")
        if "runtime_source" in spec and (
                schema.get("properties", {}).get("expected_runtime_sha", {}).get("type") != "string"
                or "expected_runtime_sha" not in schema.get("required", [])):
            raise ValueError("runtime_source requires a declared, required expected_runtime_sha string argument")
        if spec["action"] == REMOVE_ACTION:
            # TASK-1439 T2c: the row's args enum is the removal allowlist.
            validate_remove_schema(spec, schema)
        for field in ("title", "description", "risk_level"):
            if not isinstance(getattr(op, field), str):
                raise ValueError(f"{field} must be a string")
        validate_quiesce_settings(op, schema)

    def _notify_catalog(self, session, slug):
        if self.engine.dialect.name == "postgresql":
            # Delivered only if this transaction commits.
            session.execute(text("SELECT pg_notify(:channel, :slug)"),
                            {"channel": CATALOG_CHANGED_CHANNEL, "slug": slug})

    @staticmethod
    def _record_revision(session, row):
        session.add(OpsOperationRevision(operation_id=row.id, operation_slug=row.slug,
            version=row.version, snapshot_json=canonical_json(row.to_api()), actor=row.updated_by))

    def create_operation(self, data, *, actor=None):
        self._require()
        clean = self._clean_op(data)
        now = _utcnow()
        row = OpsOperation(id=uuid.uuid4().hex, created_at=now, updated_at=now,
                           created_by=actor, updated_by=actor, **clean)
        self._validate_op(row)
        row.script_hash = _script_hash(row.command_script)
        with Session(self.engine) as session:
            session.add(row)
            self._record_revision(session, row)
            self._notify_catalog(session, row.slug)
            try:
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                raise ValueError(f"operation slug already exists: {row.slug}") from exc
            session.refresh(row)
            return row

    @staticmethod
    def _lock_operation(session, slug_or_id):
        # This UPDATE acquires a write lock BEFORE any read, including SQLite.
        session.execute(update(OpsOperation).where((OpsOperation.id == slug_or_id) | (OpsOperation.slug == slug_or_id))
                        .values(version=OpsOperation.version).execution_options(synchronize_session=False))
        return session.exec(select(OpsOperation).where((OpsOperation.id == slug_or_id) | (OpsOperation.slug == slug_or_id))).first()

    def update_operation(self, slug_or_id, data, *, actor=None, _delete=False, expect_version=None):
        """``expect_version``: write only if the locked row is still at that
        version (compare-and-set); otherwise return None without writing."""
        self._require()
        clean = self._clean_op(data)
        with Session(self.engine) as session:
            row = self._lock_operation(session, slug_or_id)
            if row is None:
                return None
            if expect_version is not None and row.version != expect_version:
                return None
            if "slug" in clean and clean["slug"] != row.slug:
                raise ValueError("operation slug is immutable; create a new operation")
            if row.soft_deleted_at is not None and clean.get("enabled"):
                raise ValueError("soft-deleted operations cannot be enabled")
            for field, value in clean.items():
                setattr(row, field, value)
            if _delete:
                if row.soft_deleted_at is not None:
                    return row
                row.enabled = False
                row.soft_deleted_at = _utcnow()
            self._validate_op(row)
            row.version += 1
            row.script_hash = _script_hash(row.command_script)
            row.updated_at = _utcnow()
            row.updated_by = actor
            session.add(row)
            self._record_revision(session, row)
            from ..approval_ops_revision import invalidate_pending_ops

            invalidate_pending_ops(session, row, getattr(self, "events", None))
            self._notify_catalog(session, row.slug)
            session.commit()
            session.refresh(row)
            return row

    def soft_delete_operation(self, slug_or_id, *, actor=None):
        return self.update_operation(slug_or_id, {}, actor=actor, _delete=True) is not None

    def list_revisions(self, slug, *, limit=50, offset=0):
        if self.engine is None:
            return [], 0
        with Session(self.engine) as session:
            filters = [OpsOperationRevision.operation_slug == slug]
            total = session.exec(select(func.count()).select_from(OpsOperationRevision).where(*filters)).one()
            rows = session.exec(select(OpsOperationRevision).where(*filters).order_by(OpsOperationRevision.version.desc()).offset(offset).limit(limit)).all()
            return list(rows), total

    def seed_operation_if_missing(self, data, *, actor="system-seed"):
        if self.engine is None or self.get_operation_by_slug(data.get("slug")):
            return None
        return self.create_operation(data, actor=actor)

    def get_job_by_key(self, key):
        if self.engine is None:
            return None
        with Session(self.engine) as session:
            return session.exec(select(OpsJob).where(OpsJob.idempotency_key == key)).first()

    @staticmethod
    def verify_payload(existing, payload_json, snapshot_json=None):
        if existing.request_payload_json != payload_json:
            raise IdempotencyConflict("idempotency key is already bound to a different invocation payload")
        if snapshot_json is not None and existing.invocation_snapshot_json != snapshot_json:
            raise IdempotencyConflict("idempotency key is already bound to a different approved snapshot")

    def create_job(self, *, operation, args_json, display_args_json, idempotency_key,
                   invocation_snapshot_json=None, request_payload_json=None, caller_actor=None,
                   caller_bot_id=None, caller_user_id=None, caller_turn_id=None,
                   caller_session_key=None, caller_backend=None, approval_request_id=None,
                   fence=None, state=JOB_QUEUED, receipt_callback_json=None):
        """Insert a queued job. ``fence(session)``, when given, runs in the same
        transaction first and raises to refuse creation (an admission claim
        that is no longer held). A quiesce invocation is created directly in
        QUIESCING (TASK-1233 C5) and is never claimable from QUEUED."""
        if state not in (JOB_QUEUED, JOB_QUIESCING):
            raise ValueError(f"a job is created queued or quiescing, not {state!r}")
        if (state == JOB_QUIESCING) != is_quiesce_snapshot(invocation_snapshot_json):
            raise ValueError("only a quiesce snapshot is created quiescing, and it always is")
        self._require()
        if not isinstance(idempotency_key, str) or not 1 <= len(idempotency_key) <= 128:
            raise ValueError("idempotency_key must be 1..128 characters")
        # Legacy/internal callers are also conflict checked by args, not just key.
        payload = request_payload_json or canonical_json({"operation": operation.slug, "args": json.loads(args_json)})
        row = OpsJob(id=uuid.uuid4().hex, operation_slug=operation.slug,
            operation_version=operation.version, operation_script_hash=operation.script_hash,
            args_json=args_json, gate_plan_key=gate_plan_key(args_json), display_args_json=display_args_json,
            invocation_snapshot_json=invocation_snapshot_json, request_payload_json=payload,
            idempotency_key=idempotency_key, caller_actor=caller_actor, caller_bot_id=caller_bot_id,
            caller_user_id=caller_user_id, caller_turn_id=caller_turn_id, caller_session_key=caller_session_key,
            caller_backend=caller_backend, approval_request_id=approval_request_id, state=state,
            receipt_callback_json=receipt_callback_json)
        with Session(self.engine) as session:
            if fence is not None:
                fence(session)
            snapshot = json.loads(invocation_snapshot_json) if invocation_snapshot_json else {}
            if is_sequence_snapshot(snapshot):
                row.sequence_json = canonical_json(read_ledger(initial_ledger(snapshot)))
            operations = lock_snapshot_operations(session.connection(), snapshot, operation_slug=operation.slug)
            reason = changed_reason(operation.to_api(), operations[operation.slug])
            if is_sequence_snapshot(snapshot):
                reason = reason or snapshot_changed_reason(snapshot, operations, require_enabled=True)
            if reason:
                raise OperationChanged(reason)
            session.add(row)
            try:
                session.flush()
                if self.engine.dialect.name == "postgresql":
                    # Delivered only if this transaction commits.
                    session.execute(text("SELECT pg_notify(:channel, :job_id)"),
                                    {"channel": JOB_CREATED_CHANNEL, "job_id": row.id})
                append_job_transition(getattr(self, "events", None), session.connection(), row)
                session.commit()
            except IntegrityError:
                session.rollback()
                existing = self.get_job_by_key(idempotency_key)
                if existing is None:
                    raise
                self.verify_payload(existing, payload, invocation_snapshot_json)
                return existing
            session.refresh(row)
            return row

    def get_job(self, job_id):
        if self.engine is None:
            return None
        with Session(self.engine) as session:
            return session.get(OpsJob, job_id)

    @staticmethod
    def _job_filters(operation_slug=None, state=None):
        return ([OpsJob.operation_slug == operation_slug] if operation_slug else []) + ([OpsJob.state == state] if state else [])

    def list_jobs(self, *, operation_slug=None, state=None, limit=50, offset=0):
        if self.engine is None:
            return []
        with Session(self.engine) as session:
            return list(session.exec(select(OpsJob).where(*self._job_filters(operation_slug, state))
                .order_by(OpsJob.submitted_at.desc(), OpsJob.id.desc()).offset(offset).limit(limit)).all())

    def count_jobs(self, *, operation_slug=None, state=None):
        if self.engine is None:
            return 0
        with Session(self.engine) as session:
            return session.exec(select(func.count()).select_from(OpsJob).where(*self._job_filters(operation_slug, state))).one()

    def process_spawns_held(self) -> bool:
        """Plain authoritative read for a launcher holding its shared flock.

        Deliberately no worker-slot advisory lock; failures propagate so the
        spawn boundary defers rather than treating an unavailable DB as open.
        """
        self._require()
        with self.engine.connect() as conn:
            return process_spawn_held(conn)

    def prepare_process_spawn_hold(self, job_id, *, target_identity, deadline_at):
        from .spawn_hold_recovery import prepare_hold
        return prepare_hold(self, job_id, target_identity=target_identity, deadline_at=deadline_at)

    def classify_process_spawn_hold(self, job_id):
        from .spawn_hold_recovery import classify_hold
        return classify_hold(self, job_id)

    def release_process_spawn_hold(self, job_id, *, receipt_root, acquire_deadline, verifier):
        from .spawn_hold_recovery import release_hold
        return release_hold(self, job_id, receipt_root=receipt_root,
                            acquire_deadline=acquire_deadline, verifier=verifier)

    def claim_gate_job(self, job_id, *, turn_token):
        """Dormant gate-only authorization/CAS; no executor or production caller.

        Caller identity comes from the existing signed turn capability, never
        a supplied bot/tag. See gate_claim for first-attempt-only semantics.
        """
        from .gate_claim import claim_gate_job
        return claim_gate_job(self, job_id, turn_token=turn_token)

    def claim_job(self, job_id, *, max_concurrent=None, hold_process_spawns=False,
                  process_child_count: int | None = None):
        """QUEUED -> DISPATCHING atomically, reserving its execution slots.

        ``hold_process_spawns`` is an internal, dormant self-config seam: an
        isolated Docker claim can take the durable spawn hold only with no
        process slots and a fresh zero child count. The future action must
        hold the exclusive receipt-volume flock through this claim; no current
        dispatcher enables it. This is not a substitute for spawn-side fencing.

        Never for a quiesce snapshot (TASK-1233 C5): only the quiesce driver
        may dispatch one, after its drain, through its own fenced CAS.
        """
        self._require()
        job = self.get_job(job_id)
        if job is None or is_quiesce_snapshot(job.invocation_snapshot_json):
            return False
        if job.invocation_snapshot_json and "gate_claim_schema" in json.loads(job.invocation_snapshot_json):
            return False  # No generic claim bypass of the dormant gate boundary.
        with Session(self.engine) as session:
            snapshot = json.loads(job.invocation_snapshot_json) if job.invocation_snapshot_json else {}
            conn = session.connection()
            operations = lock_snapshot_operations(conn, snapshot, operation_slug=job.operation_slug)
            op = operations[job.operation_slug]
            if op is None or not op["enabled"] or op["soft_deleted_at"] is not None:
                return False
            if (op["version"] != job.operation_version or op["script_hash"] != job.operation_script_hash
                    or snapshot_changed_reason(snapshot, operations)):
                return False  # the driver records operation_changed on its next pass
            if sequence_conflict(conn, snapshot, operation_slug=job.operation_slug, exclude_job_id=job_id):
                return False
            if slot_cap_reached(slot_holders(conn, job.operation_slug), max_concurrent):
                return False
            values = dict(state=JOB_DISPATCHING, dispatched_at=_utcnow(), host_unit_name=f"llm-bawt-ops-{job_id}", error_text=None)
            if hold_process_spawns:
                kind = (snapshot.get("execution") or {}).get("executor_kind")
                if (kind != EXECUTOR_DOCKER
                        or snapshot_execution_mode(snapshot, kind) != EXECUTION_ISOLATED
                        or not process_spawn_hold_available(conn, child_count=process_child_count)):
                    return False
            worker_slot = worker_slot_free(conn, snapshot, job_id=job_id) if snapshot else None
            if worker_slot is False:
                return False  # Waiting on a slot only: no error_text, so the S1e deadline never runs.
            if worker_slot:
                values["worker_slot_at"] = values["dispatched_at"]
            conditions = [OpsJob.id == job_id, OpsJob.state == JOB_QUEUED]
            if hold_process_spawns:
                # Take every potentially blocking lock before observing expiry:
                # operation -> worker slots -> job. PostgreSQL transaction time
                # is frozen before those waits; even an UPDATE's wall-clock
                # predicate could be evaluated before its target row lock.
                if session.exec(select(OpsJob.id).where(OpsJob.id == job_id).with_for_update()).first() is None:
                    return False
                now = func.clock_timestamp() if conn.dialect.name == "postgresql" else func.current_timestamp()
                # A prepared policy's absolute budget cannot be renewed/bypassed
                # by a stale queued reader. Legacy dormant ledger tests remain
                # unchanged; any future enabled hold action must prepare first.
                conditions.append(or_(OpsJob.process_spawn_hold_deadline_at.is_(None),
                                      OpsJob.process_spawn_hold_deadline_at > now))
                stamp = session.exec(select(now)).one() if conn.dialect.name == "postgresql" else _utcnow()
                values["dispatched_at"] = values["process_spawn_hold_at"] = stamp
            else:
                # A prepared hold action must not run through ordinary dispatch.
                conditions.append(OpsJob.process_spawn_hold_target_json.is_(None))
            result = session.execute(update(OpsJob).where(*conditions)
                .values(**values).execution_options(synchronize_session=False))
            if result.rowcount == 1:
                append_job_transition(getattr(self, "events", None), session.connection(), session.get(OpsJob, job_id))
            session.commit()
            return result.rowcount == 1

    def _transition(self, job_id, from_states, values):
        if self.engine is None:
            return None
        with Session(self.engine) as session:
            old_state = session.exec(select(OpsJob.state).where(OpsJob.id == job_id).with_for_update()).first()
            result = session.execute(update(OpsJob).where(OpsJob.id == job_id, OpsJob.state.in_(from_states))
                                     .values(**values).execution_options(synchronize_session=False))
            if result.rowcount == 1 and "state" in values and old_state != values["state"]:
                append_job_transition(getattr(self, "events", None), session.connection(), session.get(OpsJob, job_id))
            session.commit()
        return self.get_job(job_id)

    def mark_dispatching(self, job_id, *, host_unit_name, status_file_path=None, log_file_path=None):
        # Compatibility for store consumers. Service always claims BEFORE I/O.
        return self._transition(job_id, (JOB_QUEUED, JOB_DISPATCHING), dict(state=JOB_DISPATCHING,
            host_unit_name=host_unit_name, status_file_path=status_file_path, log_file_path=log_file_path,
            dispatched_at=_utcnow()))

    def mark_accepted(self, job_id, *, host_unit_name=None, status_file_path=None, log_file_path=None):
        values = {"state": JOB_ACCEPTED}
        for key, value in (("host_unit_name", host_unit_name), ("status_file_path", status_file_path), ("log_file_path", log_file_path)):
            if value is not None:
                values[key] = value
        return self._transition(job_id, (JOB_DISPATCHING, JOB_ACCEPTED), values)

    def mark_running(self, job_id, *, started_at=None):
        return self._transition(job_id, (JOB_DISPATCHING, JOB_ACCEPTED), {"state": JOB_RUNNING, "started_at": started_at or _utcnow()})

    def mark_sequence(self, job_id, ledger):
        return self._transition(job_id, _OPEN_STATES, {"sequence_json": canonical_json(read_ledger(ledger))})

    def mark_terminal(self, job_id, *, state, exit_code=None, output_tail=None, error_text=None, started_at=None,
                      finished_at=None, sequence=None):
        if state not in JOB_TERMINAL_STATES:
            raise ValueError(f"not a terminal state: {state!r}")
        values = dict(state=state, exit_code=exit_code, output_tail=output_tail,
                      error_text=error_text, finished_at=finished_at or _utcnow())
        if started_at is not None:
            values["started_at"] = started_at
        if sequence is not None:
            values["sequence_json"] = canonical_json(read_ledger(sequence))
        elif state != JOB_SUCCEEDED:
            # Admission/quiesce/preflight can end a parent before a worker exists.
            # Those callbacks still carry every step, all explicitly unsubmitted.
            job = self.get_job(job_id)
            if job and job.sequence_json:
                ledger = read_ledger(json.loads(job.sequence_json))
                values["sequence_json"] = canonical_json(interrupted(
                    ledger, error_text or state, unknown=state in ("lost", "timed_out")))
        job = self._transition(job_id, _OPEN_STATES, values)
        self._settle_gpu_reconcile(job)
        return job

    def _settle_gpu_reconcile(self, job):
        """TASK-1844: a terminal GPU-service reconcile releases its GPU handoff
        reservation on a known outcome, else records why it stays held."""
        if job is None or job.state not in JOB_TERMINAL_STATES or not job.invocation_snapshot_json:
            return
        from .gpu_reconcile import is_gpu_reconcile, settle

        if is_gpu_reconcile(json.loads(job.invocation_snapshot_json).get("spec")):
            settle(self.engine, job)

    def note_queued_error(self, job_id, error):
        return self._transition(job_id, (JOB_QUEUED,), {"error_text": error})

    @contextmanager
    def preflight_lock(self, job_id):
        """Serialize preflight passes on one queued job (TASK-1169 S1e r1).

        Yields True when this pass owns the job's preflight, False when another
        pass (another process, or a concurrent read) is preflighting it now.
        Without it, an older failing pass could note its error and expire the
        job after a newer pass had found preflight passing. A session-level
        PostgreSQL advisory try-lock on its own connection: never waits, and a
        dead process's lock ends with its connection. Other engines run
        single-process and always own it.
        """
        self._require()
        if self.engine.dialect.name != "postgresql":
            yield True
            return
        key = {"k": f"ops-preflight:{job_id}"}
        conn = self.engine.connect().execution_options(isolation_level="AUTOCOMMIT")
        try:
            owned = bool(conn.execute(text("SELECT pg_try_advisory_lock(hashtextextended(:k, 0))"), key).scalar_one())
            try:
                yield owned
            finally:
                if owned:
                    try:
                        conn.execute(text("SELECT pg_advisory_unlock(hashtextextended(:k, 0))"), key)
                    except Exception:
                        conn.invalidate()  # Never pool a connection that may still hold it.
                        raise
        finally:
            conn.close()

    def fail_stale_queued(self, job_id, *, deadline_seconds, code=PREFLIGHT_FAILED):
        """QUEUED -> FAILED once preflight has kept failing past the deadline
        (TASK-1169 S1e). One atomic CAS on persisted state and the DB clock:

        - ``state='queued' AND dispatched_at IS NULL`` races ``claim_job``'s own
          QUEUED CAS; whichever commits first wins, so a claimed job never fails
          here, and nothing is submitted before a claim.
        - ``error_text IS NOT NULL`` limits it to a job whose latest preflight
          failed (``claim_job`` clears it): one waiting on a slot never expires.
        - ``submitted_at`` is the start; the caller folds the snapshot's start
          delay into ``deadline_seconds``.

        PostgreSQL only (``now()``/``make_interval``), which is every deployment;
        any other engine keeps the job QUEUED, as before. True when it failed it.
        """
        self._require()
        if self.engine.dialect.name != "postgresql":
            return False
        with self.engine.begin() as conn:
            result = conn.execute(text(
                "UPDATE ops_jobs SET state = :failed, finished_at = now(),"
                " error_text = :prefix || error_text"
                " WHERE id = :id AND state = :queued AND dispatched_at IS NULL AND error_text IS NOT NULL"
                " AND submitted_at + make_interval(secs => :deadline) < now()"),
                {"failed": JOB_FAILED, "queued": JOB_QUEUED, "id": job_id,
                 "prefix": f"{code}: ", "deadline": float(deadline_seconds)})
            if result.rowcount == 1:
                append_job_transition_by_id(getattr(self, "events", None), conn, job_id)
        if result.rowcount == 1:
            self._settle_gpu_reconcile(self.get_job(job_id))  # TASK-1844: never dispatched
        return result.rowcount == 1

    def touch_reconcile(self, job_id):
        self._transition(job_id, _OPEN_STATES, {"last_reconcile_at": _utcnow()})

    def find_active_jobs(self, *, limit=100, offset=0, recent_terminal_since=None):
        """One ledger read for all active jobs, optionally with recent outcomes.

        The global UI passes limit=None so older long-running jobs cannot fall
        outside a recent-page window; other reconciler callers keep their cap.
        """
        if self.engine is None:
            return []
        where = OpsJob.state.notin_(JOB_TERMINAL_STATES)
        if recent_terminal_since is not None:
            where = or_(where, OpsJob.finished_at >= recent_terminal_since)
        with Session(self.engine) as session:
            ordering = (OpsJob.submitted_at.desc(), OpsJob.id.desc()) if recent_terminal_since is not None else (OpsJob.submitted_at, OpsJob.id)
            query = select(OpsJob).where(where).order_by(*ordering).offset(offset)
            if limit is not None:
                query = query.limit(limit)
            return list(session.exec(query).all())
