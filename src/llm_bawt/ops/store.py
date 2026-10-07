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
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, inspect, text, update
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, SQLModel, select

from ..utils.config import Config, has_database_credentials
from ..utils.schema import SchemaBootstrapGuard
from .executor import validate_spec
from .release_spec import is_release_spec
from .models import (EXECUTOR_DOCKER, EXECUTOR_RELEASE, JOB_ACCEPTED, JOB_DISPATCHING, JOB_QUEUED, JOB_RUNNING,
                     JOB_TERMINAL_STATES, OpsJob, OpsOperation, OpsOperationRevision)
from .validation import canonical_json, validate_catalog

logger = logging.getLogger(__name__)


def _utcnow():
    return datetime.now(timezone.utc)


def _script_hash(script):
    return hashlib.sha256(script.encode()).hexdigest()


_OP_WRITABLE = {"slug", "title", "description", "enabled", "executor_kind", "target_host",
                "run_as_user", "working_directory", "command_script", "args_schema_json",
                "args_defaults_json", "timeout_seconds", "start_delay_seconds", "max_output_bytes",
                "max_concurrent", "risk_level", "category", "approval_prompt_prefix"}
_ACTIVE_CLAIMED = (JOB_DISPATCHING, JOB_ACCEPTED, JOB_RUNNING)


class OpsStoreUnavailable(RuntimeError):
    pass


class IdempotencyConflict(ValueError):
    pass


class OpsStore:
    _schema_guard = SchemaBootstrapGuard()

    def __init__(self, config: Config, engine: Any = None):
        self.config = config
        self.engine = engine
        if engine is not None:
            self._ensure_tables_exist()
        elif has_database_credentials(config):
            try:
                from ..utils.db import get_shared_engine
                self.engine = get_shared_engine(config)
                self._ensure_tables_exist()
            except Exception as exc:
                self.engine = None
                logger.warning("Ops store DB unavailable: %s", exc)

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
        existing = {c["name"] for c in inspect(conn).get_columns("ops_jobs")}
        for name, kind in (("invocation_snapshot_json", "TEXT"), ("request_payload_json", "TEXT"), ("caller_actor", "VARCHAR(128)")):
            if name not in existing:
                clause = " IF NOT EXISTS" if conn.dialect.name == "postgresql" else ""
                conn.execute(text(f"ALTER TABLE ops_jobs ADD COLUMN{clause} {name} {kind}"))

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
        if op.target_host or op.run_as_user or op.working_directory:
            raise ValueError("Operations do not support target_host/run_as_user/working_directory; no SSH or shell runner")
        spec = validate_spec(op.command_script)
        # TASK-1030: the release coordinator runs ONLY the pinned release spec,
        # and that spec never reaches the Docker executor.
        if (op.executor_kind == EXECUTOR_RELEASE) != is_release_spec(spec):
            raise ValueError("release_orchestrate specs require executor_kind 'release' and vice versa")
        for key in ("container_name_from_arg", "compose_service_from_arg"):
            if key in spec and schema.get("properties", {}).get(spec[key], {}).get("type") != "string":
                raise ValueError(f"{key} must reference a declared string argument")
        for field in ("title", "description", "risk_level"):
            if not isinstance(getattr(op, field), str):
                raise ValueError(f"{field} must be a string")

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

    def update_operation(self, slug_or_id, data, *, actor=None, _delete=False):
        self._require()
        clean = self._clean_op(data)
        with Session(self.engine) as session:
            row = self._lock_operation(session, slug_or_id)
            if row is None:
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
                   caller_session_key=None, caller_backend=None, approval_request_id=None):
        self._require()
        if not isinstance(idempotency_key, str) or not 1 <= len(idempotency_key) <= 128:
            raise ValueError("idempotency_key must be 1..128 characters")
        # Legacy/internal callers are also conflict checked by args, not just key.
        payload = request_payload_json or canonical_json({"operation": operation.slug, "args": json.loads(args_json)})
        row = OpsJob(id=uuid.uuid4().hex, operation_slug=operation.slug,
            operation_version=operation.version, operation_script_hash=operation.script_hash,
            args_json=args_json, display_args_json=display_args_json,
            invocation_snapshot_json=invocation_snapshot_json, request_payload_json=payload,
            idempotency_key=idempotency_key, caller_actor=caller_actor, caller_bot_id=caller_bot_id,
            caller_user_id=caller_user_id, caller_turn_id=caller_turn_id, caller_session_key=caller_session_key,
            caller_backend=caller_backend, approval_request_id=approval_request_id)
        with Session(self.engine) as session:
            session.add(row)
            try:
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

    def claim_job(self, job_id, *, max_concurrent=None):
        """QUEUED -> DISPATCHING atomically, reserving a per-operation slot."""
        self._require()
        job = self.get_job(job_id)
        if job is None:
            return False
        with Session(self.engine) as session:
            op = self._lock_operation(session, job.operation_slug)
            if op is None or not op.enabled or op.soft_deleted_at is not None:
                return False
            active = session.exec(select(OpsJob).where(OpsJob.operation_slug == job.operation_slug, OpsJob.state.in_(_ACTIVE_CLAIMED))).all()
            limits = [max_concurrent] if max_concurrent is not None else []
            for other in active:
                if other.invocation_snapshot_json:
                    cap = json.loads(other.invocation_snapshot_json)["execution"].get("max_concurrent")
                    if cap is not None:
                        limits.append(cap)
            if limits and len(active) >= min(limits):
                return False
            result = session.execute(update(OpsJob).where(OpsJob.id == job_id, OpsJob.state == JOB_QUEUED)
                .values(state=JOB_DISPATCHING, dispatched_at=_utcnow(), host_unit_name=f"llm-bawt-ops-{job_id}", error_text=None)
                .execution_options(synchronize_session=False))
            session.commit()
            return result.rowcount == 1

    def _transition(self, job_id, from_states, values):
        if self.engine is None:
            return None
        with Session(self.engine) as session:
            session.execute(update(OpsJob).where(OpsJob.id == job_id, OpsJob.state.in_(from_states))
                            .values(**values).execution_options(synchronize_session=False))
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

    def mark_terminal(self, job_id, *, state, exit_code=None, output_tail=None, error_text=None, started_at=None, finished_at=None):
        if state not in JOB_TERMINAL_STATES:
            raise ValueError(f"not a terminal state: {state!r}")
        values = dict(state=state, exit_code=exit_code, output_tail=output_tail,
                      error_text=error_text, finished_at=finished_at or _utcnow())
        if started_at is not None:
            values["started_at"] = started_at
        return self._transition(job_id, (JOB_QUEUED, *_ACTIVE_CLAIMED), values)

    def note_queued_error(self, job_id, error):
        return self._transition(job_id, (JOB_QUEUED,), {"error_text": error})

    def touch_reconcile(self, job_id):
        self._transition(job_id, (JOB_QUEUED, *_ACTIVE_CLAIMED), {"last_reconcile_at": _utcnow()})

    def find_active_jobs(self, *, limit=100, offset=0):
        if self.engine is None:
            return []
        with Session(self.engine) as session:
            return list(session.exec(select(OpsJob).where(OpsJob.state.notin_(JOB_TERMINAL_STATES))
                .order_by(OpsJob.submitted_at, OpsJob.id).offset(offset).limit(limit)).all())
