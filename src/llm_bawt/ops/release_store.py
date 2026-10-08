"""Transactional persistence and claims for durable release orchestration."""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from sqlalchemy import func, inspect, or_, text, update
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, SQLModel, select

from ..utils.config import Config, has_database_credentials
from ..utils.schema import SchemaBootstrapGuard
from .release_models import (ReleaseEvent, ReleaseRun, RELEASE_AWAITING_DEPLOY_APPROVAL,
                             RELEASE_BUILD_COMPLETE, RELEASE_DEPLOYING, RELEASE_DEPLOYED,
                             RELEASE_TERMINAL_STATES)
from .validation import canonical_json


class ReleaseStoreUnavailable(RuntimeError):
    pass


_REMOVED_COLUMNS = frozenset({
    "llm_bawt_mode", "llm_bawt_repository", "llm_bawt_branch", "llm_bawt_sha", "warning_text",
})


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _release_id(parent_job_id: str) -> str:
    return hashlib.sha256(f"bawthub-release:{parent_job_id}".encode()).hexdigest()[:32]


def _request_id(parent_job_id: str) -> str:
    return "release-" + hashlib.sha256(f"github:{parent_job_id}".encode()).hexdigest()[:32]


def _as_states(states: str | Iterable[str]) -> tuple[str, ...]:
    return (states,) if isinstance(states, str) else tuple(states)


class ReleaseStore:
    """Release rows, append-only events, CAS transitions and expiring claims."""

    _schema_guard = SchemaBootstrapGuard()
    _MUTABLE = {
        "state",
        "source_sha",
        "release_plan_json",
        "dispatch_started_at",
        "github_run_id",
        "github_run_attempt",
        "github_run_url",
        "rerun_count",
        "receipt_json",
        "receipt_verified_at",
        "deployable",
        "version",
        "tag",
        "digest",
        "image_repository",
        "deploy_approval_request_id",
        "deploy_job_id",
        "error_code",
        "error_text",
        "finished_at",
        "next_check_at",
        "phase_deadline_at",
    }

    def __init__(self, config: Config | None, engine: Any = None):
        self.config = config
        self.engine = engine
        if engine is None and config is not None and has_database_credentials(config):
            from ..utils.db import get_shared_engine

            self.engine = get_shared_engine(config)
        if self.engine is not None:
            self._ensure_tables_exist()

    def _ensure_tables_exist(self) -> None:
        def bootstrap(conn):
            SQLModel.metadata.create_all(
                bind=conn,
                tables=[ReleaseRun.__table__, ReleaseEvent.__table__],
            )
            # llm-bawt release tagging was removed; drop its columns from
            # databases created before that. Historical values remain in each
            # run's plan/receipt JSON and events.
            existing = {col["name"] for col in inspect(conn).get_columns(ReleaseRun.__tablename__)}
            for column in sorted(_REMOVED_COLUMNS & existing):
                conn.execute(text(f"ALTER TABLE {ReleaseRun.__tablename__} DROP COLUMN {column}"))

        self._schema_guard.run(self.engine, "ops-release-store-task1030-v3", bootstrap)

    def _require(self) -> None:
        if self.engine is None:
            raise ReleaseStoreUnavailable("release store has no DB engine")

    @staticmethod
    def _append_event(
        session: Session,
        row: ReleaseRun,
        *,
        event_type: str,
        detail: dict[str, Any] | None = None,
    ) -> ReleaseEvent:
        row.event_seq = int(row.event_seq or 0) + 1
        row.updated_at = _utcnow()
        session.add(row)
        event = ReleaseEvent(
            release_run_id=row.id,
            sequence=row.event_seq,
            event_type=event_type,
            state=row.state,
            detail_json=canonical_json(detail or {}),
        )
        session.add(event)
        return event

    def create(
        self,
        *,
        parent_job_id: str,
        release_task: str,
        bump: str,
        github_repository: str,
        workflow_path: str,
        canonical_branch: str,
    ) -> ReleaseRun:
        """Create once per parent job; duplicate delivery returns the same row."""
        self._require()
        release_id = _release_id(parent_job_id)
        request_id = _request_id(parent_job_id)
        row = ReleaseRun(
            id=release_id,
            parent_job_id=parent_job_id,
            release_request_id=request_id,
            release_task=release_task,
            bump=bump,
            github_repository=github_repository,
            workflow_path=workflow_path,
            canonical_branch=canonical_branch,
        )
        with Session(self.engine) as session:
            session.add(row)
            self._append_event(session, row, event_type="release.created")
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                existing = self.get_by_parent_job(parent_job_id)
                if existing is None:
                    raise
                expected = (release_task, bump, github_repository, workflow_path, canonical_branch)
                actual = (
                    existing.release_task,
                    existing.bump,
                    existing.github_repository,
                    existing.workflow_path,
                    existing.canonical_branch,
                )
                if actual != expected:
                    raise ValueError("parent ops job is already bound to another release plan")
                return existing
            session.refresh(row)
            return row

    def get(self, release_run_id: str) -> ReleaseRun | None:
        if self.engine is None:
            return None
        with Session(self.engine) as session:
            return session.get(ReleaseRun, release_run_id)

    def get_by_parent_job(self, parent_job_id: str) -> ReleaseRun | None:
        if self.engine is None:
            return None
        with Session(self.engine) as session:
            return session.exec(
                select(ReleaseRun).where(ReleaseRun.parent_job_id == parent_job_id)
            ).first()

    def get_by_request_id(self, release_request_id: str) -> ReleaseRun | None:
        if self.engine is None:
            return None
        with Session(self.engine) as session:
            return session.exec(
                select(ReleaseRun).where(
                    ReleaseRun.release_request_id == release_request_id
                )
            ).first()

    def list(self, *, state: str | None = None, limit: int = 50, offset: int = 0) -> list[ReleaseRun]:
        if self.engine is None:
            return []
        with Session(self.engine) as session:
            stmt = select(ReleaseRun)
            if state:
                stmt = stmt.where(ReleaseRun.state == state)
            return list(
                session.exec(
                    stmt.order_by(ReleaseRun.created_at.desc(), ReleaseRun.id)
                    .offset(max(0, offset))
                    .limit(min(max(1, limit), 200))
                ).all()
            )

    def count(self, *, state: str | None = None) -> int:
        if self.engine is None:
            return 0
        with Session(self.engine) as session:
            stmt = select(func.count()).select_from(ReleaseRun)
            if state:
                stmt = stmt.where(ReleaseRun.state == state)
            return session.exec(stmt).one()

    def events(self, release_run_id: str) -> list[ReleaseEvent]:
        if self.engine is None:
            return []
        with Session(self.engine) as session:
            return list(
                session.exec(
                    select(ReleaseEvent)
                    .where(ReleaseEvent.release_run_id == release_run_id)
                    .order_by(ReleaseEvent.sequence)
                ).all()
            )

    def claim(self, release_run_id: str, *, lease_seconds: int = 30) -> ReleaseRun | None:
        """Claim a nonterminal release; expired leases can be recovered."""
        self._require()
        if not 5 <= lease_seconds <= 300:
            raise ValueError("lease_seconds must be 5..300")
        now = _utcnow()
        token = uuid.uuid4().hex
        with Session(self.engine) as session:
            result = session.execute(
                update(ReleaseRun)
                .where(
                    ReleaseRun.id == release_run_id,
                    ReleaseRun.state.notin_(RELEASE_TERMINAL_STATES),
                    or_(
                        ReleaseRun.claim_token.is_(None),
                        ReleaseRun.claim_expires_at.is_(None),
                        ReleaseRun.claim_expires_at <= now,
                    ),
                )
                .values(
                    claim_token=token,
                    claim_expires_at=now + timedelta(seconds=lease_seconds),
                    reconcile_attempts=ReleaseRun.reconcile_attempts + 1,
                    updated_at=now,
                )
                .execution_options(synchronize_session=False)
            )
            session.commit()
            if result.rowcount != 1:
                return None
            return session.get(ReleaseRun, release_run_id)

    def renew_claim(self, release_run_id: str, claim_token: str, *, lease_seconds: int = 30) -> bool:
        self._require()
        now = _utcnow()
        with Session(self.engine) as session:
            result = session.execute(
                update(ReleaseRun)
                .where(
                    ReleaseRun.id == release_run_id,
                    ReleaseRun.claim_token == claim_token,
                    ReleaseRun.state.notin_(RELEASE_TERMINAL_STATES),
                )
                .values(claim_expires_at=now + timedelta(seconds=lease_seconds), updated_at=now)
                .execution_options(synchronize_session=False)
            )
            session.commit()
            return result.rowcount == 1

    def release_claim(
        self,
        release_run_id: str,
        claim_token: str,
        *,
        next_check_at: datetime | None = None,
    ) -> bool:
        """Drop the lease; optionally schedule the next pump (no event row)."""
        self._require()
        values: dict[str, Any] = {"claim_token": None, "claim_expires_at": None, "updated_at": _utcnow()}
        if next_check_at is not None:
            values["next_check_at"] = next_check_at
        with Session(self.engine) as session:
            result = session.execute(
                update(ReleaseRun)
                .where(ReleaseRun.id == release_run_id, ReleaseRun.claim_token == claim_token)
                .values(**values)
                .execution_options(synchronize_session=False)
            )
            session.commit()
            return result.rowcount == 1

    def transition(
        self,
        release_run_id: str,
        *,
        from_states: str | Iterable[str],
        to_state: str,
        claim_token: str,
        event_type: str,
        detail: dict[str, Any] | None = None,
        **values: Any,
    ) -> ReleaseRun | None:
        """Claim-fenced CAS transition and event in one transaction."""
        self._require()
        unknown = set(values) - self._MUTABLE
        if unknown or "state" in values:
            raise ValueError(f"unsupported release fields: {sorted(unknown or {'state'})}")
        normalized = dict(values)
        for field in ("release_plan_json", "receipt_json"):
            if field in normalized and isinstance(normalized[field], dict):
                normalized[field] = canonical_json(normalized[field])
        normalized.update(state=to_state, updated_at=_utcnow())
        if to_state in RELEASE_TERMINAL_STATES and "finished_at" not in normalized:
            normalized["finished_at"] = _utcnow()
        with Session(self.engine) as session:
            result = session.execute(
                update(ReleaseRun)
                .where(
                    ReleaseRun.id == release_run_id,
                    ReleaseRun.state.in_(_as_states(from_states)),
                    ReleaseRun.claim_token == claim_token,
                )
                .values(**normalized)
                .execution_options(synchronize_session=False)
            )
            if result.rowcount != 1:
                session.rollback()
                return None
            row = session.get(ReleaseRun, release_run_id)
            self._append_event(session, row, event_type=event_type, detail=detail)
            session.commit()
            session.refresh(row)
            return row

    def note(
        self,
        release_run_id: str,
        *,
        claim_token: str,
        event_type: str,
        detail: dict[str, Any] | None = None,
        **values: Any,
    ) -> ReleaseRun | None:
        """Claim-fenced same-state update plus append-only event."""
        row = self.get(release_run_id)
        if row is None:
            return None
        return self.transition(
            release_run_id,
            from_states=row.state,
            to_state=row.state,
            claim_token=claim_token,
            event_type=event_type,
            detail=detail,
            **values,
        )

    def active_ids(self, *, limit: int = 100, offset: int = 0) -> list[str]:
        if self.engine is None:
            return []
        with Session(self.engine) as session:
            return list(
                session.exec(
                    select(ReleaseRun.id)
                    .where(ReleaseRun.state.notin_(RELEASE_TERMINAL_STATES))
                    .order_by(ReleaseRun.updated_at, ReleaseRun.id)
                    .offset(max(0, offset))
                    .limit(min(max(1, limit), 500))
                ).all()
            )

    def verified_binding(self, release_run_id: str) -> dict[str, Any]:
        row = self.get(release_run_id)
        if row is None:
            raise ValueError("release run does not exist")
        if (row.state not in {RELEASE_BUILD_COMPLETE, RELEASE_AWAITING_DEPLOY_APPROVAL,
                              RELEASE_DEPLOYING, RELEASE_DEPLOYED}
                or not row.deployable or not row.receipt_json or not row.receipt_verified_at):
            raise ValueError("release run has no active verified deployable receipt")
        receipt = json.loads(row.receipt_json)
        if (receipt.get("schema") != "bawthub.release-receipt/v1"
                or receipt.get("status") != "complete"
                or receipt.get("repository") != row.github_repository):
            raise ValueError("stored release receipt is not deployable or belongs to another repository")
        # GitHub exposes run id/attempt as strings in env; the row stores the
        # attempt as an int. Compare their canonical text, never the types.
        required = {
            "workflow_run_id": str(row.github_run_id or ""),
            "workflow_run_attempt": str(row.github_run_attempt or ""),
            "release_request_id": row.release_request_id,
            "source_sha": row.source_sha,
            "version": row.version,
            "tag": row.tag,
            "digest": row.digest,
            "image_repository": row.image_repository,
        }
        for key, expected in required.items():
            got = receipt.get(key)
            if key in {"workflow_run_id", "workflow_run_attempt"} and got is not None:
                got = str(got)
            if not expected or got != expected:
                raise ValueError(f"stored release binding mismatch for {key}")
        return {
            "release_run_id": row.id,
            "github_repository": row.github_repository,
            "workflow_path": row.workflow_path,
            "workflow_run_id": row.github_run_id,
            "workflow_run_attempt": str(row.github_run_attempt or ""),
            "workflow_run_url": row.github_run_url or "",
            "trigger_sha": receipt.get("base_sha"),
            "source_sha": row.source_sha,
            "version": row.version,
            "tag": row.tag,
            "digest": row.digest,
            "image_repository": row.image_repository,
            "image_ref": f"{row.image_repository}@{row.digest}",
            "verified_at": _aware(row.receipt_verified_at).isoformat(),
            "receipt_status": receipt.get("status"),
        }


__all__ = ["ReleaseStore", "ReleaseStoreUnavailable"]
