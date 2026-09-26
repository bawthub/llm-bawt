"""Durable, one-use consent and ownership ledger for GPU handoffs.

This is a state machine, not a container controller. External actions must be
recorded before dispatch and reconciled on restart; a lease timeout never
implicitly authorizes re-executing a stop/start operation.
"""
from __future__ import annotations

import json
import secrets
import time
from dataclasses import dataclass

from sqlalchemy import Column, Float, Integer, MetaData, String, Table, Text, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine

from llm_bawt.utils.schema import SchemaBootstrapGuard

_metadata = MetaData()
# These actions require a durable ops receipt before the cursor can advance.
OPS_ACTIONS = frozenset({
    "stop_moshi_stt", "stop_moshi_tts", "start_moshi_stt", "start_moshi_tts",
})
_video_jobs = Table(
    "gpu_handoff_video_jobs", _metadata,
    Column("id", String(32), primary_key=True),
    Column("generation", Integer, nullable=False),
)
_state = Table(
    "gpu_handoff_state", _metadata,
    Column("id", Integer, primary_key=True),
    Column("owner", String(32), nullable=False),
    Column("phase", String(32), nullable=False),
    Column("generation", Integer, nullable=False),
    Column("offer_hash", String(64)),
    Column("offer_user", String(128)),
    Column("offer_target", String(16)),
    Column("offer_expires", Float),
    Column("actions", Text),
    Column("next_action", Integer, nullable=False, default=0),
    Column("pending_action", String(128)),
    Column("last_job_id", String(128)),
    Column("last_error", Text),
)


class HandoffConflict(ValueError):
    """An offer, generation, or transition is no longer current."""


@dataclass(frozen=True)
class HandoffState:
    owner: str
    phase: str
    generation: int
    target: str | None
    actions: tuple[str, ...]
    next_action: int
    pending_action: str | None
    expires_at: float | None
    last_job_id: str | None
    last_error: str | None


class GpuHandoffStore:
    """Serialize transitions on one database row; fail closed on ambiguity."""

    _guard = SchemaBootstrapGuard()

    def __init__(self, engine: Engine):
        self.engine = engine
        self._guard.run(engine, "gpu-handoff-v1", lambda conn: _metadata.create_all(conn))
        with engine.begin() as conn:
            # A deployment cannot infer ownership from running services. Reconcile
            # observations before offering a transition from unknown. Upsert so
            # concurrent app processes cannot both attempt the first insert.
            values = dict(id=1, owner="unknown", phase="idle", generation=0, next_action=0)
            insert = (pg_insert(_state) if conn.dialect.name == "postgresql" else
                      sqlite_insert(_state) if conn.dialect.name == "sqlite" else None)
            if insert is None:
                raise RuntimeError("GPU handoff requires PostgreSQL or SQLite")
            conn.execute(insert.values(**values).on_conflict_do_nothing(index_elements=["id"]))

    @staticmethod
    def _view(row) -> HandoffState:
        return HandoffState(
            owner=row["owner"], phase=row["phase"], generation=row["generation"],
            target=row["offer_target"], actions=tuple(json.loads(row["actions"] or "[]")),
            next_action=row["next_action"], pending_action=row["pending_action"],
            expires_at=row["offer_expires"], last_job_id=row["last_job_id"],
            last_error=row["last_error"],
        )

    def _locked(self, conn):
        stmt = select(_state).where(_state.c.id == 1)
        if conn.dialect.name == "postgresql":
            stmt = stmt.with_for_update()
        return conn.execute(stmt).mappings().one()

    def status(self) -> HandoffState:
        with self.engine.connect() as conn:
            return self._view(conn.execute(select(_state).where(_state.c.id == 1)).mappings().one())

    def active_video_jobs(self) -> int:
        """Count durable render claims, including orphaned or uncertain jobs."""
        with self.engine.connect() as conn:
            return len(conn.execute(select(_video_jobs.c.id)).all())

    def recover_orphaned_video(self, *, worker_restarted: bool = False) -> HandoffState:
        """Fence orphaned render claims or a lost video worker after restart.

        A fresh worker is not evidence that the prior GPU owner and residency
        survived. No claim or lease timeout automatically grants a new owner.
        """
        with self.engine.begin() as conn:
            row = self._locked(conn)
            has_claims = conn.execute(select(_video_jobs.c.id).limit(1)).first() is not None
            lost_owner = worker_restarted and row["owner"] == "video"
            if (has_claims or lost_owner) and row["phase"] != "recovery_required":
                reason = ("Video worker restarted with unresolved render claims; verify GPU and jobs"
                          if has_claims else "Video worker restarted; verify model residency and GPU owner")
                conn.execute(_state.update().where(_state.c.id == 1).values(
                    owner="unknown", phase="recovery_required",
                    generation=row["generation"] + 1, last_error=reason,
                ))
                row = self._locked(conn)
            return self._view(row)

    def claim_video(self, job_id: str) -> int:
        """Authorize a render against the current owner, across processes."""
        if not job_id or len(job_id) > 32:
            raise ValueError("Invalid video job ID")
        with self.engine.begin() as conn:
            row = self._locked(conn)
            if row["phase"] != "idle" or row["owner"] != "video":
                raise HandoffConflict("GPU does not belong to video; request a handoff")
            # Claims cover queued requests too: no handoff may begin until the
            # entire queue drains, even when the worker is between renders.
            conn.execute(_video_jobs.insert().values(id=job_id, generation=row["generation"]))
            return row["generation"]

    def release_video(self, job_id: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(_video_jobs.delete().where(_video_jobs.c.id == job_id))

    def offer(self, *, user: str, target: str, actions: tuple[str, ...],
              expected_generation: int, ttl: int = 120) -> tuple[str, HandoffState]:
        if not user or target not in ("video", "voice") or not actions or not 1 <= ttl <= 300:
            raise ValueError("A scoped user, target, ordered actions, and short expiry are required")
        if any(not action or len(action) > 128 for action in actions):
            raise ValueError("Invalid transition action")
        token = secrets.token_urlsafe(32)
        import hashlib
        with self.engine.begin() as conn:
            row = self._locked(conn)
            if row["generation"] != expected_generation or row["phase"] not in ("idle", "offered"):
                raise HandoffConflict("GPU state changed; request a fresh preflight")
            if row["owner"] == target:
                raise HandoffConflict("GPU already belongs to the requested mode")
            if conn.execute(select(_video_jobs.c.id).limit(1)).first() is not None:
                raise HandoffConflict("Active video work must finish before offering a switch")
            # Replacing an offer invalidates its old token and bumps the generation.
            conn.execute(_state.update().where(_state.c.id == 1).values(
                phase="offered", generation=row["generation"] + 1,
                offer_hash=hashlib.sha256(token.encode()).hexdigest(), offer_user=user,
                offer_target=target, offer_expires=time.time() + ttl,
                actions=json.dumps(actions), next_action=0, pending_action=None,
                last_job_id=None, last_error=None,
            ))
            result = self._locked(conn)
        return token, self._view(result)

    def confirm(self, *, user: str, token: str, expected_generation: int) -> HandoffState:
        import hashlib
        with self.engine.begin() as conn:
            row = self._locked(conn)
            if (row["phase"] != "offered" or row["generation"] != expected_generation
                    or row["offer_user"] != user or not row["offer_expires"]
                    or row["offer_expires"] <= time.time()
                    or not secrets.compare_digest(row["offer_hash"] or "", hashlib.sha256(token.encode()).hexdigest())):
                raise HandoffConflict("Consent is expired, used, or does not match this transition")
            conn.execute(_state.update().where(_state.c.id == 1).values(
                phase="transitioning", generation=row["generation"] + 1,
                offer_hash=None, offer_user=None, offer_expires=None,
            ))
            return self._view(self._locked(conn))

    def begin_action(self, *, generation: int, action: str) -> HandoffState:
        """Persist intent before external effects; ambiguous dispatch needs recovery."""
        with self.engine.begin() as conn:
            row = self._locked(conn)
            actions = json.loads(row["actions"] or "[]")
            if (row["phase"] != "transitioning" or row["generation"] != generation
                    or row["pending_action"] or row["next_action"] >= len(actions)
                    or actions[row["next_action"]] != action):
                raise HandoffConflict("Transition action is not next or requires reconciliation")
            conn.execute(_state.update().where(_state.c.id == 1).values(pending_action=action, last_job_id=None))
            return self._view(self._locked(conn))

    def record_job(self, *, generation: int, action: str, job_id: str) -> HandoffState:
        if not job_id:
            raise ValueError("An operation job ID is required")
        with self.engine.begin() as conn:
            row = self._locked(conn)
            if (row["phase"] != "transitioning" or row["generation"] != generation
                    or row["pending_action"] != action or row["last_job_id"]):
                raise HandoffConflict("Transition has changed or operation needs reconciliation")
            conn.execute(_state.update().where(_state.c.id == 1).values(last_job_id=job_id))
            return self._view(self._locked(conn))

    def finish_action(self, *, generation: int, action: str) -> HandoffState:
        """Only a verified external outcome permits advancing the cursor."""
        with self.engine.begin() as conn:
            row = self._locked(conn)
            if (row["phase"] != "transitioning" or row["generation"] != generation
                    or row["pending_action"] != action
                    or (action in OPS_ACTIONS and not row["last_job_id"])):
                raise HandoffConflict("Action is not pending or needs an operation receipt")
            conn.execute(_state.update().where(_state.c.id == 1).values(
                pending_action=None, next_action=row["next_action"] + 1,
            ))
            return self._view(self._locked(conn))

    def complete(self, *, generation: int, target: str) -> HandoffState:
        """Caller must verify real container/model health before invoking."""
        with self.engine.begin() as conn:
            row = self._locked(conn)
            actions = json.loads(row["actions"] or "[]")
            if (row["phase"] != "transitioning" or row["generation"] != generation
                    or row["offer_target"] != target or row["pending_action"]
                    or row["next_action"] != len(actions)):
                raise HandoffConflict("Transition has unfinished actions or mismatched target")
            conn.execute(_state.update().where(_state.c.id == 1).values(
                owner=target, phase="idle", generation=generation + 1,
                offer_target=None, actions=None, next_action=0, last_job_id=None,
            ))
            return self._view(self._locked(conn))

    def recover(self, *, generation: int, reason: str) -> HandoffState:
        with self.engine.begin() as conn:
            row = self._locked(conn)
            if row["phase"] != "transitioning" or row["generation"] != generation:
                raise HandoffConflict("Transition has changed")
            conn.execute(_state.update().where(_state.c.id == 1).values(
                owner="unknown", phase="recovery_required", generation=generation + 1,
                last_error=reason[:1000],
            ))
            return self._view(self._locked(conn))

    def mark_interrupted(self) -> HandoffState:
        """Startup reconciliation: never replay an in-flight side effect."""
        with self.engine.begin() as conn:
            row = self._locked(conn)
            if row["phase"] == "transitioning":
                conn.execute(_state.update().where(_state.c.id == 1).values(
                    owner="unknown", phase="recovery_required", generation=row["generation"] + 1,
                    last_error="Transition interrupted; reconcile external operations before retrying",
                ))
                row = self._locked(conn)
            return self._view(row)
