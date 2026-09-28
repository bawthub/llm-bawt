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

from sqlalchemy import Column, Float, Integer, MetaData, String, Table, Text, select, text
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
_calibrations = Table(
    "gpu_video_calibrations", _metadata,
    Column("generation", Integer, primary_key=True),
    Column("job_id", String(32), nullable=False, unique=True),
    Column("measurement_json", Text),
)
_plans = Table(
    "gpu_handoff_plans", _metadata,
    Column("generation", Integer, primary_key=True),
    Column("plan_json", Text, nullable=False),
    Column("voice_lease", String(32)),
)
_state = Table(
    "gpu_handoff_state", _metadata,
    Column("id", Integer, primary_key=True),
    Column("owner", String(32), nullable=False),
    Column("phase", String(32), nullable=False),
    Column("generation", Integer, nullable=False),
    Column("offer_hash", String(64)),
    Column("offer_user", String(128)),
    Column("offer_target", String(32)),
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


def _bootstrap_schema(conn) -> None:
    _metadata.create_all(conn)
    if conn.dialect.name == "postgresql":
        # v3 created offer_target as varchar(16); "video_calibration" is 17 chars.
        # Widening is idempotent and create_all never alters existing columns.
        conn.execute(text("ALTER TABLE gpu_handoff_state ALTER COLUMN offer_target TYPE varchar(32)"))


class GpuHandoffStore:
    """Serialize transitions on one database row; fail closed on ambiguity."""

    _guard = SchemaBootstrapGuard()

    def __init__(self, engine: Engine):
        self.engine = engine
        self._guard.run(engine, "gpu-handoff-v4", _bootstrap_schema)
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
        elif conn.dialect.name == "sqlite":
            # SQLite has no FOR UPDATE; obtain its writer lock before reading
            # so concurrent confirmations cannot both consume the same offer.
            conn.execute(_state.update().where(_state.c.id == 1).values(id=1))
        return conn.execute(stmt).mappings().one()

    def status(self) -> HandoffState:
        with self.engine.connect() as conn:
            return self._view(conn.execute(select(_state).where(_state.c.id == 1)).mappings().one())

    def active_video_jobs(self) -> int:
        """Count durable render claims, including orphaned or uncertain jobs."""
        with self.engine.connect() as conn:
            return len(conn.execute(select(_video_jobs.c.id)).all())

    def recover_orphaned_video(self, *, worker_restarted: bool = False) -> HandoffState:
        """Fence orphaned render claims or an interrupted calibration after restart.

        Video ownership records "voice parked under a lease, Moshi stopped"; a
        clean restart of an idle worker changes neither, and every render
        re-verifies the lease and live VRAM before its claim, so it keeps the
        owner. Unresolved claims (unknown render outcome) and a calibration
        reservation (its measurement belongs to the lost process) still fence.
        No claim or lease timeout automatically grants a new owner.
        """
        with self.engine.begin() as conn:
            row = self._locked(conn)
            has_claims = conn.execute(select(_video_jobs.c.id).limit(1)).first() is not None
            lost_owner = worker_restarted and row["owner"] == "video_calibration"
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

    def claim_calibration(self, job_id: str, *, generation: int, profile: dict) -> None:
        from .gpu_profile import CALIBRATION_PROFILE, require_calibration_profile

        require_calibration_profile(profile)
        with self.engine.begin() as conn:
            row = self._locked(conn)
            if row["generation"] != generation or row["owner"] != "video_calibration" or row["phase"] != "idle":
                raise HandoffConflict("Calibration reservation is no longer current")
            plan_row = conn.execute(select(_plans.c.plan_json).where(_plans.c.generation == generation - 2)).first()
            # Consent covered this envelope; the render must fall inside it (checked above).
            if not plan_row or json.loads(plan_row[0])["calibration_profile"] != CALIBRATION_PROFILE:
                raise HandoffConflict("Calibration profile does not match the consented plan")
            if conn.execute(select(_video_jobs.c.id).limit(1)).first() or conn.execute(
                select(_calibrations.c.generation).where(_calibrations.c.generation == generation)
            ).first():
                raise HandoffConflict("Calibration was already submitted; inspect its existing job")
            conn.execute(_calibrations.insert().values(generation=generation, job_id=job_id))
            conn.execute(_video_jobs.insert().values(id=job_id, generation=generation))

    def finish_calibration(self, job_id: str, measurement: dict) -> dict:
        """Persist observed evidence, not a guessed peak; atomically release claim."""
        from .gpu_profile import CALIBRATION_PROFILE

        fields = ("total_mib", "device_total_mib", "baseline_used_mib", "sampled_peak_used_mib", "peak_reserved_mib", "peak_allocated_mib", "worker_reserved_mib")
        if (not isinstance(measurement, dict) or any(type(measurement.get(k)) is not int or measurement[k] < 0 for k in fields)
                or not isinstance(measurement.get("gpu_uuid"), str) or not measurement["gpu_uuid"].startswith("GPU-")):
            raise HandoffConflict("Calibration did not return valid GPU memory evidence")
        total, baseline, peak = measurement["total_mib"], measurement["baseline_used_mib"], measurement["sampled_peak_used_mib"]
        if (not 0 <= baseline <= peak <= total or total == 0 or measurement["peak_reserved_mib"] == 0
                or not measurement["peak_allocated_mib"] <= measurement["peak_reserved_mib"] <= total
                or measurement["worker_reserved_mib"] > measurement["peak_reserved_mib"]):
            raise HandoffConflict("Calibration memory evidence is inconsistent")
        # CUDA allocator high-water mark plus sampled driver overhead, with a
        # conservative 2 GiB minimum / 10% margin. Only this exact profile is valid.
        peak_increment = max(measurement["peak_reserved_mib"], peak - baseline)
        margin = max(2048, (peak_increment + 9) // 10)
        supported = baseline + peak_increment + margin <= total
        evidence = {**measurement, "profile": CALIBRATION_PROFILE, "margin_mib": margin,
                    "required_mib": peak_increment + margin, "supported": supported,
                    "measured_at": time.time(), "job_id": job_id}
        with self.engine.begin() as conn:
            row = self._locked(conn)
            run = conn.execute(select(_calibrations).where(_calibrations.c.job_id == job_id)).mappings().one_or_none()
            claim = conn.execute(select(_video_jobs).where(_video_jobs.c.id == job_id)).mappings().one_or_none()
            if (run is None or claim is None or row["owner"] != "video_calibration" or row["phase"] != "idle"
                    or run["generation"] != row["generation"] or claim["generation"] != row["generation"]):
                raise HandoffConflict("Calibration ownership changed; reconcile its result")
            plan_row = conn.execute(select(_plans.c.plan_json).where(_plans.c.generation == run["generation"] - 2)).first()
            original_gpu = json.loads(plan_row[0])["observed"]["gpu"] if plan_row else {}
            if (measurement["gpu_uuid"] != original_gpu.get("uuid")
                    or measurement["device_total_mib"] != original_gpu.get("total_mib")):
                raise HandoffConflict("Calibration ran on a different GPU than the consented reservation")
            conn.execute(_calibrations.update().where(_calibrations.c.job_id == job_id).values(measurement_json=json.dumps(evidence)))
            conn.execute(_video_jobs.delete().where(_video_jobs.c.id == job_id))
            conn.execute(_state.update().where(_state.c.id == 1).values(
                owner="video" if supported else "unknown", phase="idle" if supported else "recovery_required",
                generation=row["generation"] + 1,
                last_error=None if supported else "Calibration produced a video but insufficient memory margin; keep ordinary rendering blocked",
            ))
        return evidence

    def calibration(self) -> dict | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(_calibrations.c.measurement_json).where(
                _calibrations.c.measurement_json.is_not(None),
            ).order_by(_calibrations.c.generation.desc()).limit(1)).first()
            return json.loads(row[0]) if row else None

    def validate_video_profile(self, profile: dict, gpu: dict, worker_reserved_mib: int) -> None:
        from datetime import UTC, datetime
        from .gpu_profile import require_calibration_profile

        require_calibration_profile(profile)
        measured = self.calibration()
        if not measured or not measured["supported"]:
            raise HandoffConflict("A successful calibrated memory profile is required")
        try:
            age = (datetime.now(UTC) - datetime.fromisoformat(gpu["observed_at"])).total_seconds()
            free = gpu["free_mib"]
            valid = (gpu.get("ready") is True and gpu.get("uuid") == measured["gpu_uuid"]
                     and gpu.get("total_mib") == measured.get("device_total_mib") and 0 <= age <= 15
                     and type(free) is int and 0 <= free <= measured["device_total_mib"]
                     and type(worker_reserved_mib) is int and 0 <= worker_reserved_mib <= measured["total_mib"]
                     and free >= measured["margin_mib"] and free + worker_reserved_mib >= measured["required_mib"])
        except (ValueError, TypeError, KeyError):
            valid = False
        if not valid:
            raise HandoffConflict("Current GPU memory does not satisfy the measured profile and margin")

    def voice_lease(self) -> str | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(_plans.c.voice_lease).where(_plans.c.voice_lease.is_not(None))
                .order_by(_plans.c.generation.desc()).limit(1)).first()
            return row[0] if row else None

    def abandon_calibration(self, job_id: str, *, reason: str) -> HandoffState:
        """The render finished but its evidence was rejected: release, then fence."""
        with self.engine.begin() as conn:
            row = self._locked(conn)
            conn.execute(_video_jobs.delete().where(_video_jobs.c.id == job_id))
            conn.execute(_state.update().where(_state.c.id == 1).values(
                owner="unknown", phase="recovery_required", generation=row["generation"] + 1,
                last_error=f"Calibration measurement rejected: {reason}"[:1000],
            ))
            return self._view(self._locked(conn))

    def release_all_video_claims(self) -> int:
        """Only after the worker process is gone: no claim can still be rendering."""
        with self.engine.begin() as conn:
            return conn.execute(_video_jobs.delete()).rowcount

    def release_video(self, job_id: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(_video_jobs.delete().where(_video_jobs.c.id == job_id))

    def offer(self, *, user: str, target: str, actions: tuple[str, ...],
              expected_generation: int, ttl: int = 120,
              plan: dict | None = None) -> tuple[str, HandoffState]:
        if not user or target not in ("video", "voice", "video_calibration") or not actions or not 1 <= ttl <= 300:
            raise ValueError("A scoped user, target, ordered actions, and short expiry are required")
        if any(not action or len(action) > 128 for action in actions):
            raise ValueError("Invalid transition action")
        token = secrets.token_urlsafe(32)
        import hashlib
        # Restoring voice is also the recovery path: it may start from a fenced
        # state, where claims are stale by definition; its first action resets
        # the worker and clears them. Outside recovery, claims are live renders.
        restore = target == "voice"
        phases = ("idle", "offered", "recovery_required") if restore else ("idle", "offered")
        with self.engine.begin() as conn:
            row = self._locked(conn)
            if row["generation"] != expected_generation or row["phase"] not in phases:
                raise HandoffConflict("GPU state changed; request a fresh preflight")
            if row["owner"] == target:
                raise HandoffConflict("GPU already belongs to the requested mode")
            if (not (restore and row["phase"] == "recovery_required")
                    and conn.execute(select(_video_jobs.c.id).limit(1)).first() is not None):
                raise HandoffConflict("Active video work must finish before offering a switch")
            # Replacing an offer invalidates its old token and bumps the generation.
            conn.execute(_state.update().where(_state.c.id == 1).values(
                phase="offered", generation=row["generation"] + 1,
                offer_hash=hashlib.sha256(token.encode()).hexdigest(), offer_user=user,
                offer_target=target, offer_expires=time.time() + ttl,
                actions=json.dumps(actions), next_action=0, pending_action=None,
                last_job_id=None, last_error=None,
            ))
            if plan is not None:
                conn.execute(_plans.insert().values(
                    generation=row["generation"] + 1,
                    plan_json=json.dumps(plan, allow_nan=False),
                ))
            result = self._locked(conn)
        return token, self._view(result)

    def plan(self, offered_generation: int) -> dict:
        with self.engine.connect() as conn:
            row = conn.execute(select(_plans).where(
                _plans.c.generation == offered_generation,
            )).mappings().one_or_none()
            if row is None:
                raise HandoffConflict("Transition plan is unavailable; request a fresh offer")
            return {**json.loads(row["plan_json"]), "voice_lease": row["voice_lease"]}

    def record_voice_lease(self, *, generation: int, lease_id: str) -> None:
        if len(lease_id) != 32 or any(c not in "0123456789abcdef" for c in lease_id):
            raise ValueError("Invalid voice park lease")
        with self.engine.begin() as conn:
            row = self._locked(conn)
            if (row["generation"] != generation or row["phase"] != "transitioning"
                    or row["pending_action"] != "park_voice"):
                raise HandoffConflict("Voice park no longer belongs to this transition")
            changed = conn.execute(_plans.update().where(
                _plans.c.generation == generation - 1, _plans.c.voice_lease.is_(None),
            ).values(voice_lease=lease_id))
            if changed.rowcount != 1:
                raise HandoffConflict("Voice lease already recorded or plan missing")

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

    def resume_video(self, *, expected_generation: int) -> HandoffState:
        """Leave a fence back to video. Caller must first verify, live, that voice
        is still parked under the recorded lease and Moshi is stopped."""
        with self.engine.begin() as conn:
            row = self._locked(conn)
            if row["phase"] != "recovery_required" or row["generation"] != expected_generation:
                raise HandoffConflict("GPU state changed; refresh and try again")
            if conn.execute(select(_video_jobs.c.id).limit(1)).first() is not None:
                raise HandoffConflict("A render's outcome is unknown; restore voice to reset the video worker")
            conn.execute(_state.update().where(_state.c.id == 1).values(
                owner="video", phase="idle", generation=row["generation"] + 1, last_error=None,
                offer_hash=None, offer_user=None, offer_target=None, offer_expires=None,
                actions=None, next_action=0, pending_action=None, last_job_id=None,
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
