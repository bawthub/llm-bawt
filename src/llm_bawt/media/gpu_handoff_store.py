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

from sqlalchemy import Column, Float, Integer, MetaData, String, Table, Text, inspect, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine

from llm_bawt.utils.schema import SchemaBootstrapGuard

_metadata = MetaData()
# These actions require a durable ops receipt before the cursor can advance.
OPS_ACTIONS = frozenset({
    "stop_moshi_stt", "stop_moshi_tts", "start_moshi_stt", "start_moshi_tts",
})
STOP_OPERATIONS = ("bawthub.stop-moshi-stt", "bawthub.stop-moshi-tts")
START_OPERATIONS = ("bawthub.start-moshi-stt", "bawthub.start-moshi-tts")
#: The one ops operation that restarts TTS while voice owns the GPU (TASK-1171).
TTS_RESTART_OPERATION = "bawthub.restart-tts"
TTS_RESTART_KEY_PREFIX = "tts-restart:"
_START_ACTIONS = dict(zip(START_OPERATIONS, ("start_moshi_stt", "start_moshi_tts"), strict=True))
_video_jobs = Table(
    "gpu_handoff_video_jobs", _metadata,
    Column("id", String(32), primary_key=True),
    Column("generation", Integer, nullable=False),
)
_resolved_jobs = Table(
    "gpu_handoff_resolved_video_jobs", _metadata,
    Column("id", String(32), primary_key=True),
    Column("claim_generation", Integer, nullable=False),
    Column("owner_generation", Integer, nullable=False),
    Column("reviewed_by", String(128), nullable=False),
    Column("outcome", String(32), nullable=False),
    Column("evidence", Text, nullable=False),
    Column("observations_json", Text, nullable=False),
    Column("resolved_at", Float, nullable=False),
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
    # The ops idempotency key of the one TTS restart this fence admits.
    Column("tts_restart_key", String(160)),
)


class HandoffConflict(ValueError):
    """An offer, generation, or transition is no longer current."""

    def __init__(self, message: str, *, code: str = "handoff_conflict"):
        self.code = code
        super().__init__(message)


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
    tts_restart_key: str | None = None


def _bootstrap_schema(conn) -> None:
    _metadata.create_all(conn)
    if conn.dialect.name == "postgresql":
        # v3 created offer_target as varchar(16); "video_calibration" is 17 chars.
        # Widening is idempotent and create_all never alters existing columns.
        conn.execute(text("ALTER TABLE gpu_handoff_state ALTER COLUMN offer_target TYPE varchar(32)"))
        # v6 (TASK-1171): the TTS restart fence. create_all never adds columns.
        conn.execute(text("ALTER TABLE gpu_handoff_state ADD COLUMN IF NOT EXISTS tts_restart_key varchar(160)"))
    elif "tts_restart_key" not in {c["name"] for c in inspect(conn).get_columns("gpu_handoff_state")}:
        conn.execute(text("ALTER TABLE gpu_handoff_state ADD COLUMN tts_restart_key varchar(160)"))


def _lock_state(conn):
    """The single state row, write-locked for the rest of the transaction."""
    stmt = select(_state).where(_state.c.id == 1)
    if conn.dialect.name == "postgresql":
        stmt = stmt.with_for_update()
    elif conn.dialect.name == "sqlite":
        # SQLite has no FOR UPDATE; obtain its writer lock before reading
        # so concurrent confirmations cannot both consume the same offer.
        conn.execute(_state.update().where(_state.c.id == 1).values(id=1))
    return conn.execute(stmt).mappings().one()


def _start_admitted(row, key: str, operation_slug: str) -> bool:
    """A restore's own Moshi start: ``gpu-handoff:{generation}:{slug}``, while
    that restore is transitioning with exactly that start pending."""
    action = _START_ACTIONS.get(operation_slug)
    return (action is not None and key == f"gpu-handoff:{row['generation']}:{operation_slug}"
            and row["phase"] == "transitioning" and row["offer_target"] == "voice"
            and row["pending_action"] == action)


def _restart_admitted(row, key: str, operation_slug: str) -> bool:
    return (operation_slug == TTS_RESTART_OPERATION and row["phase"] == "tts_restarting"
            and row["tts_restart_key"] == key)


class GpuHandoffFence:
    """Creation fence for an ops job that starts or restarts a GPU voice service.

    ``OpsStore.create_job`` runs it inside the job-insert transaction, so the
    handoff row and the job commit together or not at all. Only the handoff
    controller builds one; the ops guard refuses a gated job without it.
    """

    def __init__(self, kind: str, *, key: str, operation_slug: str):
        if kind not in ("tts_restart", "voice_start"):
            raise ValueError(f"unknown GPU handoff fence {kind!r}")
        self.kind, self.key, self.operation_slug = kind, key, operation_slug

    @classmethod
    def tts_restart(cls, key: str) -> GpuHandoffFence:
        return cls("tts_restart", key=key, operation_slug=TTS_RESTART_OPERATION)

    @classmethod
    def voice_start(cls, *, generation: int, operation_slug: str) -> GpuHandoffFence:
        return cls("voice_start", key=f"gpu-handoff:{generation}:{operation_slug}",
                   operation_slug=operation_slug)

    def __call__(self, session) -> None:
        conn = session.connection()
        row = _lock_state(conn)
        if self.kind == "voice_start":
            # Check-only: the restore already recorded this start as pending.
            if not _start_admitted(row, self.key, self.operation_slug):
                raise HandoffConflict("Moshi start is no longer this restore's pending action",
                                      code="gpu_handoff_stale")
            return
        if row["phase"] == "tts_restarting":
            if row["tts_restart_key"] == self.key:
                return  # Same request: the unique key replays the existing job.
            raise HandoffConflict("A TTS restart is already in progress", code="tts_restart_in_progress")
        if row["owner"] != "voice" or row["phase"] != "idle":
            raise HandoffConflict("TTS restarts only while voice owns an idle GPU", code="gpu_not_voice")
        if conn.execute(select(_video_jobs.c.id).limit(1)).first() is not None:
            raise HandoffConflict("Video render claims are still open", code="active_video_claims")
        conn.execute(_state.update().where(_state.c.id == 1).values(
            phase="tts_restarting", generation=row["generation"] + 1, tts_restart_key=self.key))


class GpuHandoffStore:
    """Serialize transitions on one database row; fail closed on ambiguity."""

    _guard = SchemaBootstrapGuard()

    def __init__(self, engine: Engine):
        self.engine = engine
        self._guard.run(engine, "gpu-handoff-v6", _bootstrap_schema)
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
            last_error=row["last_error"], tts_restart_key=row["tts_restart_key"],
        )

    def _locked(self, conn):
        return _lock_state(conn)

    def admits(self, key: str, operation_slug: str) -> bool:
        """Dispatch-time re-check: the handoff row still names this gated job."""
        with self.engine.begin() as conn:
            row = _lock_state(conn)
            return _start_admitted(row, key, operation_slug) or _restart_admitted(row, key, operation_slug)

    def release_tts_restart(self, key: str) -> HandoffState:
        """CAS the TTS restart fence back to idle voice. The caller proves the
        job's outcome first; this only checks the fence is still that job's."""
        with self.engine.begin() as conn:
            row = _lock_state(conn)
            if row["phase"] != "tts_restarting" or row["tts_restart_key"] != key:
                raise HandoffConflict("TTS restart fence is no longer this request's", code="fence_changed")
            conn.execute(_state.update().where(_state.c.id == 1).values(
                phase="idle", generation=row["generation"] + 1, tts_restart_key=None))
            return self._view(_lock_state(conn))

    def status(self) -> HandoffState:
        with self.engine.connect() as conn:
            return self._view(conn.execute(select(_state).where(_state.c.id == 1)).mappings().one())

    def active_video_jobs(self) -> int:
        """Count durable render claims, including orphaned or uncertain jobs."""
        with self.engine.connect() as conn:
            return len(conn.execute(select(_video_jobs.c.id)).all())

    def video_claims(self) -> list[str]:
        """Expose identities for operator reconciliation; a reset never hides them."""
        with self.engine.connect() as conn:
            return list(conn.execute(select(_video_jobs.c.id).order_by(_video_jobs.c.id)).scalars())

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
            if (lost_owner or (has_claims and row["owner"] in ("video", "video_calibration"))) and row["phase"] != "recovery_required":
                reason = ("Video worker restarted with unresolved render claims; verify GPU and jobs"
                          if has_claims else "Video worker restarted; verify model residency and GPU owner")
                conn.execute(_state.update().where(_state.c.id == 1).values(
                    owner="unknown", phase="recovery_required",
                    generation=row["generation"] + 1, last_error=reason,
                ))
                row = self._locked(conn)
            return self._view(row)

    def validate_video_claim(self, job_id: str) -> None:
        """Admit only an already-claimed render while a voice drain is pending."""
        with self.engine.begin() as conn:
            row = self._locked(conn)
            claim = conn.execute(select(_video_jobs).where(_video_jobs.c.id == job_id)).mappings().one_or_none()
            pending_drain = (row["phase"] == "recovery_required" and row["offer_target"] == "voice"
                             and row["pending_action"] == "await_video_drain")
            if (claim is None or
                    not ((row["owner"] == "video" and row["phase"] in ("idle", "offered", "draining"))
                         or pending_drain)
                    or (row["phase"] == "draining" and row["offer_target"] != "voice")):
                raise HandoffConflict("Video claim is no longer authorized; reconcile the render")

    def claim_video(self, job_id: str) -> int:
        """Authorize a render against the current owner, across processes."""
        if not job_id or len(job_id) > 32:
            raise ValueError("Invalid video job ID")
        with self.engine.begin() as conn:
            row = self._locked(conn)
            if (row["owner"] != "video" or row["phase"] not in ("idle", "offered")
                    or (row["phase"] == "offered" and row["offer_target"] != "voice")):
                raise HandoffConflict("GPU does not belong to video; request a handoff")
            # Claims cover queued requests too: no handoff may begin until the
            # entire queue drains, even when the worker is between renders.
            conn.execute(_video_jobs.insert().values(id=job_id, generation=row["generation"]))
            return row["generation"]

    @staticmethod
    def _calibration_reservation(conn) -> int | None:
        """The forward lease anchors calibration across any number of reverse offers."""
        plan = conn.execute(select(_plans.c.generation).where(_plans.c.voice_lease.is_not(None))
                            .order_by(_plans.c.generation.desc()).limit(1)).first()
        return plan[0] + 2 if plan else None

    def calibration_reservation(self) -> int | None:
        """Return the original active calibration grant, never an offer generation."""
        with self.engine.connect() as conn:
            row = conn.execute(select(_state).where(_state.c.id == 1)).mappings().one()
            if (row["owner"] != "video_calibration" or row["phase"] not in ("idle", "offered")
                    or (row["phase"] == "offered" and row["offer_target"] != "voice")):
                return None
            return self._calibration_reservation(conn)

    def claim_calibration(self, job_id: str, *, generation: int, profile: dict) -> None:
        from .gpu_profile import CALIBRATION_PROFILE, require_calibration_profile

        require_calibration_profile(profile)
        with self.engine.begin() as conn:
            row = self._locked(conn)
            offered = row["phase"] == "offered" and row["offer_target"] == "voice"
            if (row["owner"] != "video_calibration" or
                    row["phase"] not in ("idle", "offered") or
                    (row["phase"] == "offered" and not offered) or
                    self._calibration_reservation(conn) != generation):
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
            calibration_generation = self._calibration_reservation(conn)
            interrupted_drain = (row["phase"] == "recovery_required" and row["offer_target"] == "voice"
                                 and row["pending_action"] == "await_video_drain")
            if (run is None or claim is None
                    or not ((row["owner"] == "video_calibration" and row["phase"] in ("idle", "offered", "draining"))
                            or (row["owner"] == "unknown" and interrupted_drain))
                    or run["generation"] != calibration_generation
                    or claim["generation"] != calibration_generation):
                raise HandoffConflict("Calibration ownership changed; reconcile its result")
            plan_row = conn.execute(select(_plans.c.plan_json).where(_plans.c.generation == run["generation"] - 2)).first()
            original_gpu = json.loads(plan_row[0])["observed"]["gpu"] if plan_row else {}
            if (measurement["gpu_uuid"] != original_gpu.get("uuid")
                    or measurement["device_total_mib"] != original_gpu.get("total_mib")):
                raise HandoffConflict("Calibration ran on a different GPU than the consented reservation")
            conn.execute(_calibrations.update().where(_calibrations.c.job_id == job_id).values(measurement_json=json.dumps(evidence)))
            conn.execute(_video_jobs.delete().where(_video_jobs.c.id == job_id))
            draining = row["phase"] == "draining"
            conn.execute(_state.update().where(_state.c.id == 1).values(
                owner="unknown" if interrupted_drain or not supported else "video",
                phase=("recovery_required" if interrupted_drain or not supported else
                       "draining" if draining else "idle"),
                generation=(row["generation"] if (draining or interrupted_drain) and supported
                            else row["generation"] + 1),
                last_error=("Drain executor interrupted; calibration evidence was saved. "
                            "Inspect GPU and resume video or consent to restore voice"
                            if interrupted_drain and supported else None if supported else
                            "Calibration produced a video but insufficient memory margin; keep ordinary rendering blocked"),
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
        if gpu.get("ready") is not True:
            raise HandoffConflict("GPU telemetry unavailable; cannot verify video memory margin")
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

    def release_video(self, job_id: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(_video_jobs.delete().where(_video_jobs.c.id == job_id))

    def resolve_video_claim(self, *, job_id: str, user: str, expected_generation: int,
                            outcome: str, evidence: str, observations: dict) -> dict:
        """Archive a human-reviewed uncertain outcome, then release its GPU claim atomically.

        Live worker/voice health checks belong to the controller immediately before
        this transaction. The original claim and the review remain traceable by ID.
        """
        if (not job_id or len(job_id) > 32 or not user or len(user) > 128
                or outcome not in ("completed", "failed", "unknown")
                or not isinstance(evidence, str) or not 20 <= len(evidence.strip()) <= 1000):
            raise HandoffConflict("A scoped job, reviewer, outcome and inspection evidence are required")
        with self.engine.begin() as conn:
            row = self._locked(conn)
            if row["owner"] != "voice" or row["phase"] != "idle" or row["generation"] != expected_generation:
                raise HandoffConflict("Voice must own an idle GPU; refresh before resolving this claim")
            claim = conn.execute(select(_video_jobs).where(_video_jobs.c.id == job_id)).mappings().one_or_none()
            if claim is None:
                raise HandoffConflict("Video claim is already resolved or was never accepted")
            record = dict(id=job_id, claim_generation=claim["generation"],
                          owner_generation=row["generation"], reviewed_by=user,
                          outcome=outcome, evidence=evidence.strip(),
                          observations_json=json.dumps(observations, allow_nan=False),
                          resolved_at=time.time())
            conn.execute(_resolved_jobs.insert().values(**record))
            conn.execute(_video_jobs.delete().where(_video_jobs.c.id == job_id))
            return {"job_id": job_id, "outcome": outcome, "reviewed_by": user,
                    "owner_generation": row["generation"], "resolved_at": record["resolved_at"]}

    def resolved_video_claim(self, job_id: str) -> dict | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(_resolved_jobs).where(_resolved_jobs.c.id == job_id)).mappings().one_or_none()
            return {**row, "observations": json.loads(row["observations_json"])} if row else None

    def offer(self, *, user: str, target: str, actions: tuple[str, ...],
              expected_generation: int, ttl: int = 120,
              plan: dict | None = None) -> tuple[str, HandoffState]:
        if not user or target not in ("video", "voice", "video_calibration") or not actions or not 1 <= ttl <= 300:
            raise ValueError("A scoped user, target, ordered actions, and short expiry are required")
        if any(not action or len(action) > 128 for action in actions):
            raise ValueError("Invalid transition action")
        token = secrets.token_urlsafe(32)
        import hashlib
        # A recovery restore may start with uncertain claims, but never erases
        # them: reset fences the worker and the claims stay visible for review.
        restore = target == "voice"
        phases = ("idle", "offered", "recovery_required") if restore else ("idle", "offered")
        with self.engine.begin() as conn:
            row = self._locked(conn)
            if row["generation"] != expected_generation or row["phase"] not in phases:
                raise HandoffConflict("GPU state changed; request a fresh preflight")
            if row["owner"] == target:
                raise HandoffConflict("GPU already belongs to the requested mode")
            if (not restore and conn.execute(select(_video_jobs.c.id).limit(1)).first() is not None):
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
            draining = row["offer_target"] == "voice" and row["owner"] in ("video", "video_calibration")
            conn.execute(_state.update().where(_state.c.id == 1).values(
                phase="draining" if draining else "transitioning", generation=row["generation"] + 1,
                offer_hash=None, offer_user=None, offer_expires=None,
            ))
            return self._view(self._locked(conn))

    def finish_drain(self, *, generation: int) -> HandoffState:
        """Move to reset only after every accepted claim has released."""
        with self.engine.begin() as conn:
            row = self._locked(conn)
            if (row["generation"] != generation or row["phase"] != "draining"
                    or row["owner"] not in ("video", "video_calibration") or row["offer_target"] != "voice"):
                raise HandoffConflict("Video drain is no longer current")
            if conn.execute(select(_video_jobs.c.id).limit(1)).first() is not None:
                raise HandoffConflict("Accepted video renders are still draining")
            conn.execute(_state.update().where(_state.c.id == 1).values(phase="transitioning"))
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
            if row["phase"] not in ("transitioning", "draining") or row["generation"] != generation:
                raise HandoffConflict("Transition has changed")
            conn.execute(_state.update().where(_state.c.id == 1).values(
                owner="unknown", phase="recovery_required", generation=generation + 1,
                pending_action="await_video_drain" if row["phase"] == "draining" else row["pending_action"],
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
            if row["phase"] in ("transitioning", "draining"):
                conn.execute(_state.update().where(_state.c.id == 1).values(
                    owner="unknown", phase="recovery_required", generation=row["generation"] + 1,
                    pending_action="await_video_drain" if row["phase"] == "draining" else row["pending_action"],
                    last_error=("Video drain executor interrupted; inspect accepted renders before restoring voice"
                                if row["phase"] == "draining" else
                                "Transition interrupted; reconcile external operations before retrying"),
                ))
                row = self._locked(conn)
            return self._view(row)
