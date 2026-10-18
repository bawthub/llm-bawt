"""Immutable, bounded health snapshots. Readers never touch Docker or the DB."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from time import monotonic

STALE_SECONDS = 30.0
FUTURE_TOLERANCE_SECONDS = 5.0
MAX_CONTAINERS = 1000


@dataclass(frozen=True)
class SampleTime:
    wall: datetime
    monotonic: float

    @property
    def iso(self):
        return self.wall.isoformat()


@dataclass(frozen=True)
class ContainerObservation:
    name: str
    state: str
    health: str | None
    started_at: str | None

    def to_api(self):
        return {"name": self.name, "state": self.state, "health": self.health,
                "startedAt": self.started_at}


@dataclass(frozen=True)
class HealthSample:
    sampled: SampleTime
    status: str
    reason: str | None = None
    containers: tuple[ContainerObservation, ...] = ()


class OpsHealthState:
    """Locks protect pointer swaps only, never probing or serialization."""

    def __init__(self, *, clock=monotonic, utcnow=lambda: datetime.now(UTC)):
        self.clock, self.utcnow = clock, utcnow
        self._lock = Lock()
        initial = HealthSample(self.sample_time(), "down", "not yet collected")
        self._samples = dict.fromkeys(("database", "docker", "ops.executor", "reconciler.loop"), initial)
        self._stopping = False

    def sample_time(self):
        return SampleTime(self.utcnow(), self.clock())

    def publish(self, name: str, sample: HealthSample):
        with self._lock:
            if name not in self._samples:
                raise ValueError("unknown health capability")
            if not self._stopping:
                self._samples[name] = sample

    def stop(self):
        with self._lock:
            self._stopping = True

    def _read(self):
        with self._lock:
            return self._samples.copy(), self._stopping

    def _unavailable_reason(self, sample, now, stopping):
        if stopping:
            return "ops reconciler is shutting down"
        age = now.monotonic - sample.sampled.monotonic
        wall_age = (now.wall - sample.sampled.wall).total_seconds()
        if age > STALE_SECONDS or wall_age > STALE_SECONDS:
            return "observation stale (older than 30 seconds)"
        if wall_age < -FUTURE_TOLERANCE_SECONDS:
            return "observation timestamp is in the future"
        return None

    def ready(self):
        samples, stopping = self._read()
        now = self.sample_time()
        checks = []
        for name, sample in samples.items():
            unavailable = self._unavailable_reason(sample, now, stopping)
            status = "down" if unavailable else sample.status
            checks.append({"name": name, "status": status, "reason": unavailable or sample.reason})
        status = max((c["status"] for c in checks), key={"ok": 0, "degraded": 1, "down": 2}.__getitem__)
        return (200 if status == "ok" else 503), {
            "status": status, "checkedAt": min(s.sampled.wall for s in samples.values()).isoformat(),
            "checks": checks,
        }

    def containers(self):
        samples, stopping = self._read()
        sample = samples["docker"]
        reason = self._unavailable_reason(sample, self.sample_time(), stopping) or sample.reason
        ok = sample.status == "ok" and reason is None
        return (200 if ok else 503), {
            "schemaVersion": 1, "checkedAt": sample.sampled.iso,
            "status": "ok" if ok else "down", "reason": reason,
            "containers": [c.to_api() for c in sample.containers] if ok else [],
        }
