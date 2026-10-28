"""Immutable, bounded health snapshots. Readers never touch Docker or the DB."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from threading import Lock
import time

STALE_SECONDS = 30.0
FUTURE_TOLERANCE_SECONDS = 5.0
MAX_CONTAINERS = 1000
# TASK-1169 S2a: bounds on the /v1/health/services service set. The catalog
# and the inventory are each bounded at collection; the union is bounded
# again at read time so the consumers' 512 KiB body limit always holds. The
# envelope around the services list is under 1 KiB.
MAX_SERVICES = 256
MAX_SERVICES_BODY_BYTES = 512 * 1024 - 1024


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
    # TASK-1169 S1c: typed Compose identity (None when the container has no
    # valid project/service labels) and an opaque instance id derived from
    # the Docker Id. The raw Id and the labels dict are never kept.
    project: str | None = None
    service: str | None = None
    instance_id: str | None = None
    # TASK-1171: Docker's RestartCount. An API start/restart resets it to 0
    # and only a restart-policy restart increments it, so consumers can tell
    # an operator restart from a crash recovery. None when not reported.
    restart_count: int | None = None

    def to_api(self):
        """The /v1/health/containers row; its shape is frozen for consumers."""
        return {"name": self.name, "state": self.state, "health": self.health,
                "startedAt": self.started_at}

    def instance_api(self):
        return {"instanceId": self.instance_id, **self.to_api(), "restartCount": self.restart_count}


@dataclass(frozen=True)
class HealthSample:
    sampled: SampleTime
    status: str
    reason: str | None = None
    containers: tuple[ContainerObservation, ...] = ()


@dataclass(frozen=True)
class ServiceCapability:
    """One enabled catalog operation that targets a service. ``args`` is the
    selector argument (name, value) for an enum-selected op, else empty."""

    operation: str
    action: str
    args: tuple[tuple[str, str], ...] = ()

    def to_api(self):
        return {"operation": self.operation, "action": self.action, "args": dict(self.args)}


@dataclass(frozen=True)
class ServiceTarget:
    project: str
    service: str
    capabilities: tuple[ServiceCapability, ...]

    @property
    def service_id(self):
        return f"{self.project}.{self.service}"

    def to_api(self, instances, *, capabilities_known=True):
        """One /v1/health/services entry. ``instances`` is a list or None;
        ``capabilities`` is None when the catalog snapshot is not fresh."""
        return {"id": self.service_id, "project": self.project, "service": self.service,
                "instances": instances,
                "capabilities": [c.to_api() for c in self.capabilities] if capabilities_known else None}


@dataclass(frozen=True)
class CatalogSample:
    """Services that enabled catalog ops target, read by the database worker."""

    sampled: SampleTime
    status: str
    reason: str | None = None
    services: tuple[ServiceTarget, ...] = ()


class OpsHealthState:
    """Locks protect pointer swaps only, never probing or serialization."""

    def __init__(self, *, clock=time.monotonic, utcnow=lambda: datetime.now(UTC)):
        self.clock, self.utcnow = clock, utcnow
        self._lock = Lock()
        initial = HealthSample(self.sample_time(), "down", "not yet collected")
        self._samples = dict.fromkeys(("database", "docker", "ops.executor", "reconciler.loop"), initial)
        # Not a readiness check: /health/ready keeps its four checks.
        self._catalog = CatalogSample(initial.sampled, "down", "not yet collected")
        self._stopping = False

    def sample_time(self):
        return SampleTime(self.utcnow(), self.clock())

    def publish(self, name: str, sample: HealthSample):
        with self._lock:
            if name not in self._samples:
                raise ValueError("unknown health capability")
            if not self._stopping:
                self._samples[name] = sample

    def publish_catalog(self, sample: CatalogSample):
        with self._lock:
            if not self._stopping:
                self._catalog = sample

    def stop(self):
        with self._lock:
            self._stopping = True

    def _read(self):
        with self._lock:
            return self._samples.copy(), self._stopping

    def _read_services(self):
        with self._lock:
            return self._samples["docker"], self._catalog, self._stopping

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

    def services(self, service_id: str | None = None):
        """TASK-1169 S1c/S2a: ``/v1/health/services[/{id}]``.

        The service set is every catalog target together with every Compose
        service the inventory observes (S2a); a service no enabled op targets
        has ``capabilities: []``. 200 needs a fresh inventory and a fresh
        catalog; otherwise 503 with ``inventoryFresh``/``catalogFresh``
        saying which half is usable:

        * stale catalog: the observed services are listed with real
          ``instances`` and ``capabilities: null`` (actions unknown);
        * stale inventory: the catalogued services are listed with
          ``instances: null`` (``[]`` stays "observed, none present");
        * both stale: ``services`` is null.

        Containers without valid Compose labels are counted in
        ``unlabeledCount``, never listed. A union past ``MAX_SERVICES`` or the
        body budget fails the inventory half closed rather than truncating.
        """
        inventory, catalog, stopping = self._read_services()
        now = self.sample_time()

        def unavailable(sample):
            return self._unavailable_reason(sample, now, stopping) or (
                None if sample.status == "ok" else sample.reason)

        inventory_reason, catalog_reason = unavailable(inventory), unavailable(catalog)
        catalog_ok = catalog_reason is None
        instances: dict[tuple[str, str], list[ContainerObservation]] = {}
        unlabeled = 0
        for container in inventory.containers if inventory_reason is None else ():
            if container.project is None or container.service is None:
                unlabeled += 1
            else:
                instances.setdefault((container.project, container.service), []).append(container)

        def entries(inventory_ok):
            targets = {(t.project, t.service): t for t in catalog.services} if catalog_ok else {}
            if inventory_ok:
                for key in instances:
                    targets.setdefault(key, ServiceTarget(*key, ()))
            return [
                t.to_api([c.instance_api() for c in instances.get(key, [])] if inventory_ok else None,
                         capabilities_known=catalog_ok)
                for key, t in sorted(targets.items(), key=lambda item: item[1].service_id)]

        rendered = None
        if inventory_reason is None:
            rendered = entries(True)
            if (len(rendered) > MAX_SERVICES or len(json.dumps(
                    rendered, separators=(",", ":")).encode()) > MAX_SERVICES_BODY_BYTES):
                inventory_reason, rendered = "service inventory exceeds bounded snapshot", None
        inventory_ok = inventory_reason is None
        if rendered is None:
            # Catalog-only (or nothing): bounded by the catalog's own budget.
            rendered = entries(False) if catalog_ok else None

        fresh = inventory_ok and catalog_ok
        body = {
            "schemaVersion": 1,
            "observedAt": min(inventory.sampled.wall, catalog.sampled.wall).isoformat(),
            "fresh": fresh,
            "inventoryFresh": inventory_ok,
            "catalogFresh": catalog_ok,
            "reason": inventory_reason or catalog_reason,
            "unlabeledCount": unlabeled if inventory_ok else None,
        }
        code = 200 if fresh else 503
        if service_id is None:
            body["services"] = rendered
            return code, body
        found = next((e for e in rendered or () if e["id"] == service_id), None)
        if found is None and fresh:
            return 404, {"error": "not found"}
        # Not fresh: an id missing from the usable half is unknown, not absent.
        body["service"] = found
        return code, body
