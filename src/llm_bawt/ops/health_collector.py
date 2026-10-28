"""Read-only collectors with fixed inventory projection and bounded RPCs.

Each collector has one worker (no per-request or retry-thread fan-out). If a
library call defeats its timeout, only that worker stalls; cached data expires
and HTTP remains available. No health path calls preflight or reconciliation.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime

from sqlalchemy import text

from .executor import DockerExecutor
from .health_state import (
    MAX_CONTAINERS,
    CatalogSample,
    ContainerObservation,
    HealthSample,
    ServiceCapability,
    ServiceTarget,
)
from .service_identity import PROJECT_LABEL, SERVICE_LABEL, ComposeServiceRef

COLLECTION_BUDGET_SECONDS = 15.0
DOCKER_TIMEOUT_SECONDS = 2
CONTAINER_STATES = frozenset({"created", "running", "paused", "restarting", "removing", "exited", "dead"})
CONTAINER_HEALTH = frozenset({"healthy", "unhealthy", "starting"})
# Catalog bounds (S1c): an oversized catalog fails the whole sample closed
# rather than publishing a truncated service set. 512 capabilities keep the
# largest /v1/health/services body under the consumers' 512 KiB limit next
# to a full 1000-container inventory.
MAX_OPERATIONS = 500
MAX_SERVICES = 256
MAX_CAPABILITIES = 512
MAX_ENUM_VALUES = 64
# Serialized catalog budget (services rendered with empty instances). Every
# emitted field is already length-bounded, but 512 maximum-width capabilities
# plus a maximum inventory would still exceed 512 KiB, so the rendered catalog
# itself is bounded: 120 KiB + a 1000-row maximum-width inventory (~388 KB)
# + the envelope stays under the consumers' limit.
MAX_CATALOG_BYTES = 120 * 1024
OPERATION_SLUG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
ACTION_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")
ARG_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
CATALOG_QUERY = text(
    "SELECT slug, command_script, args_schema_json FROM ops_operations "
    "WHERE enabled IS TRUE AND soft_deleted_at IS NULL ORDER BY slug LIMIT :limit")


def _compose_identity(attrs):
    """(project, service) from the Compose labels, or (None, None). Only the
    two label values are read; the labels dict itself is never kept."""
    labels = (attrs.get("Config") or {}).get("Labels") or {}
    if not isinstance(labels, dict):
        return None, None
    try:
        ref = ComposeServiceRef(labels.get(PROJECT_LABEL), labels.get(SERVICE_LABEL))
    except ValueError:
        return None, None
    return ref.project, ref.service


def instance_id(container_id):
    """Opaque, stable per container: sha256(Docker Id)[:16]. A recreate gets a
    new Id, so consumers see a replacement as an instance_id change."""
    if not isinstance(container_id, str) or not 0 < len(container_id) <= 128:
        raise ValueError("invalid container id")
    return hashlib.sha256(container_id.encode()).hexdigest()[:16]


def observe_container(attrs):
    """Allowlist projection: no raw IDs, label dicts, logs, env, images,
    ports or mounts. Compose project/service are typed values."""
    name = attrs["Name"].removeprefix("/")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,254}", name):
        raise ValueError("invalid container name")
    state = attrs["State"]
    status = state["Status"]
    if status not in CONTAINER_STATES:
        raise ValueError("unknown container state")
    health = state.get("Health", {}).get("Status")
    if health is not None and health not in CONTAINER_HEALTH:
        raise ValueError("unknown container health")
    started = state.get("StartedAt")
    if started is not None:
        if not isinstance(started, str) or len(started) > 64:
            raise ValueError("invalid container start time")
        parsed = datetime.fromisoformat(started.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("container start time lacks timezone")
        started = None if parsed.year == 1 else parsed.isoformat()
    project, service = _compose_identity(attrs)
    return ContainerObservation(name, status, health, started, project, service,
                                instance_id(attrs.get("Id")))


def _operation_capabilities(slug, command_script, args_schema_json):
    """(ComposeServiceRef, ServiceCapability) pairs one enabled op contributes.

    A fixed ``compose_service`` gives one pair. ``compose_service_from_arg``
    gives one pair per value of that argument's bounded ``enum``; without a
    bounded enum it contributes nothing (never "all services"). Any malformed
    op contributes nothing and does not affect the others.
    """
    if not isinstance(slug, str) or not OPERATION_SLUG.fullmatch(slug):
        return []
    try:
        spec = json.loads(command_script)
        schema = json.loads(args_schema_json) if isinstance(args_schema_json, str) else args_schema_json
    except (TypeError, ValueError):
        return []
    if not isinstance(spec, dict):
        return []
    action = spec.get("action")
    if not isinstance(action, str) or not ACTION_NAME.fullmatch(action):
        return []
    if "compose_service" in spec and "compose_service_from_arg" not in spec:
        selectors = [(spec["compose_service"], ())]
    elif "compose_service_from_arg" in spec and "compose_service" not in spec:
        arg = spec["compose_service_from_arg"]
        # Type-check every level before any lookup: an unhashable arg or a
        # non-mapping schema/properties must only drop this op.
        if not isinstance(arg, str) or not ARG_NAME.fullmatch(arg) or not isinstance(schema, dict):
            return []
        props = schema.get("properties")
        prop = props.get(arg) if isinstance(props, dict) else None
        enum = prop.get("enum") if isinstance(prop, dict) else None
        if (not isinstance(enum, list)
                or not 0 < len(enum) <= MAX_ENUM_VALUES
                or not all(isinstance(v, str) for v in enum)):
            return []
        selectors = [(value, ((arg, value),)) for value in dict.fromkeys(enum)]
    else:
        return []
    pairs = []
    for service, args in selectors:
        try:
            ref = ComposeServiceRef(spec.get("compose_project"), service)
        except ValueError:
            continue
        pairs.append((ref, ServiceCapability(slug, action, args)))
    return pairs


def catalog_targets(rows):
    """Service targets from enabled-op rows ``(slug, command_script,
    args_schema_json)``, sorted by service id. Raises past any bound."""
    if len(rows) > MAX_OPERATIONS:
        raise ValueError("catalog exceeds bounded snapshot")
    by_service: dict[ComposeServiceRef, list[ServiceCapability]] = {}
    total = 0
    for slug, command_script, args_schema_json in rows:
        try:
            pairs = _operation_capabilities(slug, command_script, args_schema_json)
        except Exception:
            # Defence in depth: an unforeseen malformed shape (e.g. JSON
            # nested past the recursion limit) drops only this op.
            pairs = []
        for ref, capability in pairs:
            by_service.setdefault(ref, []).append(capability)
            total += 1
    if total > MAX_CAPABILITIES or len(by_service) > MAX_SERVICES:
        raise ValueError("catalog exceeds bounded snapshot")
    targets = tuple(
        ServiceTarget(ref.project, ref.service,
                      tuple(sorted(caps, key=lambda c: (c.operation, c.args))))
        for ref, caps in sorted(by_service.items(), key=lambda item: item[0].service_id))
    rendered = json.dumps([t.to_api([]) for t in targets], separators=(",", ":")).encode()
    if len(rendered) > MAX_CATALOG_BYTES:
        raise ValueError("catalog exceeds bounded snapshot")
    return targets


def docker_client():
    import docker
    return docker.from_env(timeout=DOCKER_TIMEOUT_SECONDS)


class OpsHealthCollector:
    def __init__(self, state, engine_factory, *, client_factory=docker_client):
        self.state = state
        self.engine_factory = engine_factory
        self.client_factory = client_factory

    def collect_database(self):
        sampled = self.state.sample_time()
        try:
            engine = self.engine_factory()
            if engine is None:
                raise RuntimeError("database not configured")
            with engine.connect() as conn:
                if engine.dialect.name == "postgresql":
                    conn.execute(text("SET TRANSACTION READ ONLY"))
                    conn.execute(text("SET LOCAL statement_timeout = '2000ms'"))
                    conn.execute(text("SET LOCAL lock_timeout = '2000ms'"))
                if conn.execute(text("SELECT 1")).scalar_one() != 1:
                    raise RuntimeError("database probe failed")
            sample = HealthSample(sampled, "ok")
        except Exception:
            # Never leak DSNs, passwords, queries or connection exception text.
            sample = HealthSample(sampled, "down", "database probe unavailable")
        self.state.publish("database", sample)

    def collect_catalog(self):
        """Read-only snapshot of the services enabled catalog ops target."""
        sampled = self.state.sample_time()
        try:
            engine = self.engine_factory()
            if engine is None:
                raise RuntimeError("database not configured")
            with engine.connect() as conn:
                if engine.dialect.name == "postgresql":
                    conn.execute(text("SET TRANSACTION READ ONLY"))
                    conn.execute(text("SET LOCAL statement_timeout = '2000ms'"))
                    conn.execute(text("SET LOCAL lock_timeout = '2000ms'"))
                rows = [tuple(row) for row in
                        conn.execute(CATALOG_QUERY, {"limit": MAX_OPERATIONS + 1}).all()]
            self._budget(sampled)
            sample = CatalogSample(sampled, "ok", services=catalog_targets(rows))
        except Exception:
            # Same sanitization as the database probe: no DSN or query text.
            sample = CatalogSample(sampled, "down", "operation catalog unavailable")
        self.state.publish_catalog(sample)

    def collect_storage(self):
        """The database worker's tick: liveness probe, then catalog snapshot."""
        self.collect_database()
        self.collect_catalog()

    def collect_docker(self):
        sampled = self.state.sample_time()
        client = None
        try:
            client = self.client_factory()
            rows = client.api.containers(all=True, limit=MAX_CONTAINERS + 1)
            if len(rows) > MAX_CONTAINERS:
                raise ValueError("inventory exceeds bounded snapshot")
            observations = []
            for row in rows:
                self._budget(sampled)
                observations.append(observe_container(client.api.inspect_container(row["Id"])))
            self._budget(sampled)
            if len({c.name for c in observations}) != len(observations):
                raise ValueError("ambiguous inventory")
            sample = HealthSample(sampled, "ok", containers=tuple(sorted(observations, key=lambda c: c.name)))
        except Exception:
            sample = HealthSample(sampled, "down", "container observation unavailable")
        self.state.publish("docker", sample)
        # Inventory stays independently usable if prerequisites fail/block.
        prerequisite_start = self.state.sample_time()
        try:
            if client is None:
                raise RuntimeError("Docker unavailable")
            executor = DockerExecutor(client_factory=lambda: client)
            reason = executor.readiness()  # read-only shared prerequisite logic, NOT preflight()
            self._budget(prerequisite_start)
            ready = HealthSample(prerequisite_start, "down", "Docker executor prerequisites unavailable") if reason else HealthSample(prerequisite_start, "ok")
        except Exception:
            ready = HealthSample(prerequisite_start, "down", "Docker executor prerequisites unavailable")
        self.state.publish("ops.executor", ready)
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    def _budget(self, sampled):
        if self.state.clock() - sampled.monotonic > COLLECTION_BUDGET_SECONDS:
            raise TimeoutError("health collection budget exceeded")

    @staticmethod
    def run(collect, stop, *, interval=5.0):
        while not stop.is_set():
            collect()
            stop.wait(interval)
