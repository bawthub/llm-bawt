"""Read-only collectors with fixed inventory projection and bounded RPCs.

Each collector has one worker (no per-request or retry-thread fan-out). If a
library call defeats its timeout, only that worker stalls; cached data expires
and HTTP remains available. No health path calls preflight or reconciliation.
"""
from __future__ import annotations

import re
from datetime import datetime

from sqlalchemy import text

from .executor import DockerExecutor
from .health_state import MAX_CONTAINERS, ContainerObservation, HealthSample

COLLECTION_BUDGET_SECONDS = 15.0
DOCKER_TIMEOUT_SECONDS = 2
CONTAINER_STATES = frozenset({"created", "running", "paused", "restarting", "removing", "exited", "dead"})
CONTAINER_HEALTH = frozenset({"healthy", "unhealthy", "starting"})


def observe_container(attrs):
    """Allowlist projection: no IDs, labels, logs, env, images, ports or mounts."""
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
    return ContainerObservation(name, status, health, started)


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
