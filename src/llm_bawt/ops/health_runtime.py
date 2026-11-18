"""Fixed-worker lifecycle: health HTTP survives blocked reconciliation/collectors."""
from __future__ import annotations

import logging
import math
import os
from contextlib import asynccontextmanager
from threading import Event, Thread
import time

from .health_collector import OpsHealthCollector
from .health_state import HealthSample, OpsHealthState

logger = logging.getLogger(__name__)


def database_engine():
    from ..utils.config import Config
    from ..utils.db import get_shared_engine
    return get_shared_engine(Config())  # reuse process-wide pool; no schema bootstrap


def build_full_service():
    """The ops service's full OpsService: executors, the release coordinator,
    the quiesce driver and its own approval store. Returns (service, approvals)."""
    from ..approval_policy_store import ToolApprovalPolicyStore
    from ..utils.config import Config
    from .release_executor import build_ops_service
    from .store import OpsStore

    config = Config()
    store = OpsStore(config)
    if store.engine is None:
        raise RuntimeError("ops reconciler requires a configured canonical database")
    # This process's own approval store: building it never imports the app's
    # FastAPI dependency wiring (TASK-1220).
    approvals = ToolApprovalPolicyStore(config, engine=store.engine)
    service = build_ops_service(config, store, approvals=lambda: approvals)
    # Quiesce jobs are driven only here, never by the app (TASK-1233).
    from .quiesce import build_quiesce_driver
    service.quiesce_driver = build_quiesce_driver(store.engine)
    return service, approvals


def _attach_gate_driver(service):
    from .gate_driver import build_gate_driver
    service.gate_driver = build_gate_driver(service.store)
    return service


def build_control_service():
    return build_full_service()[0]


def build_reconciler(interval):
    from .approved_admission import ApprovedOpsAdmission
    from .reconciler import JobCreatedWaiter, OpsReconciler

    service, approvals = build_full_service()
    _attach_gate_driver(service)  # Only the reconciler loop drives batch gates.
    engine = service.store.engine
    # The app's job creation notifies; wake on it rather than the interval.
    waiter = JobCreatedWaiter(engine) if engine.dialect.name == "postgresql" else None
    return OpsReconciler(service, interval=interval, waiter=waiter,
                         admission=ApprovedOpsAdmission(approvals, service))


WEDGE_EXIT_CODE = 70


def _wedge_seconds() -> float:
    """``LLM_BAWT_OPS_WEDGE_SECONDS`` (default 180; 0 disables the watchdog)."""
    raw = os.getenv("LLM_BAWT_OPS_WEDGE_SECONDS", "180")
    value = float(raw)
    if not math.isfinite(value) or (0 < value < 60) or value < 0:
        raise ValueError("LLM_BAWT_OPS_WEDGE_SECONDS must be 0 or at least 60")
    return value


class _IntervalWaiter:
    """No LISTEN available (no database / not PostgreSQL): plain interval."""

    def wait(self, stop, timeout):
        stop.wait(timeout)
        return False

    def close(self):
        pass


class OpsHealthRuntime:
    def __init__(self, *, interval=5.0, state=None, collector=None,
                 reconciler_factory=build_reconciler,
                 control_address=None, control_factory=build_control_service,
                 wedge_seconds=None, exit=None):
        self.state = state or OpsHealthState()
        self.stop = Event()
        self.docker_wake = Event()
        self.collector = collector or OpsHealthCollector(self.state, database_engine)
        self.reconciler_factory = reconciler_factory
        self.interval = interval
        self._started = False
        self._closed = False
        # The app's prepare/readiness channel (TASK-1169 S3a): its own listener,
        # so a slow Docker read there never delays the health surface.
        self.control = None
        if control_address is not None:
            from .control_http import OpsControlServer
            self.control = OpsControlServer(control_factory, control_address)
        self.workers: list[Thread] = []
        self.wedge_seconds = wedge_seconds if wedge_seconds is not None else _wedge_seconds()
        self.exit = exit or os._exit

    def _watchdog(self):
        """Exit the process when a producer stays silent past the limit.

        A wedged thread keeps the HTTP health host alive but stops publishing;
        Docker's healthcheck can only mark that unhealthy. Exiting lets the
        container's restart policy recover it, with no extra Docker access
        (TASK-1991). Workers and receipts are durable across the restart."""
        while not self.stop.wait(min(self.interval, self.wedge_seconds)):
            silent = self.state.silent(self.wedge_seconds)
            if silent:
                logger.critical("ops producers silent for over %ss (%s); exiting for restart",
                                self.wedge_seconds, ", ".join(silent))
                self.exit(WEDGE_EXIT_CODE)
                return

    def _reconcile(self):
        while not self.stop.is_set():
            sampled = self.state.sample_time()
            try:
                pump = self.reconciler_factory(self.interval)
                pump.run(self.stop, self.state)
                return
            except Exception:
                self.state.publish("reconciler.loop", HealthSample(
                    sampled, "down", "reconciler initialization unavailable"))
                self.stop.wait(self.interval)

    @asynccontextmanager
    async def lifespan(self, _app):
        # FastAPI owns these existing workers, not per-request BackgroundTasks.
        # Even a partial startup must release listeners and signal producers.
        try:
            self.start()
            yield
        finally:
            self.close()

    def start(self):
        if self._started or self.stop.is_set():
            raise RuntimeError("ops runtime is single-use")
        self._started = True
        # Initialization stays in fixed daemon producers, off the HTTP loop.
        if self.control is not None:
            self.control.start()
        for name, target in (
            # Timer: DB liveness (5s); a dead database emits no event.
            ("ops-health-database", lambda: self.collector.run(self.collector.collect_database, self.stop)),
            # Catalog writes notify; the 60s wait is reconciliation only.
            ("ops-health-catalog", lambda: self.collector.watch_catalog(self.stop, self._catalog_waiter())),
            # Docker events push changes; the full snapshot is reconciliation.
            ("ops-health-docker", lambda: self.collector.run(
                self.collector.collect_docker, self.stop,
                interval=self.collector.docker_interval, wake=self.docker_wake)),
            ("ops-health-docker-events", lambda: self.collector.watch_docker(self.stop, self.docker_wake)),
            ("ops-reconciler", self._reconcile),
        ) + ((("ops-watchdog", self._watchdog),) if self.wedge_seconds > 0 else ()):
            worker = Thread(name=name, target=target, daemon=True)
            worker.start()
            self.workers.append(worker)

    @staticmethod
    def _catalog_waiter():
        from .reconciler import JobCreatedWaiter
        from .store import CATALOG_CHANGED_CHANNEL
        try:
            engine = database_engine()
        except Exception:  # noqa: BLE001 - the reconcile interval still runs
            engine = None
        if engine is None or engine.dialect.name != "postgresql":
            return _IntervalWaiter()
        return JobCreatedWaiter(engine, channel=CATALOG_CHANGED_CHANNEL)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.state.stop()
        self.stop.set()
        self.docker_wake.set()
        if self.control is not None:
            self.control.close()
        # A stuck RPC cannot hold SIGTERM indefinitely. One fixed daemon per
        # producer; never spawn replacement threads for a blocked collector.
        deadline = time.monotonic() + 2
        for worker in self.workers:
            worker.join(timeout=max(0, deadline - time.monotonic()))
