"""Fixed-worker lifecycle: health HTTP survives blocked reconciliation/collectors."""
from __future__ import annotations

from threading import Event, Thread
from time import monotonic

from .health_collector import OpsHealthCollector
from .health_http import OpsHealthServer
from .health_state import HealthSample, OpsHealthState


def database_engine():
    from ..utils.config import Config
    from ..utils.db import get_shared_engine
    return get_shared_engine(Config())  # reuse process-wide pool; no schema bootstrap


def build_reconciler(interval):
    from ..approval_policy_store import ToolApprovalPolicyStore
    from ..utils.config import Config
    from .approved_admission import ApprovedOpsAdmission
    from .reconciler import OpsReconciler
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
    return OpsReconciler(service, interval=interval,
                         admission=ApprovedOpsAdmission(approvals, service))


class OpsHealthRuntime:
    def __init__(self, *, interval=5.0, state=None, collector=None,
                 reconciler_factory=build_reconciler, address=("0.0.0.0", 8650)):
        self.state = state or OpsHealthState()
        self.stop = Event()
        self.collector = collector or OpsHealthCollector(self.state, database_engine)
        self.reconciler_factory = reconciler_factory
        self.interval = interval
        self.http = OpsHealthServer(self.state, address)
        self.workers: list[Thread] = []

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

    def start(self):
        # No database/Docker initialization on the HTTP thread or before listen.
        self.http.start()
        for name, target in (
            ("ops-health-database", lambda: self.collector.run(self.collector.collect_database, self.stop)),
            ("ops-health-docker", lambda: self.collector.run(self.collector.collect_docker, self.stop)),
            ("ops-reconciler", self._reconcile),
        ):
            worker = Thread(name=name, target=target, daemon=True)
            self.workers.append(worker)
            worker.start()

    def close(self):
        self.state.stop()
        self.stop.set()
        self.http.close()
        # A stuck RPC cannot hold SIGTERM indefinitely. One fixed daemon per
        # producer; never spawn replacement threads for a blocked collector.
        deadline = monotonic() + 2
        for worker in self.workers:
            worker.join(timeout=max(0, deadline - monotonic()))
