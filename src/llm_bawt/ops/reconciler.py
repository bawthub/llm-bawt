"""Independent DB reconciliation/queued-dispatch pump.

Run in its own container using the app image and configuration; one-shot action
workers never load this module or receive database credentials. Multiple pumps
are safe (conditional claims), though a single configured replica is sufficient.
"""
from __future__ import annotations

import logging
import math
import os
import signal
from threading import Event

from .health_state import HealthSample
from .service import OpsService

logger = logging.getLogger(__name__)


class OpsReconciler:
    def __init__(self, service: OpsService, *, interval=5.0, batch_size=100):
        self.service = service
        self.interval = interval
        self.batch_size = batch_size
        self.offset = 0

    def tick(self):
        count = self.service.reconcile_active_jobs(limit=self.batch_size, offset=self.offset)
        # Jobs may leave the active set while paginating. Cycle back to zero
        # after the final page; skipped shifted rows are seen on the next cycle.
        self.offset = self.offset + count if count == self.batch_size else 0
        return count

    def run(self, stop=None, health=None):
        stop = stop or Event()
        while not stop.is_set():
            sampled = health.sample_time() if health else None
            try:
                self.tick()
                if health:
                    health.publish("reconciler.loop", HealthSample(sampled, "ok"))
            except Exception:
                logger.exception("ops reconciliation cycle failed; durable jobs remain pending")
                if health:
                    health.publish("reconciler.loop", HealthSample(sampled, "degraded", "reconciliation cycle failed"))
            stop.wait(self.interval)


def main():
    from .health_runtime import OpsHealthRuntime

    logging.basicConfig(level=logging.INFO)
    interval = float(os.getenv("LLM_BAWT_OPS_RECONCILE_INTERVAL", "5"))
    if not math.isfinite(interval) or interval < 1:
        raise ValueError("LLM_BAWT_OPS_RECONCILE_INTERVAL must be finite and at least 1 second")
    runtime = OpsHealthRuntime(interval=interval)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_args: runtime.stop.set())
    try:
        runtime.start()
        runtime.stop.wait()
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
