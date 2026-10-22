"""Independent DB reconciliation/queued-dispatch pump.

Run in its own container using the app image and configuration; one-shot action
workers never load this module or receive database credentials. Multiple pumps
are safe (conditional claims), though a single configured replica is sufficient.

It also admits approved MCP ``ops_run`` requests into durable jobs
(TASK-1220, :mod:`.approved_admission`); the app only records the decision.
It also pushes a direct quiesce job's terminal receipt to the caller it held
(TASK-1233 e, :mod:`.receipt_callback`).
"""
from __future__ import annotations

import logging
import math
import os
import signal
from threading import Event

from .approved_admission import ApprovedOpsAdmission
from .health_state import HealthSample
from .service import OpsService

logger = logging.getLogger(__name__)


class OpsReconciler:
    def __init__(self, service: OpsService, *, interval=5.0, batch_size=100,
                 admission: ApprovedOpsAdmission | None = None, receipts=None):
        self.service = service
        self.interval = interval
        self.batch_size = batch_size
        self.offset = 0
        self.admission = admission
        # TASK-1233 e: terminal receipts of direct quiesce jobs to their callers.
        self.receipts = receipts

    def tick(self):
        # Each half runs even when the other fails; the cycle reports the first
        # failure afterwards so the loop's health sample turns degraded.
        failure = None
        count = 0
        try:
            count = self.service.reconcile_active_jobs(limit=self.batch_size, offset=self.offset)
            # Jobs may leave the active set while paginating. Cycle back to zero
            # after the final page; skipped shifted rows are seen on the next cycle.
            self.offset = self.offset + count if count == self.batch_size else 0
        except Exception as exc:
            failure = exc
        if self.admission is not None:
            try:
                self.admission.admit_due()
            except Exception as exc:
                failure = failure or exc
        if self.receipts is not None:
            # After reconciliation, so a job that just turned terminal is sent
            # in the same cycle.
            try:
                self.receipts.deliver_due()
            except Exception as exc:
                failure = failure or exc
        if failure is not None:
            raise failure
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
