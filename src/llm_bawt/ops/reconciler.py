"""Independent DB reconciliation/queued-dispatch pump.

Run in its own container using the app image and configuration; one-shot action
workers never load this module or receive database credentials. Multiple pumps
are safe (conditional claims), though a single configured replica is sufficient.

It also admits approved MCP ``ops_run`` requests into durable jobs
(TASK-1220, :mod:`.approved_admission`); the app only records the decision.
It makes quiesce jobs terminal; their receipts are pushed by the app, which
owns the delivery dispatcher (TASK-1233 (f), service/ops_receipt_outbox.py).

It is also the only dispatcher (TASK-1169 S3): the app admits a job QUEUED and
its commit notifies ``ops_job_created``, which wakes this loop at once; the
interval is the fallback when the listener is down.
"""
from __future__ import annotations

import logging
import math
import os
import select
import signal
import time
from threading import Event

from .approved_admission import ApprovedOpsAdmission
from .health_state import HealthSample
from .service import OpsService
from .store import JOB_CREATED_CHANNEL

logger = logging.getLogger(__name__)


class IntervalWaiter:
    """Waits the full interval (or until stop)."""

    def wait(self, stop: Event, timeout: float) -> bool:
        stop.wait(timeout)
        return False

    def close(self):
        pass


class JobCreatedWaiter:
    """Returns early when a job is created (``LISTEN ops_job_created``).

    The listen connection is its own, detached from the pool and autocommit,
    so it never holds a pool slot or a transaction open. Any listener failure
    is logged, the connection dropped and the rest of the interval waited out;
    the next wait reconnects. A missed notify costs at most one interval.
    """

    SLICE_SECONDS = 0.5

    def __init__(self, engine, *, channel: str = JOB_CREATED_CHANNEL):
        self.engine = engine
        self.channel = channel
        self._conn = None
        #: Payloads (job ids) drained so far, newest last; bounded, for tests.
        self.received: list[str] = []

    def _connect(self):
        raw = self.engine.raw_connection()
        # Take the driver connection first: detach() clears the pool's handle.
        conn = raw.driver_connection
        raw.detach()
        conn.autocommit = True
        with conn.cursor() as cursor:
            cursor.execute(f"LISTEN {self.channel}")
        return conn

    def _drain(self) -> bool:
        self._conn.poll()
        woke = False
        while self._conn.notifies:
            self.received.append(self._conn.notifies.pop(0).payload)
            woke = True
        del self.received[:-100]
        return woke

    def wait(self, stop: Event, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        try:
            if self._conn is None:
                self._conn = self._connect()
            if self._drain():
                return True
            while not stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                ready, _, _ = select.select([self._conn], [], [], min(self.SLICE_SECONDS, remaining))
                if ready and self._drain():
                    return True
            return False
        except Exception:  # noqa: BLE001 - the interval is the fallback
            logger.exception("ops job-created listener failed; falling back to the reconcile interval")
            self.close()
            stop.wait(max(0.0, deadline - time.monotonic()))
            return False

    def close(self):
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001 - already broken
                pass


class OpsReconciler:
    def __init__(self, service: OpsService, *, interval=5.0, batch_size=100,
                 admission: ApprovedOpsAdmission | None = None, waiter=None):
        self.service = service
        self.interval = interval
        self.batch_size = batch_size
        self.offset = 0
        self.admission = admission
        self.waiter = waiter or IntervalWaiter()

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
        if failure is not None:
            raise failure
        return count

    def run(self, stop=None, health=None):
        stop = stop or Event()
        try:
            while not stop.is_set():
                sampled = health.sample_time() if health else None
                try:
                    self.tick()
                    if health:
                        health.publish("reconciler.loop", HealthSample(sampled, "ok"))
                except Exception:
                    logger.exception("ops reconciliation cycle failed; durable jobs remain pending")
                    if health:
                        health.publish("reconciler.loop",
                                       HealthSample(sampled, "degraded", "reconciliation cycle failed"))
                self.waiter.wait(stop, self.interval)
        finally:
            self.waiter.close()


def main():
    from .control_protocol import CONTROL_PORT
    from .health_runtime import OpsHealthRuntime

    logging.basicConfig(level=logging.INFO)
    interval = float(os.getenv("LLM_BAWT_OPS_RECONCILE_INTERVAL", "5"))
    if not math.isfinite(interval) or interval < 1:
        raise ValueError("LLM_BAWT_OPS_RECONCILE_INTERVAL must be finite and at least 1 second")
    runtime = OpsHealthRuntime(interval=interval, control_address=("0.0.0.0", CONTROL_PORT))
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_args: runtime.stop.set())
    try:
        runtime.start()
        runtime.stop.wait()
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
