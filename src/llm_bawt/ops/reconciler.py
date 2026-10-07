"""Independent DB reconciliation/queued-dispatch pump.

Run in its own container using the app image and configuration; one-shot action
workers never load this module or receive database credentials. Multiple pumps
are safe (conditional claims), though a single configured replica is sufficient.
"""
from __future__ import annotations

import logging
import os
import time

from .release_executor import build_ops_service
from .service import OpsService
from .store import OpsStore
from ..utils.config import Config

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

    def run(self):
        while True:
            try:
                self.tick()
            except Exception:
                logger.exception("ops reconciliation cycle failed; durable jobs remain pending")
            time.sleep(self.interval)


def main():
    logging.basicConfig(level=logging.INFO)
    store = OpsStore(Config())
    if store.engine is None:
        raise RuntimeError("ops reconciler requires a configured canonical database")
    interval = float(os.getenv("LLM_BAWT_OPS_RECONCILE_INTERVAL", "5"))
    if interval < 1:
        raise ValueError("LLM_BAWT_OPS_RECONCILE_INTERVAL must be at least 1 second")
    OpsReconciler(build_ops_service(Config(), store), interval=interval).run()


if __name__ == "__main__":
    main()
