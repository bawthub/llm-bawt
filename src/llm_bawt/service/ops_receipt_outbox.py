"""App-side loop that pushes direct quiesce jobs' terminal receipts (TASK-1233 (f) N1).

The receipt is enqueued here, in the process that owns the inter-bot delivery
dispatcher, so the new row is announced at once: QUEUED is published and the
dispatcher woken, the same as :class:`OpsBlockedNotifier`. (Enqueued from the
ops-reconciler, which has no dispatcher, a receipt waited for the recovery
sweep and never showed QUEUED.) The ops-reconciler still makes the job
terminal; this loop only reads terminal rows.

Exactly-once is the sender's: idempotency key ``ops-receipt:<job_id>`` plus a
conditional stamp, so several app processes, or an old reconciler image still
sending during rollout, only ever produce one delivery row.
"""
from __future__ import annotations

import asyncio
import logging

from ..ops.receipt_callback import QuiesceReceiptSender
from .dependencies import get_ops_service

log = logging.getLogger(__name__)


def announce(dispatcher):
    """``on_enqueued`` hook: announce a new row, wake on a duplicate."""
    def on_enqueued(record, duplicate: bool) -> None:
        if duplicate:
            dispatcher.wake(record.target_bot_id)
        else:
            dispatcher.announce_admitted(record)
    return on_enqueued


def build_receipt_sender(service) -> QuiesceReceiptSender | None:
    """The sender for this app, or None where receipts cannot be sent (no
    dispatcher, or a non-PostgreSQL ops store)."""
    dispatcher = getattr(service, "_inter_bot_dispatcher", None)
    if dispatcher is None or getattr(dispatcher.store, "engine", None) is None:
        return None
    ops = get_ops_service(service.config)
    engine = getattr(ops.store, "engine", None)
    if getattr(getattr(engine, "dialect", None), "name", None) != "postgresql":
        return None
    return QuiesceReceiptSender(ops, dispatcher.store, on_enqueued=announce(dispatcher))


async def run_ops_receipt_outbox(service, *, idle_seconds: float = 2.0) -> None:
    """Lifespan loop; a DB outage logs and retries, it never kills the loop."""
    sender = None
    while True:
        try:
            sender = sender or build_receipt_sender(service)
            if sender is None:
                log.info("ops receipt outbox disabled: no dispatcher or no PostgreSQL ops store")
                return
            await asyncio.to_thread(sender.deliver_due)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("ops receipt outbox pass failed; retrying")
        await asyncio.sleep(idle_seconds)


__all__ = ["announce", "build_receipt_sender", "run_ops_receipt_outbox"]
