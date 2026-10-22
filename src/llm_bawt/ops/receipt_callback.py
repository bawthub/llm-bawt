"""Push a direct quiesce job's terminal receipt to the caller it held (TASK-1233 e).

A quiesce job whose scope covers its own caller tells that caller to end its
turn (:func:`.quiesce.caller_instruction`). An approved job's continuation then
delivers the terminal receipt as a new turn; a direct job has no approval row,
so nothing would ever start the turn that reads its result. Such a job records
a receipt callback at creation (:meth:`.service.OpsService.dispatch_job`), and
this sender, run by the ops reconciler, delivers it once the job is terminal:

* through the existing durable inter-bot delivery FIFO, from ``system``, as a
  new turn (never a steer) bound to the caller's session;
* ``on_gate="hold"``: a system receipt is never refused by a gate; under a
  hold it waits at the claim and is delivered after release;
* exactly once per job: the delivery's idempotency key is the job id, and the
  job is stamped with the delivery id afterwards, so a crash between the two
  re-reads the same delivery rather than sending a second.

Pre-dispatch failures (``quiesce_timeout``, ``quiesce_lost``, a disabled
operation) are terminal too, so the caller always learns how its job ended.
The framing text is the prompt-registry key :data:`PROMPT_KEY`.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import text

from ..bot_identities import SYSTEM_BOT_ID
from .models import JOB_TERMINAL_STATES

logger = logging.getLogger(__name__)

PROMPT_KEY = "ops.quiesce_receipt"
DEFAULT_QUIESCE_RECEIPT = (
    "[Ops job receipt: an operation you started finished while your turn was "
    "ended for it. The JSON below is its final receipt. Do not re-run the "
    "operation. Report the outcome, check the target's health if it ran, and "
    "continue any work it was blocking.]"
)
OUTPUT_TAIL_BYTES = 4096


def receipt_prompt(bot_id: str) -> str:
    """The live framing text for ``bot_id``, or the seed if the registry fails."""
    try:
        from ..prompt_registry import get_prompt_resolver

        resolved = get_prompt_resolver().resolve(PROMPT_KEY, scope_type="bot", scope_id=bot_id)
        if resolved and (resolved.body or "").strip():
            return resolved.body
    except Exception as error:  # registry/DB failure must not strand the receipt
        logger.warning("ops receipt prompt resolve failed, using seed: %s", error)
    return DEFAULT_QUIESCE_RECEIPT


class QuiesceReceiptSender:
    """Deliver due receipt callbacks; one job's failure never stops the rest."""

    def __init__(self, ops, deliveries):
        self.ops = ops
        self.deliveries = deliveries

    @property
    def engine(self):
        return self.ops.store.engine

    def deliver_due(self, *, limit: int = 20) -> list[str]:
        with self.engine.connect() as conn:
            due = [row[0] for row in conn.execute(text(
                "SELECT id FROM ops_jobs WHERE receipt_callback_json IS NOT NULL"
                " AND receipt_callback_delivery_id IS NULL AND state = ANY(:terminal)"
                " ORDER BY finished_at NULLS FIRST, id LIMIT :limit"
            ), {"terminal": list(JOB_TERMINAL_STATES), "limit": int(limit)}).all()]
        sent = []
        for job_id in due:
            try:
                self.deliver(job_id)
                sent.append(job_id)
            except Exception:
                logger.exception("ops receipt callback failed for job %s; retrying next cycle", job_id)
        return sent

    def deliver(self, job_id: str) -> str | None:
        job = self.ops.store.get_job(job_id)
        if job is None or not job.receipt_callback_json or job.state not in JOB_TERMINAL_STATES:
            return None
        callback: dict[str, Any] = json.loads(job.receipt_callback_json)
        bot_id = callback["bot_id"]
        # A read only: the reconciler already recorded the terminal state.
        receipt = self.ops.get_job_status(job_id, output_tail_bytes=OUTPUT_TAIL_BYTES,
                                          reconcile_if_active=False)
        message = (receipt_prompt(bot_id) + "\n"
                   + json.dumps(receipt, ensure_ascii=False, sort_keys=True, default=str))
        record, _duplicate = self.deliveries.enqueue(
            sender_bot_id=SYSTEM_BOT_ID, target_bot_id=bot_id, message=message,
            idempotency_key=f"ops-receipt:{job_id}", message_kind="OPS_RECEIPT",
            on_gate="hold",
            metadata={"ops_job_id": job_id, "ops_receipt": True},
            payload={
                "bot_id": bot_id, "user": callback.get("user_id"),
                "session_id": callback.get("session_id"),
                "messages": [{"role": "user", "content": message}],
                "prefer_steer": False, "reply_mode": "async",
                "extract_memory": False, "augment_memory": False, "stream": True,
            },
        )
        with self.engine.begin() as conn:
            conn.execute(text(
                "UPDATE ops_jobs SET receipt_callback_delivery_id=:delivery"
                " WHERE id=:id AND receipt_callback_delivery_id IS NULL"
            ), {"delivery": record.id, "id": job_id})
        return record.id


__all__ = ["DEFAULT_QUIESCE_RECEIPT", "PROMPT_KEY", "QuiesceReceiptSender", "receipt_prompt"]
