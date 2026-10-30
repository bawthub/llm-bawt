"""Resume the bots a quiesce job held once it ends (TASK-1798).

A quiesce job ends the running turns in its scope before it runs
(:mod:`.quiesce`). Nothing started them again: a bot whose turn was cut sat
idle until someone wrote to it. This sender closes that loop, from the app
process beside :class:`.receipt_callback.QuiesceReceiptSender`:

* Every bot with a member row in any of the job's gate generations gets one
  ``OPS_RESUME`` new turn, carrying its own ``ops_quiesce_ack`` note, once the
  job is terminal and its target is up: the job succeeded, or it never ran
  (``dispatched_at`` unset: a quiesce timeout or a cancel left the service
  untouched). Bots that were idle have no member row and get nothing.
* A job that ran and did not succeed may have left its target down, so its
  held bots are not resumed. Each coordinator bot (a non-private bot holding a
  role with the ``ops_resume_coordinator_capability`` setting's capability)
  gets one ``OPS_RESUME_BLOCKED`` notice naming them instead.
* The job's own caller is skipped when it already gets the terminal receipt
  (a receipt callback, or an approval continuation).

Exactly once: each delivery's idempotency key is ``ops-resume:<job>:<bot>``
(``ops-resume-blocked:`` for coordinators), and ``ops_resume_notices`` records
each send plus a ``done`` row per job, so a crash mid-job re-reads the same
deliveries rather than sending more. Only jobs that finished within
:data:`RECENT_SECONDS` are considered, and never bots
for old jobs: only jobs that finished after its first run (the ``EPOCH`` row)
count, and a held bot is one whose turn the job actually stopped. Framing text is registry-owned (:mod:`.quiesce_prompts`).
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from sqlalchemy import text

from ..bot_identities import SYSTEM_BOT_ID
from ..bot_tags.vocabulary import BehaviorTag
from .models import JOB_SUCCEEDED, JOB_TERMINAL_STATES
from .quiesce_prompts import RESUME_BLOCKED_KEY, RESUME_KEY, resolve

logger = logging.getLogger(__name__)

COORDINATOR_CAPABILITY_SETTING = "ops_resume_coordinator_capability"
RECENT_SECONDS = 6 * 3600
NOTE_LIMIT = 2000
DONE_BOT = "*"
# ops_resume_notices (EPOCH, EPOCH, done): when this sender first ran. Jobs that
# finished before it are never resumed, so deploying it resends nothing.
EPOCH = "*"

_DUE = """
    SELECT j.id FROM ops_jobs j
    WHERE j.state = ANY(:terminal) AND j.finished_at > now() - make_interval(secs => :recent)
      AND j.finished_at > (SELECT e.created_at FROM ops_resume_notices e
                           WHERE e.job_id = :epoch AND e.bot_id = :epoch AND e.kind = 'done')
      AND EXISTS (SELECT 1 FROM system_gates g JOIN gate_members m ON m.gate_id = g.id
                  WHERE g.owner_kind = 'job' AND g.owner_id = j.id)
      AND NOT EXISTS (SELECT 1 FROM ops_resume_notices n
                      WHERE n.job_id = j.id AND n.kind = 'done')
    ORDER BY j.finished_at, j.id LIMIT :limit
"""

# One row per held bot: its latest ack across every generation of the job.
# Held means the job actually stopped that turn: its control ask was sent
# (steering/cancelled) or the bot acked "stopping". A turn that ended on its
# own (not_needed), the job caller's own turn (self) and an unstarted turn the
# hold requeued (revoked_unstarted: it gets its work back) are not held.
_HELD = """
    SELECT DISTINCT ON (m.bot_id) m.bot_id, m.ack_status, m.ack_note, t.user_id
    FROM system_gates g JOIN gate_members m ON m.gate_id = g.id
    LEFT JOIN turn_logs t ON t.id = m.turn_id
    WHERE g.owner_kind = 'job' AND g.owner_id = :job
      AND m.disposition <> 'revoked_unstarted' AND m.control_state <> 'self'
      AND (m.control_state IN ('steering', 'cancelled') OR m.ack_status = 'stopping')
    ORDER BY m.bot_id, m.acked_at DESC NULLS LAST, m.id DESC
"""

_COORDINATORS = """
    SELECT DISTINCT t.bot_slug FROM bot_profile_tags t
    JOIN bot_tag_catalog c ON c.slug = t.tag_slug AND c.kind = 'role'
    WHERE c.capabilities @> jsonb_build_array(CAST(:capability AS text))
      AND NOT EXISTS (SELECT 1 FROM bot_profile_tags p
                      WHERE p.bot_slug = t.bot_slug AND p.tag_slug = :private)
    ORDER BY t.bot_slug
"""


def resumable(job) -> bool:
    """Whether ``job`` left its target up: it succeeded, or it never ran."""
    return job.state == JOB_SUCCEEDED or getattr(job, "dispatched_at", None) is None


class QuiesceResumeSender:
    """Deliver due resume notices; one job's failure never stops the rest."""

    def __init__(self, ops, deliveries, *, coordinator_capability: Callable[[], str | None],
                 on_enqueued=None):
        self.ops = ops
        self.deliveries = deliveries
        # Read live each job, so a settings edit applies without a restart.
        self.coordinator_capability = coordinator_capability
        self.on_enqueued = on_enqueued

    @property
    def engine(self):
        return self.ops.store.engine

    def deliver_due(self, *, limit: int = 20) -> list[str]:
        self._mark(EPOCH, EPOCH, "done", None)  # first run only: ON CONFLICT keeps it
        with self.engine.connect() as conn:
            due = [row[0] for row in conn.execute(text(_DUE), {
                "terminal": list(JOB_TERMINAL_STATES), "recent": RECENT_SECONDS, "epoch": EPOCH,
                "limit": int(limit)}).all()]
        done = []
        for job_id in due:
            try:
                self.deliver(job_id)
                done.append(job_id)
            except Exception:
                logger.exception("ops resume failed for job %s; retrying next cycle", job_id)
        return done

    def deliver(self, job_id: str) -> list[str]:
        """Send ``job_id``'s notices and mark it done; the delivery ids sent."""
        job = self.ops.store.get_job(job_id)
        if job is None or job.state not in JOB_TERMINAL_STATES:
            return []
        with self.engine.connect() as conn:
            held = [dict(row) for row in conn.execute(text(_HELD), {"job": job_id}).mappings()]
        skip = self._receipt_bot(job)
        held = [member for member in held if member["bot_id"] != skip]
        if resumable(job):
            sent = [self._resume(job, member) for member in held]
        else:
            sent = self._escalate(job, held) if held else []
        self._mark(job_id, DONE_BOT, "done", None)
        return sent

    # -- targets ----------------------------------------------------------

    @staticmethod
    def _receipt_bot(job) -> str | None:
        """The caller that already gets this job's terminal receipt, if any."""
        if getattr(job, "receipt_callback_json", None):
            return json.loads(job.receipt_callback_json).get("bot_id")
        if getattr(job, "approval_request_id", None):
            return getattr(job, "caller_bot_id", None)
        return None

    def _coordinators(self) -> list[str]:
        capability = self.coordinator_capability()
        if not isinstance(capability, str) or not capability.strip():
            return []
        with self.engine.connect() as conn:
            return list(conn.execute(text(_COORDINATORS), {
                "capability": capability.strip(), "private": BehaviorTag.PRIVATE.value}).scalars())

    # -- sends ------------------------------------------------------------

    @staticmethod
    def _facts(job) -> dict[str, Any]:
        return {"job_id": job.id, "operation": job.operation_slug, "state": job.state,
                "error_text": job.error_text or None}

    @staticmethod
    def _note(member: dict[str, Any]) -> dict[str, Any]:
        return {"bot_id": member["bot_id"], "ack_status": member["ack_status"],
                "ack_note": (member["ack_note"] or None) and member["ack_note"][:NOTE_LIMIT]}

    def _resume(self, job, member: dict[str, Any]) -> str:
        bot_id = member["bot_id"]
        facts = {**self._facts(job), **self._note(member)}
        record = self._send(job, bot_id, member.get("user_id"), RESUME_KEY, facts,
                            key=f"ops-resume:{job.id}:{bot_id}", kind="OPS_RESUME")
        self._mark(job.id, bot_id, "resume", record.id)
        return record.id

    def _escalate(self, job, held: list[dict[str, Any]]) -> list[str]:
        coordinators = self._coordinators()
        if not coordinators:
            logger.error("ops job %s ended %s after it ran; held bots %s were not resumed and"
                         " no coordinator is set (%s)", job.id, job.state,
                         [m["bot_id"] for m in held], COORDINATOR_CAPABILITY_SETTING)
            return []
        facts = {**self._facts(job), "held_bots": [self._note(m) for m in held]}
        sent = []
        for bot_id in coordinators:
            record = self._send(job, bot_id, None, RESUME_BLOCKED_KEY, facts,
                                key=f"ops-resume-blocked:{job.id}:{bot_id}",
                                kind="OPS_RESUME_BLOCKED")
            self._mark(job.id, bot_id, "escalation", record.id)
            sent.append(record.id)
        return sent

    def _send(self, job, bot_id: str, user_id: str | None, prompt_key: str,
              facts: dict[str, Any], *, key: str, kind: str):
        message = (resolve(prompt_key, bot_id) + "\n"
                   + json.dumps(facts, ensure_ascii=False, sort_keys=True, default=str))
        record, duplicate = self.deliveries.enqueue(
            sender_bot_id=SYSTEM_BOT_ID, target_bot_id=bot_id, message=message,
            idempotency_key=key, message_kind=kind,
            # A system notice: held behind a live gate, never refused by one.
            on_gate="hold",
            metadata={"ops_job_id": job.id, "ops_resume": True},
            payload={
                "bot_id": bot_id, "user": user_id, "session_id": None,
                "messages": [{"role": "user", "content": message}],
                "prefer_steer": False, "reply_mode": "async",
                "extract_memory": False, "augment_memory": False, "stream": True,
            },
        )
        if self.on_enqueued is not None:
            try:
                self.on_enqueued(record, duplicate)
            except Exception:  # the row is durable; the dispatcher sweep still finds it
                logger.exception("ops resume announce failed for job %s", job.id)
        return record

    def _mark(self, job_id: str, bot_id: str, kind: str, delivery_id: str | None) -> None:
        with self.engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO ops_resume_notices (job_id, bot_id, kind, delivery_id)"
                " VALUES (:job, :bot, :kind, :delivery) ON CONFLICT DO NOTHING"
            ), {"job": job_id, "bot": bot_id, "kind": kind, "delivery": delivery_id})


__all__ = ["COORDINATOR_CAPABILITY_SETTING", "QuiesceResumeSender", "RECENT_SECONDS", "resumable"]
