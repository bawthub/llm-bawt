"""Thin scheduled caller of the shared new-session lifecycle.

The scheduler owns the single JobRun. This job only selects candidates, submits
System-authored intents and returns observed outcomes. It never rotates a
session, publishes thread events, or manufactures reset confirmations.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from datetime import datetime, timezone

from ...bot_identities import SYSTEM_BOT_ID
from ...message_authorship import AuthorReference
from ..job_parameters import IdleContextResetParameters
from ..session_lifecycle.intent import (
    Admission, Continuity, NewSessionIntent, NewSessionResult, Origin, Outcome, Stage,
)
from .idle_context_candidates import IdleContextCandidates

log = logging.getLogger(__name__)


class IdleContextResetJob:
    def __init__(self, engine, service, *, protected_bot_ids=(), lifecycle=None, parameters=None):
        self.service = service
        values = (parameters or IdleContextResetParameters()).model_dump()
        values["protected_bot_ids"] = (*values["protected_bot_ids"], *protected_bot_ids)
        self.parameters = IdleContextResetParameters.model_validate(values)
        self.candidates = IdleContextCandidates(engine, protected_bot_ids=self.parameters.protected_bot_ids)
        self._lifecycle = lifecycle

    async def run(self, *, job_id: str, run_id: str, target_bot: str) -> dict:
        rows = await asyncio.to_thread(self.candidates.read, target_bot)
        results = {}
        counts = {"reset": 0, "skipped": 0, "failed": 0, "ambiguous": 0}
        for candidate in rows:
            bot_id = candidate["bot_id"]
            # Defense in depth: no submit even if a future selector is replaced.
            reason = ("protected bot" if bot_id.lower() in self.candidates.protected_bot_ids
                      else self.candidates.skip_reason(
                          candidate, now=datetime.now(timezone.utc), parameters=self.parameters,
                      ))
            if reason:
                result = NewSessionResult(outcome=Outcome.SKIPPED, stage=Stage.ADMISSION, detail=reason)
            else:
                try:
                    intent = NewSessionIntent(
                        bot_id=bot_id, user_id=candidate["user_id"],
                        author=AuthorReference.bot(SYSTEM_BOT_ID),
                        continuity=Continuity.CONFIGURED, admission=Admission.SKIP,
                        origin=Origin.SCHEDULER, job_id=job_id, run_id=run_id,
                        scheduler_parameters=self.parameters,
                        expected_session_id=candidate["session_id"],
                        expected_usage_turn_id=candidate["usage_turn_id"],
                        # Rechecking the same measured context attaches to the
                        # SAME operation even after a timeout or a later sweep.
                        idempotency_key=(f"idle-context:{job_id}:{candidate['session_id']}:"
                                         f"{candidate['usage_turn_id']}"),
                    )
                    if self._lifecycle is None:
                        from ..session_lifecycle.lifecycle import NewSessionLifecycle
                        self._lifecycle = NewSessionLifecycle(self.service)
                    result = await self._lifecycle.submit(intent)
                except TimeoutError:
                    result = NewSessionResult(
                        outcome=Outcome.AMBIGUOUS, stage=Stage.RECEIPT,
                        detail="reset completion not confirmed before timeout",
                    )
                except Exception as exc:
                    log.exception("Idle context reset submission failed for %s", bot_id)
                    result = NewSessionResult(
                        outcome=Outcome.FAILED, stage=Stage.ADMISSION, detail=str(exc),
                    )
            counts["reset" if result.outcome is Outcome.COMPLETED else result.outcome.value] += 1
            results[bot_id] = asdict(result)
        return {"scanned": len(rows), **counts, "bots": results,
                "parameters": self.parameters.model_dump(mode="json")}
