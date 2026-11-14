"""API-owned bounded terminal recovery pump; report-only unless opted in.

Per-turn database leases, not process leadership, own work across replicas.
Report-only is the default and does not claim, renew, abort or project anything.
The caller supplies the shutdown Event and owns the loop's task lifetime.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging

from agent_bridge.abort_contract import ABORT_RPC_TIMEOUT_S
from .turn_execution import TurnExecutionDetached
from .turn_terminal_observe import OBSERVATION_SECONDS
from .turn_terminal_protocol import FollowerFenceLost
from .turn_terminal_recovery import TERMINAL_READ_SECONDS, recover_terminal_turn

log = logging.getLogger(__name__)
RECOVERY_INTERVAL_SECONDS = 15.0
RECOVERY_BATCH_SIZE = 16
RECOVERY_CONCURRENCY = 4
# Covers observation, liveness probe, bounded abort, both terminal-hash reads
# and DB/projection overhead. A timeout cancels observation, never inference.
RECOVERY_ATTEMPT_SECONDS = OBSERVATION_SECONDS + ABORT_RPC_TIMEOUT_S + 2 * TERMINAL_READ_SECONDS + 30


@dataclass(frozen=True)
class RecoveryPass:
    candidates: tuple[str, ...]
    attempted: int
    errors: int


async def terminal_recovery_pass(protocol, subscriber, *, report_only: bool = True,
                                 subscriber_factory=None) -> RecoveryPass:
    ids = await asyncio.to_thread(protocol.recoverable_ids, limit=RECOVERY_BATCH_SIZE)
    if report_only or not ids:
        return RecoveryPass(tuple(ids), 0, 0)
    owned_subscriber = None
    if subscriber is None:
        if subscriber_factory is None:
            raise RuntimeError("terminal recovery transport unavailable")
        # Redis clients connect lazily on commands. Failed startup must not
        # disable recovery for this API's entire lifetime. Retry each pass;
        # private fallback pools never become inference/tool-drain subscribers.
        subscriber = owned_subscriber = subscriber_factory()
    sem = asyncio.Semaphore(RECOVERY_CONCURRENCY)

    async def recover(turn_id):
        async with sem:
            try:
                async with asyncio.timeout(RECOVERY_ATTEMPT_SECONDS):
                    await recover_terminal_turn(protocol, turn_id, subscriber)
                return 0
            except (FollowerFenceLost, TurnExecutionDetached):
                # Another worker/Stop owns the outcome. No mutable fallback.
                return 0
            except Exception:
                # Claim/renewal leaves a DB-time lease before any expensive work.
                # Failure cannot hot-loop on this row in the next 15 s sweep.
                log.exception("Terminal recovery attempt failed for turn %s", turn_id)
                return 1

    try:
        errors = await asyncio.gather(*(recover(turn_id) for turn_id in ids))
        return RecoveryPass(tuple(ids), len(ids), sum(errors))
    finally:
        if owned_subscriber is not None:
            await owned_subscriber.close()


async def run_terminal_recovery_pump(protocol, subscriber, stop: asyncio.Event, *,
                                     report_only: bool = True, mode_resolver=None, subscriber_factory=None):
    """One loop per API lifespan; dynamic DB settings are read off the API loop."""
    while not stop.is_set():
        try:
            mode = (await asyncio.to_thread(mode_resolver) if mode_resolver is not None
                    else "report" if report_only else "recover")
            if mode not in {"off", "report", "recover"}:
                log.warning("Unknown terminal recovery mode %r; reporting only", mode)
                mode = "report"
            if stop.is_set():
                break
            if mode != "off":
                transport = {"subscriber_factory": subscriber_factory} if subscriber_factory is not None else {}
                result = await terminal_recovery_pass(protocol, subscriber, report_only=mode != "recover", **transport)
                if result.candidates:
                    log.info("Terminal recovery %s: candidates=%s attempted=%s errors=%s",
                             mode, len(result.candidates), result.attempted, result.errors)
        except Exception:
            log.exception("Terminal recovery discovery failed")
        try:
            await asyncio.wait_for(stop.wait(), timeout=RECOVERY_INTERVAL_SECONDS)
        except TimeoutError:
            pass
