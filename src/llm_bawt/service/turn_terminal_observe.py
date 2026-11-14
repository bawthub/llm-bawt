"""Bounded, headless observation of an already-bound bridge run.

The caller owns a started TerminalFollower (including its lease heartbeat) and
supplies identities loaded under that fence. This helper never starts/cancels a
run, commits a terminal or publishes any event. Its result directs the caller to
fold aggregate evidence or take the explicit durable-partial fallback path.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import time

from redis.exceptions import RedisError

from agent_bridge.terminal_replay import ReplayDisposition, RetainedRun, read_retained_run

OBSERVATION_SECONDS = 60.0
POLL_SECONDS = 1.0


@dataclass(frozen=True)
class RunObservation:
    replay: RetainedRun | None
    incomplete_reason: str | None


async def observe_bound_run(
    follower, redis, *, request_id: str, session_key: str, backend: str,
    boot_id: str | None = None,
) -> RunObservation:
    """Retry a pending/unreachable stream for at most one 60 s observation.

    No inactivity-deadline resets: a perpetually streaming or unreachable run
    must not wedge admission forever. Missing/trimmed/invalid/over-limit streams
    are already conclusive incomplete evidence and do not consume the budget.
    Transport timeout is not missing-stream evidence. Fence loss always escapes
    instead of returning fallback authority to an observer that lost its lease.
    """
    deadline = time.monotonic() + OBSERVATION_SECONDS
    last_replay = None
    reason = "recovery_observation_timeout"
    try:
        # Covers every Redis call and sleep, including a hung transport. The
        # monotonic check also makes the absolute, non-resetting budget explicit.
        async with asyncio.timeout(OBSERVATION_SECONDS):
            while time.monotonic() < deadline:
                follower.check()
                try:
                    replay = await read_retained_run(
                        redis, request_id=request_id, session_key=session_key,
                        backend=backend, boot_id=boot_id,
                    )
                except (RedisError, TimeoutError):
                    reason = "recovery_transport_unavailable"
                else:
                    follower.check()
                    last_replay = replay
                    if replay.disposition is ReplayDisposition.COMPLETE:
                        return RunObservation(replay, None)
                    if replay.disposition is not ReplayDisposition.PENDING:
                        return RunObservation(replay, replay.reason)
                    reason = "recovery_run_not_terminal"
                follower.check()
                await asyncio.sleep(min(POLL_SECONDS, max(0.0, deadline - time.monotonic())))
    except TimeoutError:
        # A hanging Redis call can consume the outer budget without throwing
        # its own transport error. Never mislabel that as an absent stream.
        reason = "recovery_observation_timeout"
    follower.check()
    return RunObservation(last_replay, reason)
