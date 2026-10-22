"""The one ``done`` for a Claude bridge send (TASK-1233 run contract).

``_handle_send`` creates a :class:`RunTerminalLatch` per admitted send and
publishes it in its ``finally``, after the run's bounded teardown
(``run_teardown.RunTeardown``) has finished. The sentinel's ``teardown``
report therefore describes a run that has already stopped.

A send can tear down more than one SDK client (an auth or fresh-session
retry); each teardown is recorded and the report combines them
(:meth:`TeardownReport.combine`).

Host shutdown is the exception. It publishes first, with nothing attempted,
because the bridge's whole stop has to fit Docker's stop grace
(``ClaudeCodeBridge.SHUTDOWN_SEND_GRACE_SECONDS``). The latch makes the later
publish in ``finally`` a no-op. A send whose stream had already ended, still
tearing down when that grace runs out, is published by the shutdown itself
with ``exited=False`` (the bridge keeps every live latch for this).
"""

from __future__ import annotations

import asyncio
from typing import Any

from agent_bridge.run_contract import UNATTEMPTED, TeardownReport
from agent_bridge.session_queue import RequestPhase

from .run_teardown import RunTeardown


class RunTerminalLatch:
    """Publishes one send's ``done`` exactly once."""

    def __init__(
        self, publisher: Any, request_id: str, *, owner: asyncio.Task | None = None,
    ) -> None:
        self._publisher = publisher
        self._request_id = request_id
        self.owner = owner
        """The send's task, which publishes this latch when it ends."""
        self._reports: list[TeardownReport] = []
        self.published = False

    def record(self, teardown: RunTeardown) -> None:
        """Record one SDK client's teardown after ``run_owned`` returned or raised.

        The teardown was attempted, so anything short of a proven reap is
        ``exited=False``: no outcome (cut short) or an outcome whose CLI
        could not be identified (``TeardownOutcome.exited is None``). In the
        run contract ``None`` means "not attempted" and never applies here.
        """
        outcome = teardown.outcome
        if outcome is None:
            self._reports.append(TeardownReport(exited=False))
            return
        self._reports.append(TeardownReport(
            outcome.disconnect, outcome.signal, outcome.exited is True, outcome.elapsed_ms,
        ))

    def report(self, *, phase: RequestPhase | None, cancelled: bool) -> TeardownReport:
        """The send's teardown report once its task has finished.

        With no SDK client torn down, nothing the run spawned is still known
        to be running, except a send cancelled during preflight: a ``/new``
        seed may have had its own CLI in flight, which nothing identified.
        """
        if self._reports:
            return TeardownReport.combine(self._reports)
        if cancelled and phase is RequestPhase.PREFLIGHT:
            return UNATTEMPTED
        return TeardownReport(exited=True)

    def publish(self, teardown: TeardownReport) -> None:
        if self.published:
            return
        self.published = True
        self._publisher.publish_run_done(self._request_id, teardown=teardown)


__all__ = ["RunTerminalLatch"]
