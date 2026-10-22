"""Per-run ledger of the ``codex`` subprocesses a send spawned (TASK-1233).

The Codex SDK runs one ``codex exec`` subprocess per turn (``exec_patch``'s
``patched_run``), and a send can run more than one (a fresh-session retry).
The send binds a :class:`ProcessLedger` with :func:`bind_ledger` before it
runs; ``patched_run`` registers each process it starts in the bound ledger.
After the run, :meth:`ProcessLedger.report` says whether every one of them was
reaped, which is the run contract's ``teardown.exited`` for codex.

A ``ContextVar`` carries the ledger because the SDK's call chain between the
send and ``patched_run`` isn't ours; any task the SDK creates copies the
context, so it still sees the send's ledger.
"""

from __future__ import annotations

import signal
from contextvars import ContextVar
from typing import Any

from agent_bridge.run_contract import TeardownReport
from agent_bridge.send_admission import RunOutcome

_SIGNAL_NONE = "none"

_current: ContextVar["ProcessLedger | None"] = ContextVar("codex_process_ledger", default=None)


def _signal_name(returncode: int) -> str:
    try:
        return signal.Signals(-returncode).name.lower()
    except ValueError:
        return f"signal{-returncode}"


class ProcessLedger:
    """The subprocesses one send started, in order."""

    def __init__(self) -> None:
        self._procs: list[Any] = []

    def register(self, proc: Any) -> None:
        self._procs.append(proc)

    @property
    def procs(self) -> tuple[Any, ...]:
        return tuple(self._procs)

    def report(self, outcome: RunOutcome) -> TeardownReport:
        """``exited`` = every subprocess has been reaped.

        A send that started none falls back to the generic rule.
        """
        if not self._procs:
            return outcome.default_report()
        codes = [proc.returncode for proc in self._procs]
        signals = [_signal_name(code) for code in codes if code is not None and code < 0]
        return TeardownReport(
            disconnect=None,
            signal=signals[-1] if signals else _SIGNAL_NONE,
            exited=all(code is not None for code in codes),
        )


def bind_ledger(ledger: ProcessLedger) -> None:
    """Bind *ledger* for the rest of the calling task's context."""
    _current.set(ledger)


def register_process(proc: Any) -> None:
    """Record *proc* in the bound ledger; a no-op outside a send."""
    ledger = _current.get()
    if ledger is not None:
        ledger.register(proc)


__all__ = ["ProcessLedger", "bind_ledger", "register_process"]
