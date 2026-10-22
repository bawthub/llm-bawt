"""Bounded, request-owned teardown of one Claude CLI run (TASK-1232 P4).

Why the bridge owns this instead of trusting ``ClaudeSDKClient.disconnect()``:

- The SDK's ``SubprocessCLITransport.close()`` escalates stdin EOF -> SIGTERM
  -> SIGKILL with bounded waits, but only for anyio cancellation. A raw
  asyncio cancellation (``asyncio.wait_for`` firing, a second ``task.cancel()``)
  lands at its next await and skips the escalation, leaving the CLI running.
  The old ``wait_for(disconnect(), 20)`` plus ``except CancelledError: pass``
  did exactly that.
- The SDK only signals the CLI pid. The CLI's own children (Bash tools, stdio
  MCP servers) survive a SIGKILLed CLI.
- The CLI shares the bridge's process group (the SDK spawns it without
  ``start_new_session``, and the bridge is PID 1 in its container), so
  ``killpg`` would kill the bridge. Every signal here targets one pid.

One :class:`RunTeardown` per SDK client:

1. Capture ``client._transport._process`` and its pid *before* ``disconnect()``
   (which drops both references), and snapshot the CLI's descendants from
   ``/proc/<pid>/task/*/children`` before any signal is sent.
2. ``disconnect()``, capped at :data:`TEARDOWN_DISCONNECT_BACKSTOP_S` (above
   the SDK's own ~20s escalation, so the backstop only fires when that
   escalation itself hung).
3. Judge exit by the process's ``returncode`` (set when asyncio reaps it,
   i.e. by ``waitpid``). If the CLI is still running, re-snapshot its
   descendants and SIGKILL the CLI pid; if the CLI did not exit cleanly
   (still running, or killed by a signal), SIGKILL every snapshotted
   descendant that is still the same process (pid *and* start time match).
   Then wait up to :data:`TEARDOWN_KILL_WAIT_S` for all of them to be gone.

A CLI that exits on its own after stdin EOF keeps its descendants: it had the
chance to clean them up, and anything still running is not ours to judge.

The owner (the send's ``finally``, inside the session lock) runs it through
:meth:`RunTeardown.run_owned`, so the lock is released only after the
teardown finished; a ``chat.abort`` awaits the same outcome through
:meth:`RunTeardown.wait`. The total is bounded by
``TEARDOWN_DISCONNECT_BACKSTOP_S + TEARDOWN_KILL_WAIT_S``, which
``abort_contract`` keeps below ``ABORT_TEARDOWN_BOUND_S``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from importlib import metadata
from pathlib import Path
from typing import Any

from agent_bridge.abort_contract import (
    TEARDOWN_DISCONNECT_BACKSTOP_S,
    TEARDOWN_KILL_WAIT_S,
)

logger = logging.getLogger(__name__)

# claude-agent-sdk versions whose private layout this module was verified
# against (``client._transport._process``, ``close()`` escalation). The
# claude-code-bridge image and uv.lock both pin 0.2.160;
# tests/test_bridge_dependency_lock.py keeps them equal (TASK-1269).
VERIFIED_SDK_VERSIONS = frozenset({"0.2.160"})

DISCONNECT_OK = "ok"
DISCONNECT_TIMEOUT = "timeout"
DISCONNECT_ERROR = "error"
SIGNAL_NONE = "none"
SIGNAL_SIGKILL = "sigkill"

_POLL_S = 0.05


def loaded_sdk_version() -> str | None:
    """Return the installed claude-agent-sdk version, or ``None``."""
    try:
        return metadata.version("claude-agent-sdk")
    except metadata.PackageNotFoundError:
        return None


def _log_sdk_version() -> None:
    version = loaded_sdk_version()
    if version in VERIFIED_SDK_VERSIONS:
        logger.info("claude-agent-sdk %s loaded; run teardown verified for it", version)
    else:
        logger.warning(
            "claude-agent-sdk %s loaded; run teardown was verified only on %s. "
            "Check client._transport._process still exists before trusting kills.",
            version, ", ".join(sorted(VERIFIED_SDK_VERSIONS)),
        )


_log_sdk_version()


@dataclass(frozen=True)
class TeardownOutcome:
    """What one teardown did and whether the run's processes are gone.

    ``exited`` is true when the CLI was reaped and no descendant the bridge
    knew about is still running, false when something survived, and ``None``
    when the CLI process could not be identified (no process was ever
    spawned, or the SDK layout changed).
    """

    disconnect: str
    signal: str
    exited: bool | None
    pid: int | None = None
    survivors: tuple[int, ...] = ()
    elapsed_ms: int | None = field(default=None, compare=False)
    """Wall time of the whole teardown, set by :class:`RunTeardown`; a
    measurement, so not compared."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "disconnect": self.disconnect,
            "signal": self.signal,
            "exited": self.exited,
            "pid": self.pid,
            "survivors": list(self.survivors),
            "elapsed_ms": self.elapsed_ms,
        }

    def detail(self) -> str:
        """Compact form for the ``chat.abort`` reply ``detail``."""
        exited = "unknown" if self.exited is None else str(self.exited).lower()
        text = f"teardown:disconnect={self.disconnect};signal={self.signal};exited={exited}"
        if self.survivors:
            text += ";survivors=" + "+".join(str(pid) for pid in self.survivors)
        return text


def cli_process(client: Any) -> Any | None:
    """Return the SDK client's CLI ``Process``; read it before ``disconnect()``."""
    transport = getattr(client, "_transport", None)
    return getattr(transport, "_process", None) if transport is not None else None


class ProcTable:
    """Read-only view of ``/proc`` for one process tree (injectable for tests)."""

    def __init__(self, root: str | Path = "/proc") -> None:
        self._root = Path(root)

    def _stat_fields(self, pid: int) -> list[str] | None:
        try:
            raw = (self._root / str(pid) / "stat").read_text()
        except OSError:
            return None
        # comm may contain spaces/parens; fields resume after the last ')'.
        _, sep, rest = raw.rpartition(")")
        return rest.split() if sep else None

    def start_time(self, pid: int) -> int | None:
        """Kernel start time of *pid* (``stat`` field 22), or ``None`` if gone."""
        fields = self._stat_fields(pid)
        try:
            return int(fields[19]) if fields else None
        except (IndexError, ValueError):
            return None

    def is_alive(self, pid: int, start_time: int) -> bool:
        """True while *pid* is still the same process and not a zombie."""
        fields = self._stat_fields(pid)
        if not fields or len(fields) < 20:
            return False
        try:
            same = int(fields[19]) == start_time
        except ValueError:
            return False
        return same and fields[0] not in ("Z", "X", "x")

    def children(self, pid: int) -> list[int]:
        found: list[int] = []
        try:
            tasks = list((self._root / str(pid) / "task").iterdir())
        except OSError:
            return found
        for task in tasks:
            try:
                text = (task / "children").read_text()
            except OSError:
                continue
            found.extend(int(tok) for tok in text.split() if tok.isdigit())
        return found

    def descendants(self, pid: int) -> dict[int, int]:
        """Every live descendant of *pid* as ``{pid: start_time}``, recursively."""
        tree: dict[int, int] = {}
        stack = [pid]
        while stack:
            for child in self.children(stack.pop()):
                if child in tree or child == pid:
                    continue
                start = self.start_time(child)
                if start is None:
                    continue
                tree[child] = start
                stack.append(child)
        return tree


class RunTeardown:
    """Tear down one SDK client exactly once, within a fixed bound."""

    def __init__(
        self,
        client: Any,
        *,
        disconnect_backstop_s: float = TEARDOWN_DISCONNECT_BACKSTOP_S,
        kill_wait_s: float = TEARDOWN_KILL_WAIT_S,
        procs: ProcTable | None = None,
        kill: Callable[[int, int], None] = os.kill,
    ) -> None:
        self._client = client
        self._disconnect_backstop_s = disconnect_backstop_s
        self._kill_wait_s = kill_wait_s
        self._procs = procs or ProcTable()
        self._kill = kill
        self._task: asyncio.Task[TeardownOutcome] | None = None
        self._outcome: asyncio.Future[TeardownOutcome] = (
            asyncio.get_running_loop().create_future()
        )

    @property
    def started(self) -> bool:
        return self._task is not None

    @property
    def outcome(self) -> TeardownOutcome | None:
        """The finished teardown's outcome, or ``None`` until it finishes."""
        return self._outcome.result() if self._outcome.done() else None

    def start(self) -> asyncio.Task[TeardownOutcome]:
        """Start the teardown (idempotent); it runs to completion on its own."""
        if self._task is None:
            self._task = asyncio.ensure_future(self._run())
        return self._task

    async def run_owned(self) -> TeardownOutcome:
        """Run the teardown for the send that owns the client.

        Cancelling the owner does not cut the teardown short (PE1): a repeated
        Stop or host shutdown waits for the bounded teardown to finish, then
        the cancellation propagates. That keeps the session lock held until
        the CLI is gone.
        """
        task = self.start()
        cancelled = False
        while True:
            try:
                outcome = await asyncio.shield(task)
                break
            except asyncio.CancelledError:
                if task.cancelled():
                    raise
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError
        return outcome

    async def wait(self, timeout_s: float) -> TeardownOutcome | None:
        """Await the outcome for up to *timeout_s*; ``None`` if still running."""
        try:
            return await asyncio.wait_for(asyncio.shield(self._outcome), timeout_s)
        except asyncio.TimeoutError:
            return None

    async def _run(self) -> TeardownOutcome:
        started = time.monotonic()
        try:
            outcome = await self._teardown()
        except Exception as exc:  # never leave an abort waiting on a crash
            logger.exception("Run teardown failed")
            outcome = TeardownOutcome(
                disconnect=f"{DISCONNECT_ERROR}:{type(exc).__name__}",
                signal=SIGNAL_NONE, exited=None,
            )
        outcome = replace(outcome, elapsed_ms=round((time.monotonic() - started) * 1000))
        log = logger.warning if outcome.survivors or outcome.signal != SIGNAL_NONE else logger.debug
        log("Run teardown pid=%s %s elapsed_ms=%s", outcome.pid, outcome.detail(), outcome.elapsed_ms)
        if not self._outcome.done():
            self._outcome.set_result(outcome)
        return outcome

    async def _disconnect(self) -> str:
        task = asyncio.ensure_future(self._client.disconnect())
        done, _ = await asyncio.wait({task}, timeout=self._disconnect_backstop_s)
        if not done:
            # The SDK's own escalation hung. Cancel it without waiting: the
            # kill below doesn't depend on it, and the bound must hold.
            task.cancel()
            task.add_done_callback(_consume_result)
            return DISCONNECT_TIMEOUT
        exc = task.exception()
        if exc is not None:
            logger.debug("disconnect() raised during teardown", exc_info=exc)
            return DISCONNECT_ERROR
        return DISCONNECT_OK

    def _sigkill(self, pid: int) -> None:
        try:
            self._kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            logger.warning("SIGKILL of pid %s failed", pid, exc_info=True)

    async def _teardown(self) -> TeardownOutcome:
        proc = cli_process(self._client)
        pid = getattr(proc, "pid", None)
        # Before any signal: once the CLI dies its children are reparented
        # and can no longer be found from its pid.
        tree = self._procs.descendants(pid) if pid else {}
        cli_start = self._procs.start_time(pid) if pid else None
        disconnect = await self._disconnect()
        if proc is None or pid is None:
            return TeardownOutcome(disconnect=disconnect, signal=SIGNAL_NONE, exited=None)

        sent = SIGNAL_NONE
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._kill_wait_s
        if proc.returncode is None:
            tree.update(self._procs.descendants(pid))
            # The child watcher can reap the CLI a moment before returncode is
            # set, so the pid alone could already name another process: only
            # kill it while its start time still matches.
            if cli_start is not None and self._procs.is_alive(pid, cli_start):
                self._sigkill(pid)
                sent = SIGNAL_SIGKILL
        if proc.returncode is None or proc.returncode < 0:
            for child, start in tree.items():
                if self._procs.is_alive(child, start):
                    self._sigkill(child)
                    sent = SIGNAL_SIGKILL
        if sent == SIGNAL_SIGKILL or proc.returncode is None:
            if proc.returncode is None:
                try:
                    await asyncio.wait_for(proc.wait(), max(0.0, deadline - loop.time()))
                except Exception:  # TimeoutError included; the survivor check below decides
                    pass
            while loop.time() < deadline and any(
                self._procs.is_alive(child, start) for child, start in tree.items()
            ):
                await asyncio.sleep(_POLL_S)

        survivors = [child for child, start in tree.items() if self._procs.is_alive(child, start)]
        if proc.returncode is None and (cli_start is None or self._procs.is_alive(pid, cli_start)):
            survivors.insert(0, pid)
        return TeardownOutcome(
            disconnect=disconnect,
            signal=sent,
            exited=not survivors,
            pid=pid,
            survivors=tuple(survivors),
        )


def _consume_result(task: asyncio.Future) -> None:
    if not task.cancelled():
        task.exception()


__all__ = [
    "RunTeardown",
    "ProcTable",
    "TeardownOutcome",
    "VERIFIED_SDK_VERSIONS",
    "cli_process",
    "loaded_sdk_version",
]
