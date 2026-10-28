"""Executor errors and result records shared by executors and worker launchers.

TASK-1169 S4: split out of ``executor.py`` so a launcher module can raise and
return them without importing the executor. ``executor`` re-exports every name.
"""
from __future__ import annotations

from dataclasses import dataclass


class ExecutorError(RuntimeError):
    pass


class QuiesceFenceUnsupported(ExecutorError):
    """The worker this executor would start provably cannot enforce the quiesce
    start fence (TASK-1169 S3). Definitive, unlike a fence that cannot be read
    right now: a job pinned to that worker can never run fenced."""


class ConfigCheckError(ExecutorError):
    """A Compose config check or a container-removal check refused. ``code`` is
    fixed (``config_drift``, ``config_reconcile_unavailable``, a
    :class:`ComposeError` code or a :class:`RemoveRefused` code) and the
    message carries names only, never a value."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass
class DispatchResult:
    host_unit_name: str
    status_file_path: str = ""
    log_file_path: str = ""
    terminal_state: str | None = None  # Compatibility only; never trusted by service.
    exit_code: int | None = None
    output: str | None = None


@dataclass
class ReconcileResult:
    state: str | None
    exit_code: int | None = None
    output_tail: str | None = None
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
