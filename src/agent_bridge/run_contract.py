"""Bridge ↔ app contract for a run's acceptance and its one terminal (TASK-1233).

Accepted
--------
When a bridge admits a ``chat.send`` (``Admission.ACCEPTED``) it publishes a
``RUN_ACCEPTED`` run event whose ``raw`` is :meth:`RunAccepted.to_dict`:
``{request_id, backend, boot_id, pid1}``. ``boot_id`` names the bridge process
that accepted the request. A later ``run.status`` absence from a different boot
proves the run died with that process only when both processes were their
container's PID 1 (TASK-1233 B1(b), Vex S2;
:func:`agent_bridge.run_status.proves_restart`).

Terminal
--------
Every run ends with exactly one ``done`` sentinel on ``agent:run:<request_id>``.
``subscribe_run`` stops at the first one, so that sentinel is the run's
terminal by construction; ``ASSISTANT_DONE`` and ``ERROR`` are content events
and not every path publishes one. The sentinel's stream fields are
:meth:`RunTerminal.to_fields`::

    {done: "1", request_id, backend, boot_id, teardown: '{"disconnect", "signal", "exited"}'}

``teardown.exited``:

* ``true``: every process the run spawned was reaped (claude-code: the CLI and
  its known descendants; codex: each per-turn ``codex`` subprocess). For a
  bridge whose run spawns no process (openclaw, local-model), the run task and
  any worker it started have finished.
* ``false``: teardown ran and something survived, or a cancelled run's work
  could not be shown to have stopped.
* ``null``: not attempted before publish. Host shutdown publishes before
  teardown so the stop fits its grace; a send stopped before it ran anything
  identifiable (a Claude send stopped in preflight), or refused for a request
  this process had accepted before, never had a teardown.

``teardown.disconnect`` is free text for logs: ``"ok"``, a failure name, null,
or ``"refused"`` (:data:`REFUSED`). ``"refused"`` marks a send refused as
tombstoned by a process that never accepted that request, so nothing for it
ran there and ``exited`` is ``true``. That rests on two delivery facts (Vex
r2 N1): a request id's ``chat.send`` is published once per dedupe window
(``command_publisher.command_dedupe_ttl_s``), and bridges read commands with
``">"`` only, so a send published before a restart never reaches the new
process. If either broke, a republished send refused by a new openclaw bridge
could mark a run the gateway still holds as exited. No consumer reads ``disconnect`` as
success or failure; only ``exited`` is evidence.

Only ``exited is True`` is evidence that execution stopped
(:attr:`RunTerminal.proves_exit`). A sentinel without the v1 fields comes from a
bridge that predates this contract and parses to ``None``: never evidence.

:meth:`RunTerminal.from_fields` is the one parser, for ``subscribe_run`` (which
yields the parsed terminal as a ``RUN_TERMINAL`` event) and for any reader
taking the sentinel straight off the stream.

Terminal record
---------------
An interactive run stream expires 300s after its ``done``, which is shorter
than a quiesce can wait for a member, so the same fields are also written as a
compact Redis hash at :func:`run_terminal_key` that lives
:data:`RUN_TERMINAL_RETENTION_S` (Vex N4). Quiesce's "not_found and already
terminal" check reads that record with the same parser
(``RunTerminal.from_fields(redis.hgetall(run_terminal_key(rid)))``). Quiesce
caps its timeout at :data:`QUIESCE_TIMEOUT_MAX_S` and polls at most every
:data:`QUIESCE_SWEEP_INTERVAL_MAX_S`, so the record outlives any wait.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .abort_contract import process_boot_id, process_is_pid1

DONE_FIELD = "done"
RUN_TERMINAL_KEY_PREFIX = "agent:run_terminal:"

QUIESCE_TIMEOUT_MAX_S = 1800
"""Upper bound on a quiesce's member timeout; configured values are capped."""
QUIESCE_SWEEP_INTERVAL_MAX_S = 60
"""Longest gap between two quiesce sweeps of the same member."""
RUN_TERMINAL_RETENTION_MARGIN_S = 300
RUN_TERMINAL_RETENTION_S = (
    QUIESCE_TIMEOUT_MAX_S + QUIESCE_SWEEP_INTERVAL_MAX_S + RUN_TERMINAL_RETENTION_MARGIN_S
)
"""How long a run's terminal record stays readable after its ``done``."""


def run_terminal_key(request_id: str) -> str:
    return f"{RUN_TERMINAL_KEY_PREFIX}{request_id}"
_TERMINAL_FIELDS = ("request_id", "backend", "boot_id", "teardown")
_TEARDOWN_KEYS = ("disconnect", "signal", "exited", "elapsed_ms")
_SIGNAL_NONE = "none"
_DISCONNECT_OK = "ok"


class RunContractError(ValueError):
    """A v1 accepted/terminal payload that doesn't match the contract."""


def _optional_str(name: str, value: Any) -> str | None:
    if value is None or isinstance(value, str):
        return value
    raise RunContractError(f"teardown.{name} must be a string or null")


@dataclass(frozen=True)
class TeardownReport:
    """What the bridge did to stop one run's execution, and whether it stopped."""

    disconnect: str | None = None
    signal: str | None = None
    exited: bool | None = None
    elapsed_ms: int | None = field(default=None, compare=False)
    """Wall time spent tearing the run down before ``done``; null when the
    bridge doesn't measure a separate teardown step (it ran inside the run).
    A measurement, not part of what the teardown did, so not compared."""

    def __post_init__(self) -> None:
        _optional_str("disconnect", self.disconnect)
        _optional_str("signal", self.signal)
        if self.exited is not None and not isinstance(self.exited, bool):
            raise RunContractError("teardown.exited must be a bool or null")
        if self.elapsed_ms is not None and (
            isinstance(self.elapsed_ms, bool) or not isinstance(self.elapsed_ms, int)
            or self.elapsed_ms < 0
        ):
            raise RunContractError("teardown.elapsed_ms must be a non-negative int or null")

    def to_dict(self) -> dict[str, Any]:
        return {
            "disconnect": self.disconnect, "signal": self.signal,
            "exited": self.exited, "elapsed_ms": self.elapsed_ms,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "TeardownReport":
        if not isinstance(data, Mapping) or set(data) != set(_TEARDOWN_KEYS):
            raise RunContractError(f"teardown must have exactly {list(_TEARDOWN_KEYS)}")
        return cls(data["disconnect"], data["signal"], data["exited"], data["elapsed_ms"])

    @classmethod
    def combine(cls, reports: Iterable["TeardownReport"]) -> "TeardownReport":
        """One report for a run that tore down several attempts (retries).

        Exited only if every attempt exited; any survivor wins, then any
        unknown. The strongest signal and the first failed disconnect are kept.
        Attempts run one after another, so measured times add up. No attempts
        at all is "not attempted".
        """
        reports = list(reports)
        if not reports:
            return UNATTEMPTED
        exits = [r.exited for r in reports]
        exited = False if False in exits else (None if None in exits else True)
        signals = [r.signal for r in reports if r.signal not in (None, _SIGNAL_NONE)]
        disconnects = [r.disconnect for r in reports if r.disconnect not in (None, _DISCONNECT_OK)]
        timed = [r.elapsed_ms for r in reports if r.elapsed_ms is not None]
        return cls(
            disconnect=disconnects[0] if disconnects else reports[-1].disconnect,
            signal=signals[-1] if signals else reports[-1].signal,
            exited=exited,
            elapsed_ms=sum(timed) if timed else None,
        )


UNATTEMPTED = TeardownReport()
"""All-null: no teardown before publish, so nothing is claimed."""

DISCONNECT_REFUSED = "refused"
REFUSED = TeardownReport(disconnect=DISCONNECT_REFUSED, exited=True)
"""A send refused at receipt by a process that never accepted that request:
nothing for it ever ran there (``SessionQueue.refusal_teardown``)."""


@dataclass(frozen=True)
class RunAccepted:
    request_id: str
    backend: str
    boot_id: str
    pid1: bool | None = None
    """Was the accepting process its container's PID 1 (Vex S2)? ``None``
    from a bridge that predates the field: unknown, so never proof."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id, "backend": self.backend,
            "boot_id": self.boot_id, "pid1": self.pid1,
        }

    @classmethod
    def here(cls, request_id: str, backend: str) -> "RunAccepted":
        return cls(request_id, backend, process_boot_id(), process_is_pid1())

    @classmethod
    def from_dict(cls, data: Any) -> "RunAccepted":
        if not isinstance(data, Mapping):
            raise RunContractError("accepted payload must be an object")
        values = [data.get(name) for name in ("request_id", "backend", "boot_id")]
        if not all(isinstance(v, str) and v.strip() for v in values):
            raise RunContractError("accepted payload needs request_id, backend and boot_id")
        pid1 = data.get("pid1")
        if pid1 is not None and not isinstance(pid1, bool):
            raise RunContractError("accepted pid1 must be a bool or null")
        return cls(*values, pid1)


@dataclass(frozen=True)
class RunTerminal:
    request_id: str
    backend: str
    boot_id: str
    teardown: TeardownReport = UNATTEMPTED

    def __post_init__(self) -> None:
        for name in ("request_id", "backend", "boot_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise RunContractError(f"{name} must be a non-empty string")
        if not isinstance(self.teardown, TeardownReport):
            raise RunContractError("teardown must be a TeardownReport")

    @property
    def proves_exit(self) -> bool:
        """True only when the bridge reaped the run's execution."""
        return self.teardown.exited is True

    @classmethod
    def here(cls, request_id: str, backend: str, teardown: TeardownReport = UNATTEMPTED) -> "RunTerminal":
        return cls(request_id, backend, process_boot_id(), teardown)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id, "backend": self.backend,
            "boot_id": self.boot_id, "teardown": self.teardown.to_dict(),
        }

    def to_fields(self) -> dict[str, str]:
        """Flat str→str Redis stream fields for the ``done`` sentinel."""
        return {
            DONE_FIELD: "1", "request_id": self.request_id, "backend": self.backend,
            "boot_id": self.boot_id, "teardown": json.dumps(self.teardown.to_dict()),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "RunTerminal":
        if not isinstance(data, Mapping) or set(data) != set(_TERMINAL_FIELDS):
            raise RunContractError(f"terminal must have exactly {list(_TERMINAL_FIELDS)}")
        return cls(
            data["request_id"], data["backend"], data["boot_id"],
            TeardownReport.from_dict(data["teardown"]),
        )

    @classmethod
    def from_fields(cls, fields: Mapping[str, Any]) -> "RunTerminal | None":
        """Parse a ``done`` sentinel. ``None`` for a pre-contract (bare) sentinel.

        Raises :class:`RunContractError` for a v1 sentinel that is malformed,
        so a drifting bridge fails loudly instead of reading as "no evidence".
        """
        if not fields.get(DONE_FIELD):
            raise RunContractError("not a done sentinel")
        if not any(name in fields for name in _TERMINAL_FIELDS):
            return None
        try:
            teardown = json.loads(fields.get("teardown", ""))
        except (TypeError, ValueError) as exc:
            raise RunContractError("teardown is not JSON") from exc
        data = {name: fields.get(name) for name in _TERMINAL_FIELDS[:3]}
        data["teardown"] = teardown
        return cls.from_dict(data)


__all__ = [
    "DONE_FIELD", "QUIESCE_SWEEP_INTERVAL_MAX_S", "QUIESCE_TIMEOUT_MAX_S",
    "RUN_TERMINAL_KEY_PREFIX", "RUN_TERMINAL_RETENTION_MARGIN_S", "RUN_TERMINAL_RETENTION_S",
    "DISCONNECT_REFUSED", "REFUSED", "UNATTEMPTED", "RunAccepted", "RunContractError", "RunTerminal", "TeardownReport",
    "run_terminal_key",
]
