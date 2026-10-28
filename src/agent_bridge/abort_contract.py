"""Bridge ↔ app contract for stopping a run (TASK-1231B, TASK-1232 P3).

One module that every bridge and the app import, so the shapes below are
defined exactly once.

Terminal event cause (TASK-1231B)
---------------------------------
A cancelled run still ends with ``raw.end_reason == "aborted"``: that value is
what the app's stream consumers switch on to finalize a turn as aborted rather
than completed (``agent_backends/agent_bridge.py``,
``service/turn_stream_worker.py``). Changing it would make a killed turn
look like a finished reply. The *cause* is therefore an additive field,
``raw.abort_cause``, present only when the bridge knows the run was cancelled
for a reason other than an abort/cancel RPC. Today that is only
:data:`HOST_SHUTDOWN`: the bridge process received SIGTERM/SIGINT and cancelled
its own queued or running sends. Absence means "no machine-attributable
cause"; consumers must not infer a restart from it.

Normalized abort reply (TASK-1232 P3)
-------------------------------------
``chat.abort`` replies from every bridge use
:class:`AbortReply`. :meth:`AbortReply.from_wire` is the one parser the app
uses. It is strict (:meth:`AbortReply.from_dict`), so a bridge drifting from
the contract fails loudly instead of being read as "not confirmed". Every
bridge answers v1 since TASK-1232 P3; the pre-contract adapter was removed
(TASK-1232 #21).
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

_PROCESS_BOOT_ID = uuid.uuid4().hex


def process_boot_id() -> str:
    """Return this bridge process's boot id (fixed for the process lifetime)."""
    return _PROCESS_BOOT_ID


_PROCESS_IS_PID1 = os.getpid() == 1


def process_is_pid1() -> bool:
    """Is this bridge process its container's PID 1? (TASK-1233 Vex S2)

    Only then does a new boot id prove the old process's children are gone:
    when PID 1 exits the kernel kills the whole PID namespace. A bridge run
    under a supervisor or a shell can restart while its CLI children survive.
    """
    return _PROCESS_IS_PID1


END_REASON_ABORTED = "aborted"
"""Wire ``end_reason`` for every cancelled run, whatever the cause."""

HOST_SHUTDOWN = "host_shutdown"
"""``abort_cause`` stamped when the bridge host itself is shutting down."""

ABORT_CAUSE_KEY = "abort_cause"

ABORT_CAUSES = frozenset({HOST_SHUTDOWN})

# -- Teardown bound (TASK-1232 P4, S3) ----------------------------------------
# A Stop's chat.abort reply is published only after the run's teardown has
# finished or hit this bound, so the app's RPC timeout (bound + margin) always
# outlives it. The bridge's teardown takes its waits from these constants.

TEARDOWN_DISCONNECT_BACKSTOP_S = 22.0
"""Cap on the SDK's own ``disconnect()``. claude-agent-sdk 0.2.160 escalates
stdin EOF -> SIGTERM -> SIGKILL with 5s waits, plus up to 5s for the stdin
write lock: 20s at most."""

TEARDOWN_KILL_WAIT_S = 2.0
"""Reap wait after the bridge SIGKILLs a CLI process tree that survived
``disconnect()``."""

ABORT_TEARDOWN_BOUND_S = 25.0
"""Upper bound on one run's teardown: backstop + kill wait + 1s slack."""

ABORT_RPC_MARGIN_S = 5.0
"""How long the app waits past ``ABORT_TEARDOWN_BOUND_S`` for the reply."""

ABORT_RPC_TIMEOUT_S = ABORT_TEARDOWN_BOUND_S + ABORT_RPC_MARGIN_S
"""The app's ``chat.abort`` RPC timeout. Every caller uses this one value so
the app never gives up just as the bridge's teardown reply arrives."""


def aborted_terminal_raw(cause: str | None = None) -> dict[str, str]:
    """Return the ``raw`` payload for a cancelled run's terminal event.

    The no-cause shape is byte-for-byte the historical
    ``{"end_reason": "aborted", "status": "cancelled"}``.
    """
    raw = {"end_reason": END_REASON_ABORTED, "status": "cancelled"}
    if cause is not None:
        if cause not in ABORT_CAUSES:
            raise ValueError(f"unknown abort cause: {cause!r}")
        raw[ABORT_CAUSE_KEY] = cause
    return raw


def abort_cause_of(raw: Mapping[str, Any] | None) -> str | None:
    """Return a terminal event's known abort cause, or ``None``.

    Only an aborted terminal can carry a cause; an unknown value is ignored
    rather than trusted.
    """
    if not isinstance(raw, Mapping) or raw.get("end_reason") != END_REASON_ABORTED:
        return None
    cause = raw.get(ABORT_CAUSE_KEY)
    return cause if cause in ABORT_CAUSES else None


class AbortState(str, Enum):
    """Outcome of one ``chat.abort`` for one request id."""

    CANCELLED = "cancelled"
    """A queued or running send for this request was cancelled now."""
    NOT_FOUND = "not_found"
    """The bridge has no queued or running send for this request id."""
    TOMBSTONED = "tombstoned"
    """Not running yet; the id is recorded so a later send is refused."""
    ERROR = "error"
    """The bridge could not determine or perform the abort; see ``detail``."""


_REPLY_FIELDS = ("backend", "boot_id", "request_id", "cancelled", "state", "detail")

CONTRACT_V1 = "v1"


class AbortReplyError(ValueError):
    """A ``chat.abort`` reply that doesn't match :class:`AbortReply`."""


@dataclass(frozen=True)
class AbortReply:
    """The normalized ``chat.abort`` RPC result.

    ``cancelled`` is true exactly when ``state`` is ``cancelled``; it is kept
    as its own field so callers can branch on a bool without knowing the enum.
    ``boot_id`` identifies the bridge process that answered, so a reply from a
    process that has since restarted can be told apart from the current one.
    ``contract`` records which wire shape was parsed (only ``v1`` exists) for
    the Stop audit; it is never on the wire.
    """

    backend: str
    boot_id: str
    request_id: str
    cancelled: bool
    state: AbortState
    detail: str = ""
    contract: str = field(default=CONTRACT_V1, compare=False)

    def __post_init__(self) -> None:
        for name in ("backend", "boot_id", "request_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise AbortReplyError(f"{name} must be a non-empty string")
        if not isinstance(self.cancelled, bool):
            raise AbortReplyError("cancelled must be a bool")
        if not isinstance(self.state, AbortState):
            raise AbortReplyError("state must be an AbortState")
        if not isinstance(self.detail, str):
            raise AbortReplyError("detail must be a string")
        if self.cancelled != (self.state is AbortState.CANCELLED):
            raise AbortReplyError("cancelled must be true exactly when state is cancelled")
        if self.contract != CONTRACT_V1:
            raise AbortReplyError(f"unknown contract: {self.contract!r}")

    @classmethod
    def build(
        cls,
        *,
        backend: str,
        boot_id: str,
        request_id: str,
        state: AbortState,
        detail: str = "",
    ) -> "AbortReply":
        """Construct a reply, deriving ``cancelled`` from ``state``."""
        return cls(
            backend=backend,
            boot_id=boot_id,
            request_id=request_id,
            cancelled=state is AbortState.CANCELLED,
            state=state,
            detail=detail,
        )

    def to_dict(self) -> dict[str, Any]:
        """The v1 wire shape: exactly the contract keys (never ``contract``)."""
        data = {name: getattr(self, name) for name in _REPLY_FIELDS}
        data["state"] = self.state.value
        return data

    @classmethod
    def from_wire(cls, data: Mapping[str, Any], *, backend: str) -> "AbortReply":
        """Parse a ``chat.abort`` RPC result from the bridge for *backend*.

        Strict: the reply must be exactly the v1 shape and must name *backend*;
        anything else raises :class:`AbortReplyError`, which the caller records
        as ``error``, never as ``not_found``.
        """
        reply = cls.from_dict(data)
        if reply.backend != backend:
            raise AbortReplyError(
                f"abort reply from backend {reply.backend!r}, addressed {backend!r}"
            )
        return reply

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AbortReply":
        """Strictly parse a reply: exactly the contract keys, valid values."""
        if not isinstance(data, Mapping):
            raise AbortReplyError("abort reply must be an object")
        keys = set(data)
        missing = set(_REPLY_FIELDS) - keys
        extra = keys - set(_REPLY_FIELDS)
        if missing or extra:
            raise AbortReplyError(
                f"abort reply keys mismatch: missing={sorted(missing)} extra={sorted(extra)}"
            )
        try:
            state = AbortState(data["state"])
        except ValueError as exc:
            raise AbortReplyError(f"unknown abort state: {data['state']!r}") from exc
        return cls(
            backend=data["backend"],
            boot_id=data["boot_id"],
            request_id=data["request_id"],
            cancelled=data["cancelled"],
            state=state,
            detail=data["detail"],
        )


__all__ = [
    "ABORT_CAUSES",
    "ABORT_CAUSE_KEY",
    "ABORT_RPC_MARGIN_S",
    "ABORT_RPC_TIMEOUT_S",
    "ABORT_TEARDOWN_BOUND_S",
    "CONTRACT_V1",
    "AbortReply",
    "AbortReplyError",
    "AbortState",
    "END_REASON_ABORTED",
    "HOST_SHUTDOWN",
    "TEARDOWN_DISCONNECT_BACKSTOP_S",
    "TEARDOWN_KILL_WAIT_S",
    "abort_cause_of",
    "aborted_terminal_raw",
    "process_boot_id",
    "process_is_pid1",
]
