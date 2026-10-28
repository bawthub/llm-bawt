"""The ``run.status`` RPC: where one request is on one bridge (TASK-1232).

The app asks a bridge about a request it can no longer see executing (startup
recovery, quiesce). Only the bridge the request was addressed to can say the
run is absent, so the RPC is answered only when the caller names the backend:
a backend-less ``run.status`` gets no reply and times out, which the caller
must treat as ambiguous.

Reply (:class:`RunStatusReply`, v1)::

    {backend, boot_id, request_id, state, detail, pid1}

``pid1`` (TASK-1233 Vex S2) says whether the answering process is its
container's PID 1. It is optional on the wire: a bridge that predates it
parses to ``None`` (unknown). :func:`proves_restart` is the B1(b) rule built
on it, against a durable :class:`RunSighting` of the run on its earlier boot.

``state`` is a :class:`RunState`. Only ``not_found`` from the addressed
backend is evidence of absence, and only for this ``boot_id``: tombstones and
registrations live in process memory. A send published but not yet read by the
bridge is invisible to it; the command stream is read in order, so a
``run.status`` sent after the ``chat.send`` cannot overtake it.

:meth:`RunStatusReply.from_wire` is the one parser for callers. A reply with
no v1 field comes from a bridge that predates the RPC (it answers
``unknown method``) and maps to ``unsupported``, never to an absence.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from .abort_contract import process_boot_id, process_is_pid1
from .run_contract import RunAccepted
from .session_queue import RequestPhase

if TYPE_CHECKING:
    from .session_queue import SessionQueue

RUN_STATUS_METHOD = "run.status"


class RunState(str, Enum):
    """Where one request is on the answering bridge process."""

    PREFLIGHT = RequestPhase.PREFLIGHT.value
    """Received and registered; running its one-time preparation."""
    QUEUED = RequestPhase.QUEUED.value
    """Registered, waiting for the session lock behind another send."""
    RUNNING = RequestPhase.RUNNING.value
    """Holds the session lock; the backend is executing it."""
    STOPPING = "stopping"
    """A Stop is in progress: cancelled locally, or awaiting the backend's abort."""
    TOMBSTONED = "tombstoned"
    """Not registered, and a Stop for this id was recorded by this process."""
    NOT_FOUND = "not_found"
    """Neither registered nor tombstoned in this process."""
    ERROR = "error"
    """The bridge could not answer; see ``detail``."""
    UNSUPPORTED = "unsupported"
    """Parser-only: the bridge predates ``run.status``. Never sent by a bridge."""


_REPLY_FIELDS = ("backend", "boot_id", "request_id", "state", "detail")
_OPTIONAL_FIELDS = ("pid1",)
_V1_MARKERS = ("state", "boot_id")
_UNKNOWN_REQUEST_ID = "unknown"


class RunStatusReplyError(ValueError):
    """A ``run.status`` reply that doesn't match :class:`RunStatusReply`."""


@dataclass(frozen=True)
class RunStatusReply:
    backend: str
    boot_id: str
    request_id: str
    state: RunState
    detail: str = ""
    pid1: bool | None = None

    def __post_init__(self) -> None:
        for name in ("backend", "boot_id", "request_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise RunStatusReplyError(f"{name} must be a non-empty string")
        if not isinstance(self.state, RunState):
            raise RunStatusReplyError("state must be a RunState")
        if not isinstance(self.detail, str):
            raise RunStatusReplyError("detail must be a string")
        if self.pid1 is not None and not isinstance(self.pid1, bool):
            raise RunStatusReplyError("pid1 must be a bool or null")

    @property
    def proves_absent(self) -> bool:
        """True only for ``not_found``; the parser already checked the backend.

        Absence of the *send task*, not of its side effects: until TASK-1232
        P4 (request-owned teardown) lands, claude-code can answer
        ``not_found`` while the SDK subprocess of a cancelled send is still
        tearing down. A caller that reclaims resources on ``not_found``
        (e.g. an orphan-turn probe) must not assume the process is gone.
        """
        return self.state is RunState.NOT_FOUND

    def to_dict(self) -> dict[str, Any]:
        data = {name: getattr(self, name) for name in (*_REPLY_FIELDS, *_OPTIONAL_FIELDS)}
        data["state"] = self.state.value
        return data

    @classmethod
    def from_wire(cls, data: Mapping[str, Any], *, backend: str) -> "RunStatusReply":
        """Parse the reply from the bridge for *backend*.

        A reply with any v1 field is parsed strictly (exact keys, a sendable
        state, matching backend) and raises :class:`RunStatusReplyError`
        otherwise. A reply with none is a pre-contract bridge: ``unsupported``.
        """
        if not isinstance(data, Mapping):
            raise RunStatusReplyError("run.status reply must be an object")
        if not any(key in data for key in _V1_MARKERS):
            error = data.get("error")
            return cls(
                backend=backend, boot_id="legacy", request_id=_UNKNOWN_REQUEST_ID,
                state=RunState.UNSUPPORTED,
                detail=str(error) if error else "unrecognized_legacy_reply",
            )
        keys = set(data)
        missing = set(_REPLY_FIELDS) - keys
        extra = keys - set(_REPLY_FIELDS) - set(_OPTIONAL_FIELDS)
        if missing or extra:
            raise RunStatusReplyError(
                f"run.status reply keys mismatch: missing={sorted(missing)} extra={sorted(extra)}"
            )
        try:
            state = RunState(data["state"])
        except ValueError as exc:
            raise RunStatusReplyError(f"unknown run state: {data['state']!r}") from exc
        if state is RunState.UNSUPPORTED:
            raise RunStatusReplyError("a bridge never sends 'unsupported'")
        reply = cls(
            backend=data["backend"], boot_id=data["boot_id"],
            request_id=data["request_id"], state=state, detail=data["detail"],
            pid1=data.get("pid1"),
        )
        if reply.backend != backend:
            raise RunStatusReplyError(
                f"run.status reply from backend {reply.backend!r}, addressed {backend!r}"
            )
        return reply


RunStateFn = Callable[[str, str], RunState]


def queue_run_state(queue: "SessionQueue", session_key: str, request_id: str) -> RunState:
    """This request's state from the session queue alone."""
    phase = queue.request_phase(session_key, request_id)
    if phase is not None:
        if queue.is_stopping(session_key, request_id):
            return RunState.STOPPING
        return RunState(phase.value)
    if queue.is_tombstoned(session_key, request_id):
        return RunState.TOMBSTONED
    return RunState.NOT_FOUND


def run_status_target(params: Mapping[str, Any]) -> tuple[str, str]:
    """Return ``(session_key, request_id)`` from the RPC params."""
    if not isinstance(params, Mapping):
        raise ValueError("run.status params must be an object")
    session_key, request_id = params.get("sessionKey"), params.get("requestId")
    if (
        not isinstance(session_key, str) or not session_key.strip()
        or not isinstance(request_id, str) or not request_id.strip()
    ):
        raise ValueError("run.status requires sessionKey and requestId")
    return session_key, request_id


def answer_run_status(backend: str, params: Mapping[str, Any], state_of: RunStateFn) -> dict[str, Any]:
    """The v1 reply for one ``run.status``.

    Bad params, a raising lookup, or a lookup returning anything but a
    sendable :class:`RunState` all answer ``error`` for the named request;
    only an invalid ``backend`` (a bridge bug) raises.
    """
    named = params.get("requestId") if isinstance(params, Mapping) else None
    request_id = named.strip() if isinstance(named, str) and named.strip() else _UNKNOWN_REQUEST_ID
    boot_id = process_boot_id()
    try:
        session_key, request_id = run_status_target(params)
        state = state_of(session_key, request_id)
        if state is RunState.UNSUPPORTED:
            raise RunStatusReplyError("unsupported is parser-only, never sent")
        return RunStatusReply(backend, boot_id, request_id, state, pid1=process_is_pid1()).to_dict()
    except Exception as exc:
        return RunStatusReply(
            backend, boot_id, request_id, RunState.ERROR, f"{type(exc).__name__}: {exc}",
            pid1=process_is_pid1(),
        ).to_dict()


RESTART_PROOF_BACKENDS = frozenset({"claude-code", "codex"})
"""Backends whose run processes are the bridge process's own children.

Only for these can a bridge restart kill a run: claude-code's CLI and codex's
per-turn ``codex`` subprocess live in the bridge container's PID namespace
(TASK-1233). Any backend not named here never proves a restart."""

LIVE_STATES = frozenset({RunState.PREFLIGHT, RunState.QUEUED, RunState.RUNNING, RunState.STOPPING})
"""States in which the answering process holds the request in its registry."""

ABSENT_STATES = frozenset({RunState.NOT_FOUND, RunState.TOMBSTONED})


@dataclass(frozen=True)
class RunSighting:
    """A durable record that one process held one request (TASK-1233 B1(b)).

    Built from the RUN_ACCEPTED payload (:meth:`from_accepted`) or from a
    ``run.status`` reply in a :data:`LIVE_STATES` state (:meth:`from_reply`):
    the request is registered only in the memory of the process that accepted
    it, so a live answer names the accepting boot. ``pid1`` is ``None`` when
    the source predates the field: unknown, so never proof.
    """

    request_id: str
    backend: str
    boot_id: str
    pid1: bool | None = None

    def __post_init__(self) -> None:
        for name in ("request_id", "backend", "boot_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise RunStatusReplyError(f"sighting {name} must be a non-empty string")
        if self.pid1 is not None and not isinstance(self.pid1, bool):
            raise RunStatusReplyError("sighting pid1 must be a bool or null")

    @classmethod
    def from_accepted(cls, accepted: RunAccepted) -> "RunSighting":
        return cls(accepted.request_id, accepted.backend, accepted.boot_id, accepted.pid1)

    @classmethod
    def from_reply(cls, reply: RunStatusReply) -> "RunSighting":
        if reply.state not in LIVE_STATES:
            raise RunStatusReplyError(f"a {reply.state.value} reply does not sight the run")
        return cls(reply.request_id, reply.backend, reply.boot_id, reply.pid1)


def proves_restart(sighting: RunSighting, reply: RunStatusReply) -> bool:
    """TASK-1233 B1(b): does *reply* prove the sighted run's process is gone?

    Only when all of these hold:

    * the backend is in :data:`RESTART_PROOF_BACKENDS` (its runs die with the
      bridge);
    * the same backend answers for the same request from a different boot and
      says the request is absent there (``not_found``/``tombstoned``);
    * BOTH the sighted and the answering process were their container's PID 1.
      A PID 1 exit kills the container's whole PID namespace, so the sighted
      boot's children went with it. Without PID 1 on either side (or unknown),
      a new boot proves nothing about surviving children.

    The sighting must come from a durable record (Snark step 34), never be
    inferred; that is the caller's contract.
    """
    return (
        sighting.backend.lower() in RESTART_PROOF_BACKENDS
        and reply.backend.lower() == sighting.backend.lower()
        and reply.request_id == sighting.request_id
        and reply.boot_id != sighting.boot_id
        and reply.state in ABSENT_STATES
        and sighting.pid1 is True
        and reply.pid1 is True
    )


__all__ = [
    "ABSENT_STATES", "LIVE_STATES", "RESTART_PROOF_BACKENDS", "RUN_STATUS_METHOD", "RunSighting",
    "RunState", "RunStateFn", "RunStatusReply", "RunStatusReplyError", "answer_run_status",
    "proves_restart", "queue_run_state", "run_status_target",
]
