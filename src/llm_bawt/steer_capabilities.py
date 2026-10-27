"""What a run's bridge can do with a steer (TASK-1243).

A steer is only as capable as the run it addresses. The capabilities are
decided once, when ``bind_agent_bridge`` binds the run, and stored with it
(``turn_logs.agent_steer_caps``, copied to ``gate_members``). Every steer
decision reads the run's own pinned value: the delivery claim, the steer
routes, the quiesce producer and the control lane. Changing a bot's backend
or its config mid-run changes the next run, never the one already running.

Capabilities:

``next``
    The run accepts a message queued for its next model step
    (``priority=next``). Without it nothing about steering is supported.
``now``
    The run accepts ``priority=now``: interrupt the current step and run the
    message as its replacement.
``escalate``
    A queued ``next`` message can be promoted to an interrupt
    (``/v1/chat/steer/escalate``).
``cancel_queued``
    A queued ``next`` message can be withdrawn (``/v1/chat/steer/cancel``).

The stored form is the sorted, comma-joined names. ``NULL`` is a row bound
before TASK-1243 recorded capabilities: only the claude-code bridge could
steer then, so it means every capability on claude-code and none elsewhere.
"""
from __future__ import annotations

from typing import Any, Iterable

from agent_bridge.steer_contract import STEER_PRIORITY_UNSUPPORTED, STEER_UNSUPPORTED_TRANSPORT

NEXT = "next"
NOW = "now"
ESCALATE = "escalate"
CANCEL_QUEUED = "cancel_queued"
ALL_STEER_CAPS = frozenset({NEXT, NOW, ESCALATE, CANCEL_QUEUED})
NO_STEER_CAPS: frozenset[str] = frozenset()

# The one backend whose runs could steer before capabilities were recorded.
# The product's own backend name, not deployment data: it decodes legacy rows.
LEGACY_STEERABLE_BACKEND = "claude-code"


def encode_caps(caps: Iterable[str]) -> str:
    """The stored form: known names, sorted and comma-joined ('' for none)."""
    return ",".join(sorted(set(caps) & ALL_STEER_CAPS))


def decode_caps(raw: str | None) -> frozenset[str]:
    """Known capability names in a stored value; unknown names are dropped."""
    names = (part.strip() for part in (raw or "").split(","))
    return frozenset(name for name in names if name in ALL_STEER_CAPS)


def run_steer_caps(
    *, backend: str | None, caps: str | None, provisional: frozenset[str] | None = None,
) -> frozenset[str]:
    """The capabilities of one run, from its pinned binding.

    ``caps`` is the run's stored value. ``None`` on a bound run (``backend``
    set) is a pre-TASK-1243 row. A run with no binding yet has no pinned
    value: ``provisional`` is what its bot would bind now (default none). It
    is only a guess, so a bridge that cannot serve a steer still refuses it
    (``steer_unsupported_transport``) before anything reaches the model.
    """
    if caps is not None:
        return decode_caps(caps)
    if backend:
        return ALL_STEER_CAPS if backend.strip().lower() == LEGACY_STEERABLE_BACKEND else NO_STEER_CAPS
    return provisional if provisional is not None else NO_STEER_CAPS


def steer_refusal(caps: frozenset[str], capability: str) -> str | None:
    """The definitive refusal code for asking ``capability`` of a run, else None.

    ``steer_unsupported_transport`` when the run cannot steer at all,
    ``steer_priority_unsupported`` when it steers but not this way.
    """
    if NEXT not in caps:
        return STEER_UNSUPPORTED_TRANSPORT
    if capability not in caps:
        return STEER_PRIORITY_UNSUPPORTED
    return None


def bot_steer_caps(bot: Any) -> frozenset[str]:
    """What a run of ``bot`` would bind now: its backend's answer for its config."""
    from .agent_backends.registry import get_backend

    name = (getattr(bot, "agent_backend", None) or "").strip()
    backend = get_backend(name) if name else None
    if backend is None:
        return NO_STEER_CAPS
    return frozenset(backend.steer_capabilities(dict(getattr(bot, "agent_backend_config", None) or {})))


def turn_steer_caps(turn: Any, *, bot_backend: str) -> tuple[str, frozenset[str]]:
    """``(backend, caps)`` of a turn row's bound run.

    The row's own ``agent_backend`` decides; ``bot_backend`` is used only for
    a row bound before TASK-1232 recorded the backend.
    """
    backend = (getattr(turn, "agent_backend", None) or "").strip() or bot_backend.strip()
    return backend, run_steer_caps(backend=backend, caps=getattr(turn, "agent_steer_caps", None))


__all__ = [
    "ALL_STEER_CAPS",
    "CANCEL_QUEUED",
    "ESCALATE",
    "LEGACY_STEERABLE_BACKEND",
    "NEXT",
    "NOW",
    "NO_STEER_CAPS",
    "bot_steer_caps",
    "decode_caps",
    "encode_caps",
    "run_steer_caps",
    "steer_refusal",
    "turn_steer_caps",
]
