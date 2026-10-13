"""Transient per-turn phase for live UI status (TASK-1074).

One :class:`TurnPhaseTracker` per send request owns "what is the turn doing
right now?" — waiting on the model, reasoning, compacting, retrying, etc. It
publishes ``UPSTREAM_STATUS`` events only on transitions. Phase is display
state: it is never appended to the response text, persisted to history, or
replayed. The app keeps only the last phase per active turn (with a TTL) so a
client joining mid-turn can render it.

The tracker never tracks tools itself: the :class:`TurnWatchdog` already owns
the open-tool set, and callers pass that in.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import StrEnum
from typing import Any

# No SDK message for this long while the model (not a tool) owns the turn
# reads as a stall. Detection resolution is the watchdog health interval.
STALL_AFTER_SECONDS = 90.0

PhasePublisher = Callable[[str, dict[str, Any]], None]


class TurnPhase(StrEnum):
    WAITING_MODEL = "waiting_model"
    REASONING = "reasoning"
    COMPACTING = "compacting"
    STALLED = "stalled"
    RETRYING = "retrying"
    RECONNECTING = "reconnecting"
    STREAMING = "streaming"


PHASE_TEXT: dict[TurnPhase, str] = {
    TurnPhase.WAITING_MODEL: "Waiting on model",
    TurnPhase.REASONING: "Thinking",
    TurnPhase.COMPACTING: "Compacting conversation",
    TurnPhase.STALLED: "Model stalled",
    TurnPhase.RETRYING: "Retrying model request",
    TurnPhase.RECONNECTING: "Upstream stalled; reconnecting",
    TurnPhase.STREAMING: "Streaming",
}

# Phases that already explain a silent SDK; a quiet stretch is not a stall.
_SILENCE_EXPLAINED = frozenset(
    {TurnPhase.COMPACTING, TurnPhase.RETRYING, TurnPhase.RECONNECTING, TurnPhase.STALLED}
)

# Proxy ``state`` values (adapters/base.py, chatgpt_transport.py) → phase.
_PROXY_STATES: dict[str, TurnPhase] = {
    "reconnecting": TurnPhase.RECONNECTING,
    "recovered": TurnPhase.STREAMING,
}

# SDK stream content blocks → phase.
_BLOCK_PHASES: dict[str, TurnPhase] = {
    "thinking": TurnPhase.REASONING,
    "redacted_thinking": TurnPhase.REASONING,
    "text": TurnPhase.STREAMING,
    "tool_use": TurnPhase.STREAMING,
}


class TurnPhaseTracker:
    """Single owner of one request's transient phase."""

    def __init__(
        self,
        publish: PhasePublisher,
        *,
        model: str | None = None,
        context_tokens: Callable[[], int | None] | None = None,
        stall_after: float = STALL_AFTER_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._publish = publish
        self.model = model
        self._context_tokens = context_tokens
        self._stall_after = stall_after
        self._clock = clock
        self.phase: TurnPhase | None = None
        self.since: float | None = None
        self._detail: dict[str, Any] = {}
        self._resume_phase: TurnPhase | None = None

    # ── core ────────────────────────────────────────────────────────────

    def transition(self, phase: TurnPhase, *, text: str | None = None, **detail: Any) -> bool:
        """Enter ``phase`` (or refresh its detail); publish only on change."""
        detail = {k: v for k, v in detail.items() if v is not None}
        if phase == self.phase and detail == self._detail:
            return False
        if phase != self.phase:
            self.phase = phase
            self.since = self._clock()
        self._detail = detail
        self._publish(text or PHASE_TEXT[phase], self.snapshot())
        return True

    def snapshot(self) -> dict[str, Any]:
        """The ``upstream_status`` payload for the current phase."""
        assert self.phase is not None
        context_tokens = self._context_tokens() if self._context_tokens else None
        return {
            **self._detail,
            "phase": self.phase.value,
            # Back-compat: pre-phase clients switch on ``state``.
            "state": self.phase.value,
            "since": self.since,
            "model": self.model,
            "context_tokens": context_tokens,
        }

    # ── SDK observations ────────────────────────────────────────────────

    def on_turn_start(self) -> None:
        self.transition(TurnPhase.WAITING_MODEL)

    def on_stream_event(self, event: dict) -> None:
        if event.get("type") != "content_block_start":
            return
        block_type = (event.get("content_block") or {}).get("type")
        phase = _BLOCK_PHASES.get(block_type or "")
        if phase is not None:
            self.transition(phase)

    def on_tool_results(self, open_tools: int) -> None:
        """All tool results are in: the next model call is pending."""
        if open_tools == 0:
            self.transition(TurnPhase.WAITING_MODEL)

    def on_system_status(self, data: dict) -> None:
        """SDK ``SystemMessage`` compaction lifecycle."""
        if data.get("status") == "compacting":
            self.transition(TurnPhase.COMPACTING)
        elif data.get("compact_result") and self.phase == TurnPhase.COMPACTING:
            # Auto-compact resumes the model call; /compact ends right after.
            self.transition(TurnPhase.WAITING_MODEL)

    def on_api_retry(self, *, attempt: Any, max_attempts: Any, error_status: Any) -> None:
        """A user-visible SDK API retry (the caller filters silent auth retries)."""
        self.transition(
            TurnPhase.RETRYING,
            attempt=attempt,
            max_attempts=max_attempts,
            error_status=error_status,
        )

    def set_context_source(self, source: Callable[[], int | None] | None) -> None:
        """Bind the per-attempt live usage publisher's resident tokens."""
        self._context_tokens = source

    def on_proxy_status(self, status: dict) -> None:
        """In-process proxy reconnect/recover signals."""
        phase = _PROXY_STATES.get(str(status.get("state") or ""))
        if phase is None:
            return
        detail = {
            k: status.get(k)
            for k in (
                "attempt", "next_attempt", "max_attempts", "transport",
                "fallback_transport", "stall_phase", "productive_idle_seconds",
                "elapsed_seconds", "provider",
            )
        }
        self.transition(phase, text=status.get("message") or None, **detail)

    def on_health(self, health: dict) -> None:
        """Watchdog tick while waiting on the SDK: detect a model stall."""
        if health.get("phase") != "sdk_idle":
            return  # a tool owns the turn; silence is expected
        if self.phase is None or self.phase in _SILENCE_EXPLAINED:
            return
        waited = health.get("wait_elapsed_seconds")
        if isinstance(waited, (int, float)) and waited >= self._stall_after:
            self._resume_phase = self.phase
            self.transition(TurnPhase.STALLED)

    def on_sdk_message(self) -> None:
        """Any SDK message ends a stall; resume the phase it interrupted.

        Call before the type-specific observers so a fresh block start can
        still move on from the resumed phase.
        """
        if self.phase == TurnPhase.STALLED:
            self.transition(self._resume_phase or TurnPhase.WAITING_MODEL)
