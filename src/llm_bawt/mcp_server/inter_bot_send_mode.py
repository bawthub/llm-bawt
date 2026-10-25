"""Single resolver for ``bots_send_message`` mode flags (TASK-1015, TASK-1431).

Every send is durable. The flags only choose three orthogonal things:

* **delivery** — ``steer_or_idle`` (steer an active Claude turn, else one idle
  turn), ``when_idle`` (never steer; one separate idle turn) or ``steer_only``
  (steer an active Claude turn, else settle as not needed; never a turn).
* **wait** — whether the caller blocks (bounded) for the target turn's reply.
* **addressing** — exactly one of ``target_bot_id`` or ``targets``
  (:func:`require_one_recipient`).

Conflicting combinations are rejected here, before any side effect. The
2026-10-04 incident (``delivery="when_idle"`` + ``fire_and_forget=False``
silently selected a non-durable synchronous path that ignored the idempotency
key) is exactly what this module exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..inter_bot_steer_only import STEER_ONLY, STEER_OR_IDLE, WHEN_IDLE

__all__ = [
    "STEER_ONLY", "STEER_OR_IDLE", "WHEN_IDLE", "SendMode", "SendModeError",
    "require_one_recipient", "resolve_send_mode",
]

_DELIVERY_ALIASES = {
    "steer_or_idle": STEER_OR_IDLE,
    "immediate": STEER_OR_IDLE,
    "when_idle": WHEN_IDLE,
    "queued": WHEN_IDLE,
    "queue_if_busy": WHEN_IDLE,
    "steer_only": STEER_ONLY,
}


class SendModeError(ValueError):
    """Flags are contradictory or unsafe; nothing was dispatched."""


@dataclass(frozen=True)
class SendMode:
    delivery: str
    wait: bool

    @property
    def prefer_steer(self) -> bool:
        return self.delivery in (STEER_OR_IDLE, STEER_ONLY)

    @property
    def steer_only(self) -> bool:
        return self.delivery == STEER_ONLY

    @property
    def label(self) -> str:
        return "waited" if self.wait else "async"


def require_one_recipient(target_bot_id: str | None, targets: Any) -> None:
    """Exactly one of ``target_bot_id`` or ``targets`` (design r3 §7)."""
    has_target = bool((target_bot_id or "").strip())
    if has_target == (targets is not None):
        raise SendModeError(
            "Pass exactly one of target_bot_id or targets."
            if has_target else "Name a recipient: target_bot_id or targets."
        )


def resolve_send_mode(
    *,
    delivery: str | None,
    queue_if_busy: bool,
    fire_and_forget: bool | None,
    wait_for_reply: bool,
    task_id: str | None,
    idempotency_key: str | None,
) -> SendMode:
    """Return the one effective mode or raise :class:`SendModeError`."""
    explicit_delivery: str | None = None
    if delivery is not None and str(delivery).strip():
        normalized = str(delivery).strip().lower().replace("-", "_")
        explicit_delivery = _DELIVERY_ALIASES.get(normalized)
        if explicit_delivery is None:
            raise SendModeError(
                f"Unknown delivery mode '{delivery}'. "
                "Use 'steer_or_idle', 'when_idle' or 'steer_only'."
            )

    if wait_for_reply and fire_and_forget is True:
        raise SendModeError(
            "wait_for_reply=true contradicts fire_and_forget=true; pass only one."
        )
    if queue_if_busy and explicit_delivery in (STEER_OR_IDLE, STEER_ONLY):
        raise SendModeError(
            f"queue_if_busy=true contradicts delivery='{delivery}'; "
            "use delivery='when_idle' alone."
        )

    wait = bool(wait_for_reply or fire_and_forget is False)
    resolved_delivery = (
        WHEN_IDLE if queue_if_busy else explicit_delivery or STEER_OR_IDLE
    )

    if wait:
        if explicit_delivery in (STEER_OR_IDLE, STEER_ONLY):
            raise SendModeError(
                "A waited send needs its own target turn; it cannot steer. "
                "Drop wait_for_reply/fire_and_forget=false, or use delivery='when_idle'."
            )
        # A waited reply only exists for a separate turn.
        resolved_delivery = WHEN_IDLE
        if task_id and not (idempotency_key or "").strip():
            raise SendModeError(
                "Task handoffs must not use an unkeyed waited send. Use the default "
                "async mode, or pass a stable idempotency_key (e.g. 'TASK-N:START') "
                "so an interrupted wait can be retried without a second target turn."
            )
    return SendMode(delivery=resolved_delivery, wait=wait)
