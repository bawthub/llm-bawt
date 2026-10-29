"""Prompt-registry text an agent reads about a quiesce job (TASK-1233).

Each body here is seed data only: it is registered in
:mod:`llm_bawt.prompt_definitions` and resolved per bot at call time, so Nick
can see and edit it in BawtHub's Prompt Registry. A blank body or a registry
failure falls back to the seed, so an instruction is never lost.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

RECEIPT_KEY = "ops.quiesce_receipt"
DEFAULT_RECEIPT = (
    "[Ops job receipt: an operation you started finished while your turn was "
    "ended for it. The JSON below is its final receipt. Do not re-run the "
    "operation. Report the outcome, check the target's health if it ran, and "
    "continue any work it was blocking.]"
)

# caller_instruction.message when the terminal receipt arrives as a new turn.
END_TURN_NEW_TURN_KEY = "ops.quiesce_end_turn.new_turn"
DEFAULT_END_TURN_NEW_TURN = (
    "This operation waits for your current turn to exit before it runs. End your turn "
    "now and do not poll in this turn: polling keeps the turn alive until "
    "quiesce_timeout fails the job with no action. The terminal receipt arrives "
    "as a new turn."
)

# caller_instruction.message when nothing will deliver the receipt (a direct
# job whose caller had no session to bind a callback to).
END_TURN_NO_RECEIPT_KEY = "ops.quiesce_end_turn.no_receipt"
DEFAULT_END_TURN_NO_RECEIPT = (
    "This operation waits for your current turn to exit before it runs. End your turn "
    "now and do not poll in this turn: polling keeps the turn alive until "
    "quiesce_timeout fails the job with no action. No receipt will be pushed to you "
    "for this job. To learn its outcome, call ops_job_status with its job_id in a "
    "later turn."
)

# The control steer (B3) the app's quiesce control pump sends into each held
# running turn, at the next tool boundary.
QUIESCE_REQUEST_KEY = "ops.quiesce_request"
DEFAULT_QUIESCE_REQUEST = (
    "[Ops quiesce request: an approved operation (for example a restart of the "
    "service your turn runs on) is waiting for your current turn to end before it "
    "runs. New work is held until it finishes. Bring the step you are on to a safe "
    "stopping point, save or note anything that would be lost, and end your turn "
    "now. Do not start new work and do not poll. If you can, first call "
    "ops_quiesce_ack with status \"stopping\", or \"cannot_stop\" with a short note "
    "saying what you are waiting on. You can pick the work up again in a later turn.]"
)

# TASK-1748: the replacement query when the quiesce request was escalated
# into a running tool. ``{messages}`` is where the request itself goes. It
# must end the turn: the chat interrupt frame says "continue the task".
QUIESCE_INTERRUPT_FRAME_KEY = "ops.quiesce_interrupt_frame"
QUIESCE_INTERRUPT_SLOT = "{messages}"
DEFAULT_QUIESCE_INTERRUPT_FRAME = (
    "[System interruption: the system stopped the tool you were running because an "
    "approved operation could not wait for it to finish. Nobody rejected your work "
    "and this is not a message from the user. The system's request is below.]\n"
    "{messages}\n\n"
    "End your turn now. Do not continue the interrupted task and do not re-run the "
    "tool in this turn; note where you stopped so you can pick it up in a later turn."
)

SEEDS = {
    RECEIPT_KEY: DEFAULT_RECEIPT,
    END_TURN_NEW_TURN_KEY: DEFAULT_END_TURN_NEW_TURN,
    END_TURN_NO_RECEIPT_KEY: DEFAULT_END_TURN_NO_RECEIPT,
    QUIESCE_REQUEST_KEY: DEFAULT_QUIESCE_REQUEST,
    QUIESCE_INTERRUPT_FRAME_KEY: DEFAULT_QUIESCE_INTERRUPT_FRAME,
}


def resolve(key: str, bot_id: str | None) -> str:
    """The live body of ``key`` for ``bot_id``, or its seed on a blank body or failure."""
    try:
        from ..prompt_registry import get_prompt_resolver

        if bot_id:
            resolved = get_prompt_resolver().resolve(key, scope_type="bot", scope_id=bot_id)
        else:
            resolved = get_prompt_resolver().resolve(key)
        if resolved and (resolved.body or "").strip():
            return resolved.body
    except Exception as error:  # registry/DB failure must not lose the instruction
        logger.warning("ops prompt %s resolve failed, using seed: %s", key, error)
    return SEEDS[key]


def render_quiesce_interrupt_frame(bot_id: str | None) -> str:
    """The escalation frame for ``bot_id`` with the bridge's item slot in place.

    A body without the ``{messages}`` slot would drop the request itself, so
    it falls back to the seed rather than send an unframed interrupt.
    """
    from agent_bridge.steer_contract import STEER_ITEMS_SLOT

    body = resolve(QUIESCE_INTERRUPT_FRAME_KEY, bot_id)
    if QUIESCE_INTERRUPT_SLOT not in body:
        logger.warning("ops prompt %s has no %s slot, using seed", QUIESCE_INTERRUPT_FRAME_KEY,
                       QUIESCE_INTERRUPT_SLOT)
        body = DEFAULT_QUIESCE_INTERRUPT_FRAME
    return body.replace(QUIESCE_INTERRUPT_SLOT, STEER_ITEMS_SLOT, 1)


__all__ = [
    "DEFAULT_END_TURN_NEW_TURN", "DEFAULT_END_TURN_NO_RECEIPT", "DEFAULT_QUIESCE_INTERRUPT_FRAME",
    "DEFAULT_QUIESCE_REQUEST", "DEFAULT_RECEIPT", "END_TURN_NEW_TURN_KEY", "END_TURN_NO_RECEIPT_KEY",
    "QUIESCE_INTERRUPT_FRAME_KEY", "QUIESCE_INTERRUPT_SLOT", "QUIESCE_REQUEST_KEY", "RECEIPT_KEY", "SEEDS",
    "render_quiesce_interrupt_frame", "resolve",
]
