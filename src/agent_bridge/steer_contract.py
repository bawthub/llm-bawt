"""Wire contract for app-rendered steer prompts (TASK-1300).

Every model-facing word of a steer comes from the prompt registry
(``chat.steer.next_aside`` and ``chat.steer.interrupt_replacement``). The app
renders them, because only the app can read the registry, and sends the
results on ``chat.steer``. The bridge adds no wording of its own.

``chat.steer`` fields:

``framed_message``
    The exact text the model receives. On ``priority=next`` it is the CLI
    command content. On ``priority=now`` it is the whole replacement query.
``interrupt_item``
    ``priority=next`` only: this message rendered as one item of an
    interruption. If an escalation later sweeps the command, the bridge replays
    this string. It is rendered when the steer is sent because only the app
    knows the sender's label.
``interrupt_frame``
    Escalation only: the ``chat.steer.interrupt_replacement`` frame, rendered
    with :data:`STEER_ITEMS_SLOT` in the place of its item list. The bridge
    substitutes the swept items, in send order, joined with
    :data:`STEER_ITEMS_SEPARATOR`.

``system``-origin steers (the quiesce request) are already final registry
text: ``message`` is sent verbatim and the fields are optional. For any other
origin a bridge that gets no framing refuses with :data:`STEER_UNFRAMED`
before anything reaches the CLI.
"""

from __future__ import annotations

FRAMED_MESSAGE_FIELD = "framed_message"
INTERRUPT_ITEM_FIELD = "interrupt_item"
INTERRUPT_FRAME_FIELD = "interrupt_frame"

STEER_ITEMS_SLOT = "<<steer_items>>"
STEER_ITEMS_SEPARATOR = "\n\n"

STEER_UNFRAMED = "steer_unframed"

# The run already handed its final ResultMessage to the send handler and is
# finalizing or tearing down (TASK-1344). Nothing can serve a steer there: the
# CLI would run it as an unseen follow-up. Raised before anything reaches the
# CLI, so it is a definite non-delivery; callers send the message as a fresh
# turn, the same as a steer to an idle bot.
STEER_RUN_FINISHING = "run_finishing"

# TASK-1243: the run's transport cannot steer at all (a codex exec run), or
# can steer but not with the asked priority (``now``, escalate, cancel of a
# queued message). Refused before anything reaches the model, by the app from
# the run's pinned capabilities (``llm_bawt.steer_capabilities``) or by the
# bridge as its backstop, so both are definite non-deliveries: callers send
# the message as a fresh turn.
STEER_UNSUPPORTED_TRANSPORT = "steer_unsupported_transport"
STEER_PRIORITY_UNSUPPORTED = "steer_priority_unsupported"
STEER_CAPABILITY_REFUSALS = frozenset({STEER_UNSUPPORTED_TRANSPORT, STEER_PRIORITY_UNSUPPORTED})

# ``steer_lifecycle`` event states (the CLI's ``command_lifecycle`` frames,
# plus the bridge's own ``uncertain``) and the rejection proofs that a
# command can never run. Published by ``claude_code_bridge.active_run``;
# the app re-queues a bot delivery only on a proven rejection.
LIFECYCLE_STARTED = frozenset({"started", "completed"})
LIFECYCLE_REJECTED = frozenset({"cancelled", "discarded", "refused"})
LIFECYCLE_UNCERTAIN = "uncertain"
PROOF_CLI_TERMINAL = "cli_terminal"
PROOF_SWEEP_RECEIPT = "sweep_receipt"
NOT_RUN_PROOFS = frozenset({PROOF_CLI_TERMINAL, PROOF_SWEEP_RECEIPT})


def render_interrupt(frame: str, items: list[str]) -> str:
    """Fill an app-rendered interruption frame with its swept items."""
    if STEER_ITEMS_SLOT not in frame:
        raise RuntimeError(STEER_UNFRAMED)
    return frame.replace(STEER_ITEMS_SLOT, STEER_ITEMS_SEPARATOR.join(items), 1)
