"""Prompt-registry seed for the cancelled-delivery sender notice (TASK-1327 D2).

Seed data only: registered in :mod:`llm_bawt.prompt_definitions` and resolved
per bot at send time (``delivery_cancel_notice.notice_text``), so operators can
edit it in BawtHub's Prompt Registry.
"""
from __future__ import annotations

PROMPT_KEY = "deliveries.cancel_notice"
DEFAULT_CANCEL_NOTICE = (
    "[Delivery cancelled: your message {delivery_id} to {target_bot_id} (task: {task_id}) "
    "was cancelled before it ran and will not be delivered. Reason: {reason}. "
    "It began: \"{excerpt}\"]\n"
    "Nothing was done with it. If the reason says the user stopped or cancelled it, "
    "do not resend it on your own; tell the user or the task owner what did not happen "
    "if that matters. Otherwise decide whether the work still needs doing and resend "
    "it, or report that it was dropped."
)

STARTED_PROMPT_KEY = "deliveries.stopped_after_start_notice"
DEFAULT_STARTED_NOTICE = (
    "[Delivery stopped: your message {delivery_id} to {target_bot_id} (task: {task_id}) "
    "was stopped by user after it ran. Reason: {reason}. "
    "It began: \"{excerpt}\"]\n"
    "The target started working on this delivery but the user stopped its turn; "
    "some work may already have happened. Do not resend it on your own. "
    "Check the turn or ask the user before reporting what was completed."
)

__all__ = ["DEFAULT_CANCEL_NOTICE", "DEFAULT_STARTED_NOTICE", "PROMPT_KEY", "STARTED_PROMPT_KEY"]
