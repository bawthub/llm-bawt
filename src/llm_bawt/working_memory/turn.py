"""Working memory on one started stream turn (docs/WORKING_MEMORY.md 8.1, 9.1-9.3, OI-R5).

``turn_stream_worker`` makes three calls, in order:

1. :meth:`WorkingMemoryTurn.begin` before ``prepare_messages_for_query``:
   renders the section onto the (cached, per-turn) ``llm_bawt`` instance,
   always fresh, so a stale render never leaks into another turn.
2. :attr:`WorkingMemoryTurn.sections` into ``apply_session_pin``, so the pin
   pointer records this render's content-free metadata.
3. :meth:`WorkingMemoryTurn.after_pin` once the pin transaction committed and
   before the command is published: folds the unverified floor and appends
   the notice to the outbound user message when it fires.

Every step is exception-safe and policy-gated. A bot the policy excludes (the
default, and always a protected bot) gets no section, no note, no notice and no
write, so its prompt is byte-identical with the feature deployed.
"""
from __future__ import annotations

import copy
import logging
from typing import Any

from agent_bridge.maintenance import is_clear_command, is_compact_command, is_new_command

from . import metrics
from .notice import prepare_notice
from .prompts import scheduled_note_text
from .render import SECTION_NAME, SectionRender, render_section
from .session_state import SUPPRESSED

logger = logging.getLogger(__name__)

SCHEDULER_SENDER_PREFIX = "scheduler:"


def is_maintenance_command(prompt: str | None) -> bool:
    """``/compact``, ``/new`` or ``/clear`` as the leading token, with or without arguments (9.1)."""
    return is_compact_command(prompt) or is_new_command(prompt) or is_clear_command(prompt)


def scheduler_turn(service: Any, request: Any) -> bool | None:
    """Whether the claim-validated delivery behind this turn came from the scheduler.

    Identity comes from the delivery record the dispatcher validated, never
    from an author label or ``message_kind`` (9.1). None means "can't tell";
    the caller then treats the turn as ineligible for a notice.
    """
    delivery_id = getattr(request, "inter_bot_delivery_id", None)
    if not delivery_id:
        return False
    store = getattr(getattr(service, "_inter_bot_dispatcher", None), "store", None)
    if store is None:
        return None
    try:
        record = store.get(delivery_id)
    except Exception as error:
        logger.warning("working memory: delivery %s unreadable: %s", delivery_id, error)
        return None
    if record is None:
        return None
    return str(getattr(record, "sender_bot_id", "") or "").startswith(SCHEDULER_SENDER_PREFIX)


def _session_id(ctx: Any) -> str:
    return str((ctx.thread_binding or {}).get("thread_session_id") or "").strip()


def _pins(llm_bawt: Any) -> bool:
    return bool(getattr(getattr(llm_bawt.client, "_backend", None), "pins_system_prompt", False))


class WorkingMemoryTurn:
    """One turn's working-memory state between prompt assembly and publish."""

    def __init__(self, ctx: Any, render: SectionRender | None = None, *, scheduler: bool | None = False):
        self.ctx = ctx
        self.render = render
        self.scheduler = scheduler

    @classmethod
    def begin(cls, ctx: Any) -> WorkingMemoryTurn:
        """Render this turn's section and set it on ``ctx.llm_bawt`` (None when none applies)."""
        llm_bawt = ctx.llm_bawt
        llm_bawt._working_memory_section = None
        if not ctx.is_agent_backend or ctx.bridge_reset_action == "reset_only":
            return cls(ctx)
        session_id = _session_id(ctx)
        try:
            if not session_id or not _pins(llm_bawt):
                return cls(ctx)
            render = render_section(
                ctx.svc._turn_log_store.engine, bot_id=ctx.bot_id, user_id=ctx.user_id,
                session_id=session_id,
            )
        except Exception:
            # render_section is already failure-safe; this guards the lookups around it.
            logger.warning("working memory section skipped for %s", ctx.bot_id, exc_info=True)
            return cls(ctx)
        if render is None:
            return cls(ctx)
        llm_bawt._working_memory_section = render.text
        return cls(ctx, render, scheduler=scheduler_turn(ctx.svc, ctx.request))

    @property
    def active(self) -> bool:
        return self.render is not None

    @property
    def sections(self) -> dict[str, dict] | None:
        """``apply_session_pin``'s ``sections`` argument, or None when the bot is excluded."""
        return {SECTION_NAME: dict(self.render.meta)} if self.render is not None else None

    def extend_suffix(self, suffix: str | None) -> str | None:
        """Add the scheduled-automation note to a scheduler turn's outbound suffix (OI-R5).

        Only when the policy renders a section and the session isn't
        suppressed; never on a maintenance command, whose bare text any
        trailing note would break.
        """
        if (self.render is None or self.render.state == SUPPRESSED or self.scheduler is not True
                or is_maintenance_command(self.ctx.user_prompt)):
            return suffix
        note = scheduled_note_text(self.ctx.bot_id)
        return "\n\n".join(part for part in (suffix, note) if part) or None

    def after_pin(self, messages: list, decision: Any) -> list:
        """Fold the floor and append the notice when it fires; returns the messages to send."""
        if self.render is None:
            return messages
        if decision is None:
            metrics.record_no_pin_decision(self.ctx.bot_id)
            return messages
        notice = prepare_notice(
            self.ctx.svc._turn_log_store.engine,
            bot_id=self.ctx.bot_id, user_id=self.ctx.user_id, session_id=_session_id(self.ctx),
            decision=decision,
            eligible=self.scheduler is False and not is_maintenance_command(self.ctx.user_prompt),
        )
        return append_to_last_user(messages, notice) if notice else messages


def append_to_last_user(messages: list, text: str) -> list:
    """A copy of ``messages`` whose last user message ends with ``text``.

    The message is copied, not mutated: the original may be the one history
    persistence holds, and the notice is outbound-only (9.3).
    """
    out = list(messages)
    for index in range(len(out) - 1, -1, -1):
        message = out[index]
        if getattr(message, "role", None) == "user" and isinstance(message.content, str):
            patched = copy.copy(message)
            patched.content = f"{message.content}\n\n{text}" if message.content else text
            out[index] = patched
            return out
    logger.warning("working memory notice had no user message to ride; dropped")
    return messages


__all__ = [
    "SCHEDULER_SENDER_PREFIX", "WorkingMemoryTurn", "append_to_last_user", "is_maintenance_command",
    "scheduler_turn",
]
