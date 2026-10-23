"""Prompt-registry text for mid-turn steers (TASK-1300).

The claude-code bridge has no registry access, so the app renders every
model-facing word of a steer here and sends the results on ``chat.steer``
(:mod:`agent_bridge.steer_contract`). The bridge only joins these strings.

Each body is seed data: it is registered in :mod:`llm_bawt.prompt_definitions`
(its loader returns the ``DEFAULT_*`` text below) and resolved per bot at call
time, so Nick can edit it in BawtHub's Prompt Registry. Every read goes through
the registry: a blank live body, a resolve failure, or a live body that no
longer renders its slot falls back to the registered definition's default body,
never to text held here. If that default can't render either, the steer fails
with :class:`SteerPromptUnavailable` instead of going out unframed.

``system``-origin steers (the quiesce request) are already final registry text
and never pass through here.
"""
from __future__ import annotations

import logging

from agent_bridge.steer_contract import STEER_ITEMS_SLOT

logger = logging.getLogger(__name__)

# A ``priority=next`` message, delivered at the next tool boundary.
NEXT_ASIDE_KEY = "chat.steer.next_aside"
DEFAULT_NEXT_ASIDE = (
    "[Mid-turn message — delivered at a tool boundary; your running "
    "work was not interrupted]\n"
    "{message}\n\n"
    "Address it as needed, then continue the unfinished task."
)

# The replacement query after an interrupt: every interrupting message, in
# the order sent, then the instruction to resume.
INTERRUPT_REPLACEMENT_KEY = "chat.steer.interrupt_replacement"
DEFAULT_INTERRUPT_REPLACEMENT = (
    "[Mid-turn interruption]\n"
    "{messages}\n\n"
    "Respond to this interruption as needed, then continue the unfinished "
    "task from the interrupted turn, incorporating any new direction above. "
    "Do not conclude the turn merely because a message was a status question. "
    "If the user explicitly asks you to stop, pause, abandon, or only report "
    "status without continuing, honor that instruction instead."
)

# Stands in for the slot while a body is formatted (see ``_render``).
_MARKER = "\x00steer-slot\x00"


class SteerPromptUnavailable(RuntimeError):
    """No registered body for ``key`` can frame this steer, so it is not sent."""

    def __init__(self, key: str, reason: str):
        super().__init__(f"steer_prompt_unavailable: {key} ({reason})")
        self.key = key


def _definition(key: str):
    """The registered definition for ``key``; an unregistered key is refused.

    Checked before any live body is used, so a stale DB row for a key the
    code no longer registers can't frame a steer.
    """
    try:
        from .prompt_registry import PromptResolver

        definition = PromptResolver.definition_for(key)
    except Exception as error:
        logger.error("steer prompt %s: registry unavailable: %s", key, error)
        raise SteerPromptUnavailable(key, "registry unavailable") from error
    if definition is None:
        logger.error("steer prompt %s: not registered", key)
        raise SteerPromptUnavailable(key, "not registered")
    return definition


def _registered_default(key: str, definition) -> str:
    """The definition's default body; a failing loader is ``SteerPromptUnavailable``."""
    try:
        return definition.default_body()
    except Exception as error:  # a failing loader must still refuse visibly
        logger.error("steer prompt %s: registered default failed to load: %s", key, error)
        raise SteerPromptUnavailable(key, "registered default failed to load") from error


def _live_body(key: str, bot_id: str | None) -> str:
    """The body the registry resolves for ``bot_id``, or ``""`` when it can't."""
    try:
        from .prompt_registry import get_prompt_resolver

        resolver = get_prompt_resolver()
        resolved = (
            resolver.resolve(key, scope_type="bot", scope_id=bot_id)
            if bot_id else resolver.resolve(key)
        )
    except Exception as error:  # a DB failure falls back to the registered default
        logger.warning("steer prompt %s resolve failed, using its registered default: %s", key, error)
        return ""
    return (resolved.body if resolved else "") or ""


def _fill(body: str, slot: str) -> str | None:
    """``body`` with ``slot`` set to a marker, or None if it can't carry the slot."""
    try:
        rendered = body.format(**{slot: _MARKER})
    except Exception:  # any malformed field (KeyError, ValueError, AttributeError on {slot.x}, ...)
        return None
    return rendered if _MARKER in rendered else None


def _render(key: str, bot_id: str | None, *, slot: str, value: str) -> str:
    """Render ``key`` with ``slot`` from the live body, else the registered default."""
    definition = _definition(key)
    live = _live_body(key, bot_id)
    rendered = _fill(live, slot) if live.strip() else None
    if rendered is None:
        if live.strip():
            logger.warning("steer prompt %s does not render {%s}, using its registered default", key, slot)
        rendered = _fill(_registered_default(key, definition), slot)
        if rendered is None:
            logger.error("steer prompt %s: registered default does not render {%s}", key, slot)
            raise SteerPromptUnavailable(key, f"no body renders {{{slot}}}")
    # Substituted after formatting, so braces in the message are never parsed.
    return rendered.replace(_MARKER, value)


def interrupt_item(message: str, *, origin: str, user_id: str | None) -> str:
    """This message as one item of an interruption, with its sender.

    A bot delivery's stored text already starts with
    ``Message from bot '<slug>':`` (``routes.inter_bot_deliveries``), so only a
    user's message gets the matching user attribution.
    """
    if origin == "user":
        return f"Message from user '{user_id or 'unknown'}': {message}"
    return message


def render_next(message: str, *, bot_id: str | None) -> str:
    """The exact CLI command text for a ``priority=next`` steer."""
    return _render(NEXT_ASIDE_KEY, bot_id, slot="message", value=message)


def render_interrupt_frame(bot_id: str | None) -> str:
    """The escalation frame, with the bridge's item slot left in place."""
    return _render(INTERRUPT_REPLACEMENT_KEY, bot_id, slot="messages", value=STEER_ITEMS_SLOT)


def render_now(item: str, *, bot_id: str | None) -> str:
    """The whole replacement query for a ``priority=now`` steer of one message."""
    return _render(INTERRUPT_REPLACEMENT_KEY, bot_id, slot="messages", value=item)


__all__ = [
    "DEFAULT_INTERRUPT_REPLACEMENT", "DEFAULT_NEXT_ASIDE", "INTERRUPT_REPLACEMENT_KEY",
    "NEXT_ASIDE_KEY", "SteerPromptUnavailable", "interrupt_item", "render_interrupt_frame", "render_next",
    "render_now",
]
