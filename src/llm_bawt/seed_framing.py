"""Registry framing for an agent continuity seed (TASK-1579).

A fresh agent session opens on a briefing turn built from app-selected history
and one synthetic assistant reply. Both texts are prompts, so they live in the
prompt registry and are resolved here at call time (bot override, then global,
then the registered default). Bridges receive them with the seed and hold no
fallback copy.
"""
from __future__ import annotations

import logging

from agent_bridge.continuity import SeedFraming

logger = logging.getLogger(__name__)

SEED_BRIEFING_KEY = "agents.session_seed_briefing"
SEED_ACK_KEY = "agents.session_seed_ack"


def resolve_seed_framing(messages: list | None, bot_id: str | None) -> SeedFraming | None:
    """Framing for a non-empty seed; ``None`` when there is nothing to frame."""
    if not messages:
        return None
    return SeedFraming(briefing=_body(SEED_BRIEFING_KEY, bot_id), ack=_body(SEED_ACK_KEY, bot_id))


def _body(key: str, bot_id: str | None) -> str:
    from .prompt_registry import PromptResolver, get_prompt_resolver

    definition = PromptResolver.definition_for(key)
    if definition is None:
        raise RuntimeError(f"seed framing prompt {key} is not registered")
    try:
        resolver = get_prompt_resolver()
        resolved = (
            resolver.resolve(key, scope_type="bot", scope_id=bot_id)
            if bot_id else resolver.resolve(key)
        )
        body = (resolved.body if resolved else "") or ""
    except Exception as error:  # a DB failure falls back to the registered default
        logger.warning("seed framing %s resolve failed, using its registered default: %s", key, error)
        body = ""
    if body.strip():
        return body
    default = definition.default_body()
    if not default.strip():
        raise RuntimeError(f"seed framing prompt {key} has no body")
    return default
