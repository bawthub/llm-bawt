"""Approximate the context loaded by a local Skill call, not its tiny ACK result."""

from __future__ import annotations

from agent_bridge.skill_sources import AmbientSource


def skill_context_tokens(arguments: dict | None) -> int | None:
    """Estimate the loaded ambient SKILL.md body.

    The SDK adds framing and referenced files may be loaded later. This is not
    provider billing or a measure of the exact incremental prompt size.
    """
    name = (arguments or {}).get("skill")
    if not isinstance(name, str):
        return None
    body = AmbientSource().read_body(name)
    return (body.tokens or None) if body is not None else None
