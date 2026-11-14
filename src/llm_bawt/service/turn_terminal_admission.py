"""Explicit DB-backed rollout for new terminal-protocol admissions only."""
from __future__ import annotations

import asyncio

from agent_bridge.maintenance import is_new_command

from ..runtime_setting_resolution import resolve_bot_runtime_setting, resolve_global_runtime_setting


def protocol_admission_enabled(config, bot_id: str, *, user_prompt: str | None,
                               session_policy: str | None = None,
                               reset_intent: bool = False, reset_only: bool = False) -> bool:
    """Run off-loop with admission work. This never upgrades an existing row.

    Reset receipts currently author history after execution; keep those turns
    legacy until their receipt text can be frozen in the terminal snapshot.
    Requiring the recovery writer prevents enabling admissions into a report-
    only pump. Turning admission off never disables already-marked recovery.
    """
    if (not bot_id or not getattr(config, "REDIS_URL", None)
            or reset_intent or reset_only or session_policy not in (None, "continue")
            or is_new_command(user_prompt or "")):
        return False
    if resolve_bot_runtime_setting(config, bot_id, "turn_terminal_protocol_enabled") is not True:
        return False
    return resolve_global_runtime_setting(config, "terminal_recovery_mode") == "recover"


async def direct_protocol_options(config, bot_id, request, user_prompt, admission_claim) -> dict:
    """Reserved/supplied IDs use their initial marker, never current settings."""
    if admission_claim is not None or getattr(request, "inter_bot_turn_id", None):
        return {}
    enabled = await asyncio.to_thread(
        protocol_admission_enabled, config, bot_id, user_prompt=user_prompt,
        session_policy=getattr(request, "inter_bot_session_policy", None),
        reset_only=bool(getattr(request, "inter_bot_reset_only", False)),
    )
    return {"terminal_protocol": True} if enabled else {}
