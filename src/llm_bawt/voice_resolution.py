"""Effective TTS voice for a bot (TASK-1702).

The one resolver for "which voice does this bot speak with": the bot's own
``default_voice`` column, else the global ``default_voice`` runtime setting,
else none. None means voice is unavailable for that bot; callers must surface
that, never substitute a voice. A per-user chat override is a client
preference and sits on top of this in the frontend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

VoiceSource = Literal["bot", "global"]


@dataclass(frozen=True)
class EffectiveVoice:
    voice: str | None
    source: VoiceSource | None


def _clean(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def resolve_bot_voice(bot_voice: Any, global_voice: Any) -> EffectiveVoice:
    """Bot voice ?? global voice ?? none."""
    own = _clean(bot_voice)
    if own:
        return EffectiveVoice(own, "bot")
    fallback = _clean(global_voice)
    if fallback:
        return EffectiveVoice(fallback, "global")
    return EffectiveVoice(None, None)


def global_default_voice(config: Any) -> str | None:
    """The global ``default_voice`` runtime setting, or None when unset."""
    from .runtime_setting_resolution import resolve_global_runtime_setting

    return _clean(resolve_global_runtime_setting(config, "default_voice"))


def effective_voice_for_bot(config: Any, bot: Any) -> EffectiveVoice:
    return resolve_bot_voice(getattr(bot, "default_voice", None), global_default_voice(config))
