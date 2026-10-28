"""Specialized resolution helpers for database-backed runtime settings."""

from __future__ import annotations

from typing import Any


def resolve_global_runtime_setting(config: Any, key: str, fallback: Any = None) -> Any:
    """Resolve a system-wide setting while deliberately ignoring bot rows."""
    from .runtime_settings import RuntimeSettingsResolver
    from .setting_definitions import SETTING_DEFINITIONS

    definition = SETTING_DEFINITIONS.get(key)
    default = definition.default if definition is not None else fallback
    return RuntimeSettingsResolver(config=config, bot=None).resolve(key, fallback=default)


def resolve_bot_runtime_setting(
    config: Any, bot_id: str | None, key: str, fallback: Any = None,
) -> Any:
    """Resolve a setting as bot row > global row > registry default."""
    from .runtime_settings import RuntimeSettingsResolver
    from .setting_definitions import SETTING_DEFINITIONS

    definition = SETTING_DEFINITIONS.get(key)
    default = definition.default if definition is not None else fallback
    return RuntimeSettingsResolver(config=config, bot_id=bot_id).resolve(key, fallback=default)
