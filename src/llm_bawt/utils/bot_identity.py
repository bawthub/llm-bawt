"""Explicit bot identity (TASK-1373).

Memory and history are namespaced per bot, so a missing ``bot_id`` must fail
loudly rather than silently land in some named bot's namespace. Botless
utility work that still needs an identity resolves the one configured default
(``DEFAULT_BOT`` from env or DB runtime settings) here, never a hardcoded slug.
"""

from __future__ import annotations

from typing import Any


class MissingBotIdentity(ValueError):
    """Raised when a bot-scoped operation has no bot identity."""


def require_bot_id(bot_id: str | None, where: str) -> str:
    """Return the normalized ``bot_id`` or raise if it is empty."""
    slug = (bot_id or "").strip().lower()
    if not slug:
        raise MissingBotIdentity(f"{where} requires an explicit bot_id")
    return slug


def optional_default_bot_id(config: Any) -> str | None:
    """The configured ``DEFAULT_BOT``, or ``None`` on a botless install.

    For startup and status paths only (TASK-2404): a fresh install has no bots
    until first-run setup, so those must not require one.
    """
    if config is None:
        return None
    slug = (getattr(config, "DEFAULT_BOT", "") or "").strip().lower()
    if slug:
        return slug
    try:
        from ..runtime_setting_resolution import resolve_global_runtime_setting
        from .config import has_database_credentials

        if has_database_credentials(config):
            slug = str(resolve_global_runtime_setting(config, "default_bot") or "").strip().lower()
    except Exception:  # noqa: BLE001 - an unreadable setting means "not configured"
        slug = ""
    return slug or None


def configured_default_bot_id(config: Any) -> str:
    """The configured ``DEFAULT_BOT``; raises if none is configured."""
    slug = optional_default_bot_id(config)
    if not slug:
        raise MissingBotIdentity("DEFAULT_BOT is not configured (set LLM_BAWT_DEFAULT_BOT or the DB runtime setting)")
    return slug
