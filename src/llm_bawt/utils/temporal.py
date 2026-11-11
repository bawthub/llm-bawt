"""Helpers for temporal context injection into prompts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence


def _relative_time(seconds: float) -> str:
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        minutes = int(seconds // 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    if seconds < 86400:
        hours = int(seconds // 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = int(seconds // 86400)
    if days == 1:
        return "yesterday"
    return f"{days} days ago"


def _get_last_contact_epoch(history_messages: Sequence[Any]) -> float | None:
    """Best-effort 'last time we spoke' from user/assistant messages."""
    candidates: list[Any] = [
        m for m in history_messages if getattr(m, "role", None) in ("user", "assistant")
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda m: float(getattr(m, "timestamp", 0.0)))

    # Current turn often adds a fresh user message before prompt assembly.
    # Use the prior turn marker when available so "last contact" is meaningful.
    latest = candidates[-1]
    if getattr(latest, "role", None) == "user" and len(candidates) >= 2:
        return float(getattr(candidates[-2], "timestamp", 0.0))
    return float(getattr(latest, "timestamp", 0.0))


def build_temporal_context(history_messages: Sequence[Any] | None = None) -> str:
    """Return a concise temporal context block for prompt grounding."""
    now_local = datetime.now().astimezone()
    tz_name = now_local.tzname() or "local"

    # Human-readable date/time: "Wednesday, February 19, 2026 8:09 PM EST"
    date_str = now_local.strftime("%A, %B %-d, %Y %-I:%M %p") + f" {tz_name}"

    parts = [f"Current date/time: {date_str}"]

    if history_messages:
        last_contact = _get_last_contact_epoch(history_messages)
        if last_contact is not None and last_contact > 0:
            delta = max(0.0, now_local.timestamp() - last_contact)
            parts.append(f"Last conversation: {_relative_time(delta)}")

    return " | ".join(parts)
