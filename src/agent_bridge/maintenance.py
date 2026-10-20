"""Exact explicit maintenance-command classification, shared by app and bridge."""


def is_compact_command(message: str | None) -> bool:
    """Recognize a leading slash command, never mentions or /compactness."""
    parts = (message or "").lstrip().split(maxsplit=1)
    return bool(parts and parts[0] == "/compact")


def is_new_command(message: str | None) -> bool:
    """Recognize /new as a complete leading token, not /newspaper or mentions."""
    parts = (message or "").lstrip().split(maxsplit=1)
    return bool(parts and parts[0] == "/new")
