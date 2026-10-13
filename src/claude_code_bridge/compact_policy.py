"""Bare slash-command policy; prompt bodies live only in the app registry."""

from __future__ import annotations


def is_bare_compact(message: str) -> bool:
    """Recognize only a slash command with no non-whitespace user suffix."""
    command = message.lstrip()
    return (command == "/compact" or (
        command.startswith("/compact")
        and command[len("/compact"):len("/compact") + 1].isspace()
        and not command[len("/compact"):].strip()
    ))


def apply_compact_default(message: str, guidance: str | None = None) -> str:
    """Use a resolved prompt only for bare commands; preserve all other bytes."""
    if not is_bare_compact(message) or not guidance:
        return message
    return f"{message.rstrip()} {guidance}"
