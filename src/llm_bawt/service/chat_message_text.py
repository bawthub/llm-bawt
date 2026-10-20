"""Extract user command text identically from API messages and stored payloads."""
from __future__ import annotations


def message_content_text(content: object) -> str | None:
    """Join only OpenAI text parts; image and other parts are not command text."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    pieces: list[str] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "text":
            continue
        text = part.get("text")
        if not isinstance(text, str):
            return None
        pieces.append(text)
    return "".join(pieces)


def last_user_text(messages: object) -> str | None:
    """Read the last user message, even if a non-user message follows it."""
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
        if role == "user":
            content = (message.get("content") if isinstance(message, dict)
                       else getattr(message, "content", None))
            return message_content_text(content)
    return None
