"""Shared rough token estimates (no tokenizer dependency).

~4 characters per token, within 10-15% of tiktoken for English text.
"""

CHARS_PER_TOKEN = 4
MESSAGE_OVERHEAD_TOKENS = 4


def estimate_tokens(text: str) -> int:
    """Rough token estimate for a text string."""
    return len(text) // CHARS_PER_TOKEN


def estimate_message_tokens(content: str) -> int:
    """Rough token estimate for one message body plus role/framing overhead."""
    return estimate_tokens(content) + MESSAGE_OVERHEAD_TOKENS
