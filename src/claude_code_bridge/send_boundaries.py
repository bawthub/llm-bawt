from __future__ import annotations


BLOCK_SEPARATOR = "\n\n"


def separator_before_new_block(previous_tail: str) -> str:
    """Return a Markdown paragraph boundary between consecutive stream blocks.

    The Claude SDK exposes each reasoning summary section as a distinct thinking
    block. Their text does not necessarily include a leading/trailing newline, so
    concatenating raw deltas can produce malformed Markdown such as
    ``**first****second**``. Existing newline-terminated blocks already carry an
    intentional boundary and must not receive another one.
    """
    if not previous_tail or previous_tail.endswith("\n"):
        return ""
    return BLOCK_SEPARATOR
