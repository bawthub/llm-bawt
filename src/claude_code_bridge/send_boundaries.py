from __future__ import annotations


BLOCK_SEPARATOR = "\n\n"


def should_read_native_context_usage(*, use_proxy: bool) -> bool:
    """Return whether terminal context usage needs an SDK control request.

    Proxy-backed turns already receive an authoritative final-iteration usage
    snapshot from the upstream stream. Asking the Claude CLI for the same data
    after its terminal ResultMessage can never improve that accounting and may
    sit on the control-channel timeout before ASSISTANT_DONE is published.
    """
    return not use_proxy


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
