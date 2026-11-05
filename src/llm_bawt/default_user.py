"""The deployment's default user id, instead of a hard-coded person.

``DEFAULT_USER`` (env ``LLM_BAWT_DEFAULT_USER``) names the user a request
belongs to when it doesn't say. Without one configured, the result is empty
and callers that need a user reject the request.
"""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def default_user_id() -> str:
    """The configured default user id, or ``""``."""
    from llm_bawt.utils.config import Config

    return (getattr(Config(), "DEFAULT_USER", None) or "").strip()


def resolve_user_id(user_id: str | None) -> str:
    """``user_id`` if given, else the default; raises when neither exists."""
    resolved = (user_id or "").strip() or default_user_id()
    if not resolved:
        raise ValueError("user_id is required: pass one or configure LLM_BAWT_DEFAULT_USER")
    return resolved
