"""Which user a bridge event belongs to, without a hard-coded default user.

Bridge session keys are ``<bot>:<user>``. When a key carries no user, the
deployment's ``LLM_BAWT_DEFAULT_USER`` is used; with neither, the result is
empty and the caller decides (usually: skip the user-scoped publish).
"""
from __future__ import annotations

import os

DEFAULT_USER_ENV = "LLM_BAWT_DEFAULT_USER"


def configured_default_user() -> str:
    """The deployment's default user id, or ``""`` when none is configured."""
    return (os.getenv(DEFAULT_USER_ENV) or "").strip()


def session_user(session_key: str | None, explicit: str | None = None) -> str:
    """The explicit user, else the user part of ``session_key``, else the configured default."""
    user = (explicit or "").strip()
    if user:
        return user
    _, sep, tail = (session_key or "").partition(":")
    if sep and tail.strip():
        return tail.strip()
    return configured_default_user()
