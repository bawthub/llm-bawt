"""Resolve the owner for a durable inter-bot delivery."""

from __future__ import annotations

from typing import Any


def resolve_delivery_user(config: Any, payload_user: Any = None) -> str:
    """Prefer an explicit owner; never invent one from an absent default."""
    for candidate in (payload_user, config.DEFAULT_USER):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    raise ValueError("Delivery user is required: provide a user or configure DEFAULT_USER")
