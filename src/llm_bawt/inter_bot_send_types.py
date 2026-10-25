"""The send path's shared leaf types (TASK-1431).

``inter_bot_send`` (the sender) and ``inter_bot_fanout_batch`` (one batch)
both need these; keeping them here means neither module imports the other
in a cycle. ``inter_bot_send`` re-exports both.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fastapi import HTTPException

from .message_authorship import AuthorReference


class DeliveryRefused(HTTPException):
    """A send refused before any row was written.

    An HTTPException so the routes report it unchanged; typed so a batch can
    record it per target and carry on.
    """


@dataclass(frozen=True)
class Principal:
    """Who is sending: ``bot`` and ``user`` arrive over HTTP, ``system`` in-process."""

    kind: Literal["bot", "user", "system"]
    entity_id: str

    @classmethod
    def of_author(cls, author: AuthorReference) -> Principal:
        return cls("user" if author.entity_type == "user" else "bot", author.entity_id or "")
