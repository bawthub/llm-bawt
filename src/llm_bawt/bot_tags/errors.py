"""Typed bot-tag write failures; each carries the HTTP status its route returns."""
from __future__ import annotations


class BotTagError(Exception):
    status_code = 400


class TagValidationError(BotTagError):
    """An assignment or catalog value is malformed or names an unknown tag."""

    status_code = 422


class TagNotFound(BotTagError):
    status_code = 404


class TagConflict(BotTagError):
    """The write would change a system tag or delete a referenced one."""

    status_code = 409
