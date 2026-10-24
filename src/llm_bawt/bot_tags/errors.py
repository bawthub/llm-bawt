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


class TagSetChanged(TagConflict):
    """A tag replace named the set it expected, and the bot's tags have since changed.

    ``current`` is the bot's assignment set read under the bot lock, so the
    caller can recompute its edit against it instead of overwriting blind.
    """

    code = "tags_changed"

    def __init__(self, bot: str, current: list[str]):
        super().__init__(f"{bot}'s tags changed since they were read; nothing was saved")
        self.bot = bot
        self.current = current
