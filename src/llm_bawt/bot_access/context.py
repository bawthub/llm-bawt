"""Invocation-local authorization shared with internal storage consumers."""
from contextlib import contextmanager
from contextvars import ContextVar

from .policy import BotAccess

_current_access: ContextVar[BotAccess | None] = ContextVar("bot_access", default=None)


def current_bot_access() -> BotAccess | None:
    return _current_access.get()


@contextmanager
def bind_bot_access(access: BotAccess):
    token = _current_access.set(access)
    try:
        yield access
    finally:
        _current_access.reset(token)
