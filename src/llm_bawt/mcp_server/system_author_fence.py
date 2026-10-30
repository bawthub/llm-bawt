"""Service-only gate for first-party system authors on ``messages_add`` (TASK-1807).

The public ``messages_add`` tool keeps its caller-facing contract: author
fields are ``user`` or ``bot`` only (``normalize_author``). The one exception
is the app's own server-mode write of an approval-continuation turn, which
arrives through the signed ``MemoryClient`` transport. That call carries a
service capability bound to the exact tool arguments, which only the app can
mint (the Fernet key lives in the app, not in agent bridges).

A ``system`` author is therefore accepted only when ALL hold:

- the verified caller for this invocation is the app service principal
  (``TrustedCaller.kind == "service"``; an approved or turn identity on the
  same call resolves first and is refused);
- the caller's bot is the target bot namespace;
- the id is an allowlisted system author.

Anything else is refused before storage is touched.
"""
from __future__ import annotations

from llm_bawt.bot_access import BotAccessError
from llm_bawt.bot_access.context import current_bot_access
from llm_bawt.message_authorship import AuthorReference

SYSTEM_ENTITY_TYPE = "system"


def is_system_author_request(author_entity_type: str | None) -> bool:
    return (author_entity_type or "").strip().lower() == SYSTEM_ENTITY_TYPE


def fenced_system_author(author_entity_id: str | None, *, bot_id: str) -> AuthorReference:
    """Return the trusted system author, or raise ``access_denied``."""
    access = current_bot_access()
    caller = access.caller if access is not None else None
    if caller is None or caller.kind != "service":
        raise BotAccessError("access_denied")
    caller.validate()
    if caller.bot_id != (bot_id or "").strip().lower():
        raise BotAccessError("access_denied")
    try:
        return AuthorReference.system(author_entity_id or "")
    except ValueError:
        raise BotAccessError("access_denied") from None
