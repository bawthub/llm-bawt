"""User ownership for ordinary history tools, independent of bot visibility."""
from __future__ import annotations

from llm_bawt.bot_access import BotAccessError
from llm_bawt.bot_access.context import current_bot_access
from llm_bawt.bot_access.service import is_service_self


def history_caller_user(bot_id: str, user_id: str | None, *, operator: bool = False) -> str | None:
    """Resolve only from registered-call authority, never a tool's asserted user.

    The app's existing user-less, signed service-self history reader/clearer is
    an operator operation. A turn/approved/user-bound service caller cannot
    acquire that authority by omitting ``user_id`` or choosing a public bot.
    """
    access = current_bot_access()
    if access is None or access.caller is None:
        raise BotAccessError("caller_unavailable")
    target = access.require_target(bot_id)
    caller = access.caller
    caller.validate()
    if operator and is_service_self(access, target):
        if user_id is not None and not user_id.strip():
            raise BotAccessError("access_denied")
        return user_id
    if not caller.user_id:
        raise BotAccessError("caller_unavailable")
    if user_id is not None and user_id != caller.user_id:
        raise BotAccessError("access_denied")
    return caller.user_id
