"""Common registered-callable authority for ordinary AND approved MCP calls."""
from __future__ import annotations

import inspect
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

from sqlalchemy import text

from llm_bawt.bot_access import BotAccessError, BotPolicyResolver
from llm_bawt.bot_access.context import bind_bot_access
from llm_bawt.bot_access.service import SERVICE_SELF_TOOLS, is_service_self
from llm_bawt.utils.bounded_worker import BoundedWorker
from llm_bawt.utils.config import Config
from llm_bawt.utils.db import get_shared_engine
from .bot_access_identity import trusted_caller
from .bot_access_manifest import AGGREGATE_TOOLS, BOT_TARGET_TOOLS, GUARDED_TOOLS

_invocation: ContextVar[tuple[str, dict] | None] = ContextVar("bot_access_invocation", default=None)
_METADATA_WORKER = BoundedWorker(4, "bot-ownership")


@contextmanager
def invocation_context(name: str, arguments: dict):
    token = _invocation.set((name, dict(arguments)))
    try:
        yield
    finally:
        _invocation.reset(token)


class ToolBotAccess:
    def __init__(self, resolver=None):
        self._resolver = resolver
        self._signatures: dict[str, inspect.Signature] = {}

    @property
    def resolver(self):
        if self._resolver is None:
            self._resolver = BotPolicyResolver(get_shared_engine(Config()))
        return self._resolver

    def register(self, name, fn):
        if name not in GUARDED_TOOLS:
            return fn
        signature = inspect.signature(fn, eval_str=True)
        self._signatures[name] = signature

        @wraps(fn)
        async def guarded(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            raw = _invocation.get()
            received = raw[1] if raw is not None and raw[0] == name else dict(bound.arguments)
            access = await self.authorize(name, received, bound=bound)
            with bind_bot_access(access):
                result = fn(*bound.args, **bound.kwargs)
                return await result if inspect.isawaitable(result) else result

        guarded.__signature__ = signature
        guarded.__annotations__ = inspect.get_annotations(fn, eval_str=True)
        return guarded

    async def preflight(self, name, arguments):
        if name in self._signatures:
            await self.authorize(name, arguments)

    async def authorize(self, name, arguments, *, bound=None):
        if bound is None:
            bound = self._signatures[name].bind(**arguments)
            bound.apply_defaults()
        values = bound.arguments
        access = await self.resolver.resolve(trusted_caller(name, arguments))
        target = None
        if name in BOT_TARGET_TOOLS or name == "sessions_list":
            target = access.require_target(values["bot_id"])
            values["bot_id"] = target
        elif name == "memory_search_source":
            target = access.require_target(values["source"])
            values["source"] = target
        elif name == "self_fwd":
            target = access.require_target(values["sender_bot_id"])
            values["sender_bot_id"] = target
        elif name == "profile" and values.get("entity_type") != "user":
            target = access.require_target(values["entity_id"])
            values["entity_id"] = target
        elif name == "bots_deliveries_list":
            for key in ("bot_id", "sender_bot_id", "target_bot_id"):
                if values.get(key):
                    values[key] = access.require_target(values[key])
        elif name in AGGREGATE_TOOLS:
            access.aggregate_scope(values.get("bot_id") or None,
                                   require_user=name == "memory_search_all")
        service_filter = False
        if name in SERVICE_SELF_TOOLS and access.caller is not None and access.caller.user_id is None:
            service_filter = is_service_self(access, target)
            if not service_filter:
                raise BotAccessError("access_denied")
        if target is not None and access.caller is not None:
            if target in access.private_bots or access.caller.kind == "service":
                if "user_id" in values and not service_filter:
                    supplied = values["user_id"]
                    if supplied is not None and supplied != access.caller.user_id:
                        raise BotAccessError("access_denied")
                    values["user_id"] = access.caller.user_id
                if values.get("accept_pointer") and target != access.caller.bot_id:
                    raise BotAccessError("access_denied")
        # Session IDs cannot override the explicit bot namespace. This early
        # check reads ownership only; scoped storage binds get/close SQL too.
        if target is not None and values.get("session_id"):
            try:
                owner = await _METADATA_WORKER.run(self._session_owner, values["session_id"])
            except Exception:
                raise BotAccessError("source_unavailable") from None
            if owner is not None and (owner[0] != target or (
                target in access.private_bots and not service_filter
                and owner[1] != access.caller.user_id
            )):
                raise BotAccessError("access_denied")
            if owner is None and name == "messages_add":
                raise BotAccessError("access_denied")
        return access

    def _session_owner(self, session_id):
        with self.resolver.engine.connect() as conn, conn.begin():
            conn.execute(text("SET TRANSACTION READ ONLY"))
            conn.execute(text("SET LOCAL statement_timeout = '2000ms'"))
            conn.execute(text("SET LOCAL lock_timeout = '500ms'"))
            return conn.execute(text("SELECT bot_id, user_id FROM sessions WHERE id=:id"),
                                {"id": session_id}).first()
