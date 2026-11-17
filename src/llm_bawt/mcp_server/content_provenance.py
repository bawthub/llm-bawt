"""Admission on registered MCP callables, including approved execution."""
from __future__ import annotations

import inspect
from contextvars import copy_context
from functools import wraps

from agent_bridge.content_provenance import external_source, provenance_envelope, write_kind
from agent_bridge.mcp_call_context import canonical_invocation_hash
from mcp.server.mcpserver.exceptions import ToolError

from ..content_provenance import ContentTurn
from ..utils.bounded_worker import BoundedWorker
from .approval_interceptor import current_approved_caller_context
from .task_association import current_task_turn_capability, current_task_turn_context

_WORKER = BoundedWorker(4, "content-provenance")


class ToolContentProvenance:
    def __init__(self, store_provider=None):
        self._store_provider = store_provider

    def _store(self):
        if self._store_provider is not None:
            return self._store_provider()
        from ..service.dependencies import get_content_provenance_store
        from ..utils.config import Config

        return get_content_provenance_store(Config())

    def _turn(self, store):
        approved = current_approved_caller_context()
        if approved is not None:
            return store.resolve(approved.bot_id, approved.user_id, approved.turn_id)
        # No asserted bot_id, public source IDs, role=user or boolean can
        # substitute for the application's verified turn capability.
        if not current_task_turn_capability():
            raise ValueError("verified turn provenance required")
        context = current_task_turn_context()
        return ContentTurn(
            context.bot_id, context.user_id, context.turn_id, context.trigger_message_id,
        )

    async def check(self, name, arguments, *, preflight=False):
        source, kind = external_source(name), write_kind(name, arguments)
        if source is None and kind is None:
            return
        try:
            if kind and self._service_write(name, arguments, preflight=preflight):
                # The app's exact-invocation service principal is shared
                # storage plumbing (including automatic human extraction),
                # not a model-initiated MCP turn. Native bot tools have their
                # own executor admission before invoking this plumbing.
                return
            await _WORKER.run(
                copy_context().run, self._check, name, arguments, source, kind, preflight,
            )
        except ToolError:
            raise
        except Exception:
            raise ToolError(
                "Held: verified content provenance is unavailable. Nothing was executed."
            ) from None

    @staticmethod
    def _service_write(name, arguments, *, preflight):
        from ..bot_access.context import current_bot_access
        from .bot_access_identity import trusted_caller

        access = current_bot_access()
        if not preflight and access is not None:
            access.assert_current()
            caller = access.caller
        else:
            caller = trusted_caller(name, arguments)
        return (
            caller is not None and caller.kind == "service"
            and current_approved_caller_context() is None
            and current_task_turn_capability() is None
        )

    def _check(self, name, arguments, source, kind, preflight):
        store = self._store()
        turn = self._turn(store)
        # Preflight precedes approval persistence. A tainted proposal must not
        # become a retained approval payload. Clean preflight also registers the
        # verified trigger identity for later approved execution.
        seen = store.seen(turn)
        if source and not preflight:
            store.expose(turn)  # Durable before fn(), including exceptions.
        if kind and (not preflight or seen):
            receipt = store.admit(
                turn, tool_name=name, write_kind=kind,
                invocation_hash=canonical_invocation_hash(name, arguments),
                action=str(arguments.get("action", "")),
            )
            if receipt.held:
                raise ToolError(
                    f"Held: {kind} write refused after untrusted content. "
                    f"Receipt {receipt.receipt_id}; source turn {receipt.source_turn}; "
                    "untrusted_seen=true. Nothing was written; no proposal was retained."
                )

    def register(self, name, fn):
        # facts_extract/profile depend on their action; bind defaults at the
        # callable boundary rather than guessing omitted argument values.
        if external_source(name) is None and write_kind(name, {}) is None and name != "profile":
            return fn
        signature = inspect.signature(fn, eval_str=True)

        @wraps(fn)
        async def guarded(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            await self.check(name, dict(bound.arguments))
            try:
                result = fn(*bound.args, **bound.kwargs)
                return await result if inspect.isawaitable(result) else result
            except Exception as error:
                source = external_source(name)
                if source is not None:
                    raise ToolError(provenance_envelope(str(error), source)) from None
                raise

        guarded.__signature__ = signature
        guarded.__annotations__ = inspect.get_annotations(fn, eval_str=True)
        return guarded
