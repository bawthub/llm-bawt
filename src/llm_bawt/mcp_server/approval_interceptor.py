"""Catalog approval dispatch for the first-party BawtHub server.

Ops requests are attested by the app and handed to ops for catalog evaluation
and durable card creation. Turns carry a signed reserved input field that binds the exact SDK tool-use id,
name, and original arguments to the opaque server-minted turn capability. The
field is stripped before FastMCP schema validation/function binding.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import inspect
import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from agent_bridge.mcp_call_context import (
    MCP_CALL_CONTEXT_KEY,
    McpCallContext,
    McpCallContextError,
    canonical_invocation_hash,
    derive_mcp_call_context,
    verify_mcp_call_context,
)
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.shared.exceptions import MCPError
from mcp.types import CallToolResult, TextContent

from ..task_turn_context import (
    TaskTurnContext,
    TaskTurnContextError,
    open_task_turn_context,
)
from .task_association import (
    current_mcp_request_context,
    current_task_turn_capability,
)

logger = logging.getLogger(__name__)

PolicyProvider = Callable[[], Any]
ApprovalPublisher = Callable[[dict[str, Any]], Awaitable[None] | None]


def _approval_context_error(
    tool: str, *, invalid: bool, message: str
) -> dict[str, Any]:
    return {
        "status": "approval_context_invalid" if invalid else "approval_context_missing",
        "tool": tool,
        "message": message,
        "is_error": True,
    }


class ToolRegistryUnavailable(RuntimeError):
    """Approved execution refused: tools were not registered at startup."""


@dataclass(frozen=True)
class ApprovedCallerContext:
    """Who originally asked for an approved call, replayed at execution time.

    Server-side execution happens in the FastAPI process, long after the
    agent's turn-scoped contextvars are gone, so a tool that wants to record
    provenance has nothing to read. This carries it without adding anything to
    the tool's public MCP schema — an agent must never be able to supply its
    own caller identity. (Approved ``ops_run`` is not executed here at all;
    ops admission reads provenance from the approval row, TASK-1220.)
    """

    bot_id: str = ""
    user_id: str = ""
    turn_id: str = ""
    session_key: str = ""
    backend: str = ""
    approval_request_id: str = ""


_approved_caller_context: contextvars.ContextVar[ApprovedCallerContext | None] = (
    contextvars.ContextVar("llm_bawt_approved_caller_context", default=None)
)


def current_approved_caller_context() -> ApprovedCallerContext | None:
    """Caller provenance for the approved call being executed, if any."""
    return _approved_caller_context.get()


def _v1_tool_semantics(fn: Callable[..., Any]) -> Callable[..., Any]:
    """``fn`` as an async tool that behaves as it did under mcp 1.x.

    mcp 2.x changes two things this server relies on:
    - It replaces the text of any exception that isn't a ``ToolError`` with a
      bare ``Error executing tool <name>``. Our tools raise ordinary exceptions
      (``BotAccessError``, ``ValueError``) as reasons the model reads and acts on,
      so re-raise them as ``ToolError``. The client then sees
      ``Error executing tool <name>: <reason>`` exactly as before.
    - It runs sync tools on a worker thread. 1.x ran them inline on the event
      loop; keep that, so moving tools to threads stays a separate decision.
    """
    if getattr(fn, "_v1_tool_semantics", False):
        return fn
    signature = inspect.signature(fn, eval_str=True)

    @functools.wraps(fn)
    async def tool(*args, **kwargs):
        try:
            result = fn(*args, **kwargs)
            return await result if inspect.isawaitable(result) else result
        except (ToolError, MCPError):
            raise
        except Exception as error:
            raise ToolError(str(error)) from error

    tool.__signature__ = signature
    tool.__annotations__ = inspect.get_annotations(fn, eval_str=True)
    tool._v1_tool_semantics = True
    return tool


class PolicyResult(CallToolResult):
    """The interceptor's own outcome (denial, approval card, context error).

    The catalog leaves it as is: under mcp 1.x its JSON text was never compacted.
    """


def policy_result(result: Any, *, has_output_schema: bool = False) -> Any:
    """A policy outcome dict rendered as mcp 1.x rendered it on the wire.

    mcp 1.x serialized a dict returned from ``call_tool`` as indented JSON text
    plus the same dict as structured content, without ``isError``. mcp 2.x puts
    ``call_tool``'s return value on the wire as-is, so build that result here.

    One deliberate difference, for a tool with an output schema: 1.x validated
    this dict against that schema, failed, and sent ``isError`` with only
    ``Output validation error``, hiding the denial or approval reply from the
    model. It is still ``isError`` (clients validate structured content against
    the schema unless the result is an error), but now carries the outcome text.
    """
    if not isinstance(result, dict):
        return result
    text = [TextContent(type="text", text=json.dumps(result, indent=2))]
    if has_output_schema:
        return PolicyResult(content=text, is_error=True)
    return PolicyResult(content=text, structured_content=result, is_error=False)


class ApprovalAwareFastMCP(MCPServer):
    """MCP server whose ``call_tool`` routes verified ops invocations to ops."""

    def __init__(
        self,
        *args,
        approval_store_provider: PolicyProvider | None = None,
        approval_publisher: ApprovalPublisher | None = None,
        operations_requester: Callable[[dict], dict] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._approval_store_provider = approval_store_provider
        self._approval_publisher = approval_publisher
        self._operations_requester = operations_requester
        self._tools_registered = False

    def add_tool(self, fn, name=None, *args, **kwargs):
        return super().add_tool(_v1_tool_semantics(fn), name, *args, **kwargs)

    def mark_tools_registered(self) -> None:
        """Every tool module has registered on this instance (``registry``)."""
        self._tools_registered = True

    @property
    def tools_registered(self) -> bool:
        return self._tools_registered

    def _approval_store(self):
        if self._approval_store_provider is not None:
            return self._approval_store_provider()
        from ..service.dependencies import get_tool_approval_policy_store
        from ..utils.config import Config

        config = getattr(self, "_approval_config", None)
        if config is None:
            config = Config()
            self._approval_config = config
        return get_tool_approval_policy_store(config)

    async def _publish_approval_required(self, payload: dict[str, Any]) -> None:
        if self._approval_publisher is not None:
            result = self._approval_publisher(payload)
            if inspect.isawaitable(result):
                await result
            return
        # The DB row is authoritative. Redis is a live fanout optimization and
        # may be unavailable during exactly the restart scenarios ops exists for.
        try:
            import redis.asyncio as aioredis

            from agent_bridge.user_events import publish_user_event
            from ..utils.config import Config

            config = getattr(self, "_approval_config", None) or Config()
            self._approval_config = config
            if not config.REDIS_URL:
                return
            client = aioredis.from_url(config.REDIS_URL, decode_responses=True)
            try:
                await publish_user_event(
                    client, payload["bot_id"], payload["user_id"],
                    json.dumps(payload, ensure_ascii=False, default=str),
                )
            finally:
                await client.aclose()
        except Exception:
            logger.warning(
                "MCP approval committed but live publish failed id=%s",
                payload.get("request_id"),
                exc_info=True,
            )

    async def call_approved_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        expected_invocation_hash: str,
        caller_context: ApprovedCallerContext | None = None,
    ):
        """Invoke an already-approved exact stored call without policy recursion.

        The persisted arguments are hash-verified and replayed unchanged — a
        caller-supplied ``idempotency_key`` survives approval (TASK-959).
        Server-owned provenance travels in ``caller_context``, never in args.
        """
        # Registration happens once in the app lifespan, never here: a late
        # import from the live tree is the stale-module mismatch (TASK-1220).
        if not self._tools_registered:
            raise ToolRegistryUnavailable(
                "[tool_registry_unavailable] MCP tools were not registered at startup; "
                "the approved call was not run")
        received = dict(arguments or {})
        actual_hash = canonical_invocation_hash(name, received)
        if actual_hash != str(expected_invocation_hash or ""):
            raise ValueError("approved MCP invocation hash mismatch")
        token = _approved_caller_context.set(caller_context)
        try:
            return await super().call_tool(name, received)
        finally:
            _approved_caller_context.reset(token)

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: Context | None = None,
    ):
        tool = self._tool_manager.get_tool(name)
        return policy_result(
            await self._intercept_call(name, arguments, context),
            has_output_schema=bool(tool is not None and tool.output_schema),
        )

    async def _intercept_call(
        self,
        name: str,
        arguments: dict[str, Any],
        context: Context | None,
    ):
        received = dict(arguments or {})
        raw_call_context = received.pop(MCP_CALL_CONTEXT_KEY, None)
        capability = current_task_turn_capability()
        raw_request_context = current_mcp_request_context()
        call_context: McpCallContext | None = None
        turn_context: TaskTurnContext | None = None

        has_any_context = any(
            value is not None
            for value in (raw_call_context, capability, raw_request_context)
        )
        if has_any_context:
            if not capability:
                return _approval_context_error(
                    name,
                    invalid=False,
                    message="Trusted approval context is incomplete; the tool was not executed.",
                )
            if (raw_call_context is None) == (raw_request_context is None):
                return _approval_context_error(
                    name,
                    invalid=raw_call_context is not None,
                    message="Exactly one trusted MCP caller context is required; the tool was not executed.",
                )
            try:
                turn_context = open_task_turn_context(capability)
                if raw_call_context is not None:
                    call_context = verify_mcp_call_context(
                        capability=capability,
                        tool_name=name,
                        tool_input=received,
                        raw_context=raw_call_context,
                    )
                else:
                    if context is None:
                        raise LookupError("no protocol request context")
                    call_context = derive_mcp_call_context(
                        capability=capability,
                        raw_request_context=raw_request_context,
                        protocol_request_id=context.request_id,
                        tool_name=name,
                        tool_input=received,
                    )
            except (
                AttributeError,
                LookupError,
                McpCallContextError,
                RuntimeError,
                TaskTurnContextError,
            ) as error:
                return _approval_context_error(
                    name,
                    invalid=True,
                    message=f"Trusted approval context is invalid: {error}. The tool was not executed.",
                )

        if name != "ops_run":
            # Ordinary tools are neither policy-evaluated nor recorded.
            return await super().call_tool(name, received, context)

        if call_context is None or turn_context is None:
            return _approval_context_error(
                name,
                invalid=False,
                message=(
                    "Approval is required, but trusted caller correlation is missing; "
                    "the tool was not executed and no approval was created."
                ),
            )

        if name == "ops_run":
            from ..ops.approved_admission import ops_run_arguments_valid

            if not ops_run_arguments_valid(received):
                # TASK-1630: refuse before a card exists, with the same check
                # admission applies after approval.
                return {
                    "status": "args_invalid",
                    "tool": name,
                    "is_error": True,
                    "message": (
                        "[args_invalid] ops_run accepts only operation, args and "
                        "idempotency_key; no approval was created. Fix the call "
                        "(operation arguments belong inside args) and retry."
                    ),
                }

        invocation_hash = canonical_invocation_hash(name, received)
        caller_context = {
            "session_id": turn_context.session_id,
            "turn_id": turn_context.turn_id,
            "trigger_message_id": turn_context.trigger_message_id,
            "bot_id": turn_context.bot_id,
            "user_id": turn_context.user_id,
            "issued_at": turn_context.issued_at,
            "agent_request_id": call_context.agent_request_id,
            "session_key": call_context.session_key,
            "backend": call_context.backend,
            "tool_use_id": call_context.tool_use_id,
        }
        from ..ops.invocation_attestation import mint
        from ..ops.service import OpsDispatchError

        try:
            envelope = mint(received, caller_context, invocation_hash)
            requester = self._operations_requester
            if requester is None:
                from .ops_tools import _get_ops_service

                requester = _get_ops_service().request_invocation
            outcome = await asyncio.to_thread(requester, envelope)
        except OpsDispatchError as error:
            return {"status": error.code, "tool": name, "is_error": True, "message": str(error)}
        except Exception:
            logger.exception("Ops invocation handoff unavailable")
            return {"status": "ops_request_unknown", "tool": name, "is_error": True,
                    "message": "The ops request outcome is unknown. An approval may exist; "
                               "do not retry automatically or create a new intent."}
        if outcome["status"] == "catalog_allow":
            return await super().call_tool(name, received, context)
        if outcome["status"] != "approval_required":
            return outcome

        if outcome.get("created"):
            if outcome.get("singleton"):
                from ..approval_ops_supersede import superseded_by
                from ..service.routes.approval_policies import resolve_mcp_card

                store = self._approval_store()
                try:
                    for previous in superseded_by(store, outcome["approval_request_id"]):
                        await resolve_mcp_card(
                            store, previous, outcome="deny",
                            message=previous.resolution_message, resolved_by=previous.resolved_by,
                        )
                except Exception:
                    logger.exception("Could not project superseded ops cards for %s",
                                     outcome["approval_request_id"])
            await self._publish_approval_required(outcome["card"])
        return {
            "status": "approval_required",
            "approval_request_id": outcome["approval_request_id"],
            "tool": name,
            "subject": outcome["subject"],
            "continuation_capable": outcome["continuation_capable"],
            "message": (
                "Approval is required. The call was not executed. Do not retry or "
                "re-issue this tool; wait for the persisted approval result."
            ),
        }


__all__ = [
    "ApprovalAwareFastMCP",
    "ApprovedCallerContext",
    "ToolRegistryUnavailable",
    "current_approved_caller_context",
]
