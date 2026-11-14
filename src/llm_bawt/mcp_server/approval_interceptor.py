"""Approval-aware FastMCP dispatch for the first-party BawtHub server.

Every registered tool passes through the shared pure policy evaluator. Claude
turns carry a signed reserved input field that binds the exact SDK tool-use id,
name, and original arguments to the opaque server-minted turn capability. The
field is stripped before FastMCP schema validation/function binding.
"""

from __future__ import annotations

import contextvars
import functools
import hashlib
import inspect
import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from agent_bridge.approval import PolicyAction, evaluate
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

from ..approval_policies import (
    ApprovalDuplicatePending,
    ApprovalPersistError,
    ApprovalStoreUnavailable,
)
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


def _approval_request_id(call_context: McpCallContext) -> str:
    identity = f"{call_context.backend}\0{call_context.tool_use_id}\0{call_context.invocation_hash}"
    return "mcp-appr-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:40]


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
    """MCP server whose ``call_tool`` performs app-owned policy interception."""

    def __init__(
        self,
        *args,
        approval_store_provider: PolicyProvider | None = None,
        approval_publisher: ApprovalPublisher | None = None,
        operations_preparer: Callable[[str, dict], dict] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._approval_store_provider = approval_store_provider
        self._approval_publisher = approval_publisher
        self._operations_preparer = operations_preparer
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

            from ..utils.config import Config

            config = getattr(self, "_approval_config", None) or Config()
            self._approval_config = config
            if not config.REDIS_URL:
                return
            client = aioredis.from_url(config.REDIS_URL, decode_responses=True)
            try:
                await client.xadd(
                    f"events:{payload['bot_id']}:{payload['user_id']}",
                    {"payload": json.dumps(payload, ensure_ascii=False, default=str)},
                    maxlen=5000,
                    approximate=True,
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

        store = self._approval_store()
        try:
            bundle = store.compile_bundle()
        except ApprovalStoreUnavailable:
            return {
                "status": "policy_unavailable",
                "tool": name,
                "is_error": True,
                "message": "Approval policy source unavailable; tool was not executed.",
            }
        decision = evaluate(
            bundle.policies,
            call_context.backend if call_context else "mcp",
            name,
            received,
        )
        store.record_decision(
            decision=decision,
            bundle=bundle,
            backend=call_context.backend if call_context else "mcp",
            tool_name=name,
            invocation_hash=canonical_invocation_hash(name, received),
            bot_id=turn_context.bot_id if turn_context else None,
            user_id=turn_context.user_id if turn_context else None,
            turn_id=turn_context.turn_id if turn_context else None,
        )
        if decision.action is PolicyAction.ALLOW:
            return await super().call_tool(name, received, context)
        if decision.action is PolicyAction.DENY:
            return {
                "status": "denied",
                "tool": name,
                "subject": decision.subject,
                "message": "This MCP call is blocked by policy. Do not retry it.",
                "is_error": True,
            }

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

        request_id = _approval_request_id(call_context)
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
        try:
            operations_snapshot = None
            if name == "ops_run":
                prepare = self._operations_preparer
                if prepare is None:
                    from .ops_tools import _get_ops_service

                    prepare = _get_ops_service().prepare_invocation
                operations_snapshot = prepare(
                    received.get("operation", ""), received.get("args") or {}
                )
                if not isinstance(operations_snapshot, dict):
                    raise ValueError("Operation preparer must return a snapshot object")
                from ..ops.gpu_guard import snapshot_handoff_only

                service = snapshot_handoff_only(operations_snapshot)
                if service is not None:
                    # TASK-1707: an approval could never run it, so file no card.
                    return {
                        "status": "gpu_handoff_required",
                        "tool": name,
                        "is_error": True,
                        "message": (
                            f"This operation starts, restarts or stops the GPU voice service "
                            f"{service!r}. Only a GPU handoff can run it, so no approval was "
                            "created. Do not retry; ask the user to use the voice/video switch "
                            "or the TTS page's Restart button."
                        ),
                    }
            approval_prompt = decision.prompt
            if name == "ops_run" and operations_snapshot and operations_snapshot.get("release_source"):
                source = operations_snapshot["release_source"]
                spec = operations_snapshot["spec"]
                approval_prompt = (
                    f"{decision.prompt or 'Authorize BawtHub release and production deployment?'}\n"
                    f"Source: {spec['github_repository']}@{source['expected_sha']}\n"
                    f"Target: {spec['deploy_operation']} (currently {source.get('expected_current_image_id', 'unknown')})\n"
                    "Deploys only if the build, release identity and production baseline verify. "
                    "Unknown or changed outcomes stop for inspection; rollback remains separate."
                ) if source.get("deploy_authorization") == "after_verified_build" else (
                    f"{decision.prompt or 'Authorize remote BawtHub release build?'}\n"
                    f"BawtHub {spec['github_repository']}@{source['expected_sha']}\n"
                    "Production deployment requires a separate approval."
                )
            if name == "ops_run" and operations_snapshot and operations_snapshot.get("routing"):
                routing = operations_snapshot["routing"]
                spec = operations_snapshot["spec"]
                approval_prompt = (
                    f"{decision.prompt or 'Authorize same-image routing repair?'}\n"
                    f"Target: {spec['container_name']} ({routing['container_id']})\n"
                    f"Unchanged image: {routing['image_ref']}\n"
                    f"Configuration fingerprint: {routing['config_hash']}\n"
                    f"Before: {routing['before_rule']}\nAfter: {routing['after_rule']}\n"
                    "No build, pull, environment, mount or permission changes. "
                    "Verifies the hostname-routed health report; restores the previous container "
                    "if verification fails. Changed target/configuration cancels execution."
                )
            if name == "ops_run" and operations_snapshot and operations_snapshot.get("remove"):
                from ..ops import container_remove  # TASK-1439 T2c: one stopped orphan

                approval_prompt = container_remove.approval_text(operations_snapshot["remove"], decision.prompt)
            if name == "ops_run" and operations_snapshot and operations_snapshot.get("config"):
                from ..ops import (  # TASK-1518, TASK-1474, TASK-1476, TASK-1649
                    app_config_reconcile,
                    config_reconcile,
                    dev_config_reconcile,
                    tagged_config_reconcile,
                )

                if operations_snapshot["spec"].get("action") == app_config_reconcile.ACTION:
                    approval_prompt = app_config_reconcile.snapshot_approval_text(
                        operations_snapshot, decision.prompt
                    )
                elif operations_snapshot["spec"].get("action") == tagged_config_reconcile.ACTION:
                    approval_prompt = tagged_config_reconcile.approval_text(
                        operations_snapshot["spec"], operations_snapshot["config"], decision.prompt
                    )
                elif operations_snapshot["spec"].get("action") == dev_config_reconcile.ACTION:
                    approval_prompt = dev_config_reconcile.approval_text(
                        operations_snapshot["spec"], operations_snapshot["config"], decision.prompt
                    )
                else:
                    approval_prompt = config_reconcile.approval_text(
                        operations_snapshot["spec"], operations_snapshot["config"], decision.prompt
                    )
            if name == "ops_run" and operations_snapshot and operations_snapshot.get("plan"):
                from ..ops import plan_approval  # TASK-1476

                approval_prompt = plan_approval.approval_text(operations_snapshot, decision.prompt)
            if name == "ops_run" and operations_snapshot and operations_snapshot.get("sequence"):
                from ..ops import sequence_approval

                approval_prompt = sequence_approval.approval_text(operations_snapshot, decision.prompt)
            if name == "ops_run" and operations_snapshot:
                from ..ops import execution_mode  # TASK-1169 S4
                from ..ops.operation_revision import with_safety_lines

                approval_prompt = execution_mode.with_card_lines(approval_prompt, operations_snapshot)
                approval_prompt = with_safety_lines(approval_prompt, operations_snapshot)
            row, created = store.record_mcp_request(
                request_id=request_id,
                tool_use_id=call_context.tool_use_id,
                mcp_server="bawthub",
                bot_id=turn_context.bot_id,
                user_id=turn_context.user_id,
                turn_id=turn_context.turn_id,
                backend=call_context.backend,
                tool_name=name,
                tool_arguments=received,
                subject=decision.subject,
                grant_key=decision.grant_key,
                policy_id=getattr(decision.policy, "id", None),
                severity=decision.severity.value,
                prompt=approval_prompt,
                invocation_hash=invocation_hash,
                operations_snapshot=operations_snapshot,
                caller_context_json=json.dumps(
                    caller_context, ensure_ascii=False, sort_keys=True
                ),
                continuation_capable=True,
                trigger_message_id=turn_context.trigger_message_id,
                session_key=call_context.session_key,
                with_created=True,
            )
        except ApprovalDuplicatePending as duplicate:
            return {
                "status": "approval_pending",
                "approval_request_id": duplicate.existing_request_id,
                "tool": name,
                "is_error": True,
                "message": (
                    "An identical request is already awaiting approval; the call "
                    "was not executed and no new approval was created. Wait for "
                    "its result instead of re-sending."
                ),
            }
        except ApprovalPersistError as error:
            logger.error("Could not persist MCP approval for %s: %s", name, error)
            return {
                "status": "approval_persist_failed",
                "tool": name,
                "message": (
                    "The approval request could not be saved, so the tool was not "
                    "executed. Do not retry automatically."
                ),
                "is_error": True,
            }

        if created:
            if name == "ops_run":
                from ..approval_ops_supersede import singleton_target, superseded_by

                if singleton_target(operations_snapshot) is not None:
                    # Project already-committed terminal results through the
                    # normal resolver: no approval/grant or tool execution.
                    # The durable outbox repairs history after an app crash.
                    from ..service.routes.approval_policies import resolve_mcp_card

                    try:
                        for previous in superseded_by(store, row.id):
                            await resolve_mcp_card(
                                store, previous, outcome="deny",
                                message=previous.resolution_message, resolved_by=previous.resolved_by,
                            )
                    except Exception:
                        logger.exception("Could not project superseded ops cards for %s", row.id)
            await self._publish_approval_required(
                {
                    "_type": "tool_approval_required",
                    "request_id": row.id,
                    "tool_use_id": row.tool_use_id,
                    "turn_id": row.turn_id,
                    "trigger_message_id": row.trigger_message_id,
                    "bot_id": row.bot_id,
                    "user_id": row.user_id,
                    "tool_name": row.tool_name,
                    "arguments": received,
                    "subject": row.subject,
                    "label": decision.label,
                    "prompt": row.prompt,
                    "severity": row.severity,
                    "policy_id": row.policy_id,
                    "session_key": row.session_key or "",
                    "provider": row.backend,
                    "request_kind": "mcp",
                    "continuation_capable": bool(row.continuation_capable),
                }
            )
        return {
            "status": "approval_required",
            "approval_request_id": row.id,
            "tool": name,
            "subject": row.subject,
            "continuation_capable": bool(row.continuation_capable),
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
