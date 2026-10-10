from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
import uuid
from collections.abc import AsyncIterable
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import httpx
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, StreamEvent
from claude_agent_sdk.types import (
    AssistantMessage,
    HookMatcher,
    MirrorErrorMessage,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    SystemMessage,
    TaskNotificationMessage,
    TaskProgressMessage,
    TaskStartedMessage,
    TaskUpdatedMessage,
    ToolPermissionContext,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from agent_bridge.approval import (
    ApprovalDecision,
    PolicyAction,
    PolicyBundle,
    evaluate as evaluate_policies,
)
from agent_bridge.mcp_call_context import (
    MCP_CALL_CONTEXT_KEY,
    mint_mcp_call_context,
)
from agent_bridge.events import AgentEvent, AgentEventKind, synthesize_event_id
from agent_bridge.publisher import COMMANDS_STREAM, RedisPublisher
from agent_bridge.session_queue import SessionQueue
from claude_code_bridge.tool_events import normalize_tool_result
from claude_code_bridge.tool_policy import effective_disallowed_tools

from ._bridge_helpers import (
    SESSION_PREFIX,
    MCP_TOOL_CONTEXT_KEY,
    _MCP_TOOL_CONTEXT_FALLBACK,
    _SEED_CLI_VERSION,
    _SEED_SANITIZE_RE,
    _XAI_RATES,
    _XAI_DEFAULT_RATES,
    _REFRESH_BUFFER_MS,
    _bot_slug_from_session_key,
    _fmt_tokens,
    _usage_input_total,
    _estimate_proxy_cost_usd,
    _pick_iteration_usage,
    _read_latest_compact_metadata,
    _token_expired_or_stale,
    _get_fresh_oauth_token,
    _is_cli_crash,
    _is_auth_failure,
)

logger = logging.getLogger("claude_code_bridge.bridge")


@dataclass
class _PendingApproval:
    """Process-local authority minted only when this bridge gates a real call."""

    grant_key: str
    origin_request_id: str
    expires_at: float
    state: str = "pending"
    grant_expires_at: float = 0.0
    continuation_request_id: str = ""


class ClaudeApprovalMixin:
    """Claude tool approval gate + permission hooks (TASK-555).

    Split out of ``ClaudeCodeBridge`` (TASK-555); composed back via
    inheritance so ``self.*`` state and sibling-mixin methods resolve
    on the assembled instance.
    """

    async def _get_policy_bundle(self) -> PolicyBundle:
        """Return the approval-policy bundle, TTL-cached, fetched from the app.

        Conditional on the cached etag so an unchanged bundle costs one cheap
        round-trip and no re-parse. On any fetch error the cached bundle is
        kept; if there's no cache yet, an empty bundle is returned (no policies
        → default-allow), which is the fail-open posture.
        """
        now = time.monotonic()
        if (
            self._policy_bundle is not None
            and (now - self._policy_bundle_fetched_at) < self._policy_bundle_ttl
        ):
            return self._policy_bundle
        if not self._app_api_url:
            self._policy_bundle = self._policy_bundle or PolicyBundle(version=1, etag="", policies=[])
            self._policy_bundle_fetched_at = now
            return self._policy_bundle
        try:
            params = {}
            if self._policy_bundle is not None and self._policy_bundle.etag:
                params["etag"] = self._policy_bundle.etag
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(
                    f"{self._app_api_url}/v1/tool-approval-policies/bundle", params=params
                )
                resp.raise_for_status()
                data = resp.json()
            self._policy_bundle_fetched_at = now
            self._policy_fetch_ok = True
            if data.get("unchanged"):
                return self._policy_bundle  # type: ignore[return-value]
            self._policy_bundle = PolicyBundle.from_dict(data)
            logger.debug(
                "Fetched approval bundle: etag=%s policies=%d",
                self._policy_bundle.etag, len(self._policy_bundle.policies),
            )
            return self._policy_bundle
        except Exception as e:  # noqa: BLE001
            # Keep the last-known bundle; only warn occasionally to avoid spam.
            logger.warning("Approval bundle fetch failed (%s); using cached/empty", e)
            self._policy_bundle_fetched_at = now  # back off until next TTL
            self._policy_fetch_ok = False
            return self._policy_bundle or PolicyBundle(version=1, etag="", policies=[])

    def _pending_approval_records(self) -> dict[tuple[str, str], _PendingApproval]:
        # Separate from the old global key->expiry map: never promote legacy
        # unscoped grants during a rolling update. Restart loses authority safely.
        if not hasattr(self, "_pending_approvals"):
            self._pending_approvals: dict[tuple[str, str], _PendingApproval] = {}
        now = time.monotonic()
        for key, record in list(self._pending_approvals.items()):
            if record.expires_at <= now:
                del self._pending_approvals[key]
        return self._pending_approvals

    def _remember_approval(
        self, grant_key: str, session_key: str, approval_request_id: str,
        origin_request_id: str,
    ) -> None:
        if not all((grant_key, session_key, approval_request_id, origin_request_id)):
            return
        records = self._pending_approval_records()
        key = (session_key, approval_request_id)
        # Never overwrite granted/consumed records on a duplicate hook/event.
        # Bounded memory; saturation denies new grants rather than evicting live
        # replay protection. Expired requests require a freshly gated tool call.
        if key not in records and len(records) < 10000:
            records[key] = _PendingApproval(
                grant_key, origin_request_id, time.monotonic() + 86400,
            )

    def _grant_approval(
        self, grant_key: str, ttl_seconds: float, *,
        session_key: str = "", request_id: str = "",
    ) -> bool:
        """Activate the exact original approval request, once, in its session.

        ``request_id`` is the approval/tool-use ID, NOT the continuation turn's
        agent request ID (the existing Redis producer contract). Duplicate
        deliveries never extend expiry or resurrect a consumed grant. Missing
        context, unsolicited grants, and pre-upgrade requests fail closed.
        """
        if not all((grant_key, session_key, request_id)) or not math.isfinite(ttl_seconds):
            return False
        if ttl_seconds <= 0:
            return False
        record = self._pending_approval_records().get((session_key, request_id))
        if record is None or record.state != "pending" or record.grant_key != grant_key:
            return False
        # Same durable identity as approval_request_store._continuation_id and
        # approval_continuations._continuation_identity. The grant transport
        # carries the approval ID, not an arbitrary future turn's request ID.
        record.continuation_request_id = (
            "req_delivery_approval_approval-cont-"
            + sha256(request_id.encode()).hexdigest()[:32]
        )
        record.state = "granted"
        record.grant_expires_at = min(
            record.expires_at, time.monotonic() + min(ttl_seconds, 600.0),
        )
        logger.info("Approval grant stored: %s session=%s request=%s", grant_key[:12], session_key, request_id)
        return True

    def _consume_grant(
        self, grant_key: str, *, session_key: str = "", request_id: str = "",
    ) -> bool:
        """Consume a grant only in its server-owned continuation request.

        Same session and invocation alone do not authorize an unrelated turn.
        The deterministic continuation ID binds consumption to the approved
        request; client-owned arbitrary turn IDs fail closed until their
        producer supplies this same binding. No global/subject-only fallback.
        """
        if not all((grant_key, session_key, request_id)):
            return False
        now = time.monotonic()
        for (session, _), record in self._pending_approval_records().items():
            if (session == session_key and record.grant_key == grant_key
                    and record.state == "granted" and record.grant_expires_at > now
                    and record.origin_request_id != request_id
                    and record.continuation_request_id == request_id):
                record.state = "consumed"
                return True
        return False

    def _decide_approval(
        self, tool_name: str, tool_input: dict, *, cwd: str | None = None,
    ) -> ApprovalDecision:
        """Evaluate matching separately from exact input/cwd authorization."""
        bundle = self._policy_bundle or PolicyBundle(version=1, etag="", policies=[])
        try:
            return evaluate_policies(
                bundle.policies, self._backend_name, tool_name, tool_input,
                cwd=cwd if cwd is not None else self._cwd, exact_invocation=True,
            )
        except (TypeError, ValueError, RecursionError):
            # Do not let an unhashable/non-JSON invocation fall through the
            # generic hook-error default allow or acquire a lossy identity.
            return ApprovalDecision(
                action=PolicyAction.DENY, subject="Invalid tool invocation identity",
            )

    async def _evaluate_tool_gate(
        self,
        tool_name: str,
        tool_input: dict,
        ctx: ToolPermissionContext,
        request_id: str,
        session_key: str,
        seq_holder: list[int],
        *,
        cwd: str | None = None,
    ):
        """Permission decision for a non-question tool (TASK-292).

        Returns a PermissionResult. On require_approval (no live grant) emits an
        APPROVAL_REQUIRED event and returns a deferred deny so the turn ends
        cleanly — the user's decision arrives as a continuation turn.
        """
        await self._get_policy_bundle()

        # Fail-closed: the app is unreachable AND the operator chose safety over
        # availability → deny every tool until policies can be fetched again.
        if self._approval_fail_closed and not self._policy_fetch_ok:
            logger.warning(
                "Approval fail-closed: policies unreachable, denying %s", tool_name
            )
            self._emit_approval_decision(
                ApprovalDecision(action=PolicyAction.DENY, subject=""),
                tool_name, (ctx.tool_use_id or "").strip(),
                request_id, session_key, seq_holder, outcome="policy_unavailable",
            )
            return PermissionResultDeny(
                message=(
                    "[Tool execution is paused: the approval-policy service is "
                    "unreachable and this bridge is configured fail-closed. "
                    "Acknowledge and end your turn.]"
                ),
                interrupt=False,
            )

        decision = self._decide_approval(tool_name, tool_input, cwd=cwd)
        self._emit_approval_decision(
            decision, tool_name, (ctx.tool_use_id or "").strip(),
            request_id, session_key, seq_holder,
        )

        if decision.action is PolicyAction.ALLOW:
            return PermissionResultAllow()

        if decision.action is PolicyAction.DENY:
            logger.info(
                "Tool DENIED by policy %s: %s %r",
                getattr(decision.policy, "id", "?"), tool_name, decision.subject[:80],
            )
            return PermissionResultDeny(
                message=(
                    f"[This action is blocked by an approval policy and cannot be "
                    f"run: {decision.subject[:200]}. Do not retry it. Continue "
                    f"without it or explain what you need.]"
                ),
                interrupt=False,
            )

        # ---- require_approval ----
        if self._consume_grant(
            decision.grant_key, session_key=session_key, request_id=request_id,
        ):
            logger.info(
                "Tool ALLOWED by prior approval grant: %s %r",
                tool_name, decision.subject[:80],
            )
            # TASK-305: mark this re-attempt as pre-approved so the UI can show
            # the gold/lock affordance on the exact card that ran with prior
            # authorization. The bridge is the single source of truth for this.
            self._emit_tool_preapproved(
                decision, tool_name, (ctx.tool_use_id or "").strip(),
                request_id, session_key, seq_holder,
            )
            return PermissionResultAllow()

        tool_use_id = (ctx.tool_use_id or "").strip()
        if not tool_use_id:
            # No id → app can't key a persistent request; defer cleanly anyway.
            logger.warning(
                "Approval-gated %s with no tool_use_id — deferring without persistence",
                tool_name,
            )
            return PermissionResultDeny(message=self._APPROVAL_PENDING_ACK, interrupt=False)

        self._emit_approval_required(
            decision, tool_name, tool_input, tool_use_id,
            request_id, session_key, seq_holder,
        )
        return PermissionResultDeny(message=self._APPROVAL_PENDING_ACK, interrupt=False)

    def _emit_approval_decision(
        self, decision, tool_name, tool_use_id, request_id, session_key,
        seq_holder, *, outcome=None,
    ) -> None:
        """Existing Redis event transport only; never wait for app HTTP/DB.

        Best-effort emission, not a durable receipt. App commits received events.
        No arguments, command text, cwd, or capability enters this audit payload.
        """
        from agent_bridge.approval_audit import decision_metadata
        try:
            seq_holder[0] += 1
            self._publish_event(
                request_id, session_key, seq_holder[0],
                kind=AgentEventKind.APPROVAL_DECISION,
                tool_name=tool_name, tool_use_id=tool_use_id,
                extra_raw=decision_metadata(decision, self._policy_bundle, outcome=outcome),
            )
        except Exception:
            logger.warning("Bridge decision audit emission failed request=%s tool_use=%s",
                           request_id, tool_use_id)

    def _emit_approval_required(
        self,
        decision: ApprovalDecision,
        tool_name: str,
        tool_input: dict,
        tool_use_id: str,
        request_id: str,
        session_key: str,
        seq_holder: list[int],
    ) -> None:
        """Publish an APPROVAL_REQUIRED event for a gated tool (TASK-292).

        Single source of truth for the approval-request event payload, shared by
        the can_use_tool gate (_evaluate_tool_gate) and the PreToolUse hook gate
        (_evaluate_tool_gate_hook). Best-effort: a publish failure is logged,
        never raised — the caller still ends the turn with the pending-ack.
        """
        self._remember_approval(decision.grant_key, session_key, tool_use_id, request_id)
        try:
            seq_holder[0] += 1
            self._publish_event(
                request_id, session_key, seq_holder[0],
                kind=AgentEventKind.APPROVAL_REQUIRED,
                tool_name=tool_name,
                tool_arguments=tool_input if isinstance(tool_input, dict) else {},
                tool_use_id=tool_use_id,
                extra_raw={
                    "policy_id": getattr(decision.policy, "id", None),
                    "severity": decision.severity.value,
                    "category": getattr(decision.policy, "category", None),
                    "subject": decision.subject,
                    "label": decision.label,
                    "prompt": decision.prompt,
                    "grant_key": decision.grant_key,
                    "action": decision.action.value,
                },
            )
        except Exception:
            logger.exception(
                "Failed to publish APPROVAL_REQUIRED for tool_use_id=%s", tool_use_id,
            )

        logger.info(
            "Tool gated (approval required): %s tool_use_id=%s policy=%s — turn ends",
            tool_name, tool_use_id, getattr(decision.policy, "id", "?"),
        )

    def _emit_tool_preapproved(
        self,
        decision: ApprovalDecision,
        tool_name: str,
        tool_use_id: str,
        request_id: str,
        session_key: str,
        seq_holder: list[int],
    ) -> None:
        """Publish a TOOL_PREAPPROVED event for a re-attempt that consumed a grant.

        Single source of truth for the pre-approved marker (TASK-305), shared by
        both gate paths (can_use_tool and the PreToolUse hook). The bridge is the
        only place that knows a tool ran because a one-shot grant was consumed —
        the client must not re-derive it. Best-effort: a publish failure is
        logged, never raised — the tool still runs.
        """
        self._emit_approval_decision(
            decision, tool_name, tool_use_id, request_id, session_key,
            seq_holder, outcome="grant_allowed",
        )
        if not tool_use_id:
            return
        try:
            seq_holder[0] += 1
            self._publish_event(
                request_id, session_key, seq_holder[0],
                kind=AgentEventKind.TOOL_PREAPPROVED,
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                extra_raw={
                    "policy_id": getattr(decision.policy, "id", None),
                    "severity": decision.severity.value,
                    "grant_key": decision.grant_key,
                },
            )
        except Exception:
            logger.exception(
                "Failed to publish TOOL_PREAPPROVED for tool_use_id=%s", tool_use_id,
            )

    @staticmethod
    def _hook_deny(reason: str) -> dict:
        """PreToolUse hook output that blocks a tool with a model-visible reason.

        The reason is surfaced to the model as the tool's failure text (verified
        live: the model reports the permissionDecisionReason as the block error).
        """
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }

    async def _evaluate_tool_gate_hook(
        self,
        tool_name: str,
        tool_input: dict,
        tool_use_id: str,
        request_id: str,
        session_key: str,
        seq_holder: list[int],
        *,
        cwd: str | None = None,
    ) -> dict:
        """Approval gate as a PreToolUse hook decision (TASK-292).

        The live gate under permission_mode="bypassPermissions": the SDK only
        consults PreToolUse hooks (never can_use_tool) for regular tools in that
        mode. Mirrors _evaluate_tool_gate's policy logic but speaks the hook
        control plane. Returns a hook JSON output dict:

          ALLOW            → {} (no decision; tool proceeds normally)
          DENY             → "deny" + block reason
          REQUIRE_APPROVAL → consume a live one-shot grant → {} (allow); else
                             emit APPROVAL_REQUIRED and "deny" with the
                             pending-ack reason so the turn ends cleanly and the
                             user's decision arrives as a continuation turn
                             (same model as TASK-269 / the can_use_tool gate).
        """
        await self._get_policy_bundle()

        # Fail-closed: app unreachable AND operator chose safety → deny all.
        if self._approval_fail_closed and not self._policy_fetch_ok:
            logger.warning(
                "Approval fail-closed: policies unreachable, denying %s", tool_name
            )
            self._emit_approval_decision(
                ApprovalDecision(action=PolicyAction.DENY, subject=""),
                tool_name, tool_use_id, request_id, session_key, seq_holder,
                outcome="policy_unavailable",
            )
            return self._hook_deny(
                "[Tool execution is paused: the approval-policy service is "
                "unreachable and this bridge is configured fail-closed. "
                "Acknowledge and end your turn.]"
            )

        decision = self._decide_approval(tool_name, tool_input, cwd=cwd)
        self._emit_approval_decision(
            decision, tool_name, tool_use_id,
            request_id, session_key, seq_holder,
        )

        if decision.action is PolicyAction.ALLOW:
            return {}

        if decision.action is PolicyAction.DENY:
            logger.info(
                "Tool DENIED by policy %s: %s %r",
                getattr(decision.policy, "id", "?"), tool_name, decision.subject[:80],
            )
            return self._hook_deny(
                f"[This action is blocked by an approval policy and cannot be "
                f"run: {decision.subject[:200]}. Do not retry it. Continue "
                f"without it or explain what you need.]"
            )

        # ---- require_approval ----
        if self._consume_grant(
            decision.grant_key, session_key=session_key, request_id=request_id,
        ):
            logger.info(
                "Tool ALLOWED by prior approval grant: %s %r",
                tool_name, decision.subject[:80],
            )
            # TASK-305: mark this re-attempt as pre-approved (gold/lock card).
            self._emit_tool_preapproved(
                decision, tool_name, (tool_use_id or "").strip(),
                request_id, session_key, seq_holder,
            )
            return {}

        tool_use_id = (tool_use_id or "").strip()
        if not tool_use_id:
            # No id → app can't key a persistent request; defer cleanly anyway.
            logger.warning(
                "Approval-gated %s with no tool_use_id — deferring without persistence",
                tool_name,
            )
            return self._hook_deny(self._APPROVAL_PENDING_ACK)

        self._emit_approval_required(
            decision, tool_name, tool_input, tool_use_id,
            request_id, session_key, seq_holder,
        )
        return self._hook_deny(self._APPROVAL_PENDING_ACK)

    async def _handle_approval_grant(self, fields: dict, msg_id: str, async_redis) -> None:
        """Store a one-shot allow from an approval.grant command, then ACK."""
        try:
            grant_key = (fields.get("grant_key") or "").strip()
            ttl = float(fields.get("ttl_seconds") or "600")
            if grant_key:
                self._grant_approval(
                    grant_key, ttl,
                    session_key=(fields.get("session_key") or "").strip(),
                    request_id=(fields.get("request_id") or "").strip(),
                )
        except Exception:
            logger.exception("Failed to handle approval.grant")
        finally:
            try:
                await async_redis.xack(COMMANDS_STREAM, "claude-code-bridge", msg_id)
            except Exception:
                pass

    async def _approval_reload_listener(self, redis_url: str) -> None:
        """Drop the cached bundle on an approval:policies:reload broadcast.

        Best-effort: any failure is logged and retried; it never crashes the
        bridge. Lets an admin edit take effect without waiting out the TTL.
        """
        import redis.asyncio as aioredis

        while True:
            client = None
            pubsub = None
            try:
                # socket_timeout=None: pubsub.listen() is a long-lived blocking
                # read; redis-py 8.0's default 5s socket timeout would fire on
                # every idle interval and spin this loop. Bound only the connect.
                # (Mirrors _command_listener's XREADGROUP handling.)
                client = aioredis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_timeout=None,
                    socket_connect_timeout=5,
                    health_check_interval=30,
                )
                pubsub = client.pubsub()
                await pubsub.subscribe("approval:policies:reload")
                logger.info("Approval reload listener subscribed")
                async for msg in pubsub.listen():
                    if msg.get("type") != "message":
                        continue
                    self._policy_bundle_fetched_at = 0.0  # force refetch next call
                    logger.info("Approval bundle cache invalidated by reload broadcast")
            except asyncio.CancelledError:
                # Clean shutdown — release the pubsub connection.
                if pubsub is not None:
                    try:
                        await pubsub.aclose()
                    except Exception:
                        pass
                if client is not None:
                    try:
                        await client.aclose()
                    except Exception:
                        pass
                raise
            except Exception:
                logger.warning("Approval reload listener error; retrying in 5s", exc_info=True)
                # Close the failed client so retries don't leak connections.
                if pubsub is not None:
                    try:
                        await pubsub.aclose()
                    except Exception:
                        pass
                if client is not None:
                    try:
                        await client.aclose()
                    except Exception:
                        pass
                await asyncio.sleep(5)

    @staticmethod
    def _is_ask_user_question(tool_name: str) -> bool:
        """Match AskUserQuestion regardless of MCP namespacing.

        The SDK built-in is just ``AskUserQuestion``, but if it ever shows up
        prefixed by an MCP server (``mcp__<server>__AskUserQuestion``) we want
        to intercept that too.
        """
        if not tool_name:
            return False
        tail = tool_name.rsplit("__", 1)[-1]
        return tail == "AskUserQuestion"

    def _make_can_use_tool(
        self,
        *,
        request_id: str,
        session_key: str,
        seq_holder: list[int],
        question_answer_wait_seconds: int = 0,
    ):
        """Build the per-run can_use_tool callback bound to this turn.

        Emit the question and briefly hold the SDK tool result if the app
        requested a live wait. On timeout, CAS the app's pending row back to
        deferred before returning the existing acknowledgement.

        A real answer ALLOWs the built-in tool with the original input plus its
        native ``answers`` field (question text -> answer); the CLI echoes them
        as a normal, non-error tool result. (TASK-269's headless 'H.map' crash
        came from an ALLOW without the input passed back.) Deferral and
        dismissal are not answers, so they DENY with a message. Everything
        else gets the SDK's default allow.
        """

        async def can_use_tool(
            tool_name: str,
            tool_input: dict,
            ctx: ToolPermissionContext,
        ):
            if not self._is_ask_user_question(tool_name):
                # TASK-292: the approval gate moved to the PreToolUse hook
                # (_make_pre_tool_use_hook). Under bypassPermissions the SDK
                # doesn't call can_use_tool for regular tools anyway, and we want
                # EXACTLY ONE gate to avoid double-emitting APPROVAL_REQUIRED.
                # So allow here; the hook is the sole policy enforcement point.
                return PermissionResultAllow()

            # QDIAG (TASK-413): the callback fired for an AskUserQuestion. If the
            # model emitted the block (see "QDIAG model-emitted") but this line is
            # ABSENT, the SDK/control-channel dropped the question before reaching
            # us — a pipeline bug, not the model. Logged at ENTRY so it survives
            # even the no-tool_use_id / exception paths below.
            logger.info(
                "QDIAG can_use_tool ENTER AskUserQuestion tool_use_id=%s session=%s",
                (ctx.tool_use_id or "").strip() or "<none>", session_key,
            )

            tool_use_id = (ctx.tool_use_id or "").strip()
            if not tool_use_id:
                # Without a tool_use_id the app can't key a persistent question
                # row, but we can still defer cleanly so the turn ends instead
                # of hanging.  The model just won't get a follow-up answer.
                logger.warning(
                    "AskUserQuestion intercepted with no tool_use_id — deferring without persistence"
                )
                return PermissionResultDeny(message=self._DEFERRED_ACK, interrupt=False)

            # Without an app URL the timeout CAS cannot run safely. Preserve
            # deferred behavior rather than claiming a live window we cannot end.
            live_wait = max(0, question_answer_wait_seconds) if self._app_api_url else 0
            future = None
            deadline = None
            if live_wait:
                from datetime import timedelta
                deadline = (datetime.now(timezone.utc) + timedelta(seconds=live_wait)).isoformat()
                future = asyncio.get_running_loop().create_future()
                self._pending_question_futures[tool_use_id] = (session_key, future)

            # Emit AWAIT_TOOL_RESULT so the app persists the question, records
            # it on the turn (end_reason="question", question_id), and fans it
            # out to the UI.  Ordered in the same seq as surrounding deltas.
            try:
                seq_holder[0] += 1
                self._publish_event(
                    request_id, session_key, seq_holder[0],
                    kind=AgentEventKind.AWAIT_TOOL_RESULT,
                    tool_name=tool_name,
                    tool_arguments=tool_input if isinstance(tool_input, dict) else {},
                    tool_use_id=tool_use_id,
                    extra_raw={"live_deadline": deadline} if deadline else None,
                )
            except Exception:
                self._pending_question_futures.pop(tool_use_id, None)
                logger.exception("Failed to publish AWAIT_TOOL_RESULT for tool_use_id=%s", tool_use_id)
                return PermissionResultDeny(message=self._DEFERRED_ACK, interrupt=False)

            if future is not None:
                try:
                    try:
                        answer = await asyncio.wait_for(asyncio.shield(future), live_wait)
                    except asyncio.TimeoutError:
                        # The app's atomic transition arbitrates answers at the
                        # boundary. If arbitration is unavailable, the deadline
                        # itself prevents later answers from claiming this Future.
                        try:
                            async with httpx.AsyncClient(timeout=10) as client:
                                response = await client.post(
                                    f"{self._app_api_url}/v1/chat/questions/{tool_use_id}/timeout",
                                    json={"session_key": session_key},
                                )
                                response.raise_for_status()
                            outcome = response.json()
                            if outcome.get("expired"):
                                answer = self._DEFERRED_ACK
                            elif isinstance(outcome.get("answers"), dict) and outcome["answers"]:
                                answer = outcome["answers"]
                            elif outcome.get("message"):
                                answer = outcome["message"]
                            else:
                                # Unknown/unfinished status must not hold the turn.
                                answer = self._DEFERRED_ACK
                        except Exception:
                            logger.warning(
                                "Question timeout arbitration failed for tool_use_id=%s",
                                tool_use_id, exc_info=True,
                            )
                            answer = future.result() if future.done() and not future.cancelled() else self._DEFERRED_ACK
                    if isinstance(answer, dict):
                        # A real answer: allow the built-in tool with its native
                        # ``answers`` input so the SDK records a normal (non-error)
                        # tool result. Original input must be passed through —
                        # the tool maps over ``questions``.
                        return PermissionResultAllow(
                            updated_input={**(tool_input or {}), "answers": answer},
                        )
                    # Not an answer (deferral/dismissal): deny with the message.
                    return PermissionResultDeny(message=answer, interrupt=False)
                finally:
                    self._pending_question_futures.pop(tool_use_id, None)
                    if not future.done():
                        future.cancel()

            return PermissionResultDeny(message=self._DEFERRED_ACK, interrupt=False)

        return can_use_tool

    def _make_pre_tool_use_hook(
        self,
        *,
        request_id: str,
        session_key: str,
        seq_holder: list[int],
        task_turn_capability: str | None = None,
    ):
        """Build the per-run PreToolUse hook bound to this turn (TASK-292).

        This is the live approval gate. It fires for every tool regardless of
        permission_mode (verified live under bypassPermissions). AskUserQuestion
        is handed back to the SDK (return {}) so its dedicated can_use_tool
        deferral (TASK-269) keeps owning it — the policy gate must not
        double-handle the question flow. Everything else goes through the policy
        engine via _evaluate_tool_gate_hook.
        """

        async def pre_tool_use(input_data, tool_use_id, context):
            tool_name = ""
            try:
                tool_name = input_data.get("tool_name") or ""
                if self._is_ask_user_question(tool_name):
                    return {}
                tool_input = input_data.get("tool_input")
                if not isinstance(tool_input, dict):
                    return self._hook_deny("[Invalid tool input; tool blocked.]")
                tuid = (tool_use_id or input_data.get("tool_use_id") or "")
                # TASK-639: this first-party MCP server is the authoritative
                # approval interception point. The bridge stamps exact call
                # identity into updatedInput but must not also policy-gate it,
                # otherwise one policy creates two approval cards.
                if tool_name.startswith("mcp__bawthub__"):
                    if not task_turn_capability:
                        # Generic/manual calls can still reach the server without
                        # continuation context; ordinary app turns should always
                        # have the capability. Do not fabricate authority here.
                        return {}
                    updated_input = dict(tool_input)
                    updated_input[MCP_CALL_CONTEXT_KEY] = mint_mcp_call_context(
                        capability=task_turn_capability,
                        tool_name=tool_name.rsplit("__", 1)[-1],
                        tool_input=tool_input,
                        tool_use_id=tuid,
                        agent_request_id=request_id,
                        session_key=session_key,
                        backend=self._backend_name,
                    )
                    return {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "updatedInput": updated_input,
                        }
                    }
                cwd = input_data.get("cwd")
                if "cwd" in input_data and (not isinstance(cwd, str) or not cwd):
                    return self._hook_deny("[Invalid execution cwd; tool blocked.]")
                return await self._evaluate_tool_gate_hook(
                    tool_name, tool_input, tuid,
                    request_id, session_key, seq_holder, cwd=cwd,
                )
            except Exception:
                logger.exception(
                    "PreToolUse approval gate errored for %s", tool_name or "?"
                )
                # Match the bundle-fetch posture: fail-closed denies on error,
                # otherwise fail-open so a gate bug can't wedge every tool.
                if self._approval_fail_closed:
                    return self._hook_deny(
                        "[Approval gate error and bridge is fail-closed; "
                        "tool blocked. Acknowledge and end your turn.]"
                    )
                return {}

        return pre_tool_use
