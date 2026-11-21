"""Successful Claude SDK ``ResultMessage`` finalization."""

from __future__ import annotations

import asyncio
import time
from typing import Callable

from agent_bridge.events import AgentEventKind

from ._bridge_helpers import _fmt_tokens, _read_latest_compact_metadata
from .context_breakdown import compact_context_breakdown
from .reasoning_ledger import record_thinking, resident_reasoning
from .send_usage import CONTEXT_BREAKDOWN_UNAVAILABLE
from .send_errors import retry_exhausted_text


class ClaudeResultMixin:
    """Normalize successful and interrupted terminal results."""

    async def _finalize_result_message(
        self,
        msg,
        *,
        request_id: str,
        session_key: str,
        seq: int,
        text_parts: list[str],
        assistant_snapshot_text: str,
        api_retry_count: int,
        api_last_error: str | None,
        actual_model: str,
        model: str,
        bot_context_window: int | None,
        latest_assistant_usage: dict | None,
        latest_stream_usage: dict | None,
        cost_baseline_usd: float | None,
        compact_status: str | None,
        compact_error_msg: str | None,
        turn_session_id: str | None,
        turn_screenshot_assets: list[dict],
        reset_ack: dict | None = None,
        compact_threshold: int | None = None,
        capture_usage: Callable[[dict | None], None] | None = None,
        turn_request_totals: dict | None = None,
        prompt_cache: dict | None = None,
        estimate_reasoning: Callable[[int], tuple[int, str] | None] | None = None,
    ) -> int:
        full_text = "".join(text_parts)
        if not full_text and assistant_snapshot_text:
            full_text = assistant_snapshot_text
        if not full_text and api_retry_count > 0:
            full_text = retry_exhausted_text(api_retry_count, api_last_error)

        token_usage_payload, ctx_window, max_output = self._compute_result_usage(
            msg,
            actual_model=actual_model,
            model=model,
            bot_context_window=bot_context_window,
            compact_threshold=compact_threshold,
            latest_assistant_usage=latest_assistant_usage,
            latest_stream_usage=latest_stream_usage,
            cost_baseline_usd=cost_baseline_usd,
        )
        if compact_status == "success":
            cm = await asyncio.to_thread(
                _read_latest_compact_metadata, turn_session_id
            )
            pre = (cm or {}).get("preTokens")
            post = (cm or {}).get("postTokens")
            if post is not None:
                freed = (
                    f" ({round(100 * (pre - post) / pre)}% freed)"
                    if pre
                    else ""
                )
                note = (
                    f"\n\n✅ Compacted: {_fmt_tokens(pre)} → "
                    f"{_fmt_tokens(post)} tokens{freed}."
                )
                token_usage_payload = {
                    **(token_usage_payload or {}),
                    "input_tokens": int(post),
                    "cache_read_tokens": 0,
                    "cache_creation_tokens": 0,
                    "output_tokens": 0,
                    "resident_tokens": int(post),
                    "resident_source": "claude_compact_metadata",
                    "context_window": ctx_window,
                    "max_output_tokens": max_output,
                    # total_cost_usd / session_cost_usd ride through from the
                    # spread above: the turn delta, never the session total.
                    "context_action": "compact",
                    "context_action_outcome": "success",
                    "context_action_pre_tokens": pre,
                    "context_action_post_tokens": post,
                }
            else:
                # The SDK said success but the transcript had no readable
                # boundary metadata. The compaction still happened: record
                # the outcome so the session prompt pin (TASK-1318) and the
                # lifecycle view see the boundary, without inventing sizes.
                note = "\n\n✅ Conversation compacted."
                token_usage_payload = {
                    **(token_usage_payload or {}),
                    "context_action": "compact",
                    "context_action_outcome": "success",
                }
            seq += 1
            text_parts.append(note)
            self._publish_event(
                request_id, session_key, seq,
                kind=AgentEventKind.ASSISTANT_DELTA,
                text=note,
            )
            full_text = "".join(text_parts)
        elif compact_status == "failed":
            token_usage_payload = {
                **(token_usage_payload or {}),
                "context_action": "compact",
                "context_action_outcome": "failed",
                "context_action_error": compact_error_msg,
            }
            note = f"\n\nℹ️ Nothing to compact — {compact_error_msg}"
            seq += 1
            text_parts.append(note)
            self._publish_event(
                request_id, session_key, seq,
                kind=AgentEventKind.ASSISTANT_DELTA,
                text=note,
            )
            full_text = "".join(text_parts)

        if token_usage_payload and turn_request_totals:
            # TASK-1961: every upstream request in the turn, not just the last.
            token_usage_payload = {**token_usage_payload, "turn_request_totals": turn_request_totals}
        if token_usage_payload and estimate_reasoning is not None:
            # TASK-1586: Anthropic doesn't split thinking out of output_tokens.
            reasoning = estimate_reasoning(int(token_usage_payload.get("output_tokens") or 0))
            if reasoning is not None:
                token_usage_payload = {
                    **token_usage_payload,
                    "reasoning_tokens": reasoning[0],
                    "reasoning_source": reasoning[1],
                }
        if prompt_cache is not None:
            token_usage_payload = {**(token_usage_payload or {}), "prompt_cache": prompt_cache}
        seq += 1
        self._publish_event(
            request_id, session_key, seq,
            kind=AgentEventKind.ASSISTANT_DONE,
            text=full_text,
            model=actual_model,
            token_usage=token_usage_payload,
            attachments=turn_screenshot_assets or None,
            **({"extra_raw": {"new_session_ack": reset_ack}} if reset_ack else {}),
        )
        if capture_usage is not None:
            capture_usage(token_usage_payload)
        return seq

    async def _publish_post_done_context(
        self, sdk_client, *, request_id: str, session_key: str, seq: int,
        model: str, final_usage: dict | None, use_proxy: bool = False,
        session_start: str | None = None,
        session_id: str | None = None,
        thinking_by_message: dict[str, int] | None = None,
    ) -> int:
        """Read /context after DONE on the still-owned SDK client.

        USAGE_UPDATE carries a complete final snapshot so the app's ordered run
        consumer can persist it without replacing terminal cost/output values.
        On proxy turns the CLI's category counts are estimates: keep the upstream
        resident total and window authoritative. A failed or slow control read
        keeps the published values and adds only ``context_breakdown_status``.
        """
        if not use_proxy and thinking_by_message:
            # TASK-2343: before the read, so this turn's thinking counts as resident.
            await asyncio.to_thread(record_thinking, session_id, thinking_by_message)
        if isinstance(final_usage, dict) and final_usage.get("context_action") == "compact":
            return seq
        started = time.monotonic()
        native = await self._read_native_context_usage(sdk_client)
        breakdown = (
            compact_context_breakdown(native, read_ms=(time.monotonic() - started) * 1000)
            if native else None
        )
        if breakdown is None:
            # Re-publish the unchanged terminal usage with the reason, so the
            # persisted turn says why it has no breakdown (TASK-1700).
            if not isinstance(final_usage, dict) or not final_usage:
                return seq
            seq += 1
            self._publish_event(
                request_id, session_key, seq, kind=AgentEventKind.USAGE_UPDATE,
                model=model, token_usage={
                    **final_usage, "usage_status": "final",
                    "context_breakdown_status": CONTEXT_BREAKDOWN_UNAVAILABLE,
                },
            )
            return seq
        if session_start in {"clean", "seeded"}:
            breakdown["session_start"] = session_start
        if not use_proxy and isinstance(breakdown.get("message_breakdown"), dict):
            # TASK-2343: earlier turns' thinking stays in context; its blocks are
            # redacted, so only the API-reported counts can size it.
            resident = await asyncio.to_thread(resident_reasoning, session_id)
            if resident is not None:
                breakdown["message_breakdown"]["reasoningTokens"] = resident[0]
                breakdown["message_breakdown"]["reasoningComplete"] = resident[1]
        usage = {**(final_usage or {}), "context_breakdown": breakdown, "usage_status": "final"}
        if not use_proxy:
            native_total = native.get("totalTokens")
            native_max = native.get("maxTokens")
            if isinstance(native_total, (int, float)) and native_total > 0:
                usage.update(resident_tokens=int(native_total), resident_source="claude_sdk_context")
            if isinstance(native_max, (int, float)) and native_max > 0:
                usage["context_window"] = int(native_max)
            usage["context_percentage"] = native.get("percentage")
            usage["auto_compact_enabled"] = native.get("isAutoCompactEnabled")
            usage["auto_compact_threshold"] = native.get("autoCompactThreshold")
        seq += 1
        self._publish_event(
            request_id, session_key, seq, kind=AgentEventKind.USAGE_UPDATE,
            model=model, token_usage=usage,
        )
        return seq
