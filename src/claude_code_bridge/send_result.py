"""Successful Claude SDK ``ResultMessage`` finalization."""

from __future__ import annotations

import asyncio
import time
from typing import Callable

from agent_bridge.events import AgentEventKind

from ._bridge_helpers import _fmt_tokens, _read_latest_compact_metadata
from .context_breakdown import compact_context_breakdown
from .send_errors import retry_exhausted_text


class ClaudeResultMixin:
    """Normalize successful and interrupted terminal results."""

    def _publish_interrupted_done(
        self,
        *,
        request_id: str,
        session_key: str,
        seq: int,
        text: str,
        actual_model: str,
        model: str,
        bot_context_window: int | None,
        latest_assistant_usage: dict | None,
        latest_stream_usage: dict | None,
        attachments: list[dict] | None = None,
        compact_threshold: int | None = None,
    ) -> tuple[int, dict | None]:
        """Publish one partial terminal event and return its usage snapshot."""
        usage = self._compute_interrupted_usage(
            actual_model=actual_model,
            model=model,
            bot_context_window=bot_context_window,
            compact_threshold=compact_threshold,
            latest_assistant_usage=latest_assistant_usage,
            latest_stream_usage=latest_stream_usage,
        )
        seq += 1
        self._publish_event(
            request_id,
            session_key,
            seq,
            kind=AgentEventKind.ASSISTANT_DONE,
            text=text,
            model=actual_model,
            token_usage=usage,
            attachments=attachments,
        )
        return seq, usage

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
        native_context_usage: dict | None,
        compact_status: str | None,
        compact_error_msg: str | None,
        turn_session_id: str | None,
        turn_screenshot_assets: list[dict],
        reset_ack: dict | None = None,
        compact_threshold: int | None = None,
        capture_usage: Callable[[dict | None], None] | None = None,
    ) -> int:
        full_text = "".join(text_parts)
        if not full_text:
            result_text = getattr(msg, "text", "") or ""
            if not result_text:
                for block in getattr(msg, "content", []):
                    if isinstance(block, dict) and block.get("type") == "text":
                        result_text += block.get("text", "")
            full_text = result_text
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
        if native_context_usage:
            native_total = int(native_context_usage.get("totalTokens") or 0)
            native_max = int(native_context_usage.get("maxTokens") or 0)
            if native_total or native_max:
                token_usage_payload = dict(token_usage_payload or {})
                token_usage_payload["resident_tokens"] = native_total or None
                token_usage_payload["resident_source"] = "claude_sdk_context"
                token_usage_payload["context_window"] = native_max or ctx_window
                token_usage_payload["context_percentage"] = native_context_usage.get(
                    "percentage"
                )
                token_usage_payload["auto_compact_enabled"] = native_context_usage.get(
                    "isAutoCompactEnabled"
                )
                token_usage_payload["auto_compact_threshold"] = native_context_usage.get(
                    "autoCompactThreshold"
                )
                ctx_window = native_max or ctx_window

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
    ) -> int:
        """Read /context after DONE on the still-owned SDK client.

        USAGE_UPDATE carries a complete final snapshot so the app's ordered run
        consumer can persist it without replacing terminal cost/output values.
        On proxy turns the CLI's category counts are estimates: keep the upstream
        resident total and window authoritative. A failed or slow control read
        leaves the already-published turn alone.
        """
        if isinstance(final_usage, dict) and final_usage.get("context_action") == "compact":
            return seq
        started = time.monotonic()
        native = await self._read_native_context_usage(sdk_client)
        if not native:
            return seq
        breakdown = compact_context_breakdown(native, read_ms=(time.monotonic() - started) * 1000)
        if breakdown is None:
            return seq
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
