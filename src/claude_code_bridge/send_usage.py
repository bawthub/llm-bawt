"""Usage / context-window accounting for the Claude send path (TASK-623).

Extracted from ``ClaudeSendMixin._handle_send``'s ResultMessage branch. The
computation is a verbatim move: given the terminal ``ResultMessage`` plus the
per-iteration / stream usage snapshots captured during the turn, produce the
``token_usage_payload`` dict and the resolved ``ctx_window`` / ``max_output``
scalars the caller stamps onto ASSISTANT_DONE.

Composed onto ``ClaudeSendMixin`` so ``self._model_provider_prefix`` and the
assembled-instance state resolve normally.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from ._bridge_helpers import (
    _estimate_proxy_cost_usd,
    _pick_iteration_usage,
    _read_session_cost_baseline,
    _usage_input_total,
)

from .context_env import effective_context_window

logger = logging.getLogger("claude_code_bridge.bridge")


def turn_cost_baseline_usd(resume_id: str | None) -> float | None:
    """Session cost already on the CLI ledger before this turn runs (TASK-1054).

    Fresh sessions start at 0; resumed ones read the transcript
    (:func:`_read_session_cost_baseline`), which also proves a never-billed
    seed transcript is 0 however it was created (cold start or bare ``/new``
    on an earlier request). ``None`` = unknown baseline.
    Blocking file I/O: call via ``asyncio.to_thread``.
    """
    if not resume_id:
        return 0.0
    return _read_session_cost_baseline(resume_id)


def native_turn_cost_usd(
    session_total_usd: object, baseline_usd: float | None
) -> float | None:
    """This turn's cost = session total − pre-turn baseline; ``None`` if unknowable."""
    if baseline_usd is None or not isinstance(session_total_usd, (int, float)):
        return None
    delta = float(session_total_usd) - baseline_usd
    # A total below the baseline means the CLI did not restore the ledger we
    # read; the true split is unknowable, so let the caller estimate instead.
    # A float-noise negative (-1e-12) is a $0 turn, not a broken ledger.
    if delta < -1e-9:
        return None
    return max(delta, 0.0)


@dataclass
class LiveUsagePublisher:
    """Per-turn merge and dedupe state for provider usage snapshots."""

    bridge: Any
    request_id: str
    session_key: str
    actual_model: str
    requested_model: str
    context_window: int | None
    stream_usage: dict | None = None
    last_signature: tuple | None = None
    compact_threshold: int | None = None
    # Latest live resident context, read by the turn phase status line.
    resident_tokens: int | None = None

    def publish(
        self,
        seq: int,
        *,
        assistant_usage: dict | None,
        stream_usage: dict | None = None,
    ) -> int:
        """Merge a sparse stream frame and publish a changed live snapshot."""
        if isinstance(stream_usage, dict) and stream_usage:
            self.stream_usage = {**(self.stream_usage or {}), **stream_usage}
        usage = self.bridge._compute_live_usage(
            actual_model=self.actual_model,
            model=self.requested_model,
            bot_context_window=self.context_window,
            compact_threshold=self.compact_threshold,
            latest_assistant_usage=assistant_usage,
            latest_stream_usage=self.stream_usage,
        )
        if usage is None:
            return seq
        resident = usage.get("resident_tokens")
        if isinstance(resident, int):
            self.resident_tokens = resident
        signature = tuple(
            usage.get(key)
            for key in (
                "input_tokens",
                "cache_read_tokens",
                "cache_creation_tokens",
                "output_tokens",
                "resident_tokens",
                "context_window",
            )
        )
        if signature == self.last_signature:
            return seq
        self.last_signature = signature
        from agent_bridge.events import AgentEventKind

        seq += 1
        self.bridge._publish_event(
            self.request_id,
            self.session_key,
            seq,
            kind=AgentEventKind.USAGE_UPDATE,
            model=self.actual_model or self.requested_model,
            token_usage=usage,
        )
        return seq


class ClaudeUsageMixin:
    """Token-usage / context-window extraction for completed and interrupted turns."""

    def _compute_live_usage(
        self,
        *,
        actual_model: str,
        model: str,
        bot_context_window: int | None,
        latest_assistant_usage: dict | None,
        latest_stream_usage: dict | None,
        compact_threshold: int | None = None,
    ) -> dict | None:
        """Build a provider-neutral usage snapshot for one live model iteration."""
        usage = self._compute_interrupted_usage(
            actual_model=actual_model,
            model=model,
            bot_context_window=bot_context_window,
            compact_threshold=compact_threshold,
            latest_assistant_usage=latest_assistant_usage,
            latest_stream_usage=latest_stream_usage,
        )
        if usage is None or int(usage.get("resident_tokens", 0) or 0) <= 0:
            return None
        return {
            **usage,
            "resident_source": "live_iteration_total_input",
            "usage_status": "partial",
        }

    def _compute_interrupted_usage(
        self,
        *,
        actual_model: str,
        model: str,
        bot_context_window: int | None,
        latest_assistant_usage: dict | None,
        latest_stream_usage: dict | None,
        compact_threshold: int | None = None,
    ) -> dict | None:
        """Build an honest usage-so-far payload from the latest model iteration.

        Interrupted turns have no cumulative ``ResultMessage``. The latest
        AssistantMessage/message_delta snapshot is still provider-reported usage,
        so preserve it for the context pill while explicitly labelling the scope
        partial. No snapshot means no payload; this never invents token counts.
        """
        proxy_model = self._model_provider_prefix(actual_model or model) is not None
        snapshot = _pick_iteration_usage(
            latest_assistant_usage,
            latest_stream_usage,
            None,
            proxy_model=proxy_model,
        )
        if not snapshot:
            return None

        input_tokens = int(snapshot.get("input_tokens", 0) or 0)
        cache_read_tokens = int(
            snapshot.get("cache_read_input_tokens", 0)
            or snapshot.get("cache_read_tokens", 0)
            or 0
        )
        cache_creation_tokens = int(
            snapshot.get("cache_creation_input_tokens", 0)
            or snapshot.get("cache_creation_tokens", 0)
            or 0
        )
        output_tokens = int(snapshot.get("output_tokens", 0) or 0)
        resident_tokens = input_tokens + cache_read_tokens + cache_creation_tokens
        if resident_tokens == 0 and output_tokens == 0:
            return None

        usage_for_cost = {
            "input_tokens": input_tokens,
            "cache_read_input_tokens": cache_read_tokens,
            "cache_creation_input_tokens": cache_creation_tokens,
            "output_tokens": output_tokens,
        }
        cost = (
            _estimate_proxy_cost_usd(actual_model or model, usage_for_cost)
            if proxy_model
            else None
        )
        return {
            "input_tokens": input_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_creation_tokens": cache_creation_tokens,
            "output_tokens": output_tokens,
            "resident_tokens": resident_tokens,
            "resident_source": "interrupted_iteration_total_input",
            "context_window": effective_context_window(bot_context_window, compact_threshold),
            "total_cost_usd": cost,
            "usage_status": "partial",
        }

    @staticmethod
    async def _read_native_context_usage(sdk_client) -> dict | None:
        if sdk_client is None:
            return None
        try:
            # The session lock stays held until teardown: never delay a queued
            # successor by more than this bounded post-DONE control request.
            return dict(await asyncio.wait_for(
                sdk_client.get_context_usage(), timeout=0.9
            ))
        except Exception as context_error:
            logger.debug("Native context usage unavailable: %s", context_error)
            return None

    def _compute_result_usage(
        self,
        msg,
        *,
        actual_model: str,
        model: str,
        bot_context_window: int | None,
        latest_assistant_usage: dict | None,
        latest_stream_usage: dict | None,
        cost_baseline_usd: float | None,
        compact_threshold: int | None = None,
    ) -> tuple[dict | None, object, object]:
        """Return ``(token_usage_payload, ctx_window, max_output)``.

        ``cost_baseline_usd`` is the session cost before this turn
        (:func:`turn_cost_baseline_usd`); ``total_cost_usd`` is this turn's
        cost, never the CLI's running session total (TASK-1054).

        Mirrors the monolith exactly: a failure mid-extraction is swallowed and
        whatever partial ``ctx_window`` / ``max_output`` were resolved (plus a
        ``None`` payload) are returned, so the caller's downstream compact
        override sees the same values it did before the split.
        """
        # Extract token usage + context window for UI surfacing.
        #
        # IMPORTANT: ResultMessage.usage is CUMULATIVE across all
        # internal API iterations in the turn — for a multi-tool-use
        # turn that re-reads cached context on each call, the summed
        # cache_read_input_tokens can exceed the context_window itself
        # and produce nonsense >100% counters in the UI. We instead
        # use the LAST AssistantMessage's per-iteration usage, which
        # represents the model's final view of the context (what the
        # user actually wants to see as "context fullness"). Cumulative
        # output_tokens still come from ResultMessage since they genuinely
        # accumulate across the turn. ResultMessage.total_cost_usd is the
        # whole SESSION's running total, so it is reduced to a turn delta.
        #
        # ResultMessage.model_usage is keyed by model id and exposes
        # the model's contextWindow + maxOutputTokens.
        token_usage_payload: dict | None = None
        ctx_window = None
        max_output = None
        try:
            cumulative_usage = getattr(msg, "usage", None) or {}
            proxy_model = self._model_provider_prefix(
                actual_model or model
            ) is not None
            # Prefer stream message_delta for proxy providers
            # (xAI/ChatGPT/z.ai) — AssistantMessage often keeps
            # message_start zeros or a partial merge.
            iter_usage = _pick_iteration_usage(
                latest_assistant_usage,
                latest_stream_usage,
                cumulative_usage,
                proxy_model=proxy_model,
            )
            # Only an actual AssistantMessage/message_delta snapshot represents
            # one model invocation's resident prompt. ResultMessage.usage is a
            # turn-wide aggregate across tool iterations and must never become a
            # context gauge merely because it was the only available fallback.
            has_iteration_snapshot = any(
                iter_usage is candidate
                for candidate in (latest_assistant_usage, latest_stream_usage)
                if isinstance(candidate, dict)
            )
            model_usage = getattr(msg, "model_usage", None) or {}
            ctx_window = None
            max_output = None
            if isinstance(model_usage, dict):
                # Prefer the actual model we ran on; fall back to any entry.
                mu_entry = (
                    model_usage.get(actual_model)
                    if actual_model
                    else None
                )
                if mu_entry is None and model_usage:
                    mu_entry = next(iter(model_usage.values()), None)
                if isinstance(mu_entry, dict):
                    ctx_window = mu_entry.get("contextWindow")
                    max_output = mu_entry.get("maxOutputTokens")
            # TASK-609: Claude Code defaults unknown (proxy-routed)
            # models to 200k. Override with the app-resolved catalog
            # window for proxy providers only — the CLI never knows
            # xAI/OpenAI/z.ai windows. Direct-Anthropic models keep the
            # SDK's own contextWindow (the app value would agree, and
            # the SDK is authoritative for its native models).
            if bot_context_window and proxy_model:
                if not ctx_window or int(ctx_window) in (200_000, 0):
                    ctx_window = bot_context_window
                elif int(ctx_window) < bot_context_window:
                    # e.g. CLI said 200k for a 500k/1M model
                    ctx_window = bot_context_window
            ctx_window = effective_context_window(
                int(ctx_window) if ctx_window else None, compact_threshold,
            )
            if iter_usage or ctx_window:
                # z.ai reports input_tokens only in message_delta, so the
                # per-iteration AssistantMessage.usage (iter_usage) carries
                # the message_start value (0). Fall back to the cumulative
                # ResultMessage.usage, which via the SDK's last-non-zero merge
                # holds the real final-context input. No-op for Anthropic,
                # where iter_usage.input_tokens is already >0 (its
                # message_delta sends explicit 0s that updateUsage ignores).
                _input_tokens = int(iter_usage.get("input_tokens", 0) or 0)
                if _input_tokens == 0:
                    _input_tokens = int(
                        cumulative_usage.get("input_tokens", 0) or 0
                    )
                _cache_read = int(
                    iter_usage.get("cache_read_input_tokens", 0) or 0
                )
                _cache_create = int(
                    iter_usage.get("cache_creation_input_tokens", 0) or 0
                )
                # If the chosen snapshot still has zero total input but
                # cumulative does not, take cache fields from cumulative too.
                if (
                    _input_tokens + _cache_read + _cache_create
                ) == 0 and isinstance(cumulative_usage, dict):
                    _cache_read = int(
                        cumulative_usage.get(
                            "cache_read_input_tokens", 0
                        )
                        or 0
                    )
                    _cache_create = int(
                        cumulative_usage.get(
                            "cache_creation_input_tokens", 0
                        )
                        or 0
                    )
                _out_tokens = int(
                    cumulative_usage.get("output_tokens", 0) or 0
                )
                if _out_tokens == 0:
                    _out_tokens = int(
                        iter_usage.get("output_tokens", 0) or 0
                    )
                # Cost must be CUMULATIVE across the turn
                # (every internal API iteration billed). Context
                # fullness above is last-iteration only.
                if isinstance(cumulative_usage, dict) and (
                    _usage_input_total(cumulative_usage) > 0
                    or int(cumulative_usage.get("output_tokens", 0) or 0) > 0
                ):
                    usage_for_cost = {
                        "input_tokens": int(
                            cumulative_usage.get("input_tokens", 0) or 0
                        ),
                        "cache_read_input_tokens": int(
                            cumulative_usage.get(
                                "cache_read_input_tokens", 0
                            )
                            or 0
                        ),
                        "cache_creation_input_tokens": int(
                            cumulative_usage.get(
                                "cache_creation_input_tokens", 0
                            )
                            or 0
                        ),
                        "output_tokens": int(
                            cumulative_usage.get("output_tokens", 0)
                            or 0
                        ),
                    }
                else:
                    usage_for_cost = {
                        "input_tokens": _input_tokens,
                        "cache_read_input_tokens": _cache_read,
                        "cache_creation_input_tokens": _cache_create,
                        "output_tokens": _out_tokens,
                    }
                # Prefer provider-accurate cost for proxy models;
                # CLI prices unknowns as Opus-tier ($5/$25).
                cost = _estimate_proxy_cost_usd(
                    actual_model or model, usage_for_cost
                )
                session_cost = getattr(msg, "total_cost_usd", None)
                if cost is None:
                    cost = native_turn_cost_usd(session_cost, cost_baseline_usd)
                token_usage_payload = {
                    "input_tokens": _input_tokens,
                    "cache_read_tokens": _cache_read,
                    "cache_creation_tokens": _cache_create,
                    # Output is still the cumulative turn total — that's
                    # what the user generated overall, regardless of how
                    # many internal iterations produced it.
                    "output_tokens": _out_tokens,
                    "context_window": ctx_window,
                    "max_output_tokens": max_output,
                    "total_cost_usd": cost,
                    # Raw CLI running total, kept for audit; never summed.
                    "session_cost_usd": session_cost
                    if isinstance(session_cost, (int, float))
                    else None,
                }
                if has_iteration_snapshot:
                    token_usage_payload["resident_tokens"] = (
                        _input_tokens + _cache_read + _cache_create
                    )
                    token_usage_payload["resident_source"] = (
                        "final_iteration_total_input"
                    )
                if actual_model and str(actual_model).startswith(
                    ("zai/", "xai/", "moonshot/", "kimi_coding/")
                ):
                    logger.info(
                        "proxy usage: model=%s iter_in=%s stream_in=%s "
                        "cum_in=%s out=%s ctx=%s cost=%s",
                        actual_model,
                        (latest_assistant_usage or {}).get(
                            "input_tokens"
                        )
                        if isinstance(
                            latest_assistant_usage, dict
                        )
                        else None,
                        (latest_stream_usage or {}).get(
                            "input_tokens"
                        )
                        if isinstance(latest_stream_usage, dict)
                        else None,
                        cumulative_usage.get("input_tokens")
                        if isinstance(cumulative_usage, dict)
                        else None,
                        _out_tokens,
                        ctx_window,
                        cost,
                    )
        except Exception as _usage_err:
            logger.debug("Failed to extract token usage: %s", _usage_err)

        return token_usage_payload, ctx_window, max_output
