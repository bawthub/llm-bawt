"""Prompt/chunk budgets for history summarization (TASK-858).

One place decides how much conversation a summarization call may carry, so the
route path, the background HISTORY_SUMMARIZATION job and the ``/new`` pre-seed
all size their prompts off the *job model's real context window* instead of a
hardcoded floor.

Before this module existed the budget logic lived only in
``service/routes/history_summaries.py``; the two in-process callers of
``summarize_session_with_client`` passed nothing and silently inherited its
conservative 6000-token default. With a 1M-context job model that rejected
ordinary sessions at ~6.3k tokens ("Summarization prompt too large ...
skipping LLM") and left them permanently heuristic.
"""

from __future__ import annotations

from typing import Any

# Hard ceiling for summarization prompts regardless of what the model
# advertises. Large advertised windows (e.g. 1M/2M) are rarely practical for
# dense output tasks.
MAX_SUMMARIZATION_PROMPT_TOKENS = 512_000

# Budget used when the model's context window is unknown. Deliberately small:
# an unknown model is assumed to be a local/short-context one.
UNKNOWN_MODEL_LIMITS = (6000, 4000)

# Output cap for one session summary. TASK-1729: 320 truncated ~1.5% of
# summaries once background effort dropped to "none" (longer prose).
SUMMARY_MAX_OUTPUT_TOKENS = 480

# Reserve for the summary itself plus a proportional safety margin.
_RESPONSE_BUDGET_TOKENS = 512
_MIN_SAFETY_MARGIN_TOKENS = 2048
_SAFETY_MARGIN_FRACTION = 0.05

# Above this window a session is summarized in one pass instead of chunked.
_NO_CHUNK_CONTEXT_WINDOW = 200_000


def resolve_summarization_limits(
    config: Any, model_alias: str | None
) -> tuple[int, int]:
    """Return ``(max_prompt_tokens, max_chunk_tokens)`` for ``model_alias``.

    ``max_chunk_tokens`` of 0 means "do not chunk" -- the window is large
    enough to summarize a session in a single pass.
    """
    try:
        context_window = int(config.get_model_context_window(model_alias) or 0)
    except Exception:
        context_window = 0

    if context_window <= 0:
        return UNKNOWN_MODEL_LIMITS

    safety_margin = max(
        _MIN_SAFETY_MARGIN_TOKENS, int(context_window * _SAFETY_MARGIN_FRACTION)
    )
    max_prompt_tokens = max(
        6000, context_window - _RESPONSE_BUDGET_TOKENS - safety_margin
    )
    max_prompt_tokens = min(max_prompt_tokens, MAX_SUMMARIZATION_PROMPT_TOKENS)

    if context_window >= _NO_CHUNK_CONTEXT_WINDOW:
        max_chunk_tokens = 0
    else:
        max_chunk_tokens = max(4000, min(64_000, int(max_prompt_tokens * 0.7)))

    return (max_prompt_tokens, max_chunk_tokens)
