"""Context-window env hints for proxy-routed Claude Code turns.

Claude Code sizes its auto-compaction from the model's context window. For a
model name it doesn't recognise (every proxy-routed upstream: ``local/…``,
``openai_chatgpt/…``, ``xai/…``) it assumes the Anthropic default of 200k and
reserves 32k of that for output. When the real window is smaller — a local
Ollama/llama-server model at 64k, say — the CLI never compacts: the upstream
silently drops the oldest messages instead (Ollama ``chatPrompt`` truncation,
verified live on qwen3.8:27b-64k: a 78,332-token prompt went in, 51,516 came
out, no error anywhere, and the model confabulated the dropped tool result).

The CLI honours two env knobs for exactly this case (both verified in the
bundled CLI source):

* ``CLAUDE_CODE_MAX_CONTEXT_TOKENS`` — the real window, applied only to model
  names that don't start with ``claude-``.
* ``CLAUDE_CODE_MAX_OUTPUT_TOKENS`` — the output reserve. It matters because the
  auto-compact threshold is ``min(0.8 * eff, eff - 13000)`` where
  ``eff = window - max_output``. Left at the 32k default on a 64k window the
  threshold lands at ~20k, i.e. compaction every turn; at 8k it lands at ~44k.

``CLAUDE_CODE_AUTO_COMPACT_WINDOW`` is NOT usable for the window itself — the
CLI floors it at 100k. It is used separately (``compact_window_env``) for the
configurable compact threshold, where the CLI takes ``min(model window,
max(100k, value))`` as its auto-compact window on every path, native or proxy.

The app resolves the true window from the model catalog and sends it down with
every command (TASK-609); this module turns that scalar into env hints. The hint
must be sent for every resolved proxy model, including windows above 200k:
otherwise the CLI's unknown-model fallback compacts large-context models early.
"""

from __future__ import annotations

#: Output reserve = window / OUTPUT_RESERVE_DIVISOR, clamped to the bounds.
OUTPUT_RESERVE_DIVISOR = 8
MIN_OUTPUT_TOKENS = 4_096
MAX_OUTPUT_TOKENS = 16_384

MAX_CONTEXT_TOKENS_ENV = "CLAUDE_CODE_MAX_CONTEXT_TOKENS"
MAX_OUTPUT_TOKENS_ENV = "CLAUDE_CODE_MAX_OUTPUT_TOKENS"
AUTO_COMPACT_WINDOW_ENV = "CLAUDE_CODE_AUTO_COMPACT_WINDOW"


def output_reserve_for_window(context_window: int) -> int:
    """Output-token reserve that leaves a usable auto-compact threshold."""
    return max(MIN_OUTPUT_TOKENS, min(MAX_OUTPUT_TOKENS, context_window // OUTPUT_RESERVE_DIVISOR))


def proxy_context_window_env(context_window: int | None) -> dict[str, str]:
    """Env hints for a resolved proxy model, or ``{}`` when unresolved.

    Claude Code does not recognize provider-qualified proxy model names, so its
    200k fallback is wrong in both directions. Propagate every positive catalog
    window: smaller models must compact before truncation, while larger models
    must not compact prematurely at the fallback threshold.
    """
    if context_window is None or context_window <= 0:
        return {}
    return {
        MAX_CONTEXT_TOKENS_ENV: str(context_window),
        MAX_OUTPUT_TOKENS_ENV: str(output_reserve_for_window(context_window)),
    }


def compact_window_env(compact_threshold: int | None) -> dict[str, str]:
    """Auto-compact window hint for any Claude Code turn, or ``{}`` when unset.

    The app resolves the value (catalog ``models.compact_threshold``, else the
    global ``agent_compact_threshold``). The CLI floors it at 100k and caps it at
    the model's context window, so no clamping is duplicated here.
    """
    if compact_threshold is None or compact_threshold <= 0:
        return {}
    return {AUTO_COMPACT_WINDOW_ENV: str(compact_threshold)}
