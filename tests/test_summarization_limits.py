"""TASK-858: summarization prompt/chunk budgets follow the model's real window.

The regression these guard: both in-process callers of
``summarize_session_with_client`` passed no ``max_prompt_tokens``, inheriting a
6000-token default. With ``maintenance_model`` on a 1M-context model, ordinary
sessions were refused at ~6.3k tokens and left heuristic forever.
"""

from types import SimpleNamespace

import pytest

from llm_bawt.memory.summarization_limits import (
    MAX_SUMMARIZATION_PROMPT_TOKENS,
    UNKNOWN_MODEL_LIMITS,
    resolve_summarization_limits,
)


def _config(window):
    """Minimal stand-in for Config: only get_model_context_window is used."""
    return SimpleNamespace(get_model_context_window=lambda alias=None: window)


def test_unknown_window_falls_back_to_conservative_budget():
    assert resolve_summarization_limits(_config(0), "mystery") == UNKNOWN_MODEL_LIMITS
    assert resolve_summarization_limits(_config(None), "mystery") == UNKNOWN_MODEL_LIMITS


def test_raising_config_does_not_propagate():
    """A catalog miss must degrade to the conservative budget, not 500."""

    def _boom(alias=None):
        raise RuntimeError("catalog unavailable")

    config = SimpleNamespace(get_model_context_window=_boom)
    assert resolve_summarization_limits(config, "grok-4.3") == UNKNOWN_MODEL_LIMITS


def test_large_window_far_exceeds_the_old_6000_default():
    """grok-4.3 (1M) must not refuse a ~6.4k-token session."""
    max_prompt, max_chunk = resolve_summarization_limits(_config(1_000_000), "grok-4.3")
    assert max_prompt > 6000
    # The prompt that triggered the bug: 25,440 chars // 4 == 6360 tokens.
    assert max_prompt > 25_440 // 4
    # Very large windows summarize in one pass.
    assert max_chunk == 0


def test_large_window_is_clamped_to_the_practical_ceiling():
    max_prompt, _ = resolve_summarization_limits(_config(2_000_000), "huge")
    assert max_prompt == MAX_SUMMARIZATION_PROMPT_TOKENS


def test_small_local_window_reserves_output_and_margin():
    """A 32k local model keeps headroom for the summary itself."""
    max_prompt, max_chunk = resolve_summarization_limits(_config(32_768), "dolphin")
    assert 6000 <= max_prompt < 32_768
    assert 0 < max_chunk <= max_prompt


@pytest.mark.parametrize("window", [1_000, 2_048, 6_000, 8_192])
def test_known_small_windows_fit_prompt_and_output_without_a_floor(window):
    """Unknown-model defaults must never override a known smaller window."""
    max_prompt, max_chunk = resolve_summarization_limits(_config(window), "tiny")
    assert 0 < max_chunk <= max_prompt
    assert max_prompt + 512 < window


@pytest.mark.parametrize("window", [1, 256, 512])
def test_window_without_room_for_summary_output_is_rejected(window):
    with pytest.raises(ValueError, match="context window"):
        resolve_summarization_limits(_config(window), "impossible")


def test_chunking_boundary_at_200k():
    below, _ = resolve_summarization_limits(_config(199_999), "below")
    _, chunk_below = resolve_summarization_limits(_config(199_999), "below")
    _, chunk_at = resolve_summarization_limits(_config(200_000), "at")
    assert chunk_below > 0
    assert chunk_at == 0
    assert below > 0


def test_route_adapter_delegates_to_the_shared_resolver():
    """The route keeps its service-shaped signature but shares the rules."""
    from llm_bawt.service.routes.history_summaries import (
        _resolve_summarization_limits,
    )

    service = SimpleNamespace(config=_config(1_000_000))
    assert _resolve_summarization_limits(service, "grok-4.3") == (
        resolve_summarization_limits(_config(1_000_000), "grok-4.3")
    )
