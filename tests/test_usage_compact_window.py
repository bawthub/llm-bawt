"""Usage denominators honor the same compact-window cap as the SDK (TASK-1056)."""

from types import SimpleNamespace

import pytest

from claude_code_bridge.context_env import effective_context_window
from claude_code_bridge.send_result import ClaudeResultMixin
from claude_code_bridge.send_usage import ClaudeUsageMixin, LiveUsagePublisher


class _Harness(ClaudeUsageMixin, ClaudeResultMixin):
    def __init__(self, proxy):
        self.proxy = proxy
        self.events = []

    def _model_provider_prefix(self, _model):
        return "openai_chatgpt" if self.proxy else None

    def _publish_event(self, *_args, **kwargs):
        self.events.append(kwargs)


@pytest.mark.parametrize(("window", "cap", "expected"), [
    (1_000_000, 250_000, 250_000),
    (200_000, 250_000, 200_000),
    (1_000_000, 50_000, 100_000),
    (65_536, 50_000, 65_536),
    (272_000, None, 272_000),
    (272_000, 0, 272_000),
    (272_000, -1, 272_000),
    (None, 250_000, None),
    (0, 250_000, None),
])
def test_effective_window_matches_cli_clamp(window, cap, expected):
    assert effective_context_window(window, cap) == expected


@pytest.mark.parametrize("proxy", [True, False])
def test_live_interrupted_and_result_fallback_share_capped_window(proxy):
    harness = _Harness(proxy)
    model = "openai_chatgpt/gpt" if proxy else "claude-opus-5-5"
    snapshot = {"input_tokens": 25_000, "cache_read_input_tokens": 175_000,
                "output_tokens": 10}
    publisher = LiveUsagePublisher(
        harness, "req", "caid:nick", model, model, 1_000_000,
        compact_threshold=250_000,
    )
    assert publisher.publish(0, assistant_usage=snapshot) == 1
    live = harness.events[-1]["token_usage"]
    assert live["context_window"] == 250_000
    assert live["resident_tokens"] / live["context_window"] == 0.8

    _, interrupted = harness._publish_interrupted_done(
        request_id="req", session_key="caid:nick", seq=1, text="partial",
        actual_model=model, model=model, bot_context_window=1_000_000,
        compact_threshold=250_000, latest_assistant_usage=snapshot,
        latest_stream_usage=None,
    )
    assert interrupted["context_window"] == live["context_window"]
    result, window, _ = harness._compute_result_usage(
        SimpleNamespace(usage=snapshot, model_usage={model: {
            "contextWindow": 1_000_000,
        }}, total_cost_usd=None),
        actual_model=model, model=model, bot_context_window=1_000_000,
        compact_threshold=250_000, latest_assistant_usage=snapshot,
        latest_stream_usage=None,
    )
    assert window == result["context_window"] == live["context_window"]
    assert result["resident_tokens"] == live["resident_tokens"]
