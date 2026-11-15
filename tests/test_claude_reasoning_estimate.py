"""TASK-1586: per-turn reasoning on the Claude bridge.

Exact when the OpenAI proxy reports it on every main-thread request,
otherwise estimated as output minus visible output (text + tool input).
"""

from claude_code_bridge.proxy.stream import _extract_reasoning
from claude_code_bridge.send_usage import LiveUsagePublisher


def _publisher() -> LiveUsagePublisher:
    return LiveUsagePublisher(None, "req", "caid:operator", "m", "m", 200_000)


def _delta(kind: str, key: str, text: str) -> dict:
    return {"type": "content_block_delta", "delta": {"type": kind, key: text}}


def _start(p: LiveUsagePublisher, source: str | None = None) -> None:
    p.observe_stream_event({"type": "message_start", "message": {"usage": {}}}, source)


def _end(p: LiveUsagePublisher, reasoning: int, source: str | None = None) -> None:
    p.observe_stream_event(
        {"type": "message_delta", "usage": {"output_tokens": 1, "reasoning_output_tokens": reasoning}},
        source,
    )


def test_estimate_counts_text_and_tool_input_but_not_thinking():
    p = _publisher()
    _start(p)
    p.observe_stream_event(_delta("thinking_delta", "thinking", "x" * 4000))
    p.observe_stream_event(_delta("text_delta", "text", "y" * 400))
    p.observe_stream_event(_delta("input_json_delta", "partial_json", "z" * 40))
    p.observe_stream_event(_delta("text_delta", "text", "s" * 999), source="toolu_sub")
    assert p.visible_output_chars == 440
    assert p.reasoning_tokens(500) == (500 - 110, "estimated")


def test_estimate_is_within_zero_and_output():
    p = _publisher()
    _start(p)
    p.observe_stream_event(_delta("text_delta", "text", "y" * 4000))
    assert p.reasoning_tokens(100) == (0, "estimated")
    assert p.reasoning_tokens(0) is None
    assert _publisher().reasoning_tokens(100) is None


def test_proxy_reported_reasoning_is_exact_summed_over_main_requests():
    p = _publisher()
    _start(p)
    _end(p, 30)
    _start(p, "toolu_sub")
    _end(p, 999, "toolu_sub")  # subagent output is not in the main ResultMessage
    _start(p)
    _end(p, 12)
    assert p.reasoning_tokens(100) == (42, "exact")
    assert p.reasoning_tokens(20) == (20, "exact")  # never above output


def test_a_main_request_without_a_report_falls_back_to_the_estimate():
    p = _publisher()
    _start(p)
    _end(p, 30)
    _start(p)  # no reasoning_output_tokens on this request
    assert p.reasoning_tokens(100)[1] == "estimated"


def test_proxy_reads_responses_reasoning_details():
    assert _extract_reasoning({"usage": {"output_tokens": 9, "output_tokens_details": {"reasoning_tokens": 7}}}) == 7
    assert _extract_reasoning({"usage": {"output_tokens": 9}}) is None
    assert _extract_reasoning({"usage": {"output_tokens_details": {"reasoning_tokens": -2}}}) == 0
