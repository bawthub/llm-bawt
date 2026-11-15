"""TASK-1586: Claude thinking estimated from output minus visible output."""

from claude_code_bridge.send_usage import LiveUsagePublisher


def _publisher() -> LiveUsagePublisher:
    return LiveUsagePublisher(None, "req", "caid:nick", "m", "m", 200_000)


def _delta(kind: str, key: str, text: str) -> dict:
    return {"type": "content_block_delta", "delta": {"type": kind, key: text}}


def test_estimate_counts_text_and_tool_input_but_not_thinking():
    p = _publisher()
    p.observe_stream_event({"type": "message_start", "message": {"usage": {}}})
    p.observe_stream_event(_delta("thinking_delta", "thinking", "x" * 4000))
    p.observe_stream_event(_delta("text_delta", "text", "y" * 400))
    p.observe_stream_event(_delta("input_json_delta", "partial_json", "z" * 40))
    p.observe_stream_event(_delta("text_delta", "text", "s" * 999), source="toolu_sub")
    assert p.visible_output_chars == 440
    assert p.estimated_reasoning_tokens(500) == 500 - 110


def test_estimate_is_within_zero_and_output():
    p = _publisher()
    p.observe_stream_event({"type": "message_start", "message": {"usage": {}}})
    p.observe_stream_event(_delta("text_delta", "text", "y" * 4000))
    assert p.estimated_reasoning_tokens(100) == 0
    assert p.estimated_reasoning_tokens(0) is None
    assert _publisher().estimated_reasoning_tokens(100) is None
