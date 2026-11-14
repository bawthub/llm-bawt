"""Tests for history summarization helpers."""

import time
from unittest.mock import Mock

import pytest

from llm_bawt.memory.summarization import (
    HistorySummarizer,
    Session,
    SUMMARIZATION_PROMPT,
    compress_structured_summary_text,
    extract_summary_sections,
    normalize_structured_summary_text,
    estimate_session_token_savings,
    find_budget_overflow_sessions,
    prioritize_summarizable_sessions,
)


def _session(start: float, end: float, messages: list[dict]) -> Session:
    return Session(
        start_timestamp=start,
        end_timestamp=end,
        messages=messages,
        message_ids=[m["id"] for m in messages],
    )


def _msg(id: str, role: str, content: str, timestamp: float) -> dict:
    return {"id": id, "role": role, "content": content, "timestamp": timestamp}


def test_find_budget_overflow_sessions_returns_empty_when_everything_fits() -> None:
    """When all messages fit in the token budget, nothing should be summarized."""
    now = time.time()
    messages = [
        _msg(f"a{i}", "user", "hello world", now + i)
        for i in range(1, 5)
    ]
    result = find_budget_overflow_sessions(
        messages,
        session_gap_seconds=3600,
        max_context_tokens=50000,  # huge budget
        protected_recent_turns=3,
        min_messages_per_session=2,
    )
    assert len(result) == 0


def test_find_budget_overflow_sessions_returns_empty_with_zero_budget() -> None:
    """With zero budget, nothing should be summarized (budget-driven means no budget = no action)."""
    now = time.time()
    messages = [
        _msg(f"a{i}", "user", "hello world", now + i)
        for i in range(1, 5)
    ]
    result = find_budget_overflow_sessions(
        messages,
        max_context_tokens=0,
    )
    assert len(result) == 0


def test_preview_ignores_non_conversation_rows() -> None:
    """Internal rows must not make a bot with no chat turns summarizable."""
    summarizer = HistorySummarizer.__new__(HistorySummarizer)
    summarizer.session_gap_seconds = 3600
    summarizer.max_context_tokens = 100
    summarizer.protected_recent_turns = 0
    summarizer.min_messages_per_session = 2
    summarizer._get_all_messages = lambda: [
        {
            "id": f"system-{i}",
            "role": "system",
            "content": "internal context " * 100,
            "timestamp": float(i),
            "summarized": False,
            "recalled_history": False,
        }
        for i in range(4)
    ]

    assert summarizer.preview_summarizable_sessions() == []


def test_find_budget_overflow_sessions_identifies_overflow() -> None:
    """When messages exceed budget, oldest sessions should be returned for summarization."""
    now = time.time()
    # Create two sessions with a gap > 3600s
    old_session_ts = now - 7200
    recent_session_ts = now - 60

    old_msgs = [
        _msg(f"old{i}", "user" if i % 2 else "assistant", "A" * 200, old_session_ts + i)
        for i in range(1, 9)
    ]
    recent_msgs = [
        _msg(f"new{i}", "user" if i % 2 else "assistant", "B" * 200, recent_session_ts + i)
        for i in range(1, 5)
    ]

    all_msgs = old_msgs + recent_msgs

    # Set a tight budget that can't fit everything
    # Each message is ~200/4 + 4 = 54 tokens, 12 messages = ~648 tokens
    # With a budget of 400, the old session should overflow
    result = find_budget_overflow_sessions(
        all_msgs,
        session_gap_seconds=3600,
        max_context_tokens=400,
        protected_recent_turns=2,
        min_messages_per_session=2,
    )
    assert len(result) >= 1
    # The overflow sessions should be from the old timestamps
    for session in result:
        assert session.end_timestamp < recent_session_ts


def test_find_budget_overflow_sessions_respects_min_messages() -> None:
    """Sessions with fewer messages than min_messages_per_session should be excluded."""
    now = time.time()
    old_ts = now - 7200

    # One message in old session (below min_messages_per_session=2)
    messages = [
        _msg("old1", "user", "A" * 1000, old_ts),
        _msg("new1", "user", "B" * 200, now - 10),
        _msg("new2", "assistant", "C" * 200, now - 5),
    ]

    result = find_budget_overflow_sessions(
        messages,
        session_gap_seconds=3600,
        max_context_tokens=300,
        protected_recent_turns=1,
        min_messages_per_session=2,
    )
    assert len(result) == 0


def test_prioritize_summarizable_sessions_orders_by_estimated_savings() -> None:
    now = time.time() - 7200
    short_session = _session(
        now,
        now + 10,
        [
            {"id": "s1", "role": "user", "content": "short", "timestamp": now + 1},
            {"id": "s2", "role": "assistant", "content": "ok", "timestamp": now + 2},
            {"id": "s3", "role": "user", "content": "tiny", "timestamp": now + 3},
            {"id": "s4", "role": "assistant", "content": "done", "timestamp": now + 4},
        ],
    )
    long_text = "very long message " * 120
    long_session = _session(
        now + 20,
        now + 40,
        [
            {"id": "l1", "role": "user", "content": long_text, "timestamp": now + 21},
            {"id": "l2", "role": "assistant", "content": long_text, "timestamp": now + 22},
            {"id": "l3", "role": "user", "content": long_text, "timestamp": now + 23},
            {"id": "l4", "role": "assistant", "content": long_text, "timestamp": now + 24},
        ],
    )

    savings_short = estimate_session_token_savings(short_session)
    savings_long = estimate_session_token_savings(long_session)
    assert savings_long > savings_short

    prioritized = prioritize_summarizable_sessions([short_session, long_session])
    assert prioritized[0].start_timestamp == long_session.start_timestamp


@pytest.mark.parametrize("allow_fallback", [False, True])
def test_partial_batch_respects_heuristic_fallback_policy(monkeypatch, allow_fallback):
    sessions = [_session(i, i + 1, [_msg(str(i), "user", "synthetic", i)])
                for i in (1, 3)]
    summarizer = HistorySummarizer.__new__(HistorySummarizer)
    summarizer._setting = lambda _key, default: default
    summarizer._summarize_batch_fn = lambda _sessions: {1: "First model summary"}
    summarizer.summarize_session = Mock(return_value={"success": True})
    heuristic = Mock(return_value="Synthetic heuristic")
    monkeypatch.setattr("llm_bawt.memory.summarization.summarize_session_heuristic", heuristic)

    results, errors, created, messages, _replaced = summarizer._summarize_sessions_batch(
        sessions, use_heuristic_fallback=allow_fallback, replace_existing=False)

    assert len(results) == created == messages == (2 if allow_fallback else 1)
    if allow_fallback:
        assert errors == []
        heuristic.assert_called_once_with(sessions[1])
    else:
        assert errors == ["Session 2: missing summary in batch output"]
        heuristic.assert_not_called()
        summarizer.summarize_session.assert_called_once()


@pytest.mark.parametrize("limit,expected", [(1, {30}), (2, {20, 30}), (0, {10, 20, 30})])
def test_rebuild_recent_selects_newest_before_savings_priority(monkeypatch, limit, expected):
    sessions = [_session(stamp, stamp + 1, [_msg(str(stamp), "user", "x" * size, stamp)])
                for stamp, size in ((10, 100), (20, 5000), (30, 1000))]
    monkeypatch.setattr("llm_bawt.memory.summarization.find_budget_overflow_sessions",
                        lambda *_args, **_kwargs: sessions)
    summarizer = HistorySummarizer.__new__(HistorySummarizer)
    summarizer.session_gap_seconds = 3600
    summarizer.max_context_tokens = 100
    summarizer.protected_recent_turns = 0
    summarizer.min_messages_per_session = 1
    summarizer._get_all_messages = lambda: []
    summarizer._summarize_batch_fn = None
    summarizer.summarize_session = Mock(return_value={"success": True})

    result = summarizer.rebuild_recent_sessions(session_limit=limit, max_tokens_per_chunk=0)

    targeted = [call.args[0].start_timestamp for call in summarizer.summarize_session.call_args_list]
    assert set(targeted) == expected
    assert result["sessions_targeted"] == len(expected)
    if limit == 2:
        assert targeted == [20, 30]  # Savings still orders work, not membership.


def test_summarization_prompt_includes_intent_and_tone_sections() -> None:
    prompt = SUMMARIZATION_PROMPT.format(messages="User: hi\nAssistant: hello")
    assert "Intent:" in prompt
    assert "Tone:" in prompt
    assert "Open Loops:" in prompt


def test_extract_summary_sections_parses_structured_output() -> None:
    text = (
        "Summary: User asked about deploys.\n"
        "Intent: Fix a broken production pipeline.\n"
        "Tone: Frustrated but focused.\n"
        "Open Loops: Confirm rollback policy."
    )
    parsed = extract_summary_sections(text)
    assert parsed["summary"] == "User asked about deploys."
    assert parsed["intent"] == "Fix a broken production pipeline."
    assert parsed["tone"] == "Frustrated but focused."
    assert parsed["open_loops"] == "Confirm rollback policy."


def test_normalize_structured_summary_text_fills_missing_sections() -> None:
    normalized = normalize_structured_summary_text("User asked about DNS propagation.")
    assert normalized.startswith("Summary: User asked about DNS propagation.")
    assert "Intent:" not in normalized
    assert "Tone:" not in normalized
    assert "Open Loops:" not in normalized


def test_compress_structured_summary_text_preserves_overlong_sections() -> None:
    source_text = (
        "Summary: " + ("A" * 500) + "\n"
        "Key Details: " + ("B" * 1200) + "\n"
        "Intent: " + ("C" * 300) + "\n"
        "Tone: " + ("D" * 260) + "\n"
        "Open Loops: " + ("E" * 400)
    )
    compact = compress_structured_summary_text(source_text)
    parsed = extract_summary_sections(compact)

    assert parsed["summary"] == "A" * 500
    assert parsed["key_details"] == "B" * 1200
    assert parsed["intent"] == "C" * 300
    assert parsed["tone"] == "D" * 260
    assert parsed["open_loops"] == "E" * 400


def test_compress_structured_summary_text_ignores_source_session_for_storage() -> None:
    now = time.time() - 7200
    session = _session(
        now,
        now + 10,
        [
            {"id": "m1", "role": "user", "content": "x" * 300, "timestamp": now + 1},
            {"id": "m2", "role": "assistant", "content": "y" * 300, "timestamp": now + 2},
        ],
    )
    source_text = (
        "Summary: " + ("A " * 200) + "\n"
        "Key Details: " + ("B " * 400) + "\n"
        "Intent: " + ("C " * 120) + "\n"
        "Tone: " + ("D " * 120) + "\n"
        "Open Loops: " + ("E " * 180)
    )
    compact = compress_structured_summary_text(source_text, source_session=session)
    parsed = extract_summary_sections(compact)
    assert parsed["summary"].startswith("A A A")
    assert "..." not in compact


def test_compress_structured_summary_text_strips_summary_boilerplate() -> None:
    compact = compress_structured_summary_text(
        "Summary: The conversation begins with a debugging pass over flaky deploys.\n"
        "Open Loops: Confirm rollback threshold."
    )
    parsed = extract_summary_sections(compact)
    assert parsed["summary"] == "A debugging pass over flaky deploys."
