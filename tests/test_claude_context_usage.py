"""Claude native context-health extraction and result merging."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from agent_bridge.events import AgentEventKind
from claude_code_bridge.send_result import ClaudeResultMixin
from claude_code_bridge.send_usage import ClaudeUsageMixin
import claude_code_bridge.send_result as send_result_module


class _Harness(ClaudeResultMixin, ClaudeUsageMixin):
    def __init__(self):
        self.events = []

    @staticmethod
    def _model_provider_prefix(_model):
        return "openai_chatgpt"

    def _publish_event(self, request_id, session_key, seq, **kwargs):
        self.events.append((request_id, session_key, seq, kwargs))


def test_proxy_iteration_usage_reports_total_resident_prompt():
    harness = _Harness()
    msg = SimpleNamespace(
        usage={
            "input_tokens": 100_000,
            "cache_read_input_tokens": 900_000,
            "cache_creation_input_tokens": 0,
            "output_tokens": 20,
        },
        model_usage={"openai_chatgpt/gpt": {
            "contextWindow": 372000,
            "maxOutputTokens": 4096,
        }},
        total_cost_usd=1.25,
    )

    usage, _, _ = harness._compute_result_usage(
        msg,
        actual_model="openai_chatgpt/gpt",
        model="openai_chatgpt/gpt",
        bot_context_window=372000,
        latest_assistant_usage=None,
        latest_stream_usage={
            "input_tokens": 22_968,
            "cache_read_input_tokens": 9_728,
            "cache_creation_input_tokens": 0,
            "output_tokens": 5,
        },
    )

    assert usage["resident_tokens"] == 32_696
    assert usage["resident_source"] == "final_iteration_total_input"
    assert usage["input_tokens"] == 22_968
    assert usage["cache_read_tokens"] == 9_728


def test_interrupted_usage_preserves_latest_iteration_snapshot():
    harness = _Harness()

    usage = harness._compute_interrupted_usage(
        actual_model="openai_chatgpt/gpt",
        model="openai_chatgpt/gpt",
        bot_context_window=372000,
        latest_assistant_usage={
            "input_tokens": 1885,
            "cache_read_input_tokens": 137728,
            "cache_creation_input_tokens": 0,
            "output_tokens": 14479,
        },
        latest_stream_usage=None,
    )

    assert usage == {
        "input_tokens": 1885,
        "cache_read_tokens": 137728,
        "cache_creation_tokens": 0,
        "output_tokens": 14479,
        "resident_tokens": 139613,
        "resident_source": "interrupted_iteration_total_input",
        "context_window": 372000,
        "total_cost_usd": usage["total_cost_usd"],
        "usage_status": "partial",
    }
    assert usage["total_cost_usd"] is None


def test_interrupted_usage_without_provider_snapshot_stays_empty():
    harness = _Harness()

    assert harness._compute_interrupted_usage(
        actual_model="openai_chatgpt/gpt",
        model="openai_chatgpt/gpt",
        bot_context_window=372000,
        latest_assistant_usage=None,
        latest_stream_usage=None,
    ) is None


def test_cumulative_only_usage_does_not_claim_resident_context():
    harness = _Harness()
    msg = SimpleNamespace(
        usage={
            "input_tokens": 623_465,
            "cache_read_input_tokens": 18_359_296,
            "cache_creation_input_tokens": 0,
            "output_tokens": 44_650,
        },
        model_usage={"openai_chatgpt/gpt": {
            "contextWindow": 372000,
            "maxOutputTokens": 4096,
        }},
        total_cost_usd=1.25,
    )

    usage, _, _ = harness._compute_result_usage(
        msg,
        actual_model="openai_chatgpt/gpt",
        model="openai_chatgpt/gpt",
        bot_context_window=372000,
        latest_assistant_usage=None,
        latest_stream_usage=None,
    )

    assert "resident_tokens" not in usage
    assert "resident_source" not in usage


def test_interrupted_done_event_carries_partial_usage():
    harness = _Harness()

    seq, usage = harness._publish_interrupted_done(
        request_id="req-interrupted",
        session_key="snark:nick",
        seq=7,
        text="partial answer",
        actual_model="openai_chatgpt/gpt",
        model="openai_chatgpt/gpt",
        bot_context_window=372000,
        latest_assistant_usage={
            "input_tokens": 10,
            "cache_read_input_tokens": 90,
            "output_tokens": 4,
        },
        latest_stream_usage=None,
    )

    assert seq == 8
    event = harness.events[-1][3]
    assert event["kind"] == AgentEventKind.ASSISTANT_DONE
    assert event["text"] == "partial answer"
    assert event["token_usage"] == usage
    assert usage["resident_tokens"] == 100
    assert usage["usage_status"] == "partial"


def test_native_context_usage_overrides_resident_not_cost_accounting():
    harness = _Harness()
    msg = SimpleNamespace(
        usage={
            "input_tokens": 1000,
            "cache_read_input_tokens": 500000,
            "cache_creation_input_tokens": 0,
            "output_tokens": 20,
        },
        model_usage={"openai_chatgpt/gpt": {
            "contextWindow": 372000,
            "maxOutputTokens": 4096,
        }},
        total_cost_usd=1.25,
        text="done",
        content=[],
    )
    seq = asyncio.run(harness._finalize_result_message(
        msg,
        request_id="req-1",
        session_key="snark:nick",
        seq=0,
        text_parts=["done"],
        assistant_snapshot_text="",
        api_retry_count=0,
        api_last_error=None,
        actual_model="openai_chatgpt/gpt",
        model="openai_chatgpt/gpt",
        bot_context_window=372000,
        latest_assistant_usage={
            "input_tokens": 1000,
            "cache_read_input_tokens": 2000,
            "cache_creation_input_tokens": 0,
            "output_tokens": 2,
        },
        latest_stream_usage=None,
        native_context_usage={
            "totalTokens": 90,
            "maxTokens": 100,
            "percentage": 90.0,
            "isAutoCompactEnabled": True,
            "autoCompactThreshold": 95,
        },
        compact_status=None,
        compact_error_msg=None,
        turn_session_id="sdk-1",
        turn_screenshot_assets=[],
    ))
    assert seq == 1
    event = harness.events[-1][3]
    assert event["kind"] == AgentEventKind.ASSISTANT_DONE
    usage = event["token_usage"]
    assert usage["resident_tokens"] == 90
    assert usage["context_window"] == 100
    assert usage["resident_source"] == "claude_sdk_context"
    assert usage["cache_read_tokens"] == 2000
    assert usage["total_cost_usd"] is not None
    assert usage["auto_compact_enabled"] is True


def test_successful_compact_persists_lifecycle_in_usage(monkeypatch):
    harness = _Harness()
    monkeypatch.setattr(
        send_result_module,
        "_read_latest_compact_metadata",
        lambda _session: {"preTokens": 1000, "postTokens": 250},
    )
    msg = SimpleNamespace(
        usage={}, model_usage={}, total_cost_usd=0.0, text="", content=[]
    )
    asyncio.run(harness._finalize_result_message(
        msg,
        request_id="req-compact",
        session_key="snark:nick",
        seq=0,
        text_parts=[],
        assistant_snapshot_text="",
        api_retry_count=0,
        api_last_error=None,
        actual_model="openai_chatgpt/gpt",
        model="openai_chatgpt/gpt",
        bot_context_window=372000,
        latest_assistant_usage=None,
        latest_stream_usage=None,
        native_context_usage={"totalTokens": 250, "maxTokens": 372000},
        compact_status="success",
        compact_error_msg=None,
        turn_session_id="sdk-compact",
        turn_screenshot_assets=[],
    ))
    usage = harness.events[-1][3]["token_usage"]
    assert usage["context_action"] == "compact"
    assert usage["context_action_outcome"] == "success"
    assert usage["context_action_pre_tokens"] == 1000
    assert usage["context_action_post_tokens"] == 250
    assert usage["resident_tokens"] == 250
    assert usage["resident_source"] == "claude_compact_metadata"
