"""Claude native context-health extraction and result merging."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from agent_bridge.events import AgentEventKind
from claude_code_bridge.context_breakdown import compact_context_breakdown
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
        cost_baseline_usd=0.0,
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
        cost_baseline_usd=0.0,
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
        cost_baseline_usd=0.0,
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
        cost_baseline_usd=0.0,
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


def test_successful_compact_without_metadata_still_records_the_boundary(monkeypatch):
    """TASK-1318: the session prompt pin moves only on a recorded success."""
    harness = _Harness()
    monkeypatch.setattr(send_result_module, "_read_latest_compact_metadata", lambda _session: None)
    msg = SimpleNamespace(usage={}, model_usage={}, total_cost_usd=0.0, text="", content=[])
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
        cost_baseline_usd=0.0,
        native_context_usage={"totalTokens": 250, "maxTokens": 372000},
        compact_status="success",
        compact_error_msg=None,
        turn_session_id="sdk-compact",
        turn_screenshot_assets=[],
    ))
    usage = harness.events[-1][3]["token_usage"]
    assert usage["context_action"] == "compact"
    assert usage["context_action_outcome"] == "success"
    assert "context_action_post_tokens" not in usage
    assert usage["resident_source"] == "claude_sdk_context"


def test_context_breakdown_keeps_bounded_facts_not_raw_payloads():
    raw = {
        "totalTokens": 53000,
        "categories": [{"name": "Skills", "tokens": 22000, "color": "red"},
                       {"name": "Deferred", "tokens": 100, "isDeferred": True}],
        "skills": {"totalSkills": 30, "includedSkills": 12, "tokens": 22000,
                   "skillFrontmatter": [
                       {"name": f"skill-{i:02}", "source": "/private/work", "tokens": i}
                       for i in range(15)
                   ]},
        "mcpTools": [{"name": f"tool-{i:02}", "serverName": "api",
                      "input_schema": {"secret": "never persist"}, "tokens": 100 + i}
                     for i in range(15)],
        "memoryFiles": [{"path": "/private/work", "type": "project", "tokens": 100}],
        "messageBreakdown": {"toolResultTokens": 900, "userMessageTokens": 42,
                             "toolCallsByType": [{"name": "private", "resultTokens": 900}]},
        "gridRows": [[{"content": "never persist"}]],
    }
    result = compact_context_breakdown(raw, read_ms=45.26)
    assert result["read_ms"] == 45.3
    assert result["cli_total_tokens"] == 53000
    assert result["categories"][1]["isDeferred"] is True
    assert result["skills"]["top"][0] == {"name": "skill-14", "tokens": 14}
    assert len(result["skills"]["top"]) == len(result["mcp_tools"]["top"]) == 10
    assert result["mcp_tools"]["total"] == 15
    assert result["memory_files"] == {"total": 1, "tokens": 100}
    assert result["message_breakdown"]["toolResultTokens"] == 900
    assert "/private/work" not in str(result)
    assert "never persist" not in str(result)
    assert compact_context_breakdown({}, read_ms=1) is None


def test_context_breakdown_excludes_non_wire_attachment_bookkeeping_from_mix():
    raw = {
        "totalTokens": 50000,
        "categories": [{"name": "Messages", "tokens": 26000}],
        "messageBreakdown": {
            "toolCallTokens": 1000, "toolResultTokens": 2000,
            "attachmentTokens": 40000, "assistantMessageTokens": 3000,
            "userMessageTokens": 12000,
            "attachmentsByType": [
                {"name": "prompt_snapshot", "tokens": 26600},
                {"name": "prompt_render_point", "tokens": 100},
                {"name": "deferred_tools_record", "tokens": 500},
                {"name": "hook_success", "tokens": 50},
                {"name": "skill_listing", "tokens": 9000},
                {"name": "other_wire_attachment", "tokens": 3750},
            ],
        },
    }
    result = compact_context_breakdown(raw, read_ms=1)
    assert result["message_breakdown"]["attachmentTokens"] == 40000
    assert result["message_breakdown"]["wireAttachmentTokens"] == 12800
    assert "attachmentsByType" not in result["message_breakdown"]
    assert "prompt_snapshot" not in str(result)

    missing_types = {**raw, "messageBreakdown": {
        key: value for key, value in raw["messageBreakdown"].items()
        if key != "attachmentsByType"
    }}
    assert "wireAttachmentTokens" not in compact_context_breakdown(
        missing_types, read_ms=1)["message_breakdown"]
    for incomplete_types in ([], [{"name": "skill_listing", "tokens": 9000}]):
        incomplete = {**raw, "messageBreakdown": {
            **raw["messageBreakdown"], "attachmentsByType": incomplete_types,
        }}
        assert "wireAttachmentTokens" not in compact_context_breakdown(
            incomplete, read_ms=1)["message_breakdown"]
    fractional = {**raw, "messageBreakdown": {
        **raw["messageBreakdown"], "attachmentTokens": 2,
        "attachmentsByType": [
            {"name": "prompt_snapshot", "tokens": 1.9},
            {"name": "skill_listing", "tokens": 1.9},
        ],
    }}
    assert "wireAttachmentTokens" not in compact_context_breakdown(
        fractional, read_ms=1)["message_breakdown"]
    fractional_total = {**raw, "messageBreakdown": {
        **raw["messageBreakdown"], "attachmentTokens": 40000.5,
    }}
    assert "wireAttachmentTokens" not in compact_context_breakdown(
        fractional_total, read_ms=1)["message_breakdown"]


def test_post_done_usage_keeps_cost_and_publishes_full_final_snapshot():
    harness = _Harness()
    final = {"input_tokens": 20, "output_tokens": 7, "total_cost_usd": 0.3,
             "resident_tokens": 50, "resident_source": "final_iteration_total_input"}

    async def read(_client):
        return {"totalTokens": 55, "maxTokens": 200, "percentage": 27.5,
                "categories": [{"name": "System prompt", "tokens": 50}],
                "mcpTools": [], "skills": {"totalSkills": 0, "includedSkills": 0,
                                         "tokens": 0, "skillFrontmatter": []}}

    harness._read_native_context_usage = read
    seq = asyncio.run(harness._publish_post_done_context(
        object(), request_id="req", session_key="session", seq=1,
        model="claude-direct", final_usage=final,
    ))
    assert seq == 2
    event = harness.events[-1][3]
    assert event["kind"] is AgentEventKind.USAGE_UPDATE
    usage = event["token_usage"]
    assert usage["usage_status"] == "final"
    assert usage["resident_tokens"] == 55
    assert usage["resident_source"] == "claude_sdk_context"
    assert usage["output_tokens"] == 7
    assert usage["total_cost_usd"] == 0.3
    assert usage["context_breakdown"]["read_ms"] >= 0
    assert "context_breakdown" not in final


def test_proxy_post_done_breakdown_preserves_provider_headline():
    harness = _Harness()
    final = {
        "input_tokens": 20, "cache_read_tokens": 80, "output_tokens": 7,
        "resident_tokens": 100, "resident_source": "final_iteration_total_input",
        "context_window": 128000, "total_cost_usd": 0.3,
    }

    async def read(_client):
        return {
            "totalTokens": 55, "maxTokens": 200, "percentage": 27.5,
            "isAutoCompactEnabled": True, "autoCompactThreshold": 95,
            "categories": [{"name": "System prompt", "tokens": 50}],
        }

    harness._read_native_context_usage = read
    seq = asyncio.run(harness._publish_post_done_context(
        object(), request_id="req", session_key="session", seq=1,
        model="openai_chatgpt/gpt", final_usage=final, use_proxy=True,
    ))
    assert seq == 2
    usage = harness.events[-1][3]["token_usage"]
    assert usage["context_breakdown"]["cli_total_tokens"] == 55
    assert usage["resident_tokens"] == 100
    assert usage["resident_source"] == "final_iteration_total_input"
    assert usage["context_window"] == 128000
    assert "context_percentage" not in usage
    assert "auto_compact_threshold" not in usage
    assert usage["output_tokens"] == 7
    assert usage["total_cost_usd"] == 0.3
    assert "context_breakdown" not in final


def test_post_done_failed_read_marks_reason_and_compact_skips_read():
    harness = _Harness()
    calls = []

    async def read(_client):
        calls.append(True)
        return None

    harness._read_native_context_usage = read
    final = {"output_tokens": 7, "total_cost_usd": 0.3}
    assert asyncio.run(harness._publish_post_done_context(
        object(), request_id="req", session_key="session", seq=1,
        model="claude-direct", final_usage=final,
    )) == 2
    usage = harness.events[-1][3]["token_usage"]
    assert usage == {**final, "usage_status": "final", "context_breakdown_status": "unavailable"}
    assert "context_breakdown_status" not in final
    harness.events.clear()
    # No terminal usage means nothing to annotate.
    assert asyncio.run(harness._publish_post_done_context(
        object(), request_id="req", session_key="session", seq=1,
        model="claude-direct", final_usage=None,
    )) == 1
    assert not harness.events
    assert asyncio.run(harness._publish_post_done_context(
        object(), request_id="req", session_key="session", seq=1,
        model="claude-direct", final_usage={"context_action": "compact"},
    )) == 1
    assert not harness.events
    assert calls == [True, True]
