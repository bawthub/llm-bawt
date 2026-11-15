"""TASK-2343: exact Claude reasoning and resident reasoning across turns.

Probes showed Claude replays every earlier turn's thinking as input (the CLI
resends the signed blocks; the API counts them in full) while the transcript
text is redacted, so the API's ``output_tokens_details.thinking_tokens`` is the
only measure.
"""

import asyncio
import json

import pytest

from claude_code_bridge import reasoning_ledger
from claude_code_bridge.send_usage import LiveUsagePublisher


def _publisher() -> LiveUsagePublisher:
    return LiveUsagePublisher(None, "req", "nova:nick", "m", "m", 200_000)


def _request(p, msg_id, thinking, source=None, details=True):
    p.observe_stream_event({"type": "message_start", "message": {"id": msg_id, "usage": {}}}, source)
    usage = {"output_tokens": thinking + 5}
    if details:
        usage["output_tokens_details"] = {"thinking_tokens": thinking}
    p.observe_stream_event({"type": "message_delta", "usage": usage}, source)


def test_anthropic_thinking_tokens_make_reasoning_exact_and_keyed_by_message():
    p = _publisher()
    _request(p, "msg_a", 233)
    _request(p, "msg_sub", 999, source="toolu_sub")
    _request(p, "msg_b", 0)
    assert p.reasoning_tokens(300) == (233, "exact")
    assert p.thinking_by_message() == {"msg_a": 233, "msg_b": 0}


def test_missing_details_on_any_main_request_falls_back_to_estimate():
    p = _publisher()
    _request(p, "msg_a", 100)
    _request(p, "msg_b", 50, details=False)
    assert p.reasoning_tokens(200)[1] == "estimated"
    assert p.thinking_by_message() == {"msg_a": 100}


def _assistant(msg_id, *, thinking=True, sidechain=False):
    content = ([{"type": "thinking", "thinking": "", "signature": "sig"}] if thinking else []) + [
        {"type": "text", "text": "hi"}
    ]
    return {"type": "assistant", "isSidechain": sidechain, "message": {"id": msg_id, "content": content}}


@pytest.fixture
def transcript(tmp_path, monkeypatch):
    path = tmp_path / "sess.jsonl"
    monkeypatch.setattr(reasoning_ledger, "_session_transcript_path", lambda sid: path if sid else None)

    def write(*entries):
        path.write_text("".join(json.dumps(e) + "\n" for e in entries))
    return write


def test_resident_reasoning_sums_ledger_after_last_compaction(transcript):
    transcript(
        _assistant("msg_old"),
        {"type": "system", "subtype": "compact_boundary", "compactMetadata": {}},
        _assistant("msg_1"),
        _assistant("msg_1"),  # the CLI writes one row per content block
        _assistant("msg_2", thinking=False),
        _assistant("msg_sub", sidechain=True),
        _assistant("msg_3"),
    )
    reasoning_ledger.record_thinking("sess", {"msg_old": 500, "msg_1": 260})
    reasoning_ledger.record_thinking("sess", {"msg_3": 40, "msg_sub": 7})
    assert reasoning_ledger.resident_reasoning("sess") == (300, True)


def test_resident_reasoning_flags_untracked_thinking_as_incomplete(transcript):
    transcript(_assistant("msg_before_deploy"), _assistant("msg_new"))
    reasoning_ledger.record_thinking("sess", {"msg_new": 12})
    assert reasoning_ledger.resident_reasoning("sess") == (12, False)


def test_no_resident_thinking_or_transcript_is_none(transcript):
    transcript(_assistant("msg_plain", thinking=False))
    assert reasoning_ledger.resident_reasoning("sess") is None
    assert reasoning_ledger.resident_reasoning(None) is None
    reasoning_ledger.record_thinking(None, {"msg": 1})  # no-op, no error


def _post_done(monkeypatch, *, use_proxy):
    from tests.test_claude_context_usage import _Harness
    import claude_code_bridge.send_result as send_result_module

    recorded = []
    monkeypatch.setattr(send_result_module, "record_thinking", lambda sid, c: recorded.append((sid, c)))
    monkeypatch.setattr(send_result_module, "resident_reasoning", lambda sid: (493, True))
    harness = _Harness()

    async def read(_client):
        return {"totalTokens": 900, "maxTokens": 200_000,
                "categories": [{"name": "Messages", "tokens": 700}],
                "messageBreakdown": {"toolCallTokens": 0, "toolResultTokens": 0, "attachmentTokens": 0,
                                     "assistantMessageTokens": 100, "userMessageTokens": 50}}

    harness._read_native_context_usage = read
    asyncio.run(harness._publish_post_done_context(
        object(), request_id="req", session_key="s", seq=1, model="claude", use_proxy=use_proxy,
        final_usage={"output_tokens": 300}, session_id="sess", thinking_by_message={"msg_1": 233},
    ))
    return recorded, harness.events[-1][3]["token_usage"]["context_breakdown"]["message_breakdown"]


def test_post_done_records_turn_thinking_and_adds_resident_reasoning(monkeypatch):
    recorded, messages = _post_done(monkeypatch, use_proxy=False)
    assert recorded == [("sess", {"msg_1": 233})]
    assert messages["reasoningTokens"] == 493
    assert messages["reasoningComplete"] is True


def test_proxy_turns_neither_record_nor_report_resident_reasoning(monkeypatch):
    recorded, messages = _post_done(monkeypatch, use_proxy=True)
    assert recorded == []
    assert "reasoningTokens" not in messages


def test_corrupt_sidecar_reads_as_empty(transcript, tmp_path):
    transcript(_assistant("msg_1"))
    (tmp_path / "sess.reasoning.json").write_text("{not json")
    assert reasoning_ledger.resident_reasoning("sess") == (0, False)
    reasoning_ledger.record_thinking("sess", {"msg_1": 9})
    assert reasoning_ledger.resident_reasoning("sess") == (9, True)
