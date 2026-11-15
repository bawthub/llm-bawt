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


def _assistant(msg_id, *, thinking=True, sidechain=False, uuid=None, parent=None):
    content = ([{"type": "thinking", "thinking": "", "signature": "sig"}] if thinking else []) + [
        {"type": "text", "text": "hi"}
    ]
    row = {"type": "assistant", "isSidechain": sidechain, "message": {"id": msg_id, "content": content}}
    if uuid is not None:
        row.update(uuid=uuid, parentUuid=parent)
    return row


def _user(uuid, parent=None):
    return {"type": "user", "uuid": uuid, "parentUuid": parent, "message": {"role": "user", "content": "go"}}


def _boundary(**metadata):
    return {"type": "system", "subtype": "compact_boundary", "uuid": "b", "compactMetadata": metadata}


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


def test_compaction_keeps_thinking_rows_listed_in_preserved_messages(transcript):
    transcript(
        _user("u1"),
        _assistant("msg_dropped", uuid="a1", parent="u1"),
        _user("u2", "a1"),
        _assistant("msg_kept", uuid="a2", parent="u2"),
        _assistant("msg_kept", uuid="a3", parent="a2"),  # second block row, same message
        _boundary(preservedMessages={"uuids": ["u2", "a2", "a3"]}),
        _assistant("msg_new", uuid="a4", parent="b"),
    )
    reasoning_ledger.record_thinking("sess", {"msg_dropped": 900, "msg_kept": 70, "msg_new": 5})
    assert reasoning_ledger.resident_reasoning("sess") == (75, True)


def test_compaction_keeps_legacy_preserved_segment_walked_by_parent(transcript):
    transcript(
        _assistant("msg_dropped", uuid="a1"),
        _user("u2", "a1"),
        _assistant("msg_kept", uuid="a2", parent="u2"),
        _assistant("msg_tail", uuid="a3", parent="a2"),
        _boundary(preservedSegment={"headUuid": "u2", "tailUuid": "a3", "anchorUuid": "x"}),
    )
    reasoning_ledger.record_thinking("sess", {"msg_dropped": 900, "msg_kept": 70, "msg_tail": 3})
    assert reasoning_ledger.resident_reasoning("sess") == (73, True)


def test_preserved_text_row_keeps_its_messages_thinking_row(transcript):
    transcript(
        _assistant("msg_kept", uuid="a1"),  # thinking block row
        _assistant("msg_kept", thinking=False, uuid="a2", parent="a1"),  # text block row
        _boundary(preservedMessages={"uuids": ["a2"]}),
    )
    reasoning_ledger.record_thinking("sess", {"msg_kept": 70})
    assert reasoning_ledger.resident_reasoning("sess") == (70, True)


def test_row_parented_across_the_boundary_keeps_its_parent(transcript):
    transcript(
        _assistant("msg_parent", uuid="a1"),
        _assistant("msg_gone", uuid="a0"),
        _boundary(),
        _user("u2", "a1"),
    )
    reasoning_ledger.record_thinking("sess", {"msg_parent": 40, "msg_gone": 900})
    assert reasoning_ledger.resident_reasoning("sess") == (40, True)


def _stripped(**attachment):
    return {"type": "attachment", "attachment": {"type": "thinking_stripped", **attachment}}


@pytest.mark.parametrize("marker, expected", [
    (_stripped(scope="all"), (5, True)),
    (_stripped(scope="partial", **{"from": {"messageId": "msg_b", "thinkingIndex": 0}}), (25, True)),
    (_stripped(scope="partial", **{"from": {"messageId": "msg_b", "thinkingIndex": 1}}), (25, False)),
    (_stripped(scope="partial", **{"from": {"messageId": "unknown"}}), (5, False)),
])
def test_thinking_stripped_marker_removes_earlier_thinking(transcript, marker, expected):
    transcript(_assistant("msg_a"), _assistant("msg_b"), marker, _assistant("msg_c"))
    reasoning_ledger.record_thinking("sess", {"msg_a": 20, "msg_b": 300, "msg_c": 5})
    assert reasoning_ledger.resident_reasoning("sess") == expected


def test_compaction_without_preservation_drops_everything_before_it(transcript):
    transcript(
        _assistant("msg_old", uuid="a1"),
        _boundary(trigger="auto", preTokens=150000),
        _assistant("msg_new", uuid="a2", parent="b"),
    )
    reasoning_ledger.record_thinking("sess", {"msg_old": 900, "msg_new": 8})
    assert reasoning_ledger.resident_reasoning("sess") == (8, True)


@pytest.mark.parametrize("metadata", [
    {"preservedMessages": {"uuids": "a1"}},
    {"preservedMessages": {"uuids": ["missing"]}},
    {"preservedMessages": {"uuids": ["a1", "missing"]}},
    {"preservedSegment": {"headUuid": "a1"}},
    {"preservedSegment": {"headUuid": "a1", "tailUuid": "missing"}},
    {"preservedSegment": {"headUuid": "not-an-ancestor", "tailUuid": "a1"}},
])
def test_unresolvable_preservation_marks_reasoning_incomplete(transcript, metadata):
    transcript(
        _assistant("msg_old", uuid="a1"),
        _boundary(**metadata),
        _assistant("msg_new", uuid="a2", parent="b"),
    )
    reasoning_ledger.record_thinking("sess", {"msg_old": 900, "msg_new": 8})
    assert reasoning_ledger.resident_reasoning("sess") == (8, False)


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
