"""TASK-2110: per-LLM-response tool grouping — identity, usage merge, durability."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from claude_code_bridge.response_tracker import FINAL, PROVISIONAL, ResponseTracker
from llm_bawt.service.llm_response_store import (
    LlmResponseRecord,
    LlmResponseStore,
    serialize_responses,
)
from llm_bawt.service.tool_call_store import ToolCallRecord, ToolCallResultPayloadRecord, ToolCallStore
from llm_bawt.service.tool_event_coordinator import ToolEventCoordinator
from llm_bawt.service.turn_logs import TurnLog


def _am(message_id, *, parent=None, usage=None, model="claude-opus-5-5"):
    return SimpleNamespace(message_id=message_id, parent_tool_use_id=parent, usage=usage, model=model)


def _start(rid, usage, model="claude-opus-5-5"):
    return {"type": "message_start", "message": {"id": rid, "model": model, "usage": usage}}


START_USAGE = {"input_tokens": 2, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 50, "output_tokens": 16}


# ── bridge tracker ────────────────────────────────────────────────────────


def test_main_thread_usage_is_final_only_on_message_stop():
    t = ResponseTracker()
    rec = t.on_stream_event(_start("msg_a", START_USAGE), source=None, seq=1)
    assert rec.usage_state == PROVISIONAL and rec.usage["output_tokens"] == 16
    # Two content-block snapshots of the same response: one record, no change.
    assert t.on_assistant_message(_am("msg_a", usage=START_USAGE), host_parent=None, seq=2) is None
    assert t.on_assistant_message(_am("msg_a", usage=START_USAGE), host_parent=None, seq=3) is None
    rec = t.on_stream_event({"type": "message_delta", "usage": {**START_USAGE, "output_tokens": 143}},
                            source=None, seq=4)
    assert rec.usage["output_tokens"] == 143 and rec.usage_state == PROVISIONAL
    rec = t.on_stream_event({"type": "message_stop"}, source=None, seq=5)
    assert rec.usage_state == FINAL and rec.first_seen_seq == 1
    # A late id-less delta after stop is not attributable to anything.
    assert t.on_stream_event({"type": "message_delta", "usage": {"output_tokens": 999}},
                             source=None, seq=6) is None
    assert t.record("msg_a").usage["output_tokens"] == 143


def test_live_stop_reason_is_never_trusted_and_subagents_stay_provisional():
    t = ResponseTracker()
    msg = _am("msg_sub", parent="toolu_agent", usage={**START_USAGE})
    msg.stop_reason = "tool_use"
    rec = t.on_assistant_message(msg, host_parent="toolu_agent", seq=1)
    assert rec.host_parent == "toolu_agent" and rec.usage_state == PROVISIONAL
    # Subagent stream frames are never used, even if they appear.
    assert t.on_stream_event({"type": "message_stop"}, source="toolu_agent", seq=2) is None
    assert t.record("msg_sub").usage_state == PROVISIONAL


def test_concurrent_siblings_sharing_a_parent_keep_separate_usage():
    t = ResponseTracker()
    t.on_assistant_message(_am("msg_A", parent="toolu_wf", usage={"input_tokens": 10, "output_tokens": 1}),
                           host_parent="toolu_wf", seq=1)
    t.on_assistant_message(_am("msg_B", parent="toolu_wf", usage={"input_tokens": 70, "output_tokens": 1}),
                           host_parent="toolu_wf", seq=2)
    t.on_assistant_message(_am("msg_A", parent="toolu_wf", usage={"input_tokens": 10, "output_tokens": 9}),
                           host_parent="toolu_wf", seq=3)
    assert t.record("msg_A").usage == {"input_tokens": 10, "output_tokens": 9}
    assert t.record("msg_B").usage == {"input_tokens": 70, "output_tokens": 1}


def test_new_main_start_leaves_an_unstopped_response_provisional():
    t = ResponseTracker()
    t.on_stream_event(_start("msg_1", START_USAGE), source=None, seq=1)
    t.on_stream_event(_start("msg_2", START_USAGE), source=None, seq=2)
    t.on_stream_event({"type": "message_stop"}, source=None, seq=3)
    assert t.record("msg_1").usage_state == PROVISIONAL
    assert t.record("msg_2").usage_state == FINAL


def test_synthetic_messages_are_not_responses():
    t = ResponseTracker()
    assert t.on_assistant_message(_am("uuid-x", model="<synthetic>"), host_parent=None, seq=1) is None
    assert t.record("uuid-x") is None


def test_first_sighting_fixes_scope_and_order():
    t = ResponseTracker()
    t.on_assistant_message(_am("msg_r", usage={"input_tokens": 1}), host_parent="toolu_resume", seq=4)
    t.on_assistant_message(_am("msg_r", usage={"input_tokens": 5}), host_parent="toolu_other", seq=9)
    rec = t.record("msg_r")
    assert (rec.host_parent, rec.first_seen_seq, rec.usage["input_tokens"]) == ("toolu_resume", 4, 5)


def _mixin():
    from claude_code_bridge import send_stream

    cls = next(v for v in vars(send_stream).values()
               if isinstance(v, type) and hasattr(v, "_on_assistant_message"))
    host = type("Host", (cls,), {})()
    host.events = []
    host._publish_event = lambda rid, sk, seq, **kw: host.events.append((seq, kw))
    host._is_ask_user_question = lambda name: False
    return host


def _tool(tid, name="Bash"):
    from claude_agent_sdk.types import ToolUseBlock
    return ToolUseBlock(id=tid, name=name, input={})


def test_tool_start_carries_response_id_for_parallel_and_sequential_responses():
    from agent_bridge.events import AgentEventKind

    host = _mixin()
    tracker = ResponseTracker()
    seq = 0
    for rid, tools in (("msg_1", ["t1", "t2"]), ("msg_2", ["t3"])):
        msg = SimpleNamespace(message_id=rid, parent_tool_use_id=None, usage=START_USAGE,
                              model="m", content=[_tool(t) for t in tools])
        seq = host._observe_llm_response(msg, request_id="r", session_key="s", seq=seq,
                                         response_tracker=tracker, usage_ledger=None)
        seq, *_ = host._on_assistant_message(
            msg, request_id="r", session_key="s", seq=seq, tool_names_by_id={},
            tool_arguments_by_id={}, latest_assistant_usage=None, assistant_snapshot_text="",
            response_tracker=tracker,
        )
    starts = [kw for _, kw in host.events if kw["kind"] == AgentEventKind.TOOL_START]
    assert [(kw["tool_use_id"], kw["extra_raw"]["response_id"]) for kw in starts] == [
        ("t1", "msg_1"), ("t2", "msg_1"), ("t3", "msg_2"),
    ]
    responses = [kw["extra_raw"] for _, kw in host.events if kw["kind"] == AgentEventKind.LLM_RESPONSE]
    assert [r["response_id"] for r in responses] == ["msg_1", "msg_2"]
    assert all(kw["extra_raw"]["exec_tracked"] for kw in starts)


def test_tool_start_without_known_response_has_no_response_id():
    host = _mixin()
    msg = SimpleNamespace(message_id=None, parent_tool_use_id=None, usage=None, model="m",
                          content=[_tool("t9")])
    host._on_assistant_message(
        msg, request_id="r", session_key="s", seq=0, tool_names_by_id={}, tool_arguments_by_id={},
        latest_assistant_usage=None, assistant_snapshot_text="", response_tracker=ResponseTracker(),
    )
    assert "response_id" not in host.events[0][1]["extra_raw"]


# ── app store ─────────────────────────────────────────────────────────────


@pytest.fixture
def engine():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine, tables=[
        TurnLog.__table__, ToolCallRecord.__table__, ToolCallResultPayloadRecord.__table__,
        LlmResponseRecord.__table__,
    ])
    with Session(engine) as session:
        session.add(TurnLog(id="turn", bot_id="al", user_id="nick", status="streaming", trigger_message_id="m"))
        session.commit()
    return engine


def _upsert(store, rid, seq, **kw):
    return store.upsert(turn_id="turn", response_id=rid, first_seen_seq=seq, bot_id="al", user_id="nick", **kw)


def test_usage_max_merges_never_sums_and_replay_keeps_known_values(engine):
    store = LlmResponseStore(engine)
    _upsert(store, "msg_a", 1, model="opus", usage=START_USAGE)
    _upsert(store, "msg_a", 1, usage=START_USAGE)  # duplicate delivery
    _upsert(store, "msg_a", 1, usage={"output_tokens": 143})  # replay omits input fields
    rec = _upsert(store, "msg_a", 7, usage={"input_tokens": 1}, usage_state="final")
    assert rec["usage"] == {"input_tokens": 152, "cache_read_tokens": 100,
                            "cache_write_tokens": 50, "output_tokens": 143}
    assert rec["usage_state"] == "final" and rec["model"] == "opus" and rec["ordinal"] == 1
    # Final never downgrades; first_seen_seq is immutable.
    rec = _upsert(store, "msg_a", 9, usage_state="provisional")
    assert rec["usage_state"] == "final"
    with Session(engine) as s:
        assert s.exec(select(LlmResponseRecord)).one().first_seen_seq == 1


def test_provisional_output_is_hidden_and_ended_turn_reads_partial(engine):
    store = LlmResponseStore(engine)
    rec = _upsert(store, "msg_sub", 3, host_parent="toolu_a", usage=START_USAGE)
    assert "output_tokens" not in rec["usage"] and rec["usage_state"] == "provisional"
    rows = LlmResponseStore(engine).for_turns(["turn"])["turn"]
    assert serialize_responses(rows, turn_ended=True)[0]["usage_state"] == "partial"


def test_ordinals_count_toolless_responses_per_host_parent(engine):
    store = LlmResponseStore(engine)
    assert _upsert(store, "m1", 1)["ordinal"] == 1          # tool-less narration response
    assert _upsert(store, "s1", 3, host_parent="toolu_a")["ordinal"] == 1
    assert _upsert(store, "m2", 5)["ordinal"] == 2
    assert _upsert(store, "s2", 6, host_parent="toolu_a")["ordinal"] == 2
    assert _upsert(store, "m1", 1, usage_state="final")["ordinal"] == 1  # re-upsert keeps number
    rows = store.for_turns(["turn"])["turn"]
    shaped = {r["response_id"]: (r["host_parent"], r["ordinal"]) for r in serialize_responses(rows, turn_ended=False)}
    assert shaped == {"m1": (None, 1), "m2": (None, 2), "s1": ("toolu_a", 1), "s2": ("toolu_a", 2)}


def test_invalid_identity_is_not_persisted(engine):
    store = LlmResponseStore(engine)
    assert _upsert(store, None, 1) is None
    assert _upsert(store, "x" * 200, 1) is None
    assert _upsert(store, "ok", None) is None


def _coord_start(coord, call_id, **extra):
    coord.start({"turn_id": "turn", "bot_id": "al", "user_id": "nick", "call_id": call_id,
                 "tool_use_id": f"tu-{call_id}", "tool_name": "Bash", "arguments": {}, "ts": 1.0, **extra})


def _row(engine, call_id):
    with Session(engine) as s:
        return s.exec(select(ToolCallRecord).where(ToolCallRecord.call_id == call_id)).one()


def test_replayed_start_without_response_id_cannot_erase_membership(engine):
    coord = ToolEventCoordinator(engine)
    _coord_start(coord, "c1", response_id="msg_1")
    _coord_start(coord, "c1")
    _coord_start(coord, "c1", response_id=None)
    assert _row(engine, "c1").response_id == "msg_1"


def test_contradictory_response_id_never_reassigns_a_tool(engine, caplog):
    coord = ToolEventCoordinator(engine)
    _coord_start(coord, "c1", response_id="msg_1")
    _coord_start(coord, "c1", response_id="msg_2")
    assert _row(engine, "c1").response_id == "msg_1"
    assert "contradictory response_id" in caplog.text


def test_tool_end_keeps_response_membership(engine):
    from agent_bridge.tool_results import ToolResultPayload

    coord = ToolEventCoordinator(engine)
    _coord_start(coord, "c1", response_id="msg_1")
    ToolCallStore(engine).save_result(
        turn_id="turn", call_id="c1", tool_use_id="tu-c1", tool_name="Bash", bot_id="al",
        user_id="nick", payload=ToolResultPayload.from_value("ok"), ended_at=2.0, is_error=False,
    )
    assert _row(engine, "c1").response_id == "msg_1"


def test_coordinator_publishes_the_merged_record(engine):
    coord = ToolEventCoordinator(engine)
    base = {"turn_id": "turn", "bot_id": "al", "user_id": "nick", "trigger_message_id": "m",
            "response_id": "msg_1", "first_seen_seq": 2, "model": "opus"}
    coord.response({**base, "usage": START_USAGE})
    public = coord.response({**base, "usage": {"output_tokens": 50}, "usage_state": "final"})
    assert public["_type"] == "llm_response" and public["trigger_message_id"] == "m"
    assert public["usage"]["output_tokens"] == 50 and public["usage"]["input_tokens"] == 152
    assert coord.response({**base, "response_id": None}) is None


def test_read_models_agree_with_live_identity(engine, monkeypatch):
    from llm_bawt.service.routes import turn_logs as routes

    coord = ToolEventCoordinator(engine)
    coord.response({"turn_id": "turn", "response_id": "m0", "first_seen_seq": 1})
    live = coord.response({"turn_id": "turn", "response_id": "m1", "first_seen_seq": 4,
                           "usage": START_USAGE, "usage_state": "final"})
    _coord_start(coord, "c1", response_id="m1")
    _coord_start(coord, "c2")  # legacy/unknown: no association
    store = SimpleNamespace(engine=engine)
    with Session(engine) as s:
        turn = s.get(TurnLog, "turn")
    responses = routes._turn_responses(store, [turn])["turn"]
    read = next(r for r in responses if r["response_id"] == "m1")
    assert {k: read[k] for k in ("ordinal", "usage", "usage_state")} == {
        k: live[k] for k in ("ordinal", "usage", "usage_state")
    }
    with Session(engine) as s:
        calls = routes._records_to_calls(list(s.exec(select(ToolCallRecord)).all()))
    assert {c["call_id"]: c["response_id"] for c in calls} == {"c1": "m1", "c2": None}
