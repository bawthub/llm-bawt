"""TASK-1614: per-bot tool-result waste report — epochs, scope, exposure, ranking."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException
from sqlmodel import Session, SQLModel, create_engine

from agent_bridge.tool_results import TOOL_RESULT_ACCOUNTING_VERSION as V
from llm_bawt.service.llm_response_store import LlmResponseRecord
from llm_bawt.service.routes import usage as usage_routes
from llm_bawt.service.tool_call_store import ToolCallRecord
from llm_bawt.service.turn_logs import TurnLog
from llm_bawt.service.usage import tool_waste as tw

T0 = datetime(2026, 11, 18, 12, tzinfo=timezone.utc)
COMPACT = json.dumps({"context_action": "compact", "context_action_outcome": "success"})


def turn(tid, *, bot="alpha", key="sess", at=0, usage=None):
    return {"id": tid, "bot_id": bot, "created_at": T0 + timedelta(minutes=at),
            "agent_session_key": key, "token_usage_json": usage}


def resp(tid, rid, seq, parent=None):
    return {"turn_id": tid, "response_id": rid, "host_parent": parent, "first_seen_seq": seq}


def call(tid, rid, *, tool="Read", tokens=100, h="h1", acc=V, bot="alpha"):
    return {"bot_id": bot, "turn_id": tid, "tool_name": tool, "response_id": rid,
            "result_total_tokens": tokens, "result_accounting": acc, "result_sha256": h}


def _bot(out, bot="alpha"):
    return next(b for b in out["bots"] if b["bot_id"] == bot)


def test_late_result_and_seq_reset_across_turns():
    # Turn 1 is long (seq up to 100); turn 2 restarts seq at 2 and 8.
    turns = [turn("t1", at=0), turn("t2", at=1)]
    resps = [resp("t1", "r1", 1), resp("t1", "r100", 100), resp("t2", "r2", 2), resp("t2", "r8", 8)]
    out = tw.compute([call("t1", "r100", tokens=10)], turns, resps)
    # Late in turn 1: still carried by both turn-2 responses despite their lower seq.
    assert _bot(out)["exposure_tokens"] == 20
    out = tw.compute([call("t1", "r1", tokens=10)], turns, resps)
    assert _bot(out)["exposure_tokens"] == 30


def test_parallel_tools_in_one_response_share_the_same_later_count():
    turns = [turn("t1")]
    resps = [resp("t1", "a", 1), resp("t1", "b", 5)]
    calls = [call("t1", "a", tool="Read", tokens=10, h="x"), call("t1", "a", tool="Grep", tokens=7, h="y")]
    assert _bot(tw.compute(calls, turns, resps))["exposure_tokens"] == 17


def test_sibling_workers_are_unknown_and_unranked():
    turns = [turn("t1")]
    resps = [resp("t1", "m", 1), resp("t1", "w1", 2, parent="toolu_wf"),
             resp("t1", "w2", 3, parent="toolu_wf"), resp("t1", "m2", 4)]
    calls = [call("t1", "w1", h="same") for _ in range(25)] + [call("t1", "w2", h="same")]
    out = tw.compute(calls, turns, resps)
    b = _bot(out)
    assert b["worker_calls"] == 26 and b["worker_verified_tokens"] == 2600
    assert b["exposure_unknown_calls"] == 26 and b["refetch_candidates"] == 0
    assert out["ranked"] == [] and out["low_sample"] == []


def test_same_hash_after_compaction_is_not_a_refetch():
    turns = [turn("t1", at=0), turn("t2", at=1, usage=COMPACT), turn("t3", at=2)]
    resps = [resp("t1", "a", 1), resp("t2", "c", 1), resp("t3", "b", 1)]
    out = tw.compute([call("t1", "a"), call("t3", "b")], turns, resps)
    assert _bot(out)["refetch_candidates"] == 0
    # Exposure stops at the boundary: the turn-1 result is carried by nothing later.
    assert _bot(out)["exposure_tokens"] == 0


def test_fresh_recheck_with_same_bytes_is_a_candidate():
    turns = [turn("t1", at=0), turn("t2", at=1)]
    resps = [resp("t1", "a", 1), resp("t2", "b", 1)]
    out = tw.compute([call("t1", "a", tool="tasks_get"), call("t2", "b", tool="tasks_get")], turns, resps)
    b = _bot(out)
    assert b["refetch_candidates"] == 1 and b["refetch_candidate_tokens"] == 100


def test_null_session_turns_are_their_own_epoch():
    turns = [turn("t1", key=None, at=0), turn("t2", key=None, at=1)]
    resps = [resp("t1", "a", 1), resp("t2", "b", 1)]
    out = tw.compute([call("t1", "a"), call("t2", "b")], turns, resps)
    assert _bot(out)["refetch_candidates"] == 0 and _bot(out)["exposure_tokens"] == 0


def test_null_hash_never_matches():
    turns = [turn("t1")]
    resps = [resp("t1", "a", 1), resp("t1", "b", 2)]
    out = tw.compute([call("t1", "a", h=None), call("t1", "b", h=None)], turns, resps)
    assert _bot(out)["refetch_candidates"] == 0 and _bot(out)["calls_without_hash"] == 2


def test_unverified_rows_are_split_out_and_never_ranked():
    turns = [turn("t1")]
    resps = [resp("t1", "a", 1), resp("t1", "b", 2)]
    calls = [call("t1", "a", acc=None, tokens=50, h=f"h{i}") for i in range(30)]
    calls.append(call("t1", "a", tokens=None, h="z"))
    out = tw.compute(calls, turns, resps)
    b = _bot(out)
    assert b["unverified_tokens"] == 1500 and b["verified_tokens"] == 0
    assert b["calls_without_tokens"] == 1 and out["ranked"] == []


def test_missing_response_rows_make_the_epoch_unknown():
    turns = [turn("t1", at=0), turn("t2", at=1)]  # t2 has no response rows
    out = tw.compute([call("t1", "a")], turns, [resp("t1", "a", 1)])
    assert _bot(out)["exposure_unknown_calls"] == 1 and out["low_sample"] == []


def test_low_sample_bucket_has_no_class():
    turns = [turn("t1")]
    resps = [resp("t1", "a", 1), resp("t1", "b", 2)]
    out = tw.compute([call("t1", "a", tokens=9000, h=f"h{i}") for i in range(5)], turns, resps)
    assert out["ranked"] == []
    [row] = out["low_sample"]
    assert row["low_sample"] and row["classes"] == [] and row["owner"] is None


def test_class_chips_and_owner_order():
    # 60 calls over 6 sessions, all one hash, large and repeated: all three chips.
    turns, resps, calls = [], [], []
    for s in range(6):
        for i in range(10):
            tid = f"s{s}t{i}"
            turns.append(turn(tid, key=f"k{s}", at=i))
            resps += [resp(tid, "a", 1), resp(tid, "b", 2)]
            calls.append(call(tid, "a", tool="ops_list_operations", tokens=9000, h="static"))
    # A small fresh tool: refetch-only.
    for i in range(20):
        calls.append(call("s0t0", "a", tool="tasks_get", tokens=10, h="same" if i % 2 else f"u{i}"))
    out = tw.compute(calls, turns, resps)
    big, small = out["ranked"]
    assert big["tool_name"] == "ops_list_operations"
    assert big["classes"] == ["oversized", "static", "refetch"]
    assert big["owner"] == tw.OWNERS["oversized"]
    assert big["calls"] == 60 and big["hash_coverage"] == 1.0
    assert small["classes"] == ["refetch"] and small["owner"] == tw.OWNERS["refetch"]


def test_orphan_calls_add_no_session_diversity_for_static():
    # One real session with 50 same-hash calls plus 4 orphans (no fetched turn).
    turns, resps, calls = [], [], []
    for i in range(50):
        tid = f"t{i}"
        turns.append(turn(tid, at=i))
        resps.append(resp(tid, "a", 1))
        calls.append(call(tid, "a", h="same"))
    calls += [call(f"gone{i}", "a", h="same") for i in range(4)]
    row = tw.compute(calls, turns, resps)["ranked"][0]
    assert "static" not in row["classes"]
    assert row["calls"] == 54


def test_protected_bot_is_excluded():
    turns = [turn("t1", bot="alpha"), turn("p1", bot="Guarded")]
    resps = [resp("t1", "a", 1), resp("p1", "a", 1)]
    out = tw.compute([call("t1", "a"), call("p1", "a", bot="Guarded")], turns, resps,
                     protected=["guarded"])
    assert [b["bot_id"] for b in out["bots"]] == ["alpha"]


# ── SQL fetch + route helper over a real (sqlite) schema ─────────────────


@pytest.fixture
def engine():
    eng = create_engine("sqlite://")
    SQLModel.metadata.create_all(eng, tables=[TurnLog.__table__, ToolCallRecord.__table__,
                                              LlmResponseRecord.__table__])
    now = datetime.now(timezone.utc)
    with Session(eng) as s:
        for bot in ("alpha", "guarded"):
            s.add(TurnLog(id=f"{bot}-t", bot_id=bot, created_at=now, agent_session_key="k"))
            s.add(LlmResponseRecord(turn_id=f"{bot}-t", response_id="a", first_seen_seq=1,
                                    created_at=now, updated_at=now))
            s.add(LlmResponseRecord(turn_id=f"{bot}-t", response_id="b", first_seen_seq=2,
                                    created_at=now, updated_at=now))
            s.add(ToolCallRecord(bot_id=bot, turn_id=f"{bot}-t", tool_name="Read", response_id="a",
                                 result_total_tokens=40, result_accounting=V, result_sha256="h",
                                 arguments_json="SECRET_ARG", result_text="SECRET_RESULT"))
        s.add(ToolCallRecord(bot_id="alpha", turn_id="old", tool_name="Read",
                             created_at=now - timedelta(days=9), result_total_tokens=1))
        s.commit()
    return eng


def test_route_excludes_protected_and_returns_aggregates_only(engine, monkeypatch):
    monkeypatch.setattr("llm_bawt.memory.guidance_run.resolve_protected", lambda conn: ["guarded"])
    with engine.connect() as conn:
        out = usage_routes.tool_waste_report(conn, 7)
    assert [b["bot_id"] for b in out["bots"]] == ["alpha"]
    assert out["bots"][0]["calls"] == 1 and out["bots"][0]["exposure_tokens"] == 40
    assert "SECRET" not in json.dumps(out, default=str) and '"h"' not in json.dumps(out)


def test_route_fails_closed_when_protected_unresolved(engine, monkeypatch):
    monkeypatch.setattr("llm_bawt.memory.guidance_run.resolve_protected", lambda conn: [])
    with engine.connect() as conn, pytest.raises(HTTPException) as err:
        usage_routes.tool_waste_report(conn, 7)
    assert err.value.status_code == 503
