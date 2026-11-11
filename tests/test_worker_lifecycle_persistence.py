"""TASK-2169: worker lifecycle survives reload through durable records."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, create_engine

from llm_bawt.service.routes.turn_logs import _turn_workers
from llm_bawt.service.tool_call_store import ToolCallRecord, ToolCallResultPayloadRecord, ToolCallStore
from llm_bawt.service.tool_event_coordinator import ToolEventCoordinator
from llm_bawt.service.turn_logs import TurnLog
from llm_bawt.service.worker_lifecycle_store import WorkerLifecycleRecord

TURN = "turn-delivery-a64e866ac9f7466fb4cc6d4cb1a935f7"
PARENT = "call_006761bcab634a38ad4c727fc2913c56"
TASK = "a30643573b719fc2e"


@pytest.fixture()
def engine():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(eng, tables=[
        TurnLog.__table__, ToolCallRecord.__table__, ToolCallResultPayloadRecord.__table__,
        WorkerLifecycleRecord.__table__,
    ])
    return eng


def _evt(kind, ts, **extra):
    return {"_type": kind, "turn_id": TURN, "bot_id": "codex", "user_id": "nick", "task_id": TASK, "ts": ts, **extra}


def _natural_sequence():
    """The exact shape of the record 473276 run: 1 start, 30 progress, 1 done."""
    events = [_evt("subagent_started", 100.0, tool_use_id=PARENT, description="Locate omitted bot identity callers",
                   task_type="local_agent")]
    for i in range(30):
        ts = 101.0 + i
        if i % 3 == 2:
            # task_updated status frames carry no tool_use_id.
            events.append(_evt("subagent_progress", ts, status="running"))
        else:
            events.append(_evt("subagent_progress", ts, tool_use_id=PARENT, last_tool_name="Grep",
                               usage={"total_tokens": 1000 + i, "tool_uses": i + 1, "cumulative_tokens": 5000 + 100 * i}))
    events.append(_evt("subagent_done", 140.0, tool_use_id=PARENT, status="completed", summary="Found 3 callers",
                       usage={"total_tokens": 2000, "tool_uses": 32, "duration_ms": 302000, "cumulative_tokens": 9000}))
    return events


def _history(engine):
    store = SimpleNamespace(engine=engine)
    return _turn_workers(store, [SimpleNamespace(id=TURN)])[TURN]


def test_single_worker_sequence_survives_reload(engine):
    coord = ToolEventCoordinator(engine)
    published = [coord.worker(e) for e in _natural_sequence()]
    # Id-less status frames are republished with the resolved spawning call.
    assert all(p["tool_use_id"] == PARENT for p in published)

    workers = _history(engine)
    assert len(workers) == 1
    w = workers[0]
    assert w["task_id"] == TASK and w["tool_use_id"] == PARENT
    assert w["status"] == "completed" and w["summary"] == "Found 3 callers"
    assert w["description"] == "Locate omitted bot identity callers"
    assert w["usage"] == {"total_tokens": 2000, "tool_uses": 32, "duration_ms": 302000, "cumulative_tokens": 9000}
    assert w["started_at"] == 100.0 and w["updated_at"] == 140.0


def test_replay_and_out_of_order_delivery_converge(engine):
    coord = ToolEventCoordinator(engine)
    seq = _natural_sequence()
    for e in seq:
        coord.worker(e)
    first = _history(engine)
    # Reconnect replay: duplicates and reversed order change nothing.
    for e in reversed(seq):
        coord.worker(e)
    assert _history(engine) == first


def test_terminal_status_is_sticky_and_partial_usage_sticky(engine):
    coord = ToolEventCoordinator(engine)
    coord.worker(_evt("subagent_started", 1.0, tool_use_id=PARENT))
    coord.worker(_evt("subagent_progress", 2.0, status="timed_out", failure={"reason": "idle"},
                      usage={"cumulative_tokens": 10, "cumulative_partial": True}))
    coord.worker(_evt("subagent_progress", 3.0, tool_use_id=PARENT, status="running",
                      usage={"cumulative_tokens": 20}))
    w = _history(engine)[0]
    assert w["status"] == "timed_out" and w["failure"] == {"reason": "idle"}
    assert w["usage"] == {"cumulative_tokens": 20, "cumulative_partial": True}


def test_idless_frame_for_unknown_worker_is_not_invented(engine):
    coord = ToolEventCoordinator(engine)
    raw = _evt("subagent_progress", 1.0, status="running")
    assert coord.worker(raw) == raw
    assert _turn_workers(SimpleNamespace(engine=engine), [SimpleNamespace(id=TURN)]) == {}


def test_worker_count_is_never_inferred_from_child_tools(engine):
    tools = ToolCallStore(engine)
    for i in range(32):
        tools.save_start(turn_id=TURN, bot_id="codex", user_id="nick", call_id=f"c{i}", tool_name="Grep",
                         arguments={}, iteration=1, started_at=1.0, tool_use_id=f"child{i}",
                         parent_tool_use_id=PARENT)
    assert _turn_workers(SimpleNamespace(engine=engine), [SimpleNamespace(id=TURN)]) == {}
    coord = ToolEventCoordinator(engine)
    for e in _natural_sequence():
        coord.worker(e)
    assert len(_history(engine)) == 1


def test_persistence_failure_still_publishes(engine, monkeypatch):
    coord = ToolEventCoordinator(engine)
    monkeypatch.setattr(coord.workers, "record", lambda _e: (_ for _ in ()).throw(RuntimeError("db down")))
    raw = _evt("subagent_started", 1.0, tool_use_id=PARENT)
    assert coord.worker(raw) == raw
