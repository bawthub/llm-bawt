"""TASK-952: terminal truth, request targeting, unknown tools and exception safety."""
from __future__ import annotations

import asyncio
import io
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock

import pytest
from rich.console import Console
from sqlmodel import Session, SQLModel, create_engine

from agent_bridge.tool_results import ToolResultPayload
from llm_bawt.service.tool_call_store import ToolCallRecord, ToolCallResultPayloadRecord, ToolCallStore
from llm_bawt.service.turn_logs import TurnLog, TurnLogStore
from llm_bawt.service.turn_termination import TurnAbortCoordinator
from llm_bawt.service.turn_stream_finalize import TurnStreamFinalizer
from llm_bawt.service.routes.turn_logs import _records_to_calls


@pytest.fixture
def store():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine, tables=[TurnLog.__table__, ToolCallRecord.__table__, ToolCallResultPayloadRecord.__table__])
    store = TurnLogStore.__new__(TurnLogStore)
    store.engine = engine
    store.ttl_hours = None
    with Session(engine) as session:
        session.add(TurnLog(id="turn", bot_id="al", user_id="nick", status="streaming", agent_session_key="al:nick", agent_request_id="req-original"))
        session.add(TurnLog(id="sibling", bot_id="al", user_id="nick", status="streaming"))
        session.add(ToolCallRecord(turn_id="turn", call_id="call", tool_use_id="tool", tool_name="Bash"))
        session.add(ToolCallRecord(turn_id="sibling", call_id="other", tool_name="Bash"))
        session.commit()
    return store


def tool(store, record_id=1):
    with Session(store.engine) as session:
        return session.get(ToolCallRecord, record_id)


@pytest.mark.parametrize("later_status", ["ok", "completed", "error", "streaming", "timeout"])
def test_abort_is_monotonic_and_settles_only_matching_tools(store, later_status):
    coordinator = TurnAbortCoordinator(store)
    assert coordinator.request("turn", source="chat_stop", peer="proxy")
    assert store.get_turn("turn").status == "cancelling"
    assert tool(store).ended_at is None
    store.update_turn(turn_id="turn", status="aborted", end_reason="aborted")
    ended = store.get_turn("turn").ended_at
    store.update_turn(turn_id="turn", status=later_status, end_reason="stop", response_text="partial", request_payload={"messages": []})
    row = store.get_turn("turn")
    assert (row.status, row.end_reason, row.ended_at) == ("aborted", "aborted", ended)
    assert row.response_text == "partial"
    assert json.loads(row.request_json)["abort_request"]["source"] == "chat_stop"
    assert tool(store).ended_at is not None
    assert tool(store).result_text is None and tool(store).is_error is None
    assert _records_to_calls([tool(store)])[0]["status"] == "interrupted"
    assert tool(store, 2).ended_at is None
    assert not coordinator.request("turn", source="bot_list_stop", peer="other")


def test_completed_turn_cannot_be_aborted(store):
    store.update_turn(turn_id="turn", status="ok", end_reason="stop")
    assert not TurnAbortCoordinator(store).request("turn", source="chat_stop", peer=None)
    assert store.get_turn("turn").status == "ok"


def test_late_tool_result_reconciles_without_reopening_or_duplicate(store):
    TurnAbortCoordinator(store).request("turn", source="unknown", peer=None)
    store.update_turn(turn_id="turn", status="aborted", end_reason="aborted")
    calls = ToolCallStore(store.engine)
    record_id, _ = calls.save_result(turn_id="turn", call_id="different-call", tool_use_id="tool", tool_name="Bash", bot_id="al", user_id="nick", payload=ToolResultPayload.from_value("real output"), ended_at=123.0, is_error=False)
    assert record_id == 1
    assert _records_to_calls([tool(store)])[0]["result"] == "real output"
    assert _records_to_calls([tool(store)])[0]["status"] == "completed"
    assert store.get_turn("turn").status == "aborted"
    assert calls.save_start(turn_id="turn", bot_id="al", user_id="nick", call_id="late-start-id", tool_use_id="tool", tool_name="Bash") == record_id
    assert tool(store).result_text == "real output"


def test_late_start_and_legacy_history_remain_terminal(store):
    store.update_turn(turn_id="turn", status="error", end_reason="upstream_error")
    calls = ToolCallStore(store.engine)
    record_id = calls.save_start(turn_id="turn", bot_id="al", user_id="nick", call_id="late", tool_name="Read")
    assert _records_to_calls([tool(store, record_id)])[0]["status"] == "interrupted"
    legacy = ToolCallRecord(tool_name="Bash", result_text=None)
    assert _records_to_calls([legacy], turn_ended_at=store.get_turn("turn").ended_at)[0]["status"] == "interrupted"
    assert legacy.ended_at is None  # projection, no manual data cleanup


def test_abort_is_persisted_before_request_scoped_rpc(store, monkeypatch):
    from llm_bawt.service.turn_abort import abort_turn
    from llm_bawt.agent_backends import agent_bridge
    import llm_bawt.bots
    publish = AsyncMock()
    monkeypatch.setattr("llm_bawt.service.turn_abort.publish_abort_snapshot", publish)
    monkeypatch.setattr(llm_bawt.bots, "BotManager", lambda config: SimpleNamespace(get_bot=lambda bot: SimpleNamespace(agent_backend="claude-code")))
    async def send_rpc(method, params, request_id, **kwargs):
        assert store.get_turn("turn").status == "cancelling"
        assert params == {"sessionKey": "al:nick", "requestId": "req-original"}
        store.update_turn(turn_id="turn", status="ok", end_reason="stop", response_text="EOF partial")
        return {"ok": True, "cancelled": True, "detail": "task_cancelled"}
    rpc = AsyncMock(side_effect=send_rpc)
    monkeypatch.setattr(agent_bridge, "get_agent_subscriber", lambda: SimpleNamespace(send_rpc=rpc))
    service = SimpleNamespace(_turn_log_store=store, config=None)
    assert asyncio.run(abort_turn(service, store.get_turn("turn"), source="chat_stop", peer="proxy")) == "aborted:task_cancelled"
    assert store.get_turn("turn").status == "aborted"
    audit = json.loads(store.get_turn("turn").request_json)["abort_request"]
    assert audit["outcome"] == "task_cancelled" and audit["actor"] is None
    asyncio.run(abort_turn(service, store.get_turn("turn"), source="unknown", peer=None))
    assert rpc.await_count == 1
    publish.assert_awaited_once()


@pytest.mark.parametrize("rpc_reply", [None, {"ok": False, "error": "worker did not acknowledge"}])
def test_unconfirmed_abort_stays_nonterminal_and_can_retry(store, monkeypatch, rpc_reply):
    from fastapi import HTTPException
    from llm_bawt.service.turn_abort import abort_turn
    from llm_bawt.agent_backends import agent_bridge
    import llm_bawt.bots
    publish = AsyncMock()
    monkeypatch.setattr("llm_bawt.service.turn_abort.publish_abort_snapshot", publish)
    monkeypatch.setattr(llm_bawt.bots, "BotManager", lambda config: SimpleNamespace(get_bot=lambda bot: SimpleNamespace(agent_backend="claude-code")))
    rpc = AsyncMock(side_effect=TimeoutError() if rpc_reply is None else None, return_value=rpc_reply)
    monkeypatch.setattr(agent_bridge, "get_agent_subscriber", lambda: SimpleNamespace(send_rpc=rpc))
    service = SimpleNamespace(_turn_log_store=store, config=None)
    with pytest.raises(HTTPException) as error:
        asyncio.run(abort_turn(service, store.get_turn("turn"), source="chat_stop", peer=None))
    assert error.value.status_code == 503
    assert store.get_turn("turn").status == "cancelling"
    assert store.get_turn("turn").ended_at is None and tool(store).ended_at is None
    publish.assert_not_awaited()
    rpc.side_effect = None
    rpc.return_value = {"ok": True, "cancelled": True, "detail": "task_cancelled"}
    asyncio.run(abort_turn(service, store.get_turn("turn"), source="chat_stop", peer=None))
    assert store.get_turn("turn").status == "aborted"
    assert tool(store).result_text is None and tool(store).ended_at is not None
    publish.assert_awaited_once()


def finalizer_context(store, monkeypatch, *, partial="partial"):
    svc = SimpleNamespace(_turn_log_store=store, _resolve_turn_token_usage=lambda *args: {}, _update_turn_log=store.update_turn)
    def persist(**kwargs):
        store.update_turn(turn_id="turn", status=kwargs.get("status", "ok"), end_reason=kwargs.get("end_reason"), response_text=kwargs["response_text"])
    svc._finalize_turn = persist
    ctx = SimpleNamespace(svc=svc, turn_log_id="turn", timing_holder=[1., 2.], cancelled_holder=[False], llm_bawt=None, token_usage_holder=[{}], full_response_holder=[partial], tool_context_holder=[""], tool_call_details_holder=[], user_prompt="test", model_alias="test-model", bot_id="al", user_id="nick", animation_holder=[None], agent_attachments_holder=[], reasoning_holder=[""], assistant_message_id="assistant", question_id_holder=[None], approval_id_holder=[None], approval_persist_failed_holder=[None], tts_scrub=False, tts_scrubber=None, thread_binding={"thread_session_id": "thread"}, loop=None, chunk_queue=[], is_agent_backend=True, done_event=threading.Event())
    monkeypatch.setattr("llm_bawt.service.turn_stream_finalize.put_queue_item_threadsafe", lambda loop, queue, value: queue.append(value))
    return ctx


@pytest.mark.parametrize("partial", ["", "partial"])
def test_finalizer_race_preserves_partial_and_cancelled_outcome(store, monkeypatch, partial):
    ctx = finalizer_context(store, monkeypatch, partial=partial)
    original = ctx.svc._finalize_turn
    def racing_persist(**kwargs):
        TurnAbortCoordinator(store).request("turn", source="unknown", peer=None)
        original(**kwargs)
    ctx.svc._finalize_turn = racing_persist
    if not partial:
        TurnAbortCoordinator(store).request("turn", source="unknown", peer=None)
    events = []
    TurnStreamFinalizer(ctx, publish_event_direct=events.append, enrich_attachment_refs=lambda refs: refs).finalize(prepared_messages=[])
    assert events[-1]["status"] == "cancelled" and events[-1]["end_reason"] == "aborted"
    assert events[-1]["response_text"] == partial
    assert ctx.chunk_queue == [None] and ctx.done_event.is_set()


def test_persistence_and_diagnostic_failure_still_emits_terminal_and_closes(store, monkeypatch):
    ctx = finalizer_context(store, monkeypatch)
    ctx.svc._finalize_turn = Mock(side_effect=RuntimeError("bad [/]"))
    monkeypatch.setattr("llm_bawt.service.turn_stream_finalize.log.error", Mock(side_effect=RuntimeError("logger failed")))
    events = []
    TurnStreamFinalizer(ctx, publish_event_direct=events.append, enrich_attachment_refs=lambda refs: refs).finalize(prepared_messages=[])
    assert events[-1]["status"] == "error"
    assert tool(store).ended_at is not None
    assert ctx.chunk_queue == [None] and ctx.done_event.is_set()


def test_publish_failure_cannot_orphan_http_stream(store, monkeypatch):
    ctx = finalizer_context(store, monkeypatch)
    with pytest.raises(RuntimeError, match="publish"):
        TurnStreamFinalizer(ctx, publish_event_direct=Mock(side_effect=RuntimeError("publish")), enrich_attachment_refs=lambda refs: refs).finalize(prepared_messages=[])
    assert ctx.chunk_queue == [None] and ctx.done_event.is_set()


def test_approval_failure_is_committed_before_terminal_guard(store, monkeypatch):
    ctx = finalizer_context(store, monkeypatch)
    ctx.approval_persist_failed_holder[0] = {"error": "approval DB unavailable"}
    # Match the real _finalize_turn default: an omitted reason becomes stop.
    ctx.svc._finalize_turn = lambda **kw: store.update_turn(
        turn_id="turn", status=kw.get("status", "ok"),
        end_reason=kw.get("end_reason") or "stop", response_text=kw["response_text"],
    )
    events = []
    TurnStreamFinalizer(ctx, publish_event_direct=events.append, enrich_attachment_refs=lambda refs: refs).finalize(prepared_messages=[])
    assert store.get_turn("turn").end_reason == "approval_persist_failed"
    assert "approval DB unavailable" in store.get_turn("turn").error_text
    assert events[-1]["end_reason"] == "approval_persist_failed"


def test_response_and_error_logging_treat_markup_as_literal(monkeypatch):
    from llm_bawt.service import logging as logging_module
    from rich.logging import RichHandler
    output = io.StringIO()
    console = Console(file=output, color_system=None)
    monkeypatch.setattr(logging_module, "_verbose", True)
    service_log = logging_module.ServiceLogger("task952-test")
    service_log._console = console
    service_log._logger.handlers = [RichHandler(console=console, markup=True)]
    service_log._logger.propagate = False
    service_log.llm_response("[Continuity] [/] Message")
    service_log.error("failure: %s", "[/]")
    assert "[Continuity] [/] Message" in output.getvalue()
    assert "failure: [/]" in output.getvalue()
