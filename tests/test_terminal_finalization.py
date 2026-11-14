"""Normal finalizers freeze before history and never publish a losing reply."""
from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import asyncio
import pytest

from llm_bawt.bots import StreamingTTSScrubber
from llm_bawt.service.turn_stream_finalize import TurnStreamFinalizer
from llm_bawt.service.turn_terminal_finalize import (
    NonstreamTerminalMetadata, commit_nonstream_terminal, prepare_terminal_content, project_execution_terminal,
)
from tests.test_turn_terminal_protocol import insert_turn, protocol, read
from tests.test_turn_termination import finalizer_context

__all__ = ["protocol"]


@pytest.fixture
def stream_case(protocol, monkeypatch):
    turn = insert_turn(protocol.engine)
    row = read(protocol.engine, turn)
    store = SimpleNamespace(
        get_turn=lambda turn_id: read(protocol.engine, turn_id),
        changed_files_summary=Mock(return_value={"files": [{"path": "frozen.txt"}]}),
        update_turn=Mock(side_effect=AssertionError("generic terminal writer forbidden")),
    )
    ctx = finalizer_context(store, monkeypatch, partial=" final **reply** ")
    ctx.turn_log_id = turn
    ctx.terminal_history_session_id = row.history_session_id
    ctx.terminal_error_text = None
    ctx.terminal_follower = Mock(protocol=protocol, lease=protocol.claim(turn))
    ctx._redis_sub, ctx.loop, ctx.publish_lock = object(), object(), object()
    ctx.stream_text_offset = 23
    ctx.llm_bawt = SimpleNamespace(
        adapter=SimpleNamespace(clean_output=lambda text: text.strip()),
        finalize_response=Mock(side_effect=AssertionError("mutable history forbidden")),
    )
    ctx.svc._finalize_turn = Mock(side_effect=AssertionError("legacy finalizer forbidden"))
    ctx.svc._extract_agent_backend_tool_calls = Mock(return_value=[])
    ctx.svc._resolve_turn_token_usage = Mock(return_value={"input_tokens": 7, "output_tokens": 3})
    project = Mock()
    monkeypatch.setattr(
        "llm_bawt.service.turn_terminal_finalize.project_execution_terminal_from_worker", project,
    )
    bump = Mock()
    monkeypatch.setattr("llm_bawt.service.turn_stream_finalize.bump_after_finalize", bump)
    direct = Mock(side_effect=AssertionError("unacknowledged terminal event forbidden"))
    finalizer = TurnStreamFinalizer(
        ctx, publish_event_direct=direct,
        enrich_attachment_refs=lambda refs: [{**ref, "url": "/frozen"} for ref in refs],
    )
    return ctx, finalizer, project, bump, direct


def snapshot(protocol, ctx):
    return json.loads(read(protocol.engine, ctx.turn_log_id).terminal_snapshot_json)


def test_stream_freezes_every_terminal_field_before_projection(protocol, stream_case):
    ctx, finalizer, project, bump, direct = stream_case
    ctx.tool_context_holder[0] = "[Read]\nfrozen result"
    ctx.reasoning_holder[0] = "frozen reasoning"
    ctx.agent_attachments_holder[:] = [{"asset_id": "image"}]
    ctx.approval_id_holder[0] = "approval"
    ctx.animation_holder[0] = "happy"
    ctx.tts_scrubber = StreamingTTSScrubber()
    prefix = ctx.tts_scrubber.feed(ctx.full_response_holder[0])
    before_offset = ctx.tts_scrubber.offset

    def observe_committed(*args, **kwargs):
        frozen = snapshot(protocol, ctx)
        assert read(protocol.engine, ctx.turn_log_id).terminal_protocol_state == "pending"
        assert frozen["response_text"] == "final **reply**"
        assert frozen["tool_context"] == "[Read]\nfrozen result"
        assert frozen["history_session_id"] == ctx.terminal_history_session_id
        assert frozen["reasoning"] == "frozen reasoning"
        completion = frozen["completion"]
        assert completion["status"] == "completed" and completion["end_reason"] == "approval"
        assert completion["approval_id"] == "approval" and completion["animation"] == "happy"
        assert completion["changed_files"] == {"files": [{"path": "frozen.txt"}]}
        assert completion["text_offset"] == 23
        assert completion["token_usage"] == {"input_tokens": 7, "output_tokens": 3}
        assert completion["attachments"] == [{"asset_id": "image", "url": "/frozen"}]
        assert frozen["tts_tail"]["tts_offset"] == before_offset
        assert completion["tts_text"] == prefix + frozen["tts_tail"]["delta"]
        ctx.terminal_follower.close.assert_called_once()

    project.side_effect = observe_committed
    finalizer.finalize(prepared_messages=[])
    project.assert_called_once_with(
        protocol, ctx.turn_log_id, ctx._redis_sub, ctx.loop, publish_lock=ctx.publish_lock,
    )
    assert ctx.full_response_holder[0] == "final **reply**"
    assert ctx.chunk_queue == [None] and ctx.done_event.is_set()
    ctx.svc._finalize_turn.assert_not_called()
    ctx.llm_bawt.finalize_response.assert_not_called()
    ctx.svc._turn_log_store.update_turn.assert_not_called()
    direct.assert_not_called()
    bump.assert_called_once()


def test_stream_stop_winner_is_the_only_history_and_publication_source(protocol, stream_case):
    ctx, finalizer, project, bump, _ = stream_case
    winner = protocol.interrupt(
        ctx.turn_log_id, status="aborted", end_reason="aborted", response_text="Stop-owned partial",
    ).snapshot
    finalizer.finalize(prepared_messages=[])
    assert snapshot(protocol, ctx) == winner
    assert ctx.full_response_holder[0] == "Stop-owned partial"
    assert ctx.cancelled_holder[0]
    bump.assert_not_called()
    project.assert_called_once()


def test_stream_error_survives_the_guarded_legacy_writer(protocol, stream_case):
    ctx, finalizer, _, _, _ = stream_case
    ctx.terminal_error_text = "backend failed"
    finalizer.finalize(prepared_messages=[])
    frozen = snapshot(protocol, ctx)
    assert frozen["completion"]["status"] == "error"
    assert frozen["completion"]["end_reason"] == "upstream_error"
    assert frozen["completion"]["error_text"] == "backend failed"
    assert frozen["response_text"].count("Turn failed") == 1
    assert "backend failed" in frozen["response_text"]


@pytest.mark.parametrize("cancelled", [False, True])
def test_empty_stream_still_commits_explicit_terminal(protocol, stream_case, cancelled):
    ctx, finalizer, _, _, _ = stream_case
    ctx.full_response_holder[0] = ""
    ctx.cancelled_holder[0] = cancelled
    finalizer.finalize(prepared_messages=[])
    frozen = snapshot(protocol, ctx)
    assert frozen["response_text"] == ""
    assert frozen["completion"]["status"] == ("cancelled" if cancelled else "timeout")


@pytest.mark.parametrize("reason", ["question", "approval"])
def test_textless_deferral_is_completed_not_timeout(protocol, stream_case, reason):
    ctx, finalizer, project, _, _ = stream_case
    ctx.full_response_holder[0] = ""
    getattr(ctx, f"{reason}_id_holder")[0] = "deferred-id"
    ctx.svc._pending_question_store = SimpleNamespace(get=lambda _id: None)
    finalizer.finalize(prepared_messages=[])
    frozen = snapshot(protocol, ctx)
    assert frozen["response_text"] == ""
    assert frozen["completion"]["status"] == "completed"
    assert frozen["completion"]["end_reason"] == reason
    assert frozen["completion"][f"{reason}_id"] == "deferred-id"
    project.assert_called_once()


def test_projection_failure_never_changes_or_republishes_the_committed_candidate(protocol, stream_case):
    ctx, finalizer, project, _, direct = stream_case
    project.side_effect = ConnectionError("Redis unavailable")
    finalizer.finalize(prepared_messages=[])
    row = read(protocol.engine, ctx.turn_log_id)
    assert row.terminal_protocol_state == "pending" and row.ended_at
    assert snapshot(protocol, ctx)["response_text"] == "final **reply**"
    direct.assert_not_called()
    assert ctx.chunk_queue == [None] and ctx.done_event.is_set()


def test_commit_failure_publishes_no_speculative_success(protocol, stream_case, monkeypatch):
    ctx, finalizer, project, _, direct = stream_case
    monkeypatch.setattr(protocol, "commit", Mock(side_effect=ConnectionError("DB unavailable")))
    finalizer.finalize(prepared_messages=[])
    assert read(protocol.engine, ctx.turn_log_id).terminal_snapshot_json is None
    project.assert_not_called()
    direct.assert_not_called()
    assert ctx.chunk_queue == [None] and ctx.done_event.is_set()


def test_terminal_preparation_has_no_history_side_effect_and_preserves_tool_inputs():
    calls = [{"tool": "Read", "result": "result"}]
    original = deepcopy(calls)
    svc = SimpleNamespace(
        _extract_agent_backend_tool_calls=Mock(return_value=[]),
        _resolve_turn_token_usage=lambda llm, usage: usage,
    )
    llm = SimpleNamespace(adapter=None, finalize_response=Mock())
    result = prepare_terminal_content(
        svc, llm, response_text="answer", tool_context="", tool_call_details=calls,
        token_usage={"input_tokens": 4},
    )
    assert result == {
        "response_text": "answer", "tool_context": "[Read]\nresult", "token_usage": {"input_tokens": 4},
    }
    assert calls == original
    llm.finalize_response.assert_not_called()


def test_nonstream_worker_commits_full_snapshot_without_mutable_history(protocol, stream_case):
    ctx, _, _, _, _ = stream_case
    result = commit_nonstream_terminal(
        ctx.svc, ctx.llm_bawt, ctx.terminal_follower, status="ok", end_reason="question",
        response_text=" answer ", tool_context="[Read]\nresult",
        attachments=[{"asset_id": "image"}], reasoning="reasoning",
        completion_fields={"question_id": "question", "approval_id": "approval"},
    )
    assert result.won
    frozen = result.snapshot
    assert frozen["response_text"] == "answer" and frozen["reasoning"] == "reasoning"
    assert frozen["history_session_id"] == ctx.terminal_history_session_id
    assert frozen["attachments"] == [{"asset_id": "image"}]
    assert frozen["completion"]["question_id"] == "question"
    assert frozen["completion"]["approval_id"] == "approval"
    assert frozen["completion"]["token_usage"]["input_tokens"] == 7
    assert frozen == snapshot(protocol, ctx)
    ctx.svc._turn_log_store.update_turn.assert_not_called()
    ctx.llm_bawt.finalize_response.assert_not_called()
    ctx.terminal_follower.close.assert_called_once()


def test_nonstream_stop_winner_does_not_prepare_a_losing_client_result(protocol, stream_case):
    ctx, _, _, _, _ = stream_case
    winner = protocol.interrupt(
        ctx.turn_log_id, status="aborted", end_reason="aborted", response_text="winner partial",
    ).snapshot
    ctx.svc._resolve_turn_token_usage.side_effect = AssertionError("no stale usage")
    result = commit_nonstream_terminal(
        ctx.svc, ctx.llm_bawt, ctx.terminal_follower, status="ok", end_reason="stop",
        response_text="losing reply",
    )
    assert not result.won and result.snapshot == winner
    ctx.svc._extract_agent_backend_tool_calls.assert_not_called()
    ctx.svc._resolve_turn_token_usage.assert_not_called()


def test_nonstream_unbound_cancellation_never_reads_a_cached_result(protocol, stream_case):
    ctx, _, _, _, _ = stream_case
    turn = insert_turn(protocol.engine, unbound=True)
    follower = Mock(protocol=protocol, lease=protocol.claim(turn))
    ctx.svc._resolve_turn_token_usage.side_effect = AssertionError("no previous usage")
    result = commit_nonstream_terminal(
        ctx.svc, ctx.llm_bawt, follower, status="aborted", end_reason="aborted",
    )
    assert result.snapshot["history_disposition"] == "unbound"
    assert result.snapshot["response_text"] == ""
    assert result.snapshot["completion"]["token_usage"] is None
    ctx.svc._extract_agent_backend_tool_calls.assert_not_called()
    follower.close.assert_called_once()


def test_unbound_stream_failure_never_prepares_cached_usage_or_invents_history(protocol, stream_case):
    ctx, finalizer, project, _, _ = stream_case
    turn = insert_turn(protocol.engine, unbound=True)
    ctx.turn_log_id, ctx.terminal_history_session_id = turn, None
    ctx.terminal_follower = Mock(protocol=protocol, lease=protocol.claim(turn))
    ctx.full_response_holder[0] = ""
    ctx.terminal_error_text = "session bind failed"
    ctx.svc._resolve_turn_token_usage.side_effect = AssertionError("no cached usage")
    finalizer.finalize(prepared_messages=[])
    frozen = snapshot(protocol, ctx)
    assert frozen["history_disposition"] == "unbound"
    assert frozen["response_text"] == "" and frozen["history_session_id"] is None
    assert frozen["completion"]["error_text"] == "session bind failed"
    project.assert_called_once()
    ctx.svc._resolve_turn_token_usage.assert_not_called()


def test_normal_nonstream_metadata_reuses_deduplicated_approval_and_question_effects():
    svc = SimpleNamespace(
        _tool_approval_policy_store=Mock(),
        _pending_question_store=Mock(),
    )
    metadata = NonstreamTerminalMetadata(
        svc, SimpleNamespace(session_id="old"), turn_id="turn", trigger_message_id="user",
        bot_id="bot", user_id="owner", loop=None,
    )
    metadata._publish_event_direct = Mock()
    approval = {
        "event": "approval_required", "tool_use_id": "approval", "tool_name": "Bash",
        "arguments": {"command": "test"}, "provider": "claude-code",
    }
    question = {
        "event": "await_tool_result", "tool_use_id": "question", "tool_name": "AskUserQuestion",
        "arguments": {}, "provider": "claude-code",
    }
    for item in [approval, approval, question, question]:
        metadata.observe(item, session_id="final-bound-session")
    svc._tool_approval_policy_store.record_request.assert_called_once()
    assert svc._tool_approval_policy_store.record_request.call_args.kwargs["session_id"] == "final-bound-session"
    svc._pending_question_store.upsert_awaiting.assert_called_once()
    assert metadata._publish_event_direct.call_count == 2
    svc._pending_question_store.get.return_value = None
    assert metadata.frozen_fields() == {
        "question_id": "question", "approval_id": "approval", "approval_persist_failed": None,
    }
    svc._pending_question_store.get.return_value = SimpleNamespace(live_deadline=1, status="answered")
    assert metadata.frozen_fields()["question_id"] is None


def test_acknowledged_projection_waits_behind_prior_live_events(monkeypatch):
    async def scenario():
        lock = asyncio.Lock()
        project = AsyncMock(return_value="ACK")
        monkeypatch.setattr(
            "llm_bawt.service.turn_terminal_finalize.project_committed_terminal", project,
        )
        subscriber = SimpleNamespace(_pub_redis=object())
        await lock.acquire()
        # Queue a real contender behind an existing live-event writer.
        started = asyncio.Event()

        async def wait_for_projection():
            started.set()
            return await project_execution_terminal("protocol", "turn", subscriber, publish_lock=lock)

        task = asyncio.create_task(wait_for_projection())
        await started.wait()
        project.assert_not_awaited()
        lock.release()
        assert await task == "ACK"
        project.assert_awaited_once_with("protocol", "turn", subscriber._pub_redis)

    asyncio.run(scenario())
