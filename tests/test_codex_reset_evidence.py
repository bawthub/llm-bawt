"""Isolated native-Codex reset proof through the real command/event handlers."""
from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from agent_bridge.continuity import RESET_SKILLS_FIELD, reset_skills_field, seed_receipt
from agent_bridge.events import AgentEventKind
from agent_bridge.run_contract import TeardownReport
from codex_bridge.bridge import CodexBridge

TASK = "  /newspaper is literal task text.\nKeep these bytes.  "
ANSWER = "The task answer, not a reset confirmation."


@pytest.fixture
def harness(monkeypatch):
    state = SimpleNamespace(events=[], writes=[], script=[], provider_id="fresh-provider",
                            put_status=200, prompts=[], abort=False, receipts=[])
    controller = SimpleNamespace(signal=object(), abort=MagicMock())
    abort_error = type("AbortError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "openai_codex_sdk", SimpleNamespace(
        AbortController=lambda: controller, AbortError=abort_error,
        LocalImageInput=SimpleNamespace, TextInput=SimpleNamespace,
    ))
    state.abort_error = abort_error
    state.controller = controller

    def handle_http(request):
        if request.method == "POST":  # continuity receipt (TASK-1579)
            state.receipts.append((str(request.url), json.loads(request.content)))
            return httpx.Response(200, json={})
        if request.method == "GET":  # registry prompt fetch: use the fallback
            return httpx.Response(404, json={})
        assert request.method == "PUT"
        state.writes.append((str(request.url), json.loads(request.content)))
        return httpx.Response(state.put_status, json={})

    client_class = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client_class(
        transport=httpx.MockTransport(handle_http), **kwargs,
    ))
    publisher = MagicMock()
    publisher.publish_run_event.side_effect = lambda _request, event: state.events.append(event)
    bridge = CodexBridge(publisher, app_api_url="http://test-app", cwd="/tmp")
    bridge._model_info.get = AsyncMock(return_value={})
    state.bridge = bridge
    state.publisher = publisher
    state.redis = SimpleNamespace(xack=AsyncMock())
    thread = SimpleNamespace(id=None)

    async def events():
        for event in state.script:
            if callable(event):
                event = event()
            if isinstance(event, BaseException):
                raise event
            if state.abort:
                bridge._session_queue.cancel_event("review-bot:nick").set()
            yield event

    async def run_streamed(prompt, options):
        assert options == {"signal": controller.signal}
        state.prompts.append(prompt)
        thread.id = state.provider_id
        return SimpleNamespace(events=events())

    thread.run_streamed = run_streamed
    state.thread = thread
    state.sdk = SimpleNamespace(start_thread=MagicMock(return_value=thread),
                                resume_thread=MagicMock(return_value=thread))
    bridge._ensure_codex = MagicMock(return_value=state.sdk)
    state.fields = {
        "request_id": "reset-run", "session_key": "review-bot:nick", "bot_id": "review-bot",
        "message": TASK, "model": "codex-fixture", "trigger_message_id": "task-trigger",
        "thread_session_id": "successor-thread", "thread_resume_id": "previous-provider",
        "bridge_reset_action": "reset_follow_up", "inject_messages": "[]",
    }
    state.script = [
        SimpleNamespace(type="thread.started", thread_id="fresh-provider"),
        SimpleNamespace(type="item.completed", item=SimpleNamespace(
            type="agent_message", id="reply", text=ANSWER)),
        SimpleNamespace(type="turn.completed", usage=None),
    ]
    return state


def send(state):
    asyncio.run(state.bridge._handle_send(state.fields, "redis-message", state.redis))
    state.redis.xack.assert_awaited_once()
    # No codex process was ledgered (the SDK is faked) and nothing was
    # cancelled, so nothing the send started is known to be running.
    state.publisher.publish_run_done.assert_called_once_with(
        state.fields["request_id"], teardown=TeardownReport(exited=True))
    assert not state.bridge._trigger_message_ids
    assert not state.bridge._session_queue.is_busy(state.fields["session_key"])
    return state.events


def assert_no_ack(state):
    assert not any(event.raw.get("new_session_ack") for event in state.events)
    assert not any(event.kind == AgentEventKind.ASSISTANT_DONE for event in state.events)
    state.publisher.publish_unified_event.assert_not_called()


@pytest.mark.parametrize("started", [True, False])
def test_clean_follow_up_proves_exact_binding_and_keeps_task_answer(harness, started):
    if not started:
        harness.script.pop(0)
    events = send(harness)
    harness.sdk.start_thread.assert_called_once()
    harness.sdk.resume_thread.assert_not_called()
    assert harness.prompts == [TASK]
    assert harness.writes == [(
        "http://test-app/v1/sessions/successor-thread/agent-session-key?bot_id=review-bot",
        {"backend": "codex", "session_key": "fresh-provider", "model": "codex-fixture"},
    )]
    done = [event for event in events if event.kind == AgentEventKind.ASSISTANT_DONE]
    assert len(done) == 1 and done[0].text == ANSWER
    assert (done[0].run_id, done[0].trigger_message_id, done[0].provider,
            done[0].session_key) == ("reset-run", "task-trigger", "codex", "review-bot:nick")
    assert done[0].raw == {"new_session_ack": {
        "thread_session_id": "successor-thread", "provider_session_id": "fresh-provider",
        "clean_start": False, **seed_receipt([]),
    }}
    harness.publisher.publish_unified_event.assert_not_called()


def test_auth_failure_refreshes_app_bundle_without_replaying_turn(harness, monkeypatch):
    refresh = MagicMock(return_value=True)
    monkeypatch.setattr("codex_bridge.command_ops.materialize_auth_from_broker", refresh)
    harness.bridge._supervisor_teardown = MagicMock()
    harness.script = [RuntimeError("invalid_grant: stale local refresh token")]

    events = send(harness)

    refresh.assert_called_once()
    harness.bridge._supervisor_teardown.assert_called_once()
    harness.sdk.start_thread.assert_called_once()
    assert len(harness.prompts) == 1
    assert len(events) == 1 and events[0].kind == AgentEventKind.ERROR
    assert_no_ack(harness)


def test_thread_started_identity_is_sufficient_when_sdk_property_missing(harness):
    harness.provider_id = None
    send(harness)
    assert harness.events[-1].raw["new_session_ack"]["provider_session_id"] == "fresh-provider"


@pytest.mark.parametrize("mode", ["http-error", "missing-api", "no-id", "reused-id", "changed-id"])
def test_unverified_binding_fails_closed_without_retry(harness, mode):
    if mode == "http-error":
        harness.put_status = 503
    elif mode == "missing-api":
        harness.bridge._app_api_url = ""
    elif mode == "no-id":
        harness.provider_id = None
        harness.script.pop(0)
    elif mode == "reused-id":
        harness.provider_id = "previous-provider"
        harness.script[0].thread_id = "previous-provider"
    else:
        harness.provider_id = "different-provider"
    send(harness)
    assert_no_ack(harness)
    assert harness.events[-1].kind == AgentEventKind.ERROR
    harness.sdk.start_thread.assert_called_once()
    harness.sdk.resume_thread.assert_not_called()


@pytest.mark.parametrize("field,value", [
    ("bridge_reset_action", "reset_only"),
    ("bridge_reset_action", "unsupported"),
    ("explicit_thread", "true"),
    ("thread_session_id", ""),
    # A retained seed needs its registry framing; it is never dropped silently.
    ("inject_messages", '[{"role":"user","content":"retained"}]'),
    ("inject_messages", '{"not":"messages"}'),
    ("inject_messages", "not-json"),
])
def test_invalid_reset_rejected_before_sdk_construction(harness, field, value):
    harness.fields[field] = value
    send(harness)
    assert_no_ack(harness)
    harness.bridge._ensure_codex.assert_not_called()
    assert not harness.prompts and not harness.writes
    assert harness.events[-1].kind == AgentEventKind.ERROR


@pytest.mark.parametrize("failure", ["eof", "failed", "timeout", "session-error", "abort", "cancel"])
def test_incomplete_follow_up_never_acknowledges_or_replays(harness, failure):
    if failure == "eof":
        harness.script.pop()
    elif failure == "failed":
        harness.script[-1] = SimpleNamespace(type="turn.failed", error={"message": "task failed"})
    elif failure == "abort":
        harness.script[-1] = harness.abort_error("aborted")
    elif failure == "cancel":
        harness.abort = True
    else:
        # Fail before any assistant text, where legacy session retry would run.
        harness.script = [TimeoutError("timeout") if failure == "timeout"
                          else RuntimeError("thread not found")]
    send(harness)
    assert_no_ack(harness)
    harness.sdk.start_thread.assert_called_once()
    harness.sdk.resume_thread.assert_not_called()
    assert harness.prompts == [TASK]


def standalone(harness):
    harness.fields.update(bridge_reset_action="reset_only", message="/new", thread_resume_id="")


def test_standalone_clean_reset_does_not_invent_a_provider_turn(harness):
    standalone(harness)
    events = send(harness)
    harness.bridge._ensure_codex.assert_not_called()
    assert not harness.prompts and not harness.writes
    assert len(events) == 1 and events[0].kind == AgentEventKind.ASSISTANT_DONE
    assert events[0].raw == {"new_session_ack": {
        "thread_session_id": "successor-thread", "provider_session_id": None,
        "clean_start": True, **seed_receipt([]),
    }}
    assert events[0].trigger_message_id == "task-trigger" and events[0].provider == "codex"
    harness.publisher.publish_unified_event.assert_not_called()


@pytest.mark.parametrize("skills,expected", [
    (None, "Session reset. Ready for a new conversation."),
    ({"refs": ["llm-bawt", "git-commits"], "tokens": 4200},
     "Session reset. Injected skills: llm-bawt, git-commits (~4.2k tokens). Ready for a new conversation."),
])
def test_standalone_clean_reset_ack_lists_injected_skills(harness, skills, expected):
    """TASK-1750: the Codex ack carries the app's skills facts, and nothing without them."""
    standalone(harness)
    field = reset_skills_field(skills)
    if field is not None:
        harness.fields[RESET_SKILLS_FIELD] = field
    events = send(harness)
    assert len(events) == 1 and events[0].text == expected


def test_standalone_clean_reset_ack_ignores_a_malformed_skills_field(harness):
    standalone(harness)
    harness.fields[RESET_SKILLS_FIELD] = "{not json"
    events = send(harness)
    assert len(events) == 1 and events[0].text == "Session reset. Ready for a new conversation."


@pytest.mark.parametrize("field,value", [
    ("inject_messages", None), ("inject_messages", "null"),
    # Unframed: refused before any provider thread is opened.
    ("inject_messages", '[{"role":"user","content":"retained"}]'),
    ("thread_resume_id", "stale-provider"), ("thread_resume_id", "bot:routing-key"),
    ("explicit_thread", "true"), ("thread_session_id", ""),
])
def test_standalone_reset_rejects_unproven_clean_state(harness, field, value):
    standalone(harness)
    harness.fields[field] = value
    send(harness)
    assert_no_ack(harness)
    harness.bridge._ensure_codex.assert_not_called()


def test_standalone_ack_waits_for_existing_session_work(harness):
    standalone(harness)

    async def run():
        ready, release = asyncio.Event(), asyncio.Event()

        async def occupy():
            async with harness.bridge._session_queue.active("review-bot:nick", request_id="existing"):
                ready.set()
                await release.wait()

        existing = asyncio.create_task(occupy())
        await ready.wait()
        reset = asyncio.create_task(harness.bridge._handle_send(harness.fields, "redis-message", harness.redis))
        await asyncio.sleep(0)
        assert not reset.done() and not harness.events
        release.set()
        await asyncio.gather(existing, reset)

    asyncio.run(run())
    assert len(harness.events) == 1 and harness.events[0].raw["new_session_ack"]["clean_start"]
    harness.bridge._ensure_codex.assert_not_called()
    harness.redis.xack.assert_awaited_once()
    assert not harness.bridge._trigger_message_ids


@pytest.mark.integration
@pytest.mark.parametrize("stale_key", [False, True])
def test_standalone_codex_receipt_requires_empty_successor_binding(harness, rig, stale_key):
    from sqlalchemy import text

    from llm_bawt.service.session_lifecycle.receipt import NewSessionReceipt, RESET_CONFIRMATION
    from tests.test_new_session_typed_pg import _admit, _dispatcher, _finish_bridge

    from tests.test_new_session_api_review_pg import bind_seed

    deliveries, record = _admit(rig)
    claim, payload, new_id, _synthetic_event = _finish_bridge(rig, deliveries, record)
    bind_seed(deliveries, claim.id, [])
    standalone(harness)
    harness.fields.update(
        request_id=payload["inter_bot_bridge_request_id"], bot_id=rig.bot,
        session_key=rig.bot, thread_session_id=new_id,
        trigger_message_id=claim.user_message_id,
    )
    if stale_key:
        with rig.engine.begin() as conn:
            conn.execute(text("""
                UPDATE sessions SET session_metadata=jsonb_set(session_metadata,
                    '{agent_session_keys}', '{"codex":"stale-provider"}'::jsonb)
                WHERE id=:id
            """), {"id": new_id})
    event = send(harness)[0]
    assert NewSessionReceipt.record_provider_ack(
        rig.engine, request_id=event.run_id, bot_id=rig.bot, event=event,
    )
    delivered = deliveries.mark_delivered(claim.id, claim.claim_token,
                                          response_model="fixture", response_chars=12)
    dispatcher = _dispatcher(deliveries)
    asyncio.run(dispatcher._finish_delivered(delivered))
    settled = deliveries.get(record.id)
    if stale_key:
        assert settled.status == "FAILED" and settled.metadata["new_session_ambiguous"]
        assert "new_session_receipt" not in settled.metadata
    else:
        assert settled.status == "DELIVERED"
        assert settled.metadata["new_session_receipt"]["new_session_id"] == new_id
        with rig.engine.connect() as conn:
            assert conn.execute(text("""
                SELECT content FROM messages WHERE bot_id=:bot AND id=:id
            """), {"bot": rig.bot, "id": payload["assistant_message_id"]}).scalar_one() == RESET_CONFIRMATION
        asyncio.run(dispatcher._finish_delivered(settled))
        with rig.engine.connect() as conn:
            assert conn.execute(text("""
                SELECT count(*) FROM new_session_event_outbox WHERE delivery_id=:id
            """), {"id": record.id}).scalar_one() == 1
    harness.bridge._ensure_codex.assert_not_called()
    assert deliveries.claim_next(rig.bot, claim_owner="replay", steer_capable=False) is None


@pytest.mark.parametrize("terminal", ["turn.failed", "error"])
def test_error_terminal_cannot_be_followed_by_reset_ack(harness, terminal):
    harness.script.insert(1, SimpleNamespace(type=terminal, message="failed", error={"message": "failed"}))
    send(harness)
    assert_no_ack(harness)


def test_repeated_completion_emits_one_ack(harness):
    harness.script.append(SimpleNamespace(type="turn.completed", usage=None))
    send(harness)
    assert sum(bool(event.raw.get("new_session_ack")) for event in harness.events) == 1


def test_second_thread_started_cannot_change_provider_identity(harness):
    harness.script.insert(1, SimpleNamespace(type="thread.started", thread_id="different-provider"))
    send(harness)
    assert_no_ack(harness)
    assert len(harness.writes) == 1


@pytest.mark.parametrize("message,is_reset,expected", [
    ("/newspaper", False, "/newspaper"),
    (" \t/newspaper keep these bytes  ", False, " \t/newspaper keep these bytes  "),
    ("/new/task", False, "/new/task"),
    ("/new​task", False, "/new​task"),
    ("/NEW task", False, "/NEW task"),
    ("mention /new here", False, "mention /new here"),
    ("/new task", True, "task"),
    (" \t/new\ttask\nsecond line  ", True, "task\nsecond line"),
    ("/new\r\ntask", True, "task"),
    ("/new task", True, "task"),
])
@pytest.mark.parametrize("explicit", [False, True])
def test_codex_legacy_new_requires_exact_command_token(harness, message, is_reset, expected, explicit):
    harness.fields.pop("bridge_reset_action")
    harness.fields.update(message=message, explicit_thread="1" if explicit else "0")
    send(harness)
    reset = is_reset and not explicit
    assert harness.prompts == [expected if reset else message]
    assert harness.sdk.start_thread.call_count == int(reset)
    assert harness.sdk.resume_thread.call_count == int(not reset)
    assert harness.publisher.publish_unified_event.call_count == int(reset)


def test_ordinary_resumed_turn_keeps_existing_behavior(harness):
    harness.fields.pop("bridge_reset_action")
    harness.fields["message"] = "ordinary task"
    send(harness)
    harness.sdk.start_thread.assert_not_called()
    harness.sdk.resume_thread.assert_called_once_with("previous-provider", {
        "model": "codex-fixture", "working_directory": "/tmp", "approval_policy": "never",
        "sandbox_mode": "danger-full-access", "skip_git_repo_check": True,
    })
    assert harness.prompts == ["ordinary task"] and not harness.writes
    assert harness.events[-1].kind == AgentEventKind.ASSISTANT_DONE
    assert harness.events[-1].text == ANSWER and harness.events[-1].raw == {}


@pytest.mark.integration
@pytest.mark.parametrize("binding", ["exact", "wrong-id", "wrong-backend"])
def test_native_codex_ack_requires_exact_persisted_policy_binding(harness, rig, real_lifecycle, binding):
    from sqlalchemy import text

    from llm_bawt.service.session_lifecycle.receipt import NewSessionReceipt, RESET_CONFIRMATION
    from tests.test_new_session_policy_review_pg import (
        TASK_REPLY, TASK_TEXT, claim_policy, finish_follow_up, settle,
    )

    from tests.test_new_session_api_review_pg import bind_seed

    claim = claim_policy(rig, real_lifecycle, "reset_without_history")
    bind_seed(real_lifecycle.store, claim.id, [])
    payload, _synthetic_event = finish_follow_up(
        rig, real_lifecycle, claim,
        backend="claude-code" if binding == "wrong-backend" else "codex",
        provider_id="wrong-provider" if binding == "wrong-id" else "fresh-provider",
    )
    harness.fields.update(
        request_id=payload["inter_bot_bridge_request_id"], bot_id=rig.bot,
        session_key=rig.bot, thread_session_id=claim.new_session_id,
        trigger_message_id=claim.user_message_id, message=TASK_TEXT,
    )
    harness.script[1].item.text = TASK_REPLY
    events = send(harness)
    done = next(event for event in events if event.kind == AgentEventKind.ASSISTANT_DONE)
    assert harness.prompts == [TASK_TEXT] and done.text == TASK_REPLY
    assert harness.writes == [(
        f"http://test-app/v1/sessions/{claim.new_session_id}/agent-session-key?bot_id={rig.bot}",
        {"backend": "codex", "session_key": "fresh-provider", "model": "codex-fixture"},
    )]
    assert NewSessionReceipt.record_provider_ack(
        rig.engine, request_id=done.run_id, bot_id=rig.bot, event=done,
    )
    settled, dispatcher = settle(real_lifecycle, claim)
    if binding != "exact":
        assert settled.status == "FAILED" and settled.metadata["new_session_ambiguous"]
        assert "new_session_receipt" not in settled.metadata
    else:
        assert settled.status == "DELIVERED" and settled.reset_status == "COMPLETED"
        receipt = settled.metadata["new_session_receipt"]
        assert receipt["provider_session_id"] == "fresh-provider"
        assert receipt["message_id"] != payload["assistant_message_id"]
        asyncio.run(dispatcher._finish_delivered(settled))
        with rig.engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT content FROM messages WHERE bot_id=:bot AND session_id=:session
            """), {"bot": rig.bot, "session": claim.new_session_id}).scalars().all()
            assert sorted(rows) == sorted([TASK_TEXT, TASK_REPLY, RESET_CONFIRMATION])
            assert conn.execute(text("""
                SELECT count(*) FROM new_session_event_outbox WHERE delivery_id=:id
            """), {"id": claim.id}).scalar_one() == 1
    assert real_lifecycle.store.claim_next(rig.bot, claim_owner="retry", steer_capable=False) is None


def test_ordinary_fresh_turn_keeps_best_effort_binding(harness):
    harness.fields.pop("bridge_reset_action")
    harness.fields.pop("thread_resume_id")
    harness.fields["message"] = "ordinary task"
    harness.put_status = 503
    send(harness)
    assert harness.events[-1].kind == AgentEventKind.ASSISTANT_DONE
    assert harness.events[-1].raw == {} and len(harness.writes) == 1
