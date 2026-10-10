"""Hermetic AskUserQuestion live-window arbitration tests (TASK-1043)."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, create_engine

from agent_bridge.events import AgentEventKind
from claude_code_bridge.approval_ops import ClaudeApprovalMixin
from claude_code_bridge.command_ops import ClaudeCommandMixin
from llm_bawt.service.chat_pending_questions import PendingQuestion, PendingQuestionStore
from llm_bawt.service.routes import chat


@pytest.fixture
def store():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine, tables=[PendingQuestion.__table__])
    obj = PendingQuestionStore.__new__(PendingQuestionStore)
    obj.engine = engine
    return obj


def add(store, *, live=True, question_id="t1"):
    deadline = datetime.now(timezone.utc) + timedelta(seconds=30) if live else None
    store.upsert_awaiting(
        tool_use_id=question_id, bot_id="loopy", user_id="nick", turn_id="turn",
        session_key="loopy:nick", arguments={"questions": []},
        origin_harness="claude-code", live_deadline=deadline,
    )
    return deadline


class FakeBridge(ClaudeApprovalMixin, ClaudeCommandMixin):
    _DEFERRED_ACK = "deferred"
    _backend_name = "claude-code"
    _app_api_url = "http://app"

    def __init__(self, store):
        self.store = store
        self._pending_question_futures = {}
        self.events = []

    def _publish_event(self, request_id, session_key, seq, **kwargs):
        self.events.append(kwargs)
        deadline = kwargs.get("extra_raw", {}).get("live_deadline") if kwargs.get("extra_raw") else None
        self.store.upsert_awaiting(
            tool_use_id=kwargs["tool_use_id"], bot_id="loopy", user_id="nick",
            turn_id="turn", session_key=session_key, arguments=kwargs["tool_arguments"],
            origin_harness="claude-code",
            live_deadline=datetime.fromisoformat(deadline) if deadline else None,
        )


class FakeRedis:
    def __init__(self):
        self.acks = []

    async def xack(self, *args):
        self.acks.append(args)

    async def xadd(self, *args, **kwargs):
        pass


@pytest.fixture
def app(monkeypatch, store):
    redis = FakeRedis()
    subscriber = SimpleNamespace(_redis=redis)

    async def send_tool_result(session_key, question_id, result, **kwargs):
        subscriber.sent.append((session_key, question_id, result))
        if subscriber.bridge:
            fields = {"session_key": session_key, "tool_use_id": question_id,
                      "result": result, "backend": "claude-code"}
            if kwargs.get("answers"):
                fields["answers_json"] = json.dumps(kwargs["answers"])
            await subscriber.bridge._handle_tool_result(fields, "cmd", redis)

    subscriber.sent = []
    subscriber.bridge = None
    subscriber.send_tool_result = send_tool_result
    monkeypatch.setattr(chat, "get_service", lambda: SimpleNamespace(_pending_question_store=store))
    monkeypatch.setattr(
        "llm_bawt.agent_backends.agent_bridge.get_agent_subscriber",
        lambda: subscriber,
    )
    return subscriber


def test_setting_resolves_each_turn_and_zero_survives_payload(monkeypatch):
    from llm_bawt.clients.agent_backend_client import AgentBackendClient
    from llm_bawt.models.message import Message
    from llm_bawt.setting_definitions import SETTING_DEFINITIONS
    from claude_code_bridge.send_request import SendRequest
    from agent_bridge.command_publisher import CommandPublisherMixin

    definition = SETTING_DEFINITIONS["question_answer_wait_seconds"]
    assert definition.default == 60 and definition.applies_to == ()
    values = iter((0, 13, 4, 9))
    monkeypatch.setattr(
        "llm_bawt.runtime_setting_resolution.resolve_global_runtime_setting",
        lambda config, key: next(values),
    )
    configs = []

    class Backend:
        def stream_raw(self, prompt, config, **kwargs):
            configs.append(dict(config))
            yield "ok"

        async def chat_full(self, prompt, config):
            configs.append(dict(config))
            return SimpleNamespace(text="ok")

    client = AgentBackendClient.__new__(AgentBackendClient)
    client._backend_name = "claude-code"
    client._bot_config = {"bot_id": "loopy"}
    client._backend = Backend()
    client.config = SimpleNamespace()
    for _ in range(2):
        assert list(client.stream_raw([Message(role="user", content="hello")])) == ["ok"]
    assert [c["question_answer_wait_seconds"] for c in configs] == [0, 13]
    for _ in range(2):
        assert asyncio.run(client._chat_full("hello")).text == "ok"
    assert [c["question_answer_wait_seconds"] for c in configs] == [0, 13, 4, 9]
    assert "question_answer_wait_seconds" not in client._bot_config

    class Redis:
        async def xadd(self, stream, fields, **kwargs):
            assert fields["question_answer_wait_seconds"] == "0"
            request = SendRequest.from_fields(fields)
            assert request.question_answer_wait_seconds == 0
            return "id"

    publisher = CommandPublisherMixin()
    publisher._pub_redis = Redis()
    asyncio.run(publisher.send_command("loopy:nick", "hello", "req", question_answer_wait_seconds=0))


def test_wait_zero_is_original_deferral(store):
    bridge = FakeBridge(store)
    cb = bridge._make_can_use_tool(request_id="r", session_key="loopy:nick", seq_holder=[0], question_answer_wait_seconds=0)
    result = asyncio.run(cb("AskUserQuestion", {"questions": []}, SimpleNamespace(tool_use_id="t1")))
    assert result.message == "deferred"
    assert bridge.events[0]["kind"] == AgentEventKind.AWAIT_TOOL_RESULT
    assert bridge.events[0]["extra_raw"] is None
    assert store.get("t1").status == "awaiting"
    assert not bridge._pending_question_futures


SHIP_INPUT = {"questions": [
    {"question": "Ship it?", "header": "Ship", "multiSelect": False,
     "options": [{"label": "Yes", "description": ""}, {"label": "No", "description": ""}]},
    {"question": "Which envs?", "header": "Envs", "multiSelect": True,
     "options": [{"label": "dev", "description": ""}, {"label": "prod", "description": ""}]},
]}


def test_live_answer_allows_tool_with_native_answers(store, app):
    """A real answer is a normal tool result (allow + answers), never an error."""
    from claude_agent_sdk.types import PermissionResultAllow
    bridge = app.bridge = FakeBridge(store)

    async def scenario():
        cb = bridge._make_can_use_tool(request_id="r", session_key="loopy:nick", seq_holder=[0], question_answer_wait_seconds=5)
        pending = asyncio.create_task(cb("AskUserQuestion", SHIP_INPUT, SimpleNamespace(tool_use_id="t1")))
        await asyncio.sleep(0)
        response = await chat.answer_question("t1", chat.QuestionAnswerRequest(
            bot_id="loopy",
            responses=[
                chat.QuestionResponseItem(question_id="Ship", selected=["Yes"]),
                chat.QuestionResponseItem(question_id="Envs", selected=["dev", "prod"], other="staging"),
            ],
        ))
        assert response.live is True
        result = await pending
        assert isinstance(result, PermissionResultAllow)
        assert result.updated_input["questions"] == SHIP_INPUT["questions"]
        assert result.updated_input["answers"] == {
            "Ship it?": "Yes", "Which envs?": "dev, prod, staging",
        }
        assert not bridge._pending_question_futures
    asyncio.run(scenario())


def test_tool_answers_mapping_fallbacks(store):
    store.upsert_awaiting(
        tool_use_id="m1", bot_id="loopy", user_id="nick", turn_id="turn",
        session_key="loopy:nick", arguments=SHIP_INPUT, origin_harness="claude-code",
    )
    # Positional fallback when question_id matches neither header nor text.
    assert store.claim_deferred_answer("m1", "x", [
        {"question_id": "?", "selected": ["No"]}, {"question_id": "?", "selected": []},
    ])
    assert store.tool_answers(store.get("m1")) == {"Ship it?": "No"}
    # Pre-formatted answer with no structure maps onto the first question.
    store.upsert_awaiting(
        tool_use_id="m2", bot_id="loopy", user_id="nick", turn_id="turn",
        session_key="loopy:nick", arguments=SHIP_INPUT, origin_harness="claude-code",
    )
    assert store.claim_deferred_answer("m2", "just ship dev", None)
    assert store.tool_answers(store.get("m2")) == {"Ship it?": "just ship dev"}


def test_live_answer_is_same_tool_result(store, app):
    bridge = app.bridge = FakeBridge(store)

    async def scenario():
        cb = bridge._make_can_use_tool(request_id="r", session_key="loopy:nick", seq_holder=[0], question_answer_wait_seconds=5)
        pending = asyncio.create_task(cb("AskUserQuestion", {"questions": []}, SimpleNamespace(tool_use_id="t1")))
        await asyncio.sleep(0)
        assert store.get("t1").status == "awaiting_live"
        assert bridge.events[0]["extra_raw"]["live_deadline"]
        response = await chat.answer_question("t1", chat.QuestionAnswerRequest(bot_id="loopy", result="Yes"))
        assert response.live is True
        assert response.continuation_prompt == ""
        assert (await pending).message == "Yes"
        assert app.sent == [("loopy:nick", "t1", "Yes")]
        assert store.get("t1").status == "answered"
        duplicate = await chat.answer_question("t1", chat.QuestionAnswerRequest(bot_id="loopy", result="Yes"))
        assert duplicate.already_answered and not duplicate.continuation_prompt
        assert len(app.sent) == 1
        assert not bridge._pending_question_futures

    asyncio.run(scenario())


def test_timeout_claim_wins_answer_race(store, app, monkeypatch):
    bridge = FakeBridge(store)

    class Response:
        def raise_for_status(self): pass
        def json(self): return {"expired": store.expire_live("t1"), "answer": None}

    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, *args, **kwargs): return Response()

    monkeypatch.setattr("claude_code_bridge.approval_ops.httpx.AsyncClient", Client)

    async def scenario():
        cb = bridge._make_can_use_tool(request_id="r", session_key="loopy:nick", seq_holder=[0], question_answer_wait_seconds=0.01)
        result = await cb("AskUserQuestion", {"questions": []}, SimpleNamespace(tool_use_id="t1"))
        assert result.message == "deferred"
        assert store.get("t1").status == "awaiting"
        response = await chat.answer_question("t1", chat.QuestionAnswerRequest(bot_id="loopy", result="Too late"))
        assert response.live is False
        assert response.continuation_prompt == "Too late"
        duplicate = await chat.answer_question("t1", chat.QuestionAnswerRequest(bot_id="loopy", result="Too late"))
        assert duplicate.already_answered and not duplicate.continuation_prompt
        assert not app.sent

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["post", "http_status", "bad_json"])
def test_timeout_arbitration_failure_defers_without_raising(store, monkeypatch, failure):
    bridge = FakeBridge(store)

    class Response:
        def raise_for_status(self):
            if failure == "http_status":
                raise RuntimeError("missing question row (404)")

        def json(self):
            if failure == "bad_json":
                raise ValueError("invalid response")
            return {"expired": True}

    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, *args, **kwargs):
            if failure == "post":
                raise ConnectionError("app restarting")
            return Response()

    monkeypatch.setattr("claude_code_bridge.approval_ops.httpx.AsyncClient", Client)

    async def scenario():
        cb = bridge._make_can_use_tool(request_id="r", session_key="loopy:nick", seq_holder=[0], question_answer_wait_seconds=0.01)
        result = await cb("AskUserQuestion", {"questions": []}, SimpleNamespace(tool_use_id="t1"))
        assert result.message == "deferred"
        assert not bridge._pending_question_futures

    asyncio.run(scenario())


def test_timeout_failure_preserves_answer_already_in_future(store, monkeypatch):
    bridge = FakeBridge(store)

    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, *args, **kwargs):
            bridge._pending_question_futures["t1"][1].set_result("answer already delivered")
            raise ConnectionError("app restarting")

    monkeypatch.setattr("claude_code_bridge.approval_ops.httpx.AsyncClient", Client)

    async def scenario():
        cb = bridge._make_can_use_tool(request_id="r", session_key="loopy:nick", seq_holder=[0], question_answer_wait_seconds=0.01)
        result = await cb("AskUserQuestion", {"questions": []}, SimpleNamespace(tool_use_id="t1"))
        assert result.message == "answer already delivered"
        assert not bridge._pending_question_futures

    asyncio.run(scenario())


def test_unexpected_timeout_status_does_not_hang(store, monkeypatch):
    bridge = FakeBridge(store)

    class Response:
        def raise_for_status(self): pass
        def json(self): return {"expired": False, "answer": None, "dismissed": False}

    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, *args, **kwargs): return Response()

    monkeypatch.setattr("claude_code_bridge.approval_ops.httpx.AsyncClient", Client)

    async def scenario():
        cb = bridge._make_can_use_tool(request_id="r", session_key="loopy:nick", seq_holder=[0], question_answer_wait_seconds=0.01)
        result = await asyncio.wait_for(
            cb("AskUserQuestion", {"questions": []}, SimpleNamespace(tool_use_id="t1")),
            timeout=1,
        )
        assert result.message == "deferred"
        assert not bridge._pending_question_futures

    asyncio.run(scenario())


def test_answer_claim_wins_bridge_timeout_without_redis_delivery(store, monkeypatch):
    bridge = FakeBridge(store)

    class Response:
        def raise_for_status(self): pass
        def json(self):
            assert not store.expire_live("t1")
            return {"expired": False, "answers": store.tool_answers(store.get("t1"))}

    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, *args, **kwargs): return Response()

    monkeypatch.setattr("claude_code_bridge.approval_ops.httpx.AsyncClient", Client)

    async def scenario():
        cb = bridge._make_can_use_tool(request_id="r", session_key="loopy:nick", seq_holder=[0], question_answer_wait_seconds=0.01)
        pending = asyncio.create_task(cb("AskUserQuestion", SHIP_INPUT, SimpleNamespace(tool_use_id="t1")))
        await asyncio.sleep(0)
        assert store.claim_live_answer("t1", "Ship: Yes", [{"question_id": "Ship", "selected": ["Yes"]}])
        assert (await pending).updated_input["answers"] == {"Ship it?": "Yes"}
        assert not bridge._pending_question_futures

    asyncio.run(scenario())


def test_delivery_mode_remains_distinguishable_at_turn_finalize(store):
    add(store, question_id="live")
    assert store.claim_live_answer("live", "yes", None)
    assert store.get("live").live_deadline is not None
    add(store, question_id="deferred")
    assert store.expire_live("deferred")
    assert store.claim_deferred_answer("deferred", "yes", None)
    assert store.get("deferred").live_deadline is None


def test_answer_claim_wins_timeout_race(store):
    add(store)
    assert store.claim_live_answer("t1", "winner", None)
    assert not store.expire_live("t1")
    assert store.get("t1").answer == "winner"


def test_timeout_endpoint_and_hydration_include_deadline(store, app):
    deadline = add(store)
    hydrated = asyncio.run(chat.list_pending_questions(bot_id="loopy", user_id="nick"))
    assert hydrated.data[0].status == "awaiting_live"
    assert hydrated.data[0].live_deadline == deadline.isoformat()
    first = asyncio.run(chat.timeout_question("t1", chat.QuestionTimeoutRequest(session_key="loopy:nick")))
    assert first["expired"] is True
    assert store.get("t1").status == "awaiting"
    second = asyncio.run(chat.timeout_question("t1", chat.QuestionTimeoutRequest(session_key="loopy:nick")))
    assert second["expired"] is True
    assert not second["answers"] and not second["message"]


def test_expired_deadline_cannot_claim_live(store, app):
    add(store)
    from sqlmodel import Session
    with Session(store.engine) as session:
        row = session.get(PendingQuestion, "t1")
        row.live_deadline = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.add(row)
        session.commit()
    assert not store.claim_live_answer("t1", "late", None)
    answer = asyncio.run(chat.answer_question("t1", chat.QuestionAnswerRequest(bot_id="loopy", result="late")))
    assert not answer.live
    assert answer.continuation_prompt == "late"
    assert not app.sent


def test_abort_cancels_future_and_preserves_deferred_answer(store, app):
    bridge = FakeBridge(store)

    async def scenario():
        cb = bridge._make_can_use_tool(request_id="r", session_key="loopy:nick", seq_holder=[0], question_answer_wait_seconds=5)
        pending = asyncio.create_task(cb("AskUserQuestion", {"questions": []}, SimpleNamespace(tool_use_id="t1")))
        await asyncio.sleep(0)
        assert store.expire_live_for_turn("turn") == 1
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert not bridge._pending_question_futures
        answer = await chat.answer_question("t1", chat.QuestionAnswerRequest(bot_id="loopy", result="Later"))
        assert not answer.live
        assert answer.continuation_prompt == "Later"

    asyncio.run(scenario())


def test_dismiss_delivery_failure_is_recovered_by_timeout(store, app):
    add(store)

    async def broken_delivery(*args, **kwargs):
        raise ConnectionError("Redis delivery failed")

    app.send_tool_result = broken_delivery
    response = asyncio.run(chat.answer_question(
        "t1", chat.QuestionAnswerRequest(bot_id="loopy", dismiss=True),
    ))
    assert response.live
    timeout = asyncio.run(chat.timeout_question(
        "t1", chat.QuestionTimeoutRequest(session_key="loopy:nick"),
    ))
    assert timeout["dismissed"]
    assert timeout["message"] == "[Question dismissed by user.]"
    assert timeout["answers"] is None


def test_live_dismiss_resolves_future(store, app):
    bridge = app.bridge = FakeBridge(store)

    async def scenario():
        cb = bridge._make_can_use_tool(request_id="r", session_key="loopy:nick", seq_holder=[0], question_answer_wait_seconds=5)
        pending = asyncio.create_task(cb("AskUserQuestion", {"questions": []}, SimpleNamespace(tool_use_id="t1")))
        await asyncio.sleep(0)
        response = await chat.answer_question("t1", chat.QuestionAnswerRequest(bot_id="loopy", dismiss=True))
        assert response.live
        assert (await pending).message == "[Question dismissed by user.]"
        assert store.get("t1").status == "skipped"
        assert not bridge._pending_question_futures

    asyncio.run(scenario())
