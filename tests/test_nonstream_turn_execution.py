"""Non-stream agent turns own a live execution handle (TASK-1015).

Before TASK-1015 ``chat_completion`` (the waited ``bots_send_message`` path)
never registered a ``TurnExecution`` and never minted the task-turn
capability, so the bridge never persisted ``agent_session_key`` /
``agent_request_id``: steer answered 409 "Active bridge run is not ready",
Stop answered 503 ``execution_not_confirmed``, the reaper could not tell the
turn was owned, and ``tasks_associate_current`` saw no trusted context.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from llm_bawt.service.background_service import BackgroundService
from llm_bawt.service.schemas import ChatCompletionRequest, ChatMessage
from llm_bawt.service.turn_execution import turn_executions


def _service(execute_llm_query, *, model_type: str = "claude-code") -> BackgroundService:
    service = BackgroundService.__new__(BackgroundService)
    service.config = SimpleNamespace(DEFAULT_BOT="nova", DEFAULT_USER="nick")
    service._redis_subscriber = SimpleNamespace(publish_tool_event=AsyncMock())
    service._resolve_request_model = Mock(return_value=("test-model", []))
    service._persist_turn_log = Mock()
    service._update_turn_log = Mock()
    service._end_generation = Mock()
    service._start_generation = AsyncMock(
        return_value=(threading.Event(), threading.Event())
    )
    service._bind_agent_thread = Mock(return_value=None)
    service._resolve_active_thread_binding = Mock(
        return_value={"thread_session_id": "sess-1"}
    )
    service._maybe_summarize_on_new = Mock()
    service._maybe_rotate_agent_session = Mock()
    service._finalize_turn = Mock()
    service._turn_log_store = SimpleNamespace(engine=object())
    service._get_llm_bawt = Mock(return_value=SimpleNamespace(
        client=SimpleNamespace(
            model_definition={"type": model_type},
            get_token_usage=Mock(return_value=None),
        ),
        bot=SimpleNamespace(tts_mode=False, agent_backend="claude-code"),
        prepare_messages_for_query=Mock(return_value=[]),
        execute_llm_query=execute_llm_query,
    ))
    return service


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role="user", content="do the task")],
        bot_id="caid",
        user="nick",
        stream=False,
        user_message_id="user-1",
    )


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr("llm_bawt.service.chat_nonstream.get_bot", lambda _b: None)
    monkeypatch.setattr(
        "llm_bawt.service.routes.history.maybe_build_session_seed",
        lambda *_a, **_k: None,
    )
    minted: list[dict] = []

    def mint(**kwargs):
        minted.append(kwargs)
        return "cap-token"

    monkeypatch.setattr("llm_bawt.task_turn_context.mint_for_agent_turn", mint)
    before = set(turn_executions.active_ids())
    yield minted
    for turn_id in set(turn_executions.active_ids()) - before:
        turn_executions.remove(turn_id)


def test_agent_nonstream_turn_registers_execution_and_capability(_isolate) -> None:
    seen: dict = {}

    def execute_llm_query(*_args, **kwargs):
        seen["kwargs"] = kwargs
        # Live while the bridge runs: Stop/steer/reaper can find the owner.
        execution = kwargs["turn_execution"]
        seen["registered"] = turn_executions.get(execution.turn_id)
        return "done", "", []

    service = _service(execute_llm_query)
    asyncio.run(service.chat_completion(_request()))

    execution = seen["kwargs"]["turn_execution"]
    assert execution is not None and seen["registered"] is execution
    assert execution.turn_id.startswith("turn-")
    assert seen["kwargs"]["task_turn_capability"] == "cap-token"
    assert _isolate[0]["turn_id"] == execution.turn_id
    assert _isolate[0]["session_id"] == "sess-1"
    # Removed once the turn ends.
    assert turn_executions.get(execution.turn_id) is None


def test_execution_is_released_when_the_query_raises(_isolate) -> None:
    seen: dict = {}

    def execute_llm_query(*_args, **kwargs):
        seen["turn_id"] = kwargs["turn_execution"].turn_id
        seen["live"] = turn_executions.get(seen["turn_id"]) is not None
        raise RuntimeError("bridge exploded")

    service = _service(execute_llm_query)
    try:
        asyncio.run(service.chat_completion(_request()))
    except Exception:
        pass
    assert seen["live"] is True
    assert turn_executions.get(seen["turn_id"]) is None


def test_chat_backend_nonstream_turn_does_not_register(_isolate) -> None:
    seen: dict = {}

    def execute_llm_query(*_args, **kwargs):
        seen["kwargs"] = kwargs
        return "done", "", []

    service = _service(execute_llm_query, model_type="openai")
    asyncio.run(service.chat_completion(_request()))
    assert seen["kwargs"]["turn_execution"] is None
    assert seen["kwargs"]["task_turn_capability"] is None
    assert _isolate == []


def test_agent_client_threads_request_local_identity_into_backend_config() -> None:
    from llm_bawt.clients.agent_backend_client import AgentBackendClient

    captured: dict = {}

    class Backend:
        async def chat_full(self, prompt, config, **_kwargs):
            captured.update(config)
            return SimpleNamespace(text="ok", tool_calls=[], model="m", usage={})

    client = AgentBackendClient.__new__(AgentBackendClient)
    client._bot_config = {"bot_id": "caid"}
    client._backend = Backend()
    marker = object()
    asyncio.run(client._chat_full(
        "hi", request_local={"turn_execution": marker, "task_turn_capability": "cap"},
    ))
    assert captured["turn_execution"] is marker
    assert captured["task_turn_capability"] == "cap"
    assert "turn_execution" not in client._bot_config  # shared config untouched
