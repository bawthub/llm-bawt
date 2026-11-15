"""Public HTTP contracts use real exact-turn observers, without model traffic."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_bawt.service.routes import chat


@pytest.mark.parametrize("path", ["delivery", "direct", "voice"])
@pytest.mark.parametrize("mode", ["comment", "data", "none"])
@pytest.mark.parametrize("animation", ["wave", None])
def test_exact_persisted_animation_reaches_negotiated_completion(monkeypatch, path, mode, animation):
    receipt = {
        "object": "chat.completion.in_flight" if path == "direct" else "chat.completion.queued",
        "delivery_id": "delivery-1", "turn_id": "turn-1", "user_message_id": "message-1",
        "bot_id": "test-bot", "user_id": "test-user", "busy": False,
    }
    turn = SimpleNamespace(
        bot_id="test-bot", user_id="test-user", trigger_message_id="message-1",
        ended_at=1, status="ok", error_text=None, response_text="own reply", animation=animation,
    )
    record = SimpleNamespace(
        id="delivery-1", status="DELIVERED", turn_id="turn-1", target_bot_id="test-bot",
        author_entity_type="user", author_entity_id="test-user", user_message_id="message-1",
        delivery_mode="queued", metadata={"ordinary_chat": {"voice_input": path == "voice"}},
    )
    turns = Mock()
    turns.get_turn.side_effect = lambda turn_id: turn if turn_id == "turn-1" else None
    deliveries = Mock()
    deliveries.get.side_effect = lambda delivery_id: record if delivery_id == "delivery-1" else None
    admissions = []

    async def stream(request):
        admissions.append(request)
        yield f"data: {json.dumps(receipt)}\n\n"
        raise AssertionError("must discard internal receipt trailer without replay")

    service = SimpleNamespace(
        chat_completion_stream=stream, _turn_log_store=turns,
        _inter_bot_dispatcher=SimpleNamespace(store=deliveries),
    )
    monkeypatch.setattr(chat, "get_service", lambda: service)
    app = FastAPI()
    app.include_router(chat.router)
    headers = {
        "comment": {"X-LLM-Bawt-Extension-Events": "comment"},
        "data": {"X-LLM-Bawt-Admission-Events": "comment"},
        "none": {},
    }[mode]
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", headers=headers, json={
            "model": "test-model", "bot_id": "test-bot", "user": "test-user",
            "stream": True, "voice_input": path == "voice",
            "messages": [{"role": "user", "content": "synthetic fixture only"}],
        })
    assert response.status_code == 200, response.text
    frames = response.text.split("\n\n")
    data = [json.loads(frame[6:]) for frame in frames
            if frame.startswith("data: ") and frame != "data: [DONE]"]
    chunks = [item for item in data if item.get("object") == "chat.completion.chunk"]
    assert chunks[0]["choices"][0]["delta"]["content"] == "own reply"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    expected = {"object": "service.animation", "animation": "wave"}
    extension = (": llm-bawt-event " if mode == "comment" else "data: ") + json.dumps(expected)
    if animation and mode != "none":
        assert frames.count(extension) == 1
        assert frames.index(extension) < next(
            index for index, frame in enumerate(frames) if '"finish_reason": "stop"' in frame
        )
    else:
        assert "service.animation" not in response.text
    assert response.text.endswith("data: [DONE]\n\n")
    assert response.text.count("data: [DONE]") == 1
    assert len(admissions) == 1
    turns.get_turn.assert_called_once_with("turn-1")
    if path != "direct":
        deliveries.get.assert_called_once_with("delivery-1")
