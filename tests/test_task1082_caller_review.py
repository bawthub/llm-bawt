"""Independent TASK-1082 caller-contract regressions; no live DB or agents."""
from __future__ import annotations

import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from rich.console import Console

ACK = {
    "object": "chat.completion.queued", "status": "QUEUED",
    "delivery_id": "delivery-review", "turn_id": "turn-delivery-review",
    "user_message_id": "00000000-0000-4000-8000-000000000001",
    "assistant_message_id": "00000000-0000-4000-8000-000000000002",
}


@pytest.mark.parametrize("stream", [True, False])
def test_cli_never_reports_accepted_ordinary_turn_as_service_unavailable(monkeypatch, stream):
    from llm_bawt.cli import display_cmd
    from llm_bawt.service.client import ServiceClient

    client = ServiceClient(http_url="http://review.invalid")
    monkeypatch.setattr(client, "is_available", lambda **_kwargs: True)

    def request(_method, path, **_kwargs):
        if path.endswith("/reply"):
            return {**ACK, "status": "DELIVERED", "response_text": "Verified reply"}
        return dict(ACK)

    monkeypatch.setattr(client, "_request", request)
    wire = f"data: {json.dumps(ACK)}\n\ndata: [DONE]\n\n".encode()
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: io.BytesIO(wire))
    config = SimpleNamespace(VERBOSE=False)
    bot = SimpleNamespace(name="Review bot", color="green")
    monkeypatch.setattr(display_cmd, "Config", lambda: config)
    monkeypatch.setattr(display_cmd, "BotManager", lambda _config: SimpleNamespace(get_bot=lambda _id: bot))
    monkeypatch.setattr(display_cmd, "get_service_client", lambda _config: client)
    output = io.StringIO()
    monkeypatch.setattr(display_cmd, "console", Console(file=output))

    handled = display_cmd.query_via_service(
        "hello", model="test-model", bot_id="review-bot", user_id="review-user",
        plaintext_output=True, stream=stream, config=config,
    )
    assert handled is True, "Already accepted work must not become a failed/unavailable service call"


def test_botchat_does_not_label_queued_acceptance_as_completed_http_200(monkeypatch):
    from llm_bawt.service.routes import botchat

    service = SimpleNamespace(chat_completion=AsyncMock(return_value=dict(ACK)))
    monkeypatch.setattr(botchat, "get_service", lambda: service)
    app = FastAPI()
    app.include_router(botchat.router)
    with TestClient(app) as client:
        result = client.post("/v1/botchat/review-bot/review-user/chat/completions", json={
            "model": "test-model", "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
        })
    # Either wait and return an actual completion, or match the main endpoint's
    # explicit 202 acceptance contract. A choices-less HTTP 200 is neither.
    assert result.status_code == 202 or (
        result.status_code == 200 and result.json().get("choices")
    ), "Botchat returned ordinary queued acceptance as a completed HTTP 200"
