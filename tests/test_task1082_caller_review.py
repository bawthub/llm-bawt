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
