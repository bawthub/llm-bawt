"""Reset receipt survives fresh HTTP history reads, without a live deployment."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_bawt.mcp_server.storage import MemoryStorage
from llm_bawt.memory.postgresql import MESSAGES_PARENT, partition_name
from llm_bawt.service.routes import history_pages
from llm_bawt.service.session_lifecycle.receipt import RESET_CONFIRMATION
from tests.test_new_session_receipt_review_pg import acknowledge, finish, pending_receipt

__all__ = ["pending_receipt"]
pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


def test_scheduled_command_and_receipt_survive_fresh_http_history_reads(pending_receipt, monkeypatch):
    state = pending_receipt
    rig = state.rig
    assert acknowledge(state)
    receipt = finish(state)
    assert receipt is not None

    # Use the production SQL reader and author resolver, with a test-owned
    # backend handle; no embedding/model/network client is constructed.
    from llm_bawt.profiles import ProfileManager
    ProfileManager(rig.config)  # Normal startup creates typed System display identity.
    storage = MemoryStorage(rig.config)
    storage._backends[rig.bot] = SimpleNamespace(
        engine=rig.engine, _messages_table_name=partition_name(MESSAGES_PARENT, rig.bot),
    )
    client = SimpleNamespace(get_messages=lambda **kwargs: storage._get_messages_raw(rig.bot, **kwargs))
    service = SimpleNamespace(
        config=rig.config, _default_bot=rig.bot,
        get_memory_client=lambda bot: client,
        _turn_log_store=SimpleNamespace(engine=rig.engine),
    )
    monkeypatch.setattr(history_pages, "get_service", lambda: service)
    for name in ("_hydrate_attachments_for_page", "_hydrate_reasoning_for_page",
                 "_hydrate_interrupt_anchors_for_page"):
        monkeypatch.setattr(history_pages, name, lambda *args: {})
    app = FastAPI()
    app.include_router(history_pages.read_router)
    pages = []
    for _ in range(2):
        # New client, no live event/optimistic bubble state from execution.
        with TestClient(app) as http:
            response = http.get("/v1/history", params={
                "bot_id": rig.bot, "user_id": rig.user, "session_id": state.successor,
            })
            assert response.status_code == 200, response.text
            pages.append(response.json()["messages"])
    assert pages[0] == pages[1]
    assert len(pages[0]) == 2
    command, confirmation = pages[0]
    assert command["id"] == state.claim.user_message_id
    assert command["role"] == "user" and command["content"] == "/new"
    assert command["author"]["entity_type"] == "bot"
    assert command["author"]["entity_id"] == "system"
    assert command["author"]["display_name"] == "System"
    assert confirmation["id"] == receipt.receipt_message_id
    assert confirmation["role"] == "assistant" and confirmation["content"] == RESET_CONFIRMATION
    assert confirmation["reply_to_message_id"] == state.claim.user_message_id
    assert confirmation["author"]["entity_type"] == "bot"
    assert confirmation["author"]["entity_id"] == rig.bot
