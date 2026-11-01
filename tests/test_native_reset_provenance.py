"""Native reset proofs require committed exact-owner backend provenance."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llm_bawt.bot_access import service as service_codec
from llm_bawt.memory.session_reset_lineage import SessionResetLineage
from llm_bawt.models.message import Message
from llm_bawt.service.schemas import ChatMessage
from llm_bawt.service.session_lifecycle.execution import NewSessionExecution, PreparedChatSession
from tests.test_nonstream_turn_execution import _claimed, _isolate, _request, _service

__all__ = ["_isolate"]


@pytest.mark.parametrize("owner", [("caid", "nick"), ("caid", "other-user"), ("other-bot", "nick"), (None, None)])
def test_only_matching_committed_rotation_can_mint(monkeypatch, owner):
    successor = SessionResetLineage.delivery_successor_id("delivery-test")
    backend = SimpleNamespace(bot_id=owner[0], user_id=owner[1],
                              get_session=lambda sid: None, get_active_session=lambda **kw: None)
    llm = SimpleNamespace(history_manager=SimpleNamespace(_db_backend=backend))
    rotated = []

    def rotate(*args, **kwargs):
        rotated.append(True)  # production backend returns only after transaction commits
        return successor

    service = SimpleNamespace(_rotate_chat_session=rotate)
    minted = []

    def mint(**kwargs):
        assert rotated == [True]
        minted.append(kwargs)
        return "signed-proof"

    monkeypatch.setattr("llm_bawt.task_turn_context.mint_task_turn_context", mint)
    prepared = NewSessionExecution.prepare_chat(service, llm, _claimed(_request()),
        bot_id="caid", user_id="nick", user_prompt="/new")
    capability = prepared.service_turn_capability(bot_id="caid", user_id="nick",
        turn_id="turn-fixture", trigger_message_id="message-fixture")
    assert capability == ("signed-proof" if owner == ("caid", "nick") else None)
    assert bool(minted) == (owner == ("caid", "nick"))
    if minted:
        assert minted[0]["session_id"] == successor
    # The same prepared reset cannot sign for another caller or altered session.
    minted.clear()
    assert prepared.service_turn_capability(bot_id="caid", user_id="other-user",
        turn_id="turn-fixture", trigger_message_id="message-fixture") is None
    assert not minted


def test_mcp_rotation_adapter_exposes_captured_not_mutable_owner():
    from llm_bawt.mcp_server.client_short_term import _MCPShortTermManager

    client = SimpleNamespace(bot_id="mutable-bot", user_id="mutable-user",
                             _service_principal=service_codec.ServicePrincipal("caid", "nick"))
    backend = _MCPShortTermManager(client)
    assert (backend.bot_id, backend.user_id) == ("caid", "nick")
    with pytest.raises(AttributeError):
        backend.user_id = "other-user"


@pytest.mark.parametrize("failure", ["raise", "wrong-successor"])
def test_failed_rotation_never_mints(monkeypatch, failure):
    mint = Mock(side_effect=AssertionError("minted before committed successor"))
    monkeypatch.setattr("llm_bawt.task_turn_context.mint_task_turn_context", mint)
    backend = SimpleNamespace(bot_id="caid", user_id="nick",
                              get_session=lambda sid: None, get_active_session=lambda **kw: None)
    llm = SimpleNamespace(history_manager=SimpleNamespace(_db_backend=backend))
    rotate = Mock(side_effect=RuntimeError("rollback")) if failure == "raise" else Mock(return_value="other-id")
    with pytest.raises(RuntimeError):
        NewSessionExecution.prepare_chat(SimpleNamespace(_rotate_chat_session=rotate), llm,
            _claimed(_request()), bot_id="caid", user_id="nick", user_prompt="/new")
    mint.assert_not_called()


def test_raw_or_unknown_owner_and_signing_failure_keep_fallback(monkeypatch):
    raw = PreparedChatSession("selected-session", None)
    assert raw.service_turn_capability(bot_id="caid", user_id="nick",
        turn_id="turn-fixture", trigger_message_id="message-fixture") is None
    trusted = PreparedChatSession("committed-session", None, ("committed-session", "caid", "nick"))
    monkeypatch.setattr("llm_bawt.task_turn_context.mint_task_turn_context", Mock(side_effect=ValueError("signing")))
    assert trusted.service_turn_capability(bot_id="caid", user_id="nick",
        turn_id="turn-fixture", trigger_message_id="message-fixture") is None


@pytest.mark.parametrize("failure", ["prepare", "query", "finalize"])
def test_native_nonstream_failure_restores_executor_context(monkeypatch, failure):
    service = _service(Mock(return_value=("answer", "", [])), model_type="openai")
    llm = service._get_llm_bawt.return_value
    llm.history_manager = SimpleNamespace(_db_backend=SimpleNamespace(
        bot_id="caid", user_id="nick", get_session=lambda sid: None,
        get_active_session=lambda **kw: None))
    service._rotate_chat_session = Mock(return_value=SessionResetLineage.delivery_successor_id("delivery-test"))
    llm.prepare_messages_for_query.return_value = [Message(role="user", content="/new task")]
    target = {"prepare": llm.prepare_messages_for_query,
              "query": llm.execute_llm_query, "finalize": service._finalize_turn}[failure]
    target.side_effect = RuntimeError("fixture failure")
    monkeypatch.setattr("llm_bawt.task_turn_context.mint_task_turn_context", lambda **kw: "proof")
    original = service_codec.bind_service_turn
    exits = []

    @contextmanager
    def track(capability):
        assert capability == "proof"
        try:
            with original(capability):
                assert service_codec._runtime_turn.get() == "proof"
                yield
        finally:
            exits.append(service_codec._runtime_turn.get())

    monkeypatch.setattr(service_codec, "bind_service_turn", track)
    request = _claimed(_request().model_copy(update={"messages": [ChatMessage(role="user", content="/new task")]}))
    with pytest.raises(Exception, match="fixture failure"):
        asyncio.run(service.chat_completion(request))
    assert exits == [None]
