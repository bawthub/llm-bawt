"""Agent nonstream history uses separate exact initial/final provenance scopes."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from unittest.mock import Mock

import pytest

from llm_bawt.bot_access import service as service_codec
from llm_bawt.models.message import Message
from llm_bawt.service.session_lifecycle.execution import NewSessionExecution, PreparedNewSession
from tests.test_nonstream_turn_execution import _claimed, _isolate, _request, _service

__all__ = ["_isolate"]


@pytest.mark.parametrize("initial_verified", [True, False])
@pytest.mark.parametrize("final_verified", [True, False])
def test_initial_and_final_owner_provenance_do_not_cross_rotation(monkeypatch, initial_verified, final_verified):
    service = _service(Mock(return_value=("answer", "", [])))
    llm = service._get_llm_bawt.return_value
    initial = {"thread_session_id": "before"}
    final = {"thread_session_id": "after"}
    if initial_verified:
        initial["_service_owner"] = ("before", "caid", "nick")
    if final_verified:
        final["_service_owner"] = ("after", "caid", "nick")
    service._bind_agent_thread.return_value = initial
    monkeypatch.setattr(NewSessionExecution, "prepare_agent",
                        lambda *a, **kw: PreparedNewSession(None, final, "after"))
    monkeypatch.setattr("llm_bawt.task_turn_context.mint_for_agent_turn",
                        lambda **kw: "proof:" + kw["session_id"])
    observed = []

    def prepare(*a, **kw):
        observed.append(("initial", service_codec._runtime_turn.get()))
        return [Message(role="user", content="do the task")]

    def execute(*a, **kw):
        observed.append(("execute", service_codec._runtime_turn.get()))
        assert kw["task_turn_capability"] == "proof:after"
        return "answer", "", []

    llm.prepare_messages_for_query.side_effect = prepare
    llm.execute_llm_query = execute
    service._finalize_turn.side_effect = lambda **kw: observed.append(("final", service_codec._runtime_turn.get()))
    asyncio.run(service.chat_completion(_claimed(_request())))
    assert observed == [("initial", "proof:before" if initial_verified else None),
                        ("execute", None),
                        ("final", "proof:after" if final_verified else None)]
    assert service_codec._runtime_turn.get() is None


@pytest.mark.parametrize("phase", ["initial", "execute", "final"])
def test_agent_nonstream_failure_clears_all_executor_scopes(monkeypatch, phase):
    service = _service(Mock(return_value=("answer", "", [])))
    llm = service._get_llm_bawt.return_value
    binding = {"thread_session_id": "bound", "_service_owner": ("bound", "caid", "nick")}
    service._bind_agent_thread.return_value = binding
    llm.prepare_messages_for_query.return_value = [Message(role="user", content="do the task")]
    monkeypatch.setattr("llm_bawt.task_turn_context.mint_for_agent_turn", lambda **kw: "proof")
    target = {"initial": llm.prepare_messages_for_query, "execute": llm.execute_llm_query,
              "final": service._finalize_turn}[phase]
    target.side_effect = RuntimeError("fixture failure")
    exits = []
    original = service_codec.bind_service_turn

    @contextmanager
    def track(capability):
        try:
            with original(capability):
                yield
        finally:
            exits.append((capability, service_codec._runtime_turn.get()))

    monkeypatch.setattr(service_codec, "bind_service_turn", track)
    with pytest.raises(Exception, match="fixture failure"):
        asyncio.run(service.chat_completion(_claimed(_request())))
    assert exits
    assert all(restored is None for _, restored in exits)
    assert exits[-1] == (None, None)


def test_mismatched_initial_owner_cannot_mint_early_history_proof(monkeypatch, _isolate):
    service = _service(Mock(return_value=("answer", "", [])))
    llm = service._get_llm_bawt.return_value
    service._bind_agent_thread.return_value = {
        "thread_session_id": "bound", "_service_owner": ("bound", "caid", "other-user")}
    observed = []
    llm.prepare_messages_for_query.side_effect = lambda *a, **kw: observed.append(service_codec._runtime_turn.get()) or []
    service._finalize_turn.side_effect = lambda **kw: observed.append(service_codec._runtime_turn.get())
    asyncio.run(service.chat_completion(_claimed(_request())))
    assert observed == [None, None]
    assert len(_isolate) == 1  # unchanged backend/task mint only, no early proof
