"""Bridge identity cannot be bound by a protocol worker whose lease was lost."""
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llm_bawt.service.turn_execution import TurnExecution, TurnExecutionDetached
from llm_bawt.service.turn_logs import TurnLogStore
from llm_bawt.service.turn_terminal_follower import TerminalFollower
from llm_bawt.service.turn_terminal_protocol import FollowerFenceLost
from tests.test_turn_terminal_protocol import protocol, insert_turn, read, expire  # noqa: F401


@pytest.fixture
def store(protocol, monkeypatch):  # noqa: F811
    value = TurnLogStore.__new__(TurnLogStore)
    value.engine = protocol.engine
    monkeypatch.setattr("llm_bawt.service.turn_logs.pin_member_request", Mock())
    return value


def bind(store, turn, **kwargs):
    store.bind_agent_bridge(turn_id=turn, backend="claude-code", session_key="test:user",
                            request_id="request", **kwargs)


def test_protocol_binding_requires_current_owner(protocol, store):  # noqa: F811
    turn = insert_turn(protocol.engine)
    lease = protocol.claim(turn)
    with pytest.raises(FollowerFenceLost):
        bind(store, turn)
    bind(store, turn, follower_lease=lease)
    assert read(protocol.engine, turn).agent_request_id == "request"
    expire(protocol.engine, turn)
    new_owner = protocol.claim(turn)
    with pytest.raises(FollowerFenceLost):
        bind(store, turn, follower_lease=lease)
    bind(store, turn, follower_lease=new_owner)


def test_binding_before_history_pin_is_refused(protocol, store):  # noqa: F811
    turn = insert_turn(protocol.engine, unbound=True)
    lease = protocol.claim(turn)
    with pytest.raises(FollowerFenceLost, match="pinned history"):
        bind(store, turn, follower_lease=lease)
    assert read(protocol.engine, turn).agent_request_id is None


def test_protocol_identity_cannot_change_under_same_request(protocol, store):  # noqa: F811
    turn = insert_turn(protocol.engine)
    lease = protocol.claim(turn)
    bind(store, turn, follower_lease=lease)
    with pytest.raises(FollowerFenceLost, match="immutable"):
        store.bind_agent_bridge(turn_id=turn, backend="other", session_key="test:user",
                                request_id="request", follower_lease=lease)


def test_legacy_binding_needs_no_protocol_lease(protocol, store):  # noqa: F811
    turn = insert_turn(protocol.engine, legacy=True)
    bind(store, turn)
    assert read(protocol.engine, turn).agent_request_id == "request"


def test_execution_passes_exact_follower_lease_and_detaches_before_rebinding():
    store = SimpleNamespace(bind_agent_bridge=Mock())
    execution = TurnExecution("turn", Event(), is_agent=True, store=store)
    from llm_bawt.service.turn_terminal_protocol import FollowerLease
    from datetime import datetime, timezone
    lease = FollowerLease("turn", "token", datetime.now(timezone.utc))
    TerminalFollower(Mock(), lease, execution)
    execution.bind_bridge(backend="bridge", session_key="session", request_id="request")
    assert store.bind_agent_bridge.call_args.kwargs["follower_lease"] is lease
    execution.detach()
    with pytest.raises(TurnExecutionDetached):
        execution.bind_bridge(backend="bridge", session_key="session", request_id="request")
    assert store.bind_agent_bridge.call_count == 1
