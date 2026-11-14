"""Canonical session binding never consumes an ephemeral manager fallback."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llm_bawt.service.turn_execution import TurnExecutionDetached
from llm_bawt.service.turn_terminal_sessions import bind_terminal_session


class Backend:
    bot_id = "test-terminal"
    user_id = "test-user"

    def __init__(self):
        self.resolve = Mock(return_value="canonical-database-session")

    @property
    def _current_session_id(self):
        raise AssertionError("cached/fallback session property must never be read")

    def get_or_create_active_session(self, *, bot_id, user_id):
        return self.resolve(bot_id=bot_id, user_id=user_id)


def setup():
    backend = Backend()
    llm = SimpleNamespace(
        history_manager=SimpleNamespace(_db_backend=backend),
        _session_id_override="stale-session",
    )
    guard = SimpleNamespace(protocol=Mock(), lease=object(), check=Mock())
    return guard, llm, backend


def bind(guard, llm, **kwargs):
    return bind_terminal_session(
        guard, llm, bot_id="test-terminal", user_id="test-user", **kwargs,
    )


def test_native_resolution_returns_write_pin_without_narrowing_read_scope():
    guard, llm, backend = setup()
    assert bind(guard, llm) == "canonical-database-session"
    backend.resolve.assert_called_once_with(bot_id="test-terminal", user_id="test-user")
    guard.protocol.bind_history_session.assert_called_once_with(
        guard.lease, session_id="canonical-database-session",
    )
    assert llm._session_id_override == "stale-session"
    assert guard.check.call_count == 2


@pytest.mark.parametrize("agent", [False, True])
def test_selected_or_reset_final_session_is_validated_without_active_resolution(agent):
    guard, llm, backend = setup()
    assert bind(guard, llm, resolved_session_id="selected-successor", is_agent=agent) == "selected-successor"
    backend.resolve.assert_not_called()
    guard.protocol.bind_history_session.assert_called_once_with(
        guard.lease, session_id="selected-successor",
    )
    assert llm._session_id_override == "stale-session"


def test_native_resolution_error_is_a_setup_failure_not_an_ephemeral_session():
    guard, llm, backend = setup()
    backend.resolve.side_effect = ConnectionError("canonical session database unavailable")
    with pytest.raises(ConnectionError):
        bind(guard, llm)
    guard.protocol.bind_history_session.assert_not_called()
    assert llm._session_id_override == "stale-session"
    assert backend.resolve.call_count == 1


@pytest.mark.parametrize("value", [None, "", "  ", " session ", "x" * 37, 123])
def test_bad_resolution_cannot_reach_user_history(value):
    guard, llm, backend = setup()
    backend.resolve.return_value = value
    with pytest.raises(ValueError, match="no usable session"):
        bind(guard, llm)
    guard.protocol.bind_history_session.assert_not_called()
    assert llm._session_id_override == "stale-session"


def test_database_rejection_of_a_fallback_uuid_prevents_user_write_destination_change():
    guard, llm, backend = setup()
    backend.resolve.return_value = "unregistered-fallback"
    guard.protocol.bind_history_session.side_effect = ValueError("history session does not exist")
    with pytest.raises(ValueError, match="does not exist"):
        bind(guard, llm)
    assert llm._session_id_override == "stale-session"


def test_agent_without_final_binding_never_creates_an_active_session():
    guard, llm, backend = setup()
    with pytest.raises(ValueError, match="final request-local"):
        bind(guard, llm, is_agent=True)
    backend.resolve.assert_not_called()
    guard.protocol.bind_history_session.assert_not_called()


@pytest.mark.parametrize("field", ["bot_id", "user_id"])
def test_wrong_history_backend_owner_is_refused_before_resolution(field):
    guard, llm, backend = setup()
    setattr(backend, field, "different-owner")
    with pytest.raises(ValueError, match="does not match"):
        bind(guard, llm)
    backend.resolve.assert_not_called()
    guard.protocol.bind_history_session.assert_not_called()


def test_lease_loss_after_binding_still_aborts_before_the_user_message_write():
    guard, llm, backend = setup()
    guard.check.side_effect = [None, TurnExecutionDetached()]
    with pytest.raises(TurnExecutionDetached):
        bind(guard, llm)
    guard.protocol.bind_history_session.assert_called_once()
    assert llm._session_id_override == "stale-session"
