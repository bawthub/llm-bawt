"""A canonical write destination must not become an explicit history read."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llm_bawt.models.message import Message
from llm_bawt.service.core import ServiceLLMBawt


def service(*, explicit=None):
    bawt = ServiceLLMBawt.__new__(ServiceLLMBawt)
    bawt._session_id_override = explicit
    history = [
        Message(role="user", content="unsummarized prior-session context"),
        Message(role="assistant", content="current-session context"),
    ]
    bawt.history_manager = SimpleNamespace(load_history=Mock(), add_message=Mock())
    bawt._refresh_history_if_stale = Mock()
    bawt.invalidate_history_cache = Mock()
    bawt._build_context_messages = Mock(return_value=history)
    return bawt, history


def test_continuous_protocol_turn_reads_continuous_pool_and_only_pins_user_write():
    bawt, history = service()
    result = bawt.prepare_messages_for_query("continue", history_write_session_id="bound-session")
    assert result is history
    assert result[0].content == "unsummarized prior-session context"
    bawt._refresh_history_if_stale.assert_called_once()
    bawt.history_manager.load_history.assert_not_called()
    bawt.invalidate_history_cache.assert_not_called()
    assert bawt.history_manager.add_message.call_args.kwargs["session_id"] == "bound-session"
    assert bawt._session_id_override is None


def test_explicit_selection_keeps_its_scoped_read_and_the_same_pinned_write():
    bawt, _ = service(explicit="selected-session")
    bawt.prepare_messages_for_query("continue", history_write_session_id="selected-session")
    bawt.history_manager.load_history.assert_called_once_with(session_id="selected-session")
    bawt.invalidate_history_cache.assert_called_once()
    bawt._refresh_history_if_stale.assert_not_called()
    assert bawt.history_manager.add_message.call_args.kwargs["session_id"] == "selected-session"


def test_conflicting_explicit_read_and_write_pin_fail_before_history_side_effects():
    bawt, _ = service(explicit="selected-session")
    with pytest.raises(ValueError, match="match explicit selection"):
        bawt.prepare_messages_for_query("continue", history_write_session_id="different-session")
    bawt.history_manager.load_history.assert_not_called()
    bawt._refresh_history_if_stale.assert_not_called()
    bawt.history_manager.add_message.assert_not_called()


@pytest.mark.parametrize("explicit", [None, "selected-session"])
def test_legacy_prepare_keeps_its_existing_read_and_write_behavior(explicit):
    bawt, _ = service(explicit=explicit)
    bawt.prepare_messages_for_query("legacy")
    assert bawt.history_manager.add_message.call_args.kwargs["session_id"] == explicit
    if explicit:
        bawt.history_manager.load_history.assert_called_once_with(session_id=explicit)
    else:
        bawt._refresh_history_if_stale.assert_called_once()


@pytest.mark.parametrize("pin", ["", " session ", "x" * 37, 123])
def test_invalid_write_pin_does_not_read_or_write_history(pin):
    bawt, _ = service()
    with pytest.raises(ValueError, match="history write pin"):
        bawt.prepare_messages_for_query("continue", history_write_session_id=pin)
    bawt.history_manager.load_history.assert_not_called()
    bawt._refresh_history_if_stale.assert_not_called()
    bawt.history_manager.add_message.assert_not_called()
