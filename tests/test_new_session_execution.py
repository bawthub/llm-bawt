"""Shared preparation parity for typed /new and delivery reset policies."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llm_bawt.service.chat_message_text import last_user_text
from llm_bawt.service.session_lifecycle.execution import NewSessionExecution
from llm_bawt.service.session_lifecycle.bridge_reset import BridgeSessionResetMixin


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("mode", ["typed", "retain", "clean"])
def test_shared_reset_preparation_uses_one_continuity_contract(monkeypatch, stream, mode):
    events = []
    predecessor = {"thread_session_id": "old-thread", "thread_resume_id": "old-sdk"}
    successor = {"thread_session_id": "new-thread"}
    service = SimpleNamespace(
        _maybe_summarize_on_new=Mock(side_effect=lambda *_a, **kw: events.append(
            ("summary", kw["thread_binding"]["thread_session_id"])
        )),
        _maybe_rotate_agent_session=Mock(side_effect=lambda *_a, **_kw: (
            events.append(("rotate", "new-thread")) or "new-thread"
        )),
        _resolve_active_thread_binding=Mock(return_value=successor),
    )
    monkeypatch.setattr(
        "llm_bawt.service.routes.history.maybe_build_session_seed",
        lambda *_a, **_kw: events.append(("seed", "old-thread")) or ["configured"],
    )
    monkeypatch.setattr(
        "llm_bawt.service.routes.history.build_context_seed",
        lambda *_a, **kw: events.append(("seed", kw["session_id"])) or {"messages": ["retained"]},
    )
    request = SimpleNamespace(
        inter_bot_session_policy={"typed": None, "retain": "reset_retain_history",
                                  "clean": "reset_without_history"}[mode],
        inter_bot_seed_session_id="old-thread" if mode != "typed" else None,
        stream=stream,
    )
    result = NewSessionExecution.prepare_agent(
        service, SimpleNamespace(), request, bot_id="fixture", model_alias="model",
        user_prompt="/new" if mode == "typed" else "Do the task",
        thread_binding=predecessor if mode == "typed" else successor,
        delivery_id="delivery-one" if mode == "typed" else None,
        backend_name="claude-code",
    )
    if mode == "typed":
        assert events == [("summary", "old-thread"), ("seed", "old-thread"),
                          ("rotate", "new-thread")]
        assert result.seed_messages == ["configured"]
        assert result.rotated_id == "new-thread"
    elif mode == "retain":
        assert events == [("summary", "old-thread"), ("seed", "old-thread")]
        assert result.seed_messages == ["retained"]
        assert result.rotated_id is None
    else:
        assert events == []
        assert result.seed_messages == []
        assert result.rotated_id is None
    assert result.thread_binding == successor
    assert result.bridge_reset_action == (
        "reset_only" if mode == "typed" else "reset_follow_up"
    )


def test_retain_without_predecessor_does_not_summarize_fresh_successor(monkeypatch):
    service = SimpleNamespace(_maybe_summarize_on_new=Mock(),
                              _maybe_rotate_agent_session=Mock())
    monkeypatch.setattr(
        "llm_bawt.service.routes.history.maybe_build_session_seed",
        Mock(side_effect=AssertionError("unexpected configured seed")),
    )
    result = NewSessionExecution.prepare_agent(
        service, SimpleNamespace(),
        SimpleNamespace(inter_bot_session_policy="reset_retain_history",
                        inter_bot_seed_session_id=None),
        bot_id="fixture", model_alias="model", user_prompt="do task",
        thread_binding={"thread_session_id": "new-thread"},
        delivery_id=None, backend_name="claude-code",
    )
    assert result.seed_messages == []
    service._maybe_summarize_on_new.assert_not_called()
    service._maybe_rotate_agent_session.assert_not_called()


def test_claimed_policy_successor_is_not_a_user_selected_explicit_thread():
    class Service(BridgeSessionResetMixin):
        def _resolve_thread_sdk_key(self, _llm_bawt, binding, _backend):
            return binding

    llm_bawt = SimpleNamespace(bot=SimpleNamespace(agent_backend="claude-code"))
    service = Service()
    policy = SimpleNamespace(session_id="fresh-thread", inter_bot_delivery_id="delivery-one",
                             inter_bot_session_policy="reset_retain_history")
    binding = service._bind_agent_thread(llm_bawt, policy)
    assert binding == {"thread_session_id": "fresh-thread"}
    explicit = SimpleNamespace(session_id="old-thread", inter_bot_delivery_id=None,
                               inter_bot_session_policy=None)
    assert service._bind_agent_thread(llm_bawt, explicit)["explicit_thread"] is True


def test_typed_follow_up_uses_only_parsed_task_for_bridge(monkeypatch):
    from llm_bawt.models.message import Message

    service = SimpleNamespace(_maybe_summarize_on_new=Mock(),
                              _maybe_rotate_agent_session=Mock(return_value="successor"))
    monkeypatch.setattr("llm_bawt.service.routes.history.maybe_build_session_seed",
                        lambda *_args, **_kwargs: ["seed"])
    monkeypatch.setattr("llm_bawt.service.agent_new_rotation.bind_new_successor",
                        lambda *_args, **_kwargs: {"thread_session_id": "successor"})
    command = "  /new\nDo the exact task  "
    result = NewSessionExecution.prepare_agent(
        service, SimpleNamespace(),
        SimpleNamespace(inter_bot_session_policy=None, inter_bot_typed_follow_up=True),
        bot_id="fixture", model_alias="model", user_prompt=command,
        thread_binding={"thread_session_id": "old"},
        delivery_id="delivery-one", backend_name="claude-code",
    )
    assert result.bridge_reset_action == "reset_follow_up"
    assert result.bridge_prompt == "Do the exact task"
    visible = Message(role="user", content=command, db_id="trigger")
    outbound = NewSessionExecution.bridge_messages([visible], result.bridge_prompt, command)
    assert outbound[0] is not visible and outbound[0].content == "Do the exact task"
    assert outbound[0].db_id == "trigger" and visible.content == command
    wrapped = Message(role="user", content=f"Voice prefix\n\n---\n\n{command}\n\nAttachment manifest")
    assert NewSessionExecution.bridge_messages(
        [wrapped], result.bridge_prompt, command,
    )[0].content == "Voice prefix\n\n---\n\nDo the exact task\n\nAttachment manifest"
    assert NewSessionExecution.typed_follow_up("/newspaper is literal") is None


@pytest.mark.parametrize("messages, command", [
    ([{"role": "user", "content": "/new task"}], "/new task"),
    ([{"role": "user", "content": [
        {"type": "text", "text": "  /new\n"},
        {"type": "image_url", "image_url": {"url": "asset"}},
        {"type": "text", "text": "keep bytes  "},
    ]}, {"role": "assistant", "content": "ignored"}], "  /new\nkeep bytes  "),
    ([{"role": "user", "content": [{"type": "text", "text": 123}]}], None),
    ([{"role": "assistant", "content": "/new task"}], None),
    ([{"role": "user", "content": {"type": "text", "text": "/new task"}}], None),
    ("invalid", None),
])
def test_stored_last_user_command_parses_text_parts_only(messages, command):
    assert last_user_text(messages) == command


def test_native_codex_standalone_clean_reset_keeps_explicit_empty_seed(monkeypatch):
    service = SimpleNamespace(_maybe_summarize_on_new=Mock(),
                              _maybe_rotate_agent_session=Mock())
    monkeypatch.setattr("llm_bawt.service.routes.history.maybe_build_session_seed",
                        Mock(side_effect=AssertionError("configured seed")))
    result = NewSessionExecution.prepare_agent(
        service, SimpleNamespace(),
        SimpleNamespace(inter_bot_session_policy="reset_without_history",
                        inter_bot_seed_session_id="old", inter_bot_reset_only=True,
                        inter_bot_delivery_id="delivery-one"),
        bot_id="fixture", model_alias="model", user_prompt="/new",
        thread_binding={"thread_session_id": "new"},
        delivery_id=None, backend_name="codex",
    )
    assert result.seed_messages == [] and result.bridge_reset_action == "reset_only"
    service._maybe_rotate_agent_session.assert_not_called()


def test_native_codex_configured_history_refuses_before_rotation(monkeypatch):
    service = SimpleNamespace(_maybe_summarize_on_new=Mock(),
                              _maybe_rotate_agent_session=Mock())
    monkeypatch.setattr("llm_bawt.service.routes.history.maybe_build_session_seed",
                        lambda *_args, **_kwargs: [{"role": "summary", "content": "previous"}])
    with pytest.raises(ValueError, match="cannot seed configured reset history"):
        NewSessionExecution.prepare_agent(
            service, SimpleNamespace(), SimpleNamespace(inter_bot_session_policy=None),
            bot_id="fixture", model_alias="model", user_prompt="/new",
            thread_binding={"thread_session_id": "old"},
            delivery_id="delivery-one", backend_name="codex",
        )
    service._maybe_rotate_agent_session.assert_not_called()


def test_policy_rejects_unknown_mode_before_side_effects():
    service = SimpleNamespace(_maybe_summarize_on_new=Mock(),
                              _maybe_rotate_agent_session=Mock())
    with pytest.raises(ValueError, match="unsupported reset policy"):
        NewSessionExecution.prepare_agent(
            service, SimpleNamespace(),
            SimpleNamespace(inter_bot_session_policy="reset_unknown"),
            bot_id="fixture", model_alias="model", user_prompt="do task",
            thread_binding={"thread_session_id": "new-thread"},
            delivery_id=None, backend_name="claude-code",
        )
    service._maybe_summarize_on_new.assert_not_called()
    service._maybe_rotate_agent_session.assert_not_called()
