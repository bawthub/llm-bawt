"""Queued CLI replies reattach to one accepted delivery, never resubmit."""
from __future__ import annotations

import io
from types import SimpleNamespace

import pytest
from rich.console import Console

from llm_bawt.service.client import ServiceClient


ACK = {
    "object": "chat.completion.queued", "delivery_id": "delivery-1",
    "user_message_id": "user-message-1", "turn_id": "turn-delivery-1",
}


def test_ordinary_receipt_exposes_backend_resolved_scope():
    from llm_bawt.service.ordinary_turn_queue import OrdinaryTurnDecision

    receipt = OrdinaryTurnDecision(
        "queued", "turn-delivery-1", "user-message-1", "assistant-message-1",
        "delivery-1", "QUEUED", bot_id="resolved-bot", user_id="resolved-user",
    ).receipt()
    assert receipt["bot_id"] == "resolved-bot"
    assert receipt["user_id"] == "resolved-user"


def test_client_resolves_missing_scope_from_exact_delivery(monkeypatch):
    client = ServiceClient(http_url="http://example.invalid")
    paths = []

    def request(method, path, **kwargs):
        paths.append((method, path, kwargs.get("params")))
        if path.endswith("/reply"):
            assert kwargs["params"] == {
                "bot_id": "resolved-bot", "user_id": "resolved-user",
                "user_message_id": "user-message-1",
            }
            return {"delivery_id": "delivery-1", "status": "DELIVERED", "response_text": "done"}
        return {"delivery_id": "delivery-1", "user_message_id": "user-message-1",
                "target_bot_id": "resolved-bot",
                "author": {"entity_type": "user", "entity_id": "resolved-user"}}

    monkeypatch.setattr(client, "_request", request)
    assert client.wait_for_queued_reply(ACK, bot_id=None, user_id=None) == {
        "delivery_id": "delivery-1", "status": "DELIVERED", "response_text": "done",
    }
    assert len(paths) == 2


def test_client_rejects_delivery_scope_mismatch(monkeypatch):
    client = ServiceClient(http_url="http://example.invalid")
    monkeypatch.setattr(client, "_request", lambda *_args, **_kwargs: {
        "delivery_id": "delivery-1", "user_message_id": "user-message-1",
        "target_bot_id": "another-bot", "author": {"entity_type": "user", "entity_id": "resolved-user"},
    })
    with pytest.raises(ValueError, match="changed scope"):
        client.wait_for_queued_reply(ACK, bot_id="selected-bot", user_id=None)


def test_queued_cli_pending_is_handled_without_second_query(monkeypatch):
    from llm_bawt.cli import display_cmd

    client = ServiceClient(http_url="http://example.invalid")
    monkeypatch.setattr(client, "is_available", lambda **_kwargs: True)
    monkeypatch.setattr(client, "chat_completion", lambda **_kwargs: dict(ACK))
    monkeypatch.setattr(client, "wait_for_queued_reply", lambda *_args, **_kwargs: None)
    config = SimpleNamespace(VERBOSE=False)
    monkeypatch.setattr(display_cmd, "Config", lambda: config)
    monkeypatch.setattr(display_cmd, "BotManager", lambda _config: SimpleNamespace(
        get_bot=lambda _id: SimpleNamespace(name="Selected bot", color="green")))
    monkeypatch.setattr(display_cmd, "get_service_client", lambda _config: client)
    output = io.StringIO()
    monkeypatch.setattr(display_cmd, "console", Console(file=output))
    assert display_cmd.query_via_service("hello", model=None, bot_id="selected-bot",
                                         user_id="selected-user", plaintext_output=True,
                                         stream=False, config=config) is True
    assert "accepted" in output.getvalue()


def _wire(*frames: str) -> bytes:
    return "".join(frames).encode()


def _comment(payload: dict) -> str:
    import json
    return f": llm-bawt-event {json.dumps(payload)}\n\n"


def _data(payload: dict) -> str:
    import json
    return f"data: {json.dumps(payload)}\n\n"


def test_cli_stream_negotiates_and_reads_extension_comments(monkeypatch):
    """TASK-1144: the route's default stream drops warnings and server-run
    tool calls, so the CLI asks for them as comments and reads them back."""
    sent = {}

    def urlopen(req, **_kwargs):
        sent.update({key.lower(): value for key, value in req.header_items()})
        return io.BytesIO(_wire(
            _comment({"object": "service.warning", "model": "m", "warnings": ["model fallback"]}),
            _data({"object": "chat.completion.chunk", "model": "m",
                   "choices": [{"index": 0, "delta": {"content": "It is "}}]}),
            _comment({"object": "chat.completion.chunk", "model": "m", "choices": [{"index": 0, "delta": {
                "tool_calls": [{"function": {"name": "time", "arguments": "{}"}}]}}]}),
            _data({"object": "chat.completion.chunk", "model": "m",
                   "choices": [{"index": 0, "delta": {"content": "noon."}}]}),
            "data: [DONE]\n\n",
        ))

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    client = ServiceClient(http_url="http://example.invalid")
    items = list(client._stream_chat_completion({"messages": [], "stream": True}))
    assert sent["x-llm-bawt-extension-events"] == "comment"
    assert items == [
        {"warnings": ["model fallback"], "model": "m"},
        {"model": "m"},
        "It is ",
        {"tool_call": "time", "tool_args": "{}", "model": "m"},
        "noon.",
    ]


def test_cli_stream_error_after_admission_is_handled_not_retried(monkeypatch):
    from llm_bawt.cli import display_cmd

    wire = _wire(_data({"error": {"message": "Reply not ready (IN_FLIGHT)", "type": "server_error",
                                  "code": 504}}))
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: io.BytesIO(wire))
    client = ServiceClient(http_url="http://example.invalid")
    monkeypatch.setattr(client, "is_available", lambda **_kwargs: True)
    config = SimpleNamespace(VERBOSE=False)
    monkeypatch.setattr(display_cmd, "Config", lambda: config)
    monkeypatch.setattr(display_cmd, "BotManager", lambda _config: SimpleNamespace(
        get_bot=lambda _id: SimpleNamespace(name="Selected bot", color="green")))
    monkeypatch.setattr(display_cmd, "get_service_client", lambda _config: client)
    output = io.StringIO()
    monkeypatch.setattr(display_cmd, "console", Console(file=output))
    # False would send the same prompt to a local model a second time.
    assert display_cmd.query_via_service("hello", model=None, bot_id="selected-bot",
                                         user_id="selected-user", plaintext_output=True,
                                         stream=True, config=config) is True
    assert "Reply not ready (IN_FLIGHT)" in output.getvalue()
    assert client.last_error == "Reply not ready (IN_FLIGHT)"


def _queued_cli(monkeypatch, client, *, stream=False):
    from llm_bawt.cli import display_cmd

    monkeypatch.setattr(client, "is_available", lambda **_kwargs: True)
    config = SimpleNamespace(VERBOSE=False)
    monkeypatch.setattr(display_cmd, "Config", lambda: config)
    monkeypatch.setattr(display_cmd, "BotManager", lambda _config: SimpleNamespace(
        get_bot=lambda _id: SimpleNamespace(name="Selected bot", color="green")))
    monkeypatch.setattr(display_cmd, "get_service_client", lambda _config: client)
    output = io.StringIO()
    monkeypatch.setattr(display_cmd, "console", Console(file=output, width=200))
    return display_cmd, config, output


DELIVERED = {"delivery_id": "delivery-1", "status": "DELIVERED",
             "turn_id": "turn-delivery-1", "response_text": "It printed hi."}


@pytest.mark.parametrize("stream", [False, True])
def test_queued_cli_reply_renders_recorded_tool_calls(monkeypatch, capsys, stream):
    """TASK-1394: an accepted agent turn shows its recorded tool calls."""
    client = ServiceClient(http_url="http://example.invalid")
    seen = {}

    def request(method, path, **kwargs):
        assert (method, path) == ("GET", "/v1/tool-calls")
        seen.update(kwargs["params"])
        return {"events": [
            {"turn_id": "other-turn", "tool_calls": [{"name": "Stray", "arguments": {"command": "nope"}}]},
            {"turn_id": "turn-delivery-1", "tool_calls": [
                {"name": "Bash", "arguments": {"command": "echo hi"}, "result": "hi"},
            ]},
        ]}

    monkeypatch.setattr(client, "_request", request)
    monkeypatch.setattr(client, "wait_for_queued_reply", lambda *_a, **_k: dict(DELIVERED))
    if stream:
        monkeypatch.setattr(client, "chat_completion", lambda **_kwargs: iter([dict(ACK)]))
    else:
        monkeypatch.setattr(client, "chat_completion", lambda **_kwargs: dict(ACK))
    display_cmd, config, output = _queued_cli(monkeypatch, client)
    assert display_cmd.query_via_service("run echo hi", model=None, bot_id="selected-bot",
                                         user_id="selected-user", plaintext_output=True,
                                         stream=stream, config=config) is True
    text = output.getvalue()
    assert "-> Bash: Bash(command=echo hi)" in text
    assert "hi" in text and "Stray" not in text
    assert seen == {"message_id": "user-message-1", "bot_id": "selected-bot", "user_id": "selected-user"}
    assert "It printed hi." in capsys.readouterr().out + text


def test_queued_cli_tool_lookup_failure_still_prints_answer(monkeypatch, capsys):
    client = ServiceClient(http_url="http://example.invalid")

    def request(*_args, **_kwargs):
        raise RuntimeError("tool lookup down")

    monkeypatch.setattr(client, "_request", request)
    monkeypatch.setattr(client, "wait_for_queued_reply", lambda *_a, **_k: dict(DELIVERED))
    monkeypatch.setattr(client, "chat_completion", lambda **_kwargs: dict(ACK))
    display_cmd, config, output = _queued_cli(monkeypatch, client)
    assert display_cmd.query_via_service("hello", model=None, bot_id="selected-bot",
                                         user_id="selected-user", plaintext_output=True,
                                         stream=False, config=config) is True
    assert "could not be loaded" in output.getvalue()
    assert "It printed hi." in capsys.readouterr().out


def test_cli_stream_reads_admission_comment(monkeypatch):
    """TASK-1394: the service resolves a queued turn itself and streams only its
    text; the CLI asks for the admission comment to learn which turn it was."""
    import json
    sent = {}

    def urlopen(req, **_kwargs):
        sent.update({key.lower(): value for key, value in req.header_items()})
        return io.BytesIO(_wire(
            f": llm-bawt-admission {json.dumps(ACK)}\n\n",
            ": llm-bawt-admission {not json\n\n",
            ": waiting for admitted turn\n\n",
            _data({"object": "chat.completion.chunk", "model": "m",
                   "choices": [{"index": 0, "delta": {"content": "done"}}]}),
            "data: [DONE]\n\n",
        ))

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    client = ServiceClient(http_url="http://example.invalid")
    items = list(client._stream_chat_completion({"messages": [], "stream": True}))
    assert sent["x-llm-bawt-admission-events"] == "comment"
    assert items == [{"admitted": ACK}, {"model": "m"}, "done"]


def _tool_api(monkeypatch, client, events_by_message):
    seen = []

    def request(method, path, **kwargs):
        assert (method, path) == ("GET", "/v1/tool-calls")
        seen.append(kwargs["params"]["message_id"])
        return {"events": events_by_message.get(kwargs["params"]["message_id"], [])}

    monkeypatch.setattr(client, "_request", request)
    return seen


def _calls(turn_id, name):
    return {"turn_id": turn_id, "tool_calls": [
        {"name": name, "arguments": {"command": name.lower()}, "result": f"{name} ran"}]}


def test_admitted_stream_renders_tool_calls_before_text(monkeypatch, capsys):
    client = ServiceClient(http_url="http://example.invalid")
    _tool_api(monkeypatch, client, {"user-message-1": [_calls("turn-delivery-1", "Bash")]})
    monkeypatch.setattr(client, "wait_for_queued_reply", lambda *_a, **_k: {
        **DELIVERED, "user_message_id": "user-message-1"})
    monkeypatch.setattr(client, "chat_completion", lambda **_kwargs: iter(
        [{"admitted": dict(ACK)}, {"model": "m"}, "It printed hi."]))
    display_cmd, config, output = _queued_cli(monkeypatch, client)
    assert display_cmd.query_via_service("run echo hi", model=None, bot_id="selected-bot",
                                         user_id="selected-user", plaintext_output=True,
                                         stream=True, config=config) is True
    text = output.getvalue()
    assert "-> Bash: Bash(command=bash)" in text
    assert "It printed hi." in capsys.readouterr().out + text


CHILD_REPLY = {"delivery_id": "delivery-1", "retry_delivery_id": "delivery-child",
               "status": "DELIVERED", "turn_id": "turn-child",
               "user_message_id": "child-message", "response_text": "Child answer."}
RETRY_EVENTS = {"user-message-1": [_calls("turn-delivery-1", "OldAttempt")],
                "child-message": [_calls("turn-child", "ChildTool")]}


@pytest.mark.parametrize("path", ["admitted-stream", "legacy-receipt"])
def test_retried_turn_shows_the_answering_childs_tools(monkeypatch, capsys, path):
    """Codex TASK-1394 blocker: a restart retry answers from a child turn with
    its own trigger message; never show the aborted original's tools."""
    client = ServiceClient(http_url="http://example.invalid")
    seen = _tool_api(monkeypatch, client, RETRY_EVENTS)
    monkeypatch.setattr(client, "wait_for_queued_reply", lambda *_a, **_k: dict(CHILD_REPLY))
    frames = ([{"admitted": dict(ACK)}, {"model": "m"}, "Child answer."]
              if path == "admitted-stream" else [dict(ACK)])
    monkeypatch.setattr(client, "chat_completion", lambda **_kwargs: iter(frames))
    display_cmd, config, output = _queued_cli(monkeypatch, client)
    assert display_cmd.query_via_service("go", model=None, bot_id="selected-bot",
                                         user_id="selected-user", plaintext_output=True,
                                         stream=True, config=config) is True
    text = output.getvalue()
    assert "ChildTool" in text and "OldAttempt" not in text
    assert seen == ["child-message"]
    assert "Child answer." in capsys.readouterr().out + text


def test_admission_then_eof_reports_accepted_pending(monkeypatch):
    client = ServiceClient(http_url="http://example.invalid")
    monkeypatch.setattr(client, "chat_completion", lambda **_kwargs: iter([{"admitted": dict(ACK)}]))
    display_cmd, config, output = _queued_cli(monkeypatch, client)
    assert display_cmd.query_via_service("go", model=None, bot_id="selected-bot",
                                         user_id="selected-user", plaintext_output=True,
                                         stream=True, config=config) is True
    assert "delivery-1 was accepted" in output.getvalue()


def test_turn_tool_calls_without_turn_id_skips_lookup(monkeypatch):
    client = ServiceClient(http_url="http://example.invalid")
    monkeypatch.setattr(client, "_request", lambda *_a, **_k: pytest.fail("no lookup"))
    assert client.turn_tool_calls({"status": "DELIVERED"}, user_message_id="m",
                                  bot_id=None, user_id=None) == []
