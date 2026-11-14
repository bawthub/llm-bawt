"""Offline regressions for memory-debug CLI confirmation and identity handling."""

from io import StringIO
from unittest.mock import Mock

import pytest
from rich.console import Console

from llm_bawt import memory_debug as cli


MEMORY_ID = "c8ee88c0-b450-4bb6-ae7a-dbb8358d51da"
ATTRIBUTE = {
    "id": 42, "category": "preferences", "key": "drink", "value": "tea",
}


@pytest.fixture
def output(monkeypatch):
    stream = StringIO()
    monkeypatch.setattr(cli, "console", Console(file=stream, width=160, color_system=None))
    # Any API operation not explicitly stubbed by a test must fail offline.
    for name in ("api_get", "api_post", "api_delete", "check_service_available"):
        monkeypatch.setattr(cli, name, Mock(side_effect=AssertionError(f"Unexpected {name}")))
    return stream


@pytest.mark.parametrize("confirmed", [False, True])
def test_attribute_delete_previews_before_confirmation(monkeypatch, output, confirmed):
    get = Mock(return_value={"attributes": [ATTRIBUTE], "total_count": 1})
    delete = Mock(return_value={"success": True, "deleted": ATTRIBUTE})
    monkeypatch.setattr(cli, "api_get", get)
    monkeypatch.setattr(cli, "api_delete", delete)

    def confirm(prompt):
        assert "preferences.drink = tea" in output.getvalue()
        assert "ID: 42" in output.getvalue()
        delete.assert_not_called()
        return confirmed

    monkeypatch.setattr(cli, "_confirm", confirm)
    cli.handle_delete_attribute(42, skip_confirm=False)

    get.assert_called_once_with("/v1/profiles/attribute", {"limit": 500, "offset": 0})
    if confirmed:
        delete.assert_called_once_with("/v1/profiles/attribute/42")
    else:
        delete.assert_not_called()


def test_attribute_delete_yes_skips_confirmation_but_previews_later_page(monkeypatch, output):
    get = Mock(side_effect=[
        {"attributes": [{"id": 1}], "total_count": 2},
        {"attributes": [ATTRIBUTE], "total_count": 2},
    ])
    monkeypatch.setattr(cli, "api_get", get)
    delete = Mock(return_value={"success": True, "deleted": ATTRIBUTE})
    monkeypatch.setattr(cli, "api_delete", delete)
    monkeypatch.setattr(cli, "_confirm", Mock(side_effect=AssertionError("--yes must not prompt")))

    cli.handle_delete_attribute(42, skip_confirm=True)

    assert get.call_args_list[1].args == ("/v1/profiles/attribute", {"limit": 500, "offset": 1})
    assert "preferences.drink = tea" in output.getvalue()
    delete.assert_called_once_with("/v1/profiles/attribute/42")


@pytest.mark.parametrize("response", [None, {"attributes": [], "total_count": 0}])
def test_attribute_delete_fails_closed_when_preview_unavailable(monkeypatch, output, response):
    monkeypatch.setattr(cli, "api_get", Mock(return_value=response))
    cli.handle_delete_attribute(42, skip_confirm=True)
    cli.api_delete.assert_not_called()


@pytest.mark.parametrize("skip_confirm,confirmed", [(True, False), (False, True), (False, False)])
def test_full_uuid_memory_delete_bypasses_limited_search_but_confirms(monkeypatch, output, skip_confirm, confirmed):
    delete = Mock(return_value={"success": True})
    monkeypatch.setattr(cli, "api_delete", delete)

    def confirm(prompt):
        assert MEMORY_ID in output.getvalue()
        assert "selected-bot" in output.getvalue()
        delete.assert_not_called()
        return confirmed

    confirmation = Mock(side_effect=confirm)
    monkeypatch.setattr(cli, "_confirm", confirmation)
    cli.handle_delete_memory("selected-bot", MEMORY_ID.upper(), skip_confirm=skip_confirm)

    cli.api_get.assert_not_called()
    cli.api_post.assert_not_called()
    assert confirmation.call_count == (0 if skip_confirm else 1)
    if skip_confirm or confirmed:
        delete.assert_called_once_with(f"/v1/memory/{MEMORY_ID}", {"bot_id": "selected-bot"})
    else:
        delete.assert_not_called()


def test_full_uuid_memory_delete_reports_server_failure(monkeypatch, output):
    monkeypatch.setattr(cli, "api_delete", Mock(return_value=None))
    cli.handle_delete_memory("selected-bot", MEMORY_ID, skip_confirm=True)
    assert "Failed to delete memory" in output.getvalue()
    assert "Memory deleted" not in output.getvalue()


def test_memory_prefix_keeps_preview_and_confirmation(monkeypatch, output):
    monkeypatch.setattr(cli, "api_post", Mock(return_value={"results": [{
        "id": MEMORY_ID, "content": "A sample memory", "tags": [],
    }]}))
    monkeypatch.setattr(cli, "_confirm", Mock(return_value=False))

    cli.handle_delete_memory("selected-bot", MEMORY_ID[:8], skip_confirm=False)

    assert "A sample memory" in output.getvalue()
    cli._confirm.assert_called_once()
    cli.api_delete.assert_not_called()


@pytest.mark.parametrize("entity,expected", [(None, "configured-user"), ("explicit-bot", "explicit-bot")])
def test_list_attrs_resolves_default_user_only_when_omitted(monkeypatch, output, entity, expected):
    monkeypatch.setattr(cli, "check_service_available", lambda: True)
    default = Mock(return_value="configured-user")
    monkeypatch.setattr(cli, "default_user_id", default)
    handler = Mock()
    monkeypatch.setattr(cli, "handle_list_attrs", handler)
    monkeypatch.setattr(cli.sys, "argv", ["llm-memory", "--list-attrs"] + ([entity] if entity else []))

    cli.main()

    handler.assert_called_once_with(expected)
    assert default.call_count == (1 if entity is None else 0)


def test_list_attrs_requires_entity_when_default_user_unconfigured(monkeypatch, output):
    monkeypatch.setattr(cli, "check_service_available", lambda: True)
    monkeypatch.setattr(cli, "default_user_id", lambda: "")
    monkeypatch.setattr(cli.sys, "argv", ["llm-memory", "--list-attrs"])

    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    assert "LLM_BAWT_DEFAULT_USER" in output.getvalue()
    cli.api_get.assert_not_called()


@pytest.mark.parametrize("handler", [cli.handle_msg_forget, cli.handle_msg_forget_since])
def test_forget_warns_before_confirmation_and_limits_restore_claim(monkeypatch, output, handler):
    monkeypatch.setattr(cli, "api_get", Mock(return_value={"messages": [{"id": "message-id"}]}))
    monkeypatch.setattr(cli, "display_messages_preview", lambda messages: None)
    post = Mock(return_value={"message": "Done"})
    monkeypatch.setattr(cli, "api_post", post)

    def confirm(prompt):
        assert "permanently deleted" in output.getvalue()
        assert "restores messages only" in output.getvalue()
        post.assert_not_called()
        return True

    monkeypatch.setattr(cli, "_confirm", confirm)
    handler("selected-bot", 5, skip_confirm=False)

    post.assert_called_once()
    assert "Derived memories are permanently deleted" in output.getvalue()
    assert "Use --msg-restore to undo" not in output.getvalue()


@pytest.mark.parametrize("handler", [cli.handle_msg_forget, cli.handle_msg_forget_since])
def test_forget_cancellation_never_mutates(monkeypatch, output, handler):
    monkeypatch.setattr(cli, "api_get", Mock(return_value={"messages": [{"id": "message-id"}]}))
    monkeypatch.setattr(cli, "display_messages_preview", lambda messages: None)
    monkeypatch.setattr(cli, "_confirm", lambda prompt: False)
    handler("selected-bot", 5, skip_confirm=False)
    cli.api_post.assert_not_called()
