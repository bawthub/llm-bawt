"""TASK-1778: exercise real handlers with trusted synthetic callers, never live history."""
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent_bridge.mcp_call_context import canonical_invocation_hash
from llm_bawt.mcp_server import server as tools
from llm_bawt.mcp_server.approval_interceptor import ApprovedCallerContext
from llm_bawt.mcp_server.catalog import CatalogFastMCP
from llm_bawt.mcp_server.client import MemoryClient, _MCPShortTermManager
from tests.test_bot_access_mcp import Resolver, identity, signer, turn  # noqa: F401
from tests.test_bot_access_service_self import service_call

pytestmark = pytest.mark.usefixtures("signer")

HANDLERS = {"messages_get": tools.get_messages, "messages_clear": tools.clear_messages,
            "messages_remove_last_partial": tools.remove_last_message_if_partial}


@pytest.fixture
def history_catalog(monkeypatch):
    calls = []

    async def get_messages(**kwargs):
        calls.append(("read", kwargs))
        return [{"id": "synthetic", "role": "assistant", "content": "fixture"}]

    async def clear_messages(**kwargs):
        calls.append(("clear", kwargs))
        return 1

    async def clear_all_messages(**kwargs):
        calls.append(("operator_clear_all", kwargs))
        return 2

    async def remove_partial(**kwargs):
        calls.append(("partial", kwargs))
        return True

    monkeypatch.setattr(tools, "_get_storage", lambda: SimpleNamespace(
        get_messages=get_messages, clear_messages=clear_messages,
        clear_all_messages=clear_all_messages, remove_last_message_if_partial=remove_partial))
    server = CatalogFastMCP("history-owners", bot_policy_resolver=Resolver())
    for name, handler in HANDLERS.items():
        server.add_tool(handler, name=name)
    server.mark_tools_registered()
    return server, calls


@contextmanager
def caller(kind, name, args, user="user_a"):
    if kind == "turn":
        with identity(turn(user=user)):
            yield
    elif kind == "service":
        with service_call(name, args, user=user):
            yield
    else:
        yield


async def invoke(server, kind, name, args, user="user_a"):
    with caller(kind, name, args, user):
        if kind == "approved":
            return await server.call_approved_tool(name, args,
                expected_invocation_hash=canonical_invocation_hash(name, args),
                caller_context=ApprovedCallerContext(bot_id="private_fixture", user_id=user))
        return await server._tool_manager.call_tool(name, args)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", HANDLERS)
@pytest.mark.parametrize("bot", ["public_fixture", "private_fixture"])
@pytest.mark.parametrize("kind", ["turn", "approved", "service"])
@pytest.mark.parametrize("user", ["user_a", "user_b"])
async def test_real_handlers_default_to_trusted_user(history_catalog, name, bot, kind, user):
    server, calls = history_catalog
    await invoke(server, kind, name, {"bot_id": bot}, user)
    assert len(calls) == 1
    assert calls[0][1]["user_id"] == user
    assert calls[0][1]["bot_id"] == bot


@pytest.mark.asyncio
@pytest.mark.parametrize("name", HANDLERS)
@pytest.mark.parametrize("bot", ["public_fixture", "private_fixture"])
@pytest.mark.parametrize("kind", ["turn", "approved", "service"])
@pytest.mark.parametrize("spoof", ["user_b", ""])
async def test_supplied_user_cannot_replace_trusted_owner(history_catalog, name, bot, kind, spoof):
    server, calls = history_catalog
    with pytest.raises(Exception, match="access_denied"):
        await invoke(server, kind, name, {"bot_id": bot, "user_id": spoof})
    assert not calls


@pytest.mark.asyncio
@pytest.mark.parametrize("name", HANDLERS)
@pytest.mark.parametrize("user", [None, "user_a"])
async def test_public_target_still_requires_identity(history_catalog, name, user):
    server, calls = history_catalog
    with pytest.raises(Exception, match="caller_unavailable"):
        await server._tool_manager.call_tool(name, {"bot_id": "public_fixture", "user_id": user})
    assert not calls


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["messages_get", "messages_clear"])
@pytest.mark.parametrize("user", [None, "user_b"])
async def test_existing_signed_operator_self_read_and_clear_remain_separate(history_catalog, name, user):
    server, calls = history_catalog
    args = {"bot_id": "private_fixture", "user_id": user}
    if name == "messages_clear" and user is not None:
        # Existing service clear-all has no arbitrary-user mutation authority.
        with service_call(name, args), pytest.raises(Exception, match="access_denied"):
            await server._tool_manager.call_tool(name, args)
        assert not calls
        return
    with service_call(name, args):
        await server._tool_manager.call_tool(name, args)
    if name == "messages_clear" and user is None:
        assert calls == [("operator_clear_all", {"bot_id": "private_fixture"})]
    else:
        assert calls[0][1]["user_id"] == user


@pytest.mark.asyncio
async def test_userless_service_cannot_do_partial_cleanup(history_catalog):
    server, calls = history_catalog
    name, args = "messages_remove_last_partial", {"bot_id": "private_fixture"}
    with service_call(name, args), pytest.raises(Exception, match="caller_unavailable"):
        await server._tool_manager.call_tool(name, args)
    assert not calls


@pytest.mark.asyncio
@pytest.mark.parametrize("name", HANDLERS)
async def test_untrusted_direct_handler_cannot_bypass_registered_context(history_catalog, name):
    _, calls = history_catalog
    with pytest.raises(Exception, match="caller_unavailable"):
        await HANDLERS[name](bot_id="public_fixture", user_id="user_a")
    assert not calls


@pytest.mark.parametrize("server_mode", [False, True])
@pytest.mark.parametrize("user", ["user_a", "user_b"])
def test_request_client_defaults_are_scoped_in_both_modes(server_mode, user):
    client = MemoryClient(SimpleNamespace(), bot_id="public_fixture", user_id=user,
                          server_url="http://mcp.invalid" if server_mode else None)
    calls = []

    async def read(**kwargs):
        calls.append(("messages_get", kwargs))
        return []

    async def clear(**kwargs):
        calls.append(("messages_clear", kwargs))
        return 0

    async def partial(**kwargs):
        calls.append(("messages_remove_last_partial", kwargs))
        return False

    def wire(name, args):
        calls.append((name, args))
        return [] if name == "messages_get" else 0

    client._initialized = True
    client._storage = SimpleNamespace(get_messages=read, clear_messages=clear,
                                      remove_last_message_if_partial=partial)
    client._call_server = wire
    client.get_messages()
    client.clear_messages()
    client.remove_last_message_if_partial("assistant")
    assert len(calls) == 3
    assert all(args["user_id"] == user for _, args in calls)
    for method in [client.get_messages, client.clear_messages,
                   lambda **kw: client.remove_last_message_if_partial("assistant", **kw)]:
        with pytest.raises(ValueError, match="client owner"):
            method(user_id="other_user")
    assert len(calls) == 3


def test_userless_adapter_cannot_clear_or_cleanup():
    client = MemoryClient(SimpleNamespace(), bot_id="public_fixture", server_url="http://mcp.invalid")
    client._call_server = Mock(side_effect=AssertionError("must reject before transport"))
    with pytest.raises(ValueError, match="requires user_id"):
        _MCPShortTermManager(client).clear()
    with pytest.raises(ValueError, match="requires user_id"):
        client.remove_last_message_if_partial("assistant")
    client._call_server.assert_not_called()


def test_history_manager_passes_its_owner_to_a_shared_backend(tmp_path):
    from llm_bawt.models.message import Message
    from llm_bawt.utils.history import HistoryManager

    backend = Mock()
    manager = HistoryManager(SimpleNamespace(console=Mock()),
        SimpleNamespace(HISTORY_FILE=str(tmp_path / "history")), db_backend=backend,
        bot_id="public_fixture", user_id="user_a")
    manager.messages = [Message(role="assistant", content="partial", timestamp=1)]
    manager.remove_last_message_if_partial("assistant")
    backend.remove_last_message_if_partial.assert_called_once_with("assistant", user_id="user_a")
    manager.clear_history()
    backend.clear.assert_called_once_with(user_id="user_a")
