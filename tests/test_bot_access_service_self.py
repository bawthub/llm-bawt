"""User-less app service reads are bot-bound filters, never user authorship."""
from __future__ import annotations

from contextlib import contextmanager

import pytest

from llm_bawt.bot_access import BotAccessError, TrustedCaller
from llm_bawt.bot_access.service import ServicePrincipal
from llm_bawt.mcp_server.bot_access import invocation_context
from llm_bawt.mcp_server.bot_access_identity import _service_capability
from llm_bawt.mcp_server.catalog import CatalogFastMCP
from tests.test_bot_access_mcp import Resolver, signer  # noqa: F401


@contextmanager
def service_call(name, arguments, *, bot="private_fixture", user=None):
    binding = _service_capability.set(ServicePrincipal(bot, user).mint(name, arguments))
    try:
        with invocation_context(name, arguments):
            yield
    finally:
        _service_capability.reset(binding)


def catalog(monkeypatch, name, owner=("private_fixture", "user_a")):
    resolver = Resolver()
    resolver.rows.append({"slug": "other_private", "include_in_global_search": False})
    server = CatalogFastMCP("service-self", bot_policy_resolver=resolver)
    calls = []

    async def handler(bot_id: str, user_id: str | None = None,
                      session_id: str | None = None, accept_pointer: bool = False):
        calls.append((bot_id, user_id, session_id))
        return "ok"

    server.add_tool(handler, name=name)
    monkeypatch.setattr(server.bot_access, "_session_owner", lambda sid: owner)
    return server, calls


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["messages_get", "sessions_get", "sessions_close",
                                 "sessions_get_active", "sessions_list"])
@pytest.mark.parametrize("user", [None, "user_a", "user_b"])
async def test_userless_service_self_retains_read_close_user_filters(monkeypatch, signer, name, user):
    server, calls = catalog(monkeypatch, name)
    args = {"bot_id": "private_fixture", "user_id": user}
    if name in {"messages_get", "sessions_get", "sessions_close"}:
        args["session_id"] = "s1"
    with service_call(name, args):
        await server._tool_manager.call_tool(name, args)
    assert calls == [("private_fixture", user, args.get("session_id"))]


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["messages_get", "sessions_get", "sessions_close",
                                 "sessions_get_active", "sessions_list"])
@pytest.mark.parametrize("target", ["public_fixture", "other_private"])
async def test_userless_service_filter_cannot_target_another_bot(monkeypatch, signer, name, target):
    server, calls = catalog(monkeypatch, name)
    args = {"bot_id": target}
    with service_call(name, args), pytest.raises(Exception, match="access_denied"):
        await server._tool_manager.call_tool(name, args)
    assert not calls


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["messages_add", "memory_store", "memory_update", "facts_extract",
                                 "sessions_get_or_create_active", "sessions_rotate"])
async def test_userless_service_never_grants_user_authorship(monkeypatch, signer, name):
    server, calls = catalog(monkeypatch, name)
    args = {"bot_id": "private_fixture", "user_id": "user_a"}
    with service_call(name, args), pytest.raises(Exception, match="access_denied"):
        await server._tool_manager.call_tool(name, args)
    assert not calls


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["messages_get", "sessions_get", "sessions_close"])
async def test_userless_service_session_owner_must_still_match_bot(monkeypatch, signer, name):
    server, calls = catalog(monkeypatch, name, owner=("public_fixture", "user_a"))
    args = {"bot_id": "private_fixture", "session_id": "other"}
    with service_call(name, args), pytest.raises(Exception, match="access_denied"):
        await server._tool_manager.call_tool(name, args)
    assert not calls


@pytest.mark.asyncio
async def test_userless_service_cannot_accept_another_bots_pointer(monkeypatch, signer):
    server, calls = catalog(monkeypatch, "sessions_rotate")
    args = {"bot_id": "public_fixture", "accept_pointer": True}
    with service_call("sessions_rotate", args), pytest.raises(Exception, match="access_denied"):
        await server._tool_manager.call_tool("sessions_rotate", args)
    assert not calls


@pytest.mark.asyncio
async def test_userless_service_still_cannot_search_aggregate_memories(monkeypatch, signer):
    server, calls = catalog(monkeypatch, "memory_search_all")
    args = {"bot_id": "private_fixture"}
    with service_call("memory_search_all", args), pytest.raises(Exception, match="caller_unavailable"):
        await server._tool_manager.call_tool("memory_search_all", args)
    assert not calls


@pytest.mark.parametrize("kind", ["turn", "approved"])
def test_nonservice_identity_requires_a_user(kind):
    with pytest.raises(BotAccessError, match="caller_unavailable"):
        TrustedCaller("private_fixture", None, kind).validate()


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["messages_get", "sessions_get", "sessions_close",
                                 "sessions_get_active", "sessions_list"])
async def test_user_bound_service_does_not_gain_filter_authority(monkeypatch, signer, name):
    server, calls = catalog(monkeypatch, name)
    args = {"bot_id": "private_fixture", "user_id": "user_b"}
    with service_call(name, args, user="user_a"), pytest.raises(Exception, match="access_denied"):
        await server._tool_manager.call_tool(name, args)
    assert not calls


@pytest.mark.parametrize("caller,target,expected", [
    (None, "private_fixture", False),
    (TrustedCaller("private_fixture", None, "service"), "private_fixture", True),
    (TrustedCaller("private_fixture", None, "service"), "public_fixture", False),
    (TrustedCaller("private_fixture", "user_a", "service"), "private_fixture", False),
    (TrustedCaller("private_fixture", "user_a", "turn"), "private_fixture", False),
    (TrustedCaller("private_fixture", "user_a", "approved"), "private_fixture", False),
])
def test_service_self_predicate_has_no_implicit_authority(caller, target, expected):
    from llm_bawt.bot_access import BotPolicyResolver
    from llm_bawt.bot_access.service import is_service_self
    from tests.test_bot_access_mcp import ROWS

    assert is_service_self(BotPolicyResolver._compile(ROWS, caller), target) is expected


def test_service_self_revalidates_expiry_and_has_exact_operation_allowlists():
    from llm_bawt.bot_access import BotPolicyResolver
    from llm_bawt.bot_access.service import SERVICE_SELF_STORAGE_OPS, SERVICE_SELF_TOOLS, is_service_self
    from tests.test_bot_access_mcp import ROWS

    access = BotPolicyResolver._compile(ROWS, TrustedCaller("private_fixture", None, "service", 1.0))
    with pytest.raises(BotAccessError, match="caller_unavailable"):
        is_service_self(access, "private_fixture")
    assert SERVICE_SELF_TOOLS == {"messages_get", "sessions_get", "sessions_close",
                                  "sessions_list", "sessions_get_active"}
    assert SERVICE_SELF_STORAGE_OPS == {"get_messages", "get_session", "close_session",
                                       "list_sessions", "get_active_session"}


@pytest.mark.asyncio
async def test_all_session_tools_require_explicit_bot_namespace():
    from llm_bawt.mcp_server.registry import ensure_tools_registered, mcp

    ensure_tools_registered()
    tools = {tool.name: tool for tool in await mcp.list_tools()}
    names = {"sessions_get", "sessions_close", "sessions_list", "sessions_get_active",
             "sessions_get_or_create_active", "sessions_rotate"}
    for name in names:
        assert "bot_id" in tools[name].inputSchema["required"], name
        assert "required bot_id" in tools[name].description, name
