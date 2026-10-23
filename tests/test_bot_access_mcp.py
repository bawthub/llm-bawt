"""Exercise the registered callable, approved dispatch, and signed identity adapters."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from cryptography.fernet import Fernet

from agent_bridge.mcp_call_context import canonical_invocation_hash
from llm_bawt import task_turn_context as turn_codec
from llm_bawt.bot_access import BotAccessError, BotPolicyResolver, TrustedCaller
from llm_bawt.bot_access.context import current_bot_access
from llm_bawt.bot_access import service as service_codec
from llm_bawt.bot_access.service import ServicePrincipal
from llm_bawt.mcp_server.approval_interceptor import ApprovedCallerContext
from llm_bawt.mcp_server.bot_access_identity import (
    BotAccessMiddleware, _service_capability, trusted_caller,
)
from llm_bawt.mcp_server.bot_access_manifest import BOT_TARGET_TOOLS, CLASSIFIED_TOOLS
from llm_bawt.mcp_server.catalog import CatalogFastMCP
from llm_bawt.mcp_server.task_association import (
    set_current_task_turn_capability, reset_current_task_turn_capability,
)

ROWS = [{"slug": "public_fixture", "include_in_global_search": True},
        {"slug": "private_fixture", "include_in_global_search": False}]


class Resolver:
    def __init__(self):
        self.calls = 0
        self.rows = [dict(row) for row in ROWS]

    async def resolve(self, caller=None):
        self.calls += 1
        return BotPolicyResolver._compile(self.rows, caller)


@pytest.fixture
def signer(monkeypatch):
    signer = Fernet(Fernet.generate_key())
    monkeypatch.setattr(turn_codec, "_get_fernet", lambda: signer)
    monkeypatch.setattr(service_codec, "_get_fernet", lambda: signer)
    return signer


def turn(bot="private_fixture", user="user_a"):
    return turn_codec.mint_task_turn_context(
        session_id=str(uuid.uuid4()), turn_id="turn-" + uuid.uuid4().hex,
        trigger_message_id=str(uuid.uuid4()), bot_id=bot, user_id=user,
    )


@contextmanager
def identity(token):
    binding = set_current_task_turn_capability(token)
    try:
        yield
    finally:
        reset_current_task_turn_capability(binding)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", sorted(BOT_TARGET_TOOLS))
async def test_every_bot_target_registered_handler_denies_before_execution(name, signer):
    server = CatalogFastMCP("test", bot_policy_resolver=Resolver())
    calls = []

    async def handler(bot_id: str = "private_fixture") -> str:
        calls.append(bot_id)
        return bot_id

    server.add_tool(handler, name=name)
    with pytest.raises(Exception, match="access_denied"):
        await server._tool_manager.call_tool(name, {})
    assert calls == []
    with identity(turn()):
        await server._tool_manager.call_tool(name, {})
    assert calls == ["private_fixture"]
    assert current_bot_access() is None


@pytest.mark.asyncio
async def test_approved_execution_checks_current_policy_and_context_conflicts(signer):
    resolver = Resolver()
    server = CatalogFastMCP("test", bot_policy_resolver=resolver)
    calls = []

    async def handler(bot_id: str) -> str:
        calls.append(bot_id)
        return bot_id

    server.add_tool(handler, name="messages_get")
    server.mark_tools_registered()
    arguments = {"bot_id": "public_fixture"}
    approved = ApprovedCallerContext(bot_id="private_fixture", user_id="user_a")
    await server.call_approved_tool("messages_get", arguments,
        expected_invocation_hash=canonical_invocation_hash("messages_get", arguments), caller_context=approved)
    resolver.rows[0]["include_in_global_search"] = False
    with pytest.raises(Exception, match="access_denied"):
        await server.call_approved_tool("messages_get", arguments,
            expected_invocation_hash=canonical_invocation_hash("messages_get", arguments), caller_context=approved)
    arguments = {"bot_id": "private_fixture"}
    for credential in ["garbage", turn("public_fixture"), turn(user="user_b")]:
        with identity(credential), pytest.raises(Exception, match="caller_unavailable"):
            await server.call_approved_tool("messages_get", arguments,
                expected_invocation_hash=canonical_invocation_hash("messages_get", arguments), caller_context=approved)
    assert calls == ["public_fixture"]
    assert trusted_caller("messages_get", {}) is None


@pytest.mark.asyncio
async def test_preflight_denial_precedes_approval_store(signer):
    def store():
        pytest.fail("denied request reached approval store")
    server = CatalogFastMCP("test", bot_policy_resolver=Resolver(), approval_store_provider=store)

    async def handler(bot_id: str):
        pytest.fail("denied request reached handler")

    server.add_tool(handler, name="memory_clear")
    with pytest.raises(BotAccessError, match="access_denied"):
        await server.call_tool("memory_clear", {"bot_id": "private_fixture"})


@pytest.mark.parametrize("change", ["tool", "args", "purpose", "expired", "turn-purpose"])
def test_service_envelope_is_purpose_expiry_and_invocation_bound(signer, change):
    principal = ServicePrincipal("private_fixture", "user_a")
    arguments = {"bot_id": "private_fixture"}
    token = principal.mint("messages_get", arguments)
    assert service_codec.open_service_capability(token, "messages_get", arguments).bot_id == "private_fixture"
    name = "messages_get"
    if change == "tool":
        name = "memory_clear"
    elif change == "args":
        arguments = {"bot_id": "public_fixture"}
    elif change == "turn-purpose":
        token = turn()
    else:
        body = signer.decrypt(token.encode())
        if change == "purpose":
            data = json.loads(body)
            data["purpose"] = "task-turn"
            token = signer.encrypt(json.dumps(data).encode()).decode()
        else:
            token = signer.encrypt_at_time(body, int(time.time()) - 121).decode()
    with pytest.raises(BotAccessError, match="caller_unavailable"):
        service_codec.open_service_capability(token, name, arguments)


@pytest.mark.asyncio
@pytest.mark.parametrize("values", [[b""], [b"\xff"], [b"first", b"second"]])
async def test_supplied_invalid_transport_identity_never_becomes_anonymous(values):
    async def app(scope, receive, send):
        with pytest.raises(BotAccessError, match="caller_unavailable"):
            trusted_caller("messages_get", {})
    scope = {"type": "http", "headers": [(turn_codec.TASK_TURN_CONTEXT_HEADER.lower().encode(), v) for v in values]}
    await BotAccessMiddleware(app)(scope, None, None)
    assert trusted_caller("messages_get", {}) is None


@pytest.mark.asyncio
async def test_concurrent_principals_and_failure_cleanup(signer):
    server = CatalogFastMCP("test", bot_policy_resolver=Resolver())
    seen = []
    arrived = 0
    barrier = asyncio.Event()

    async def handler(bot_id: str) -> str:
        nonlocal arrived
        arrived += 1
        if arrived == 2:
            barrier.set()
        await barrier.wait()
        seen.append((bot_id, current_bot_access().caller.bot_id))
        if bot_id == "private_fixture":
            raise ValueError("handler failure")
        return bot_id

    server.add_tool(handler, name="messages_get")

    async def invoke(bot):
        with identity(turn(bot)):
            try:
                await server._tool_manager.call_tool("messages_get", {"bot_id": bot})
            except Exception as error:
                assert "handler failure" in str(error)
            assert current_bot_access() is None
    await asyncio.gather(invoke("private_fixture"), invoke("public_fixture"))
    assert sorted(seen) == [("private_fixture", "private_fixture"), ("public_fixture", "public_fixture")]


@pytest.mark.asyncio
async def test_registered_catalog_requires_explicit_classification():
    from llm_bawt.mcp_server.registry import ensure_tools_registered, mcp
    ensure_tools_registered()
    registered = {tool.name for tool in await mcp.list_tools()}
    assert registered <= CLASSIFIED_TOOLS


@pytest.mark.asyncio
async def test_service_principal_never_bypasses_approval_and_raw_arguments_bind(signer):
    from agent_bridge.approval import ApprovalPolicy, MatcherType, PolicyAction
    from tests.test_mcp_approval_interceptor import FakeStore

    calls = []
    for action in (PolicyAction.ALLOW, PolicyAction.DENY, PolicyAction.REQUIRE_APPROVAL):
        policy = ApprovalPolicy(id="p", tool_name="messages_get", matcher_type=MatcherType.ALWAYS, action=action)
        store = FakeStore(policy, [])
        server = CatalogFastMCP("test", bot_policy_resolver=Resolver(), approval_store_provider=lambda: store)

        async def handler(bot_id: str, limit: int = 3) -> str:
            calls.append((bot_id, limit))
            return "ok"

        server.add_tool(handler, name="messages_get")
        arguments = {"bot_id": "private_fixture"}  # default expansion must not break hash
        token = _service_capability.set(ServicePrincipal("private_fixture", "user_a").mint("messages_get", arguments))
        try:
            result = await server.call_tool("messages_get", arguments)
            if action is PolicyAction.DENY:
                assert result["status"] == "denied"
            elif action is PolicyAction.REQUIRE_APPROVAL:
                assert result["status"] == "approval_context_missing"
            assert not store.rows
        finally:
            _service_capability.reset(token)
    assert calls == [("private_fixture", 3)]


@pytest.mark.asyncio
@pytest.mark.parametrize("name,arguments", [
    ("memory_search_source", {"source": "private_fixture", "query": "q"}),
    ("self_fwd", {"sender_bot_id": "private_fixture", "target_bot_id": "public_fixture"}),
    ("profile", {"entity_type": "bot", "entity_id": "private_fixture", "action": "get"}),
    ("sessions_list", {"bot_id": "private_fixture"}),
])
async def test_actual_special_tool_denial_precedes_handler(monkeypatch, name, arguments):
    from llm_bawt.mcp_server.registry import ensure_tools_registered, mcp
    ensure_tools_registered()
    monkeypatch.setattr(mcp.bot_access, "_resolver", Resolver())
    with pytest.raises(Exception, match="access_denied"):
        await mcp._tool_manager.call_tool(name, arguments)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["sessions_get", "sessions_close", "messages_add"])
async def test_session_indirection_denied_before_registered_handler(monkeypatch, signer, name):
    catalog = CatalogFastMCP("session-proof", bot_policy_resolver=Resolver())
    calls = []

    async def handler(bot_id: str, session_id: str):
        calls.append(session_id)

    catalog.add_tool(handler, name=name)
    monkeypatch.setattr(catalog.bot_access, "_session_owner", lambda id: ("private_fixture", "user_a"))
    with pytest.raises(Exception, match="access_denied"):
        await catalog._tool_manager.call_tool(name, {"bot_id": "public_fixture", "session_id": "hidden"})
    with identity(turn(user="user_b")), pytest.raises(Exception, match="access_denied"):
        await catalog._tool_manager.call_tool(name, {"bot_id": "private_fixture", "session_id": "hidden"})
    assert not calls


@pytest.mark.asyncio
async def test_protected_session_defaults_pin_user_and_reject_pointer_override(monkeypatch, signer):
    catalog = CatalogFastMCP("session-user", bot_policy_resolver=Resolver())
    seen = []

    async def handler(bot_id: str, user_id: str | None = None, accept_pointer: bool = False):
        seen.append((bot_id, user_id))

    catalog.add_tool(handler, name="sessions_rotate")
    with identity(turn()):
        await catalog._tool_manager.call_tool("sessions_rotate", {"bot_id": "private_fixture"})
        with pytest.raises(Exception, match="access_denied"):
            await catalog._tool_manager.call_tool("sessions_rotate", {"bot_id": "private_fixture", "user_id": "user_b"})
    assert seen == [("private_fixture", "user_a")]
