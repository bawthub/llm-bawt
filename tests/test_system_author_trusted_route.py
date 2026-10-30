"""TASK-1807 Phase 1: system authors are app-written, service-fenced and readable.

Covers the trusted write route in both MemoryClient modes (embedded in-process
storage; server mode through the signed service transport into the real
``messages_add`` tool), the fence against every other caller, and the
read-side normalization/classification of persisted system rows.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from llm_bawt.bot_access.service import ServicePrincipal
from llm_bawt.entity_presentation import EntityPresentationResolver
from llm_bawt.mcp_server import server as mcp_server_module
from llm_bawt.mcp_server.bot_access import invocation_context
from llm_bawt.mcp_server.bot_access_identity import _service_capability
from llm_bawt.mcp_server.catalog import CatalogFastMCP
from llm_bawt.mcp_server.client import MemoryClient, _MCPShortTermManager
from llm_bawt.mcp_server.storage import MemoryStorage
from llm_bawt.memory.speaker_attribution import TOOL, USER, extraction_input, speaker_of
from llm_bawt.message_authorship import (
    SYSTEM_AUTHORS,
    AuthorReference,
    LegacyAuthorResolver,
    normalize_author,
    normalize_persisted_author,
    require_system_author,
)
from tests.test_bot_access_mcp import Resolver, identity, signer, turn  # noqa: F401

MCP_ID = "approval-continuation:mcp"
HARNESS_ID = "approval-continuation:harness"
BOT, USER_ID = "private_fixture", "user_a"


# ---------------------------------------------------------------- author model

def test_system_reference_is_allowlisted():
    assert set(SYSTEM_AUTHORS) == {MCP_ID, HARNESS_ID}
    assert AuthorReference.system(MCP_ID).to_dict() == {
        "entity_type": "system", "entity_id": MCP_ID, "status": "canonical"}
    with pytest.raises(ValueError):
        AuthorReference.system("approval-continuation:other")


def test_caller_facing_normalizer_stays_user_or_bot():
    with pytest.raises(ValueError):
        normalize_author("system", MCP_ID)


@pytest.mark.parametrize("pair,expected", [
    (("system", MCP_ID), AuthorReference("system", MCP_ID)),
    (("SYSTEM", HARNESS_ID.upper()), AuthorReference("system", HARNESS_ID)),
    (("user", "Nick_Fixture"), AuthorReference("user", "nick_fixture")),
    (("bot", "byte_fixture"), AuthorReference("bot", "byte_fixture")),
    ((None, None), None),
])
def test_persisted_normalizer_accepts_allowlisted_system(pair, expected):
    assert normalize_persisted_author(*pair) == expected


def test_persisted_normalizer_rejects_unknown_system_id():
    with pytest.raises(ValueError):
        normalize_persisted_author("system", "forged")


@pytest.mark.parametrize("value", [
    AuthorReference.user("user_a"), AuthorReference.bot("bot_a"),
    AuthorReference("system", "forged"), "system", None,
])
def test_require_system_author_rejects_everything_else(value):
    with pytest.raises(ValueError):
        require_system_author(value)


def test_legacy_resolver_reads_explicit_system_rows_without_db():
    rows = [{"id": "m1", "role": "user", "author_entity_type": "system",
             "author_entity_id": MCP_ID}]
    assert LegacyAuthorResolver(engine=None).resolve(rows, bot_id=BOT) == {
        "m1": AuthorReference("system", MCP_ID)}


# ---------------------------------------------------------------- readers

def test_speaker_of_system_row_is_tool_even_without_envelope():
    row = {"role": "user", "content": "plain text, no envelope",
           "author_entity_type": "system", "author_entity_id": HARNESS_ID}
    human = {"role": "user", "content": "hi", "author_entity_type": "user",
             "author_entity_id": USER_ID}
    assert speaker_of(row).kind == TOOL
    assert speaker_of({"role": "user", "content": "x",
                       "author": AuthorReference.system(MCP_ID)}).kind == TOOL
    assert speaker_of(human).kind == USER
    assert extraction_input([row, human]) == [human]


def test_presentation_of_system_author_is_fixed_and_lookup_free():
    resolved = EntityPresentationResolver(engine=None).resolve_many(
        [AuthorReference.system(MCP_ID)])
    presentation = resolved[("system", MCP_ID)].to_dict()
    assert presentation["display_name"] == SYSTEM_AUTHORS[MCP_ID]
    assert presentation["entity_type"] == "system"
    assert presentation["status"] == "canonical"


# ---------------------------------------------------------------- storage

class _Backend:
    def __init__(self, rows=()):
        self.writes, self.rows = [], list(rows)

    def add_message(self, **kwargs):
        self.writes.append(kwargs)

    def get_unprocessed_messages(self, limit=100):
        return self.rows


def _storage(backend):
    storage = object.__new__(MemoryStorage)
    storage._get_backend = lambda bot_id, provision_schema=False: backend
    return storage


def test_storage_trusted_author_writes_system(monkeypatch):
    backend = _Backend()
    message = asyncio.run(_storage(backend).add_message(
        role="assistant", content="x", bot_id=BOT, session_id="s1",
        trusted_author=AuthorReference.system(MCP_ID)))
    assert backend.writes[0]["author"] == AuthorReference("system", MCP_ID)
    assert message.author == AuthorReference("system", MCP_ID)


@pytest.mark.parametrize("kwargs", [
    {"trusted_author": AuthorReference.user("user_a")},
    {"trusted_author": AuthorReference.system(MCP_ID), "author_entity_type": "user",
     "author_entity_id": "user_a"},
    {"author_entity_type": "system", "author_entity_id": MCP_ID},
])
def test_storage_refuses_untrusted_or_mixed_authors(kwargs):
    backend = _Backend()
    with pytest.raises(ValueError):
        asyncio.run(_storage(backend).add_message(
            role="assistant", content="x", bot_id=BOT, session_id="s1", **kwargs))
    assert backend.writes == []


def test_unprocessed_reader_hydrates_system_rows():
    backend = _Backend([{"id": "m1", "role": "user", "content": "x", "timestamp": 1.0,
                         "author_entity_type": "system", "author_entity_id": MCP_ID}])
    [message] = asyncio.run(_storage(backend).get_unprocessed_messages(bot_id=BOT))
    assert message.author == AuthorReference("system", MCP_ID)


# ---------------------------------------------------------------- MCP fence

class _FakeStorage:
    def __init__(self):
        self.calls = []

    async def add_message(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(to_dict=lambda: {"id": "m1", "role": kwargs["role"],
                                                "content": kwargs["content"]})


@pytest.fixture
def fenced(monkeypatch):
    storage = _FakeStorage()
    monkeypatch.setattr(mcp_server_module, "_get_storage", lambda: storage)
    server = CatalogFastMCP("system-author", bot_policy_resolver=Resolver())
    server.add_tool(mcp_server_module.add_message, name="messages_add")
    return server, storage


@contextmanager
def service_token(token):
    binding = _service_capability.set(token)
    try:
        yield
    finally:
        _service_capability.reset(binding)


def _args(author_id=MCP_ID, bot=BOT, **extra):
    return {"role": "user", "content": "envelope", "bot_id": bot, "user_id": USER_ID,
            "author_entity_type": "system", "author_entity_id": author_id, **extra}


async def _call(server, args):
    with invocation_context("messages_add", args):  # as CatalogFastMCP.call_tool binds it
        return await server._tool_manager.call_tool("messages_add", args)


@pytest.mark.asyncio
async def test_service_principal_writes_system_author(fenced, signer):
    server, storage = fenced
    args = _args()
    with service_token(ServicePrincipal(BOT, USER_ID).mint("messages_add", args)):
        await _call(server, args)
    [call] = storage.calls
    assert call["trusted_author"] == AuthorReference("system", MCP_ID)
    assert call["author_entity_type"] is None and call["author_entity_id"] is None


@pytest.mark.asyncio
async def test_agent_turn_cannot_write_system_author(fenced, signer):
    server, storage = fenced
    with identity(turn(BOT, USER_ID)), pytest.raises(Exception, match="access_denied"):
        await _call(server, _args())
    assert storage.calls == []


@pytest.mark.asyncio
async def test_anonymous_caller_cannot_write_system_author(fenced):
    server, storage = fenced
    with pytest.raises(Exception, match="access_denied"):
        await _call(server, _args(bot="public_fixture"))
    assert storage.calls == []


@pytest.mark.asyncio
async def test_service_cannot_use_unlisted_system_id(fenced, signer):
    server, storage = fenced
    args = _args(author_id="forged")
    with service_token(ServicePrincipal(BOT, USER_ID).mint("messages_add", args)), \
            pytest.raises(Exception, match="access_denied"):
        await _call(server, args)
    assert storage.calls == []


@pytest.mark.asyncio
async def test_service_of_another_bot_cannot_write_system_author(fenced, signer):
    server, storage = fenced
    args = _args(bot="public_fixture")
    with service_token(ServicePrincipal("other_fixture", USER_ID).mint("messages_add", args)), \
            pytest.raises(Exception, match="access_denied|caller_unavailable"):
        await _call(server, args)
    assert storage.calls == []


@pytest.mark.asyncio
async def test_service_token_replayed_onto_other_args_is_refused(fenced, signer):
    server, storage = fenced
    minted_for = _args(author_entity_type="user", author_entity_id=USER_ID)
    replayed = _args()
    with service_token(ServicePrincipal(BOT, USER_ID).mint("messages_add", minted_for)), \
            pytest.raises(Exception, match="caller_unavailable"):
        await _call(server, replayed)
    assert storage.calls == []


@pytest.mark.asyncio
async def test_ordinary_user_author_still_passes_through(fenced, signer):
    server, storage = fenced
    args = _args(author_entity_type="user", author_entity_id=USER_ID)
    with identity(turn(BOT, USER_ID)):
        await _call(server, args)
    [call] = storage.calls
    assert call["trusted_author"] is None
    assert (call["author_entity_type"], call["author_entity_id"]) == ("user", USER_ID)


# ---------------------------------------------------------------- client modes

def test_server_mode_continuation_write_round_trips_through_fence(fenced, signer, monkeypatch):
    """Adapter → signed MemoryClient → real messages_add tool → storage."""
    server, storage = fenced
    client = MemoryClient(SimpleNamespace(), bot_id=BOT, user_id=USER_ID,
                          server_url="http://mcp.invalid")

    def call_server(method, params):
        with service_token(client._service_principal.mint(method, params)):
            asyncio.run(_call(server, params))
        return {"id": "m1", "role": params["role"], "content": params["content"]}

    monkeypatch.setattr(client, "_call_server", call_server)
    _MCPShortTermManager(client).add_message(
        role="user", content="envelope", author=AuthorReference.system(HARNESS_ID))
    [call] = storage.calls
    assert call["trusted_author"] == AuthorReference("system", HARNESS_ID)


def test_embedded_mode_hands_trusted_object_to_storage(monkeypatch):
    storage = _FakeStorage()
    client = MemoryClient(SimpleNamespace(), bot_id=BOT, user_id=USER_ID)
    client._storage, client._initialized = storage, True
    client.add_message(role="user", content="envelope",
                       trusted_author=AuthorReference.system(MCP_ID))
    [call] = storage.calls
    assert call["trusted_author"] == AuthorReference("system", MCP_ID)
    assert call["author_entity_type"] is None


def test_adapter_keeps_human_and_bot_authors_on_the_narrow_strings(monkeypatch):
    seen = []
    client = SimpleNamespace(add_message=lambda **kw: seen.append(kw) or SimpleNamespace(id="m"))
    for author in (AuthorReference.user(USER_ID), AuthorReference.bot("bot_a")):
        _MCPShortTermManager(client).add_message(role="user", content="x", author=author)
    assert [(kw["author_entity_type"], "trusted_author" in kw) for kw in seen] == [
        ("user", False), ("bot", False)]


@pytest.mark.parametrize("bad", [AuthorReference.user(USER_ID), AuthorReference("system", "forged")])
def test_client_refuses_non_system_trusted_author(bad):
    client = MemoryClient(SimpleNamespace(), bot_id=BOT, user_id=USER_ID)
    client._storage, client._initialized = _FakeStorage(), True
    with pytest.raises(ValueError):
        client.add_message(role="user", content="x", trusted_author=bad)
