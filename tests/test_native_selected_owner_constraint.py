"""Native-selected envelope authentication and statement-snapshot store predicate.

Hermetic only: in-memory signer, SQLite row semantics, PostgreSQL SQL compilation.
No services, configured keys, inference, or live database access.
"""
from __future__ import annotations

import asyncio
import inspect
import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
from sqlalchemy import (Boolean, Column, DateTime, Float, JSON, MetaData, String,
                        Table, create_engine, event, select, update)
from sqlalchemy.dialects import postgresql

from llm_bawt.bot_access import BotAccessError, BotPolicyResolver
from llm_bawt.bot_access import service as service_codec
from llm_bawt.memory import message_store
from llm_bawt.memory.message_store import (
    MessageRowStore, NativeSelectedOwnerDenied, _NativeSelectedOwnerConstraint,
)
from llm_bawt.memory.postgresql_messages import PostgreSQLMessageMixin
from llm_bawt.mcp_server import bot_access_identity as identity
from llm_bawt.mcp_server.bot_access import ToolBotAccess, current_native_selected_constraint
from llm_bawt.mcp_server.storage import MemoryStorage
from llm_bawt.message_authorship import AuthorReference
from tests.test_native_selected_owner_provenance import (
    BOT, USER, SID, OTHER_SID, MID, _arguments, _llm, _proven_request, _take,
    _window, signer, turn_codec,
)

__all__ = ["signer"]  # re-export the imported pytest fixture


@contextmanager
def envelope(token):
    binding = identity._service_capability.set(token)
    try:
        yield
    finally:
        identity._service_capability.reset(binding)


def native_envelope(arguments):
    request, llm = _proven_request(), _llm()
    selected = _take(request, llm)
    with _window(selected, request, llm, "initial"):
        return service_codec.ServicePrincipal(BOT, USER).mint("messages_add", arguments)


def guards(monkeypatch, *, private=True, failure=False):
    class Resolver:
        async def resolve(self, caller):
            return BotPolicyResolver._compile([
                {"slug": BOT, "include_in_global_search": not private},
            ], caller)

    guard = ToolBotAccess(Resolver())
    owner = Mock(side_effect=AssertionError("unexpected owner roundtrip"))
    monkeypatch.setattr(guard, "_session_owner", owner)
    seen = []

    async def handler(bot_id: str, user_id: str, session_id: str,
                      role: str, content: str):
        seen.append(current_native_selected_constraint())
        if failure:
            raise ValueError("handler failure")
        return "ok"

    fn = guard.register("messages_add", handler)
    return guard, fn, owner, seen


def test_marker_is_native_only_nested_and_restored(signer):
    args = _arguments()
    request, llm = _proven_request(), _llm()
    selected = _take(request, llm)
    principal = service_codec.ServicePrincipal(BOT, USER)
    with service_codec.bind_service_turn(selected.capability):
        ordinary = principal.mint("messages_add", args)
        assert not service_codec.open_service_native_selected(ordinary, "messages_add", args)
        with pytest.raises(RuntimeError, match="cleanup"):
            with _window(selected, request, llm, "initial"):
                native = principal.mint("messages_add", args)
                assert service_codec.open_service_native_selected(native, "messages_add", args)
                with service_codec.bind_service_turn(selected.capability):
                    nested = principal.mint("messages_add", args)
                    assert not service_codec.open_service_native_selected(nested, "messages_add", args)
                assert service_codec._runtime_native_selected.get() is True
                raise RuntimeError("cleanup")
        assert service_codec._runtime_turn.get() == selected.capability
        assert service_codec._runtime_native_selected.get() is False
    assert service_codec._runtime_turn.get() is None
    assert service_codec._runtime_native_selected.get() is False
    # Raw arguments cannot create a native proof, even with a matching turn.
    raw_args = {**args, "native_selected_history": True}
    with service_codec.bind_service_turn(selected.capability):
        raw = principal.mint("messages_add", raw_args)
    assert not service_codec.open_service_native_selected(raw, "messages_add", raw_args)


@pytest.mark.parametrize("private", [True, False])
def test_both_guards_authenticate_and_deliver_private_constraint(monkeypatch, signer, private):
    args = _arguments()
    guard, fn, owner, seen = guards(monkeypatch, private=private)
    token = native_envelope(args)
    with envelope(token):
        asyncio.run(guard.preflight("messages_add", args))
        assert current_native_selected_constraint() is None
        assert asyncio.run(fn(**args)) == "ok"
    assert seen == [_NativeSelectedOwnerConstraint(SID, BOT, USER, private)]
    assert current_native_selected_constraint() is None
    owner.assert_not_called()
    assert "_native_selected_owner" not in inspect.signature(fn).parameters


@pytest.mark.parametrize("change", ["false", "null", "integer", "object", "no-proof",
                                    "bad-proof", "other-session", "expired-proof", "forged", "invocation"])
def test_bad_marker_denied_at_both_guards_without_fallback(monkeypatch, signer, change):
    args = _arguments()
    token = native_envelope(args)
    body = json.loads(signer.decrypt(token.encode()))
    if change in {"false", "null", "integer", "object"}:
        body["native_selected_history"] = {"false": False, "null": None, "integer": 1,
                                            "object": {"enabled": True}}[change]
    elif change == "no-proof":
        body.pop("conversation_context")
    elif change == "bad-proof":
        body["conversation_context"] = "not-a-proof"
    elif change == "other-session":
        body["conversation_context"] = turn_codec.mint_task_turn_context(
            session_id=OTHER_SID, bot_id=BOT, user_id=USER,
            turn_id="turn-" + "b" * 32, trigger_message_id=MID,
        )
    elif change == "expired-proof":
        raw = signer.decrypt(body["conversation_context"].encode())
        body["conversation_context"] = signer.encrypt_at_time(raw, 1).decode()
    token = signer.encrypt(json.dumps(body).encode()).decode()
    if change == "forged":
        token = token[:-8] + "forged!!"
    elif change == "invocation":
        args = {**args, "content": "different"}
    guard, fn, owner, seen = guards(monkeypatch)
    with envelope(token):
        for call in (lambda: guard.preflight("messages_add", args), lambda: fn(**args)):
            with pytest.raises(BotAccessError, match="^caller_unavailable$"):
                asyncio.run(call())
    assert not seen
    owner.assert_not_called()
    assert current_native_selected_constraint() is None


@pytest.mark.parametrize("kind", ["turn", "approved"])
@pytest.mark.parametrize("content", ["hello", " "])
def test_matching_other_proof_class_cannot_launder_native_marker(monkeypatch, signer, kind, content):
    args = _arguments(content=content)
    token = native_envelope(args)
    proof = turn_codec.mint_task_turn_context(
        session_id=SID, bot_id=BOT, user_id=USER,
        turn_id="turn-" + "b" * 32, trigger_message_id=MID,
    )
    if kind == "turn":
        monkeypatch.setattr(identity, "current_task_turn_capability", lambda: proof)
    else:
        monkeypatch.setattr(identity, "current_approved_caller_context",
                            lambda: SimpleNamespace(bot_id=BOT, user_id=USER))
    guard, fn, owner, seen = guards(monkeypatch)
    with envelope(token):
        for call in (lambda: guard.preflight("messages_add", args), lambda: fn(**args)):
            with pytest.raises(BotAccessError, match="^caller_unavailable$"):
                asyncio.run(call())
    owner.assert_not_called()
    assert not seen


@pytest.mark.parametrize("failure", [False, True])
def test_tool_constraint_nested_binding_restores_after_success_or_failure(monkeypatch, signer, failure):
    from llm_bawt.mcp_server.bot_access import _native_selected_write

    args = _arguments()
    guard, fn, owner, seen = guards(monkeypatch, failure=failure)
    parent = _NativeSelectedOwnerConstraint(OTHER_SID, BOT, USER, True)
    binding = _native_selected_write.set(parent)
    try:
        with envelope(native_envelope(args)):
            if failure:
                with pytest.raises(ValueError, match="handler failure"):
                    asyncio.run(fn(**args))
            else:
                asyncio.run(fn(**args))
        assert current_native_selected_constraint() is parent
    finally:
        _native_selected_write.reset(binding)
    assert current_native_selected_constraint() is None
    assert seen == [_NativeSelectedOwnerConstraint(SID, BOT, USER, True)]
    owner.assert_not_called()


@pytest.mark.parametrize("kind", ["agent", "reset", "turn", "approved"])
def test_other_existing_proof_classes_do_not_gain_constraint(monkeypatch, signer, kind):
    args = _arguments()
    request, llm = _proven_request(), _llm()
    selected = _take(request, llm)
    with service_codec.bind_service_turn(selected.capability):
        token = service_codec.ServicePrincipal(BOT, USER).mint("messages_add", args)
    assert not service_codec.open_service_native_selected(token, "messages_add", args)
    guard, fn, owner, seen = guards(monkeypatch)
    if kind == "turn":
        monkeypatch.setattr(identity, "current_task_turn_capability", lambda: selected.capability)
        token = None
    elif kind == "approved":
        monkeypatch.setattr(identity, "current_approved_caller_context",
                            lambda: SimpleNamespace(bot_id=BOT, user_id=USER))
        owner.side_effect = None
        owner.return_value = (BOT, USER)
    with envelope(token):
        asyncio.run(guard.preflight("messages_add", args))
        asyncio.run(fn(**args))
    assert seen == [None]
    if kind == "approved":
        assert owner.call_count == 2  # approved is not service-owner authority
    else:
        owner.assert_not_called()


@pytest.mark.parametrize("content", ["", " \n\t"])
@pytest.mark.parametrize("exists", [True, False])
def test_native_blank_uses_authoritative_fallback_at_both_guards(monkeypatch, signer, content, exists):
    args = _arguments(content=content)
    token = native_envelope(args)
    assert not service_codec.open_service_native_selected(token, "messages_add", args)
    assert service_codec.open_service_session_owner(token, "messages_add", args) is None
    guard, fn, owner, seen = guards(monkeypatch)
    owner.side_effect = None
    owner.return_value = (BOT, USER) if exists else None
    with envelope(token):
        if exists:
            asyncio.run(guard.preflight("messages_add", args))
            asyncio.run(fn(**args))
        else:
            for call in (lambda: guard.preflight("messages_add", args), lambda: fn(**args)):
                with pytest.raises(BotAccessError, match="^access_denied$"):
                    asyncio.run(call())
    assert owner.call_count == 2
    assert seen == ([None] if exists else [])


@pytest.mark.parametrize("value", [0, False, "expired"])
def test_blank_fallback_still_rejects_expired_or_malformed_signed_expiry(monkeypatch, signer, value):
    args = _arguments(content=" ")
    token = native_envelope(args)
    body = json.loads(signer.decrypt(token.encode()))
    assert type(body["conversation_expires_at"]) is int
    body["conversation_expires_at"] = value
    token = signer.encrypt(json.dumps(body).encode()).decode()
    guard, fn, owner, seen = guards(monkeypatch)
    with envelope(token):
        for call in (lambda: guard.preflight("messages_add", args), lambda: fn(**args)):
            with pytest.raises(BotAccessError, match="^caller_unavailable$"):
                asyncio.run(call())
    owner.assert_not_called()
    assert not seen


def test_signed_native_blank_cannot_force_no_dml_shortcut(signer):
    args = _arguments(content=" ")
    token = native_envelope(_arguments())
    body = json.loads(signer.decrypt(token.encode()))
    from agent_bridge.mcp_call_context import canonical_invocation_hash
    body["invocation"] = canonical_invocation_hash("messages_add", args)
    token = signer.encrypt(json.dumps(body).encode()).decode()
    with pytest.raises(BotAccessError, match="^caller_unavailable$"):
        service_codec.open_service_capability(token, "messages_add", args)


@pytest.mark.parametrize("capability", ["malformed", "expired"])
def test_blank_mint_does_not_ignore_bad_bound_turn_proof(signer, capability):
    request, llm = _proven_request(), _llm()
    selected = _take(request, llm)
    if capability == "expired":
        raw = signer.decrypt(selected.capability.encode())
        capability = signer.encrypt_at_time(raw, 1).decode()
    with service_codec.bind_service_turn(capability, _native_selected=True):
        with pytest.raises(turn_codec.TaskTurnContextError):
            service_codec.ServicePrincipal(BOT, USER).mint("messages_add", _arguments(content=" "))


@pytest.fixture
def store():
    engine = create_engine("sqlite://")
    metadata = MetaData()
    table = Table("messages_constraint_test", metadata,
                  Column("id", String, primary_key=True), Column("role", String),
                  Column("content", String), Column("timestamp", Float),
                  Column("session_id", String), Column("attachments", JSON),
                  Column("reasoning", String), Column("author_entity_type", String),
                  Column("author_entity_id", String), Column("processed", Boolean, default=False),
                  Column("summary_metadata", JSON), Column("created_at", DateTime))
    sessions = Table("sessions", metadata, Column("id", String, primary_key=True),
                     Column("bot_id", String), Column("user_id", String))
    metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(sessions.insert(), [
            {"id": SID, "bot_id": BOT, "user_id": USER},
            {"id": OTHER_SID, "bot_id": BOT, "user_id": USER},
        ])
    row_store = MessageRowStore(engine, table, table.name)
    sql, commits = [], []
    event.listen(engine, "before_cursor_execute", lambda c, cur, stmt, p, ctx, many: sql.append(stmt))
    event.listen(engine, "commit", lambda c: commits.append(True))
    yield SimpleNamespace(rows=row_store, sessions=sessions, sql=sql, commits=commits)
    engine.dispose()


def write(store, **updates):
    return store.rows.upsert(**{
        "message_id": MID, "role": "user", "content": "hello", "timestamp": 10.0,
        "session_id": SID, "_native_selected_owner": _NativeSelectedOwnerConstraint(SID, BOT, USER, True),
        **updates,
    })


def stored(store):
    with store.rows.engine.connect() as conn:
        return conn.execute(select(store.rows.table)).mappings().one()


@pytest.mark.parametrize("private,owner_bot,owner_user,allowed", [
    (True, BOT, USER, True), (True, BOT, "other", False),
    (False, BOT, "other", True), (False, "other", USER, False),
    (True, "other", USER, False),
])
def test_exact_bot_and_only_applicable_private_user_predicate(store, private, owner_bot, owner_user, allowed):
    with store.rows.engine.begin() as conn:
        conn.execute(update(store.sessions).where(store.sessions.c.id == SID)
                     .values(bot_id=owner_bot, user_id=owner_user))
    store.sql.clear()
    store.commits.clear()
    constraint = _NativeSelectedOwnerConstraint(SID, BOT, USER, private)
    if allowed:
        assert write(store, _native_selected_owner=constraint) is None
        assert len(store.commits) == 1
    else:
        with pytest.raises(NativeSelectedOwnerDenied):
            write(store, _native_selected_owner=constraint)
        assert not store.commits
    assert len(store.sql) == 1
    sql = store.sql[0].upper()
    assert sql.startswith("INSERT") and "SELECT" in sql and "FROM SESSIONS" in sql
    assert "ON CONFLICT" in sql and "RETURNING" in sql
    assert "FOR UPDATE" not in sql and "FOR KEY SHARE" not in sql


@pytest.mark.parametrize("extract_memory", [True, False])
def test_missing_source_denies_before_commit_even_on_conflict_or_bad_metadata(store, extract_memory):
    write(store)
    with store.rows.engine.begin() as conn:
        conn.execute(update(store.rows.table).values(summary_metadata=["malformed"]))
        conn.execute(store.sessions.delete().where(store.sessions.c.id == OTHER_SID))
    original = dict(stored(store))
    store.sql.clear()
    store.commits.clear()
    with pytest.raises(NativeSelectedOwnerDenied):
        write(store, session_id=OTHER_SID, content="must not update", extract_memory=extract_memory,
              _native_selected_owner=_NativeSelectedOwnerConstraint(OTHER_SID, BOT, USER, True))
    assert not store.commits and len(store.sql) == 1
    assert dict(stored(store)) == original


def test_conflict_retry_keeps_stored_identity_and_optional_fields(store):
    author = AuthorReference("user", USER)
    write(store, attachments=[{"asset_id": "old"}], reasoning="old", author=author)
    before = dict(stored(store))
    write(store, role="assistant", timestamp=20.0, content="edited", session_id=OTHER_SID,
          _native_selected_owner=_NativeSelectedOwnerConstraint(OTHER_SID, BOT, USER, True))
    after = dict(stored(store))
    for key in ("id", "role", "session_id", "created_at", "timestamp", "attachments",
                "reasoning", "author_entity_type", "author_entity_id"):
        assert after[key] == before[key]
    assert after["content"] == "edited"
    write(store, attachments=[], reasoning="", author=AuthorReference("bot", BOT))
    after = dict(stored(store))
    assert after["attachments"] == [] and after["reasoning"] == ""
    assert (after["author_entity_type"], after["author_entity_id"]) == ("bot", BOT)


def test_assistant_retry_and_json_null_returning_are_not_missing_source(store):
    write(store, role="assistant")  # default SQL/JSON null metadata is still one row
    write(store, role="user", timestamp=30.0)
    assert stored(store)["timestamp"] == 30.0


@pytest.mark.parametrize("metadata", [None, False, 0, "", [], {}, {"keep": True}])
def test_extraction_false_normalizes_falsy_and_merges_objects(store, metadata):
    write(store)
    with store.rows.engine.begin() as conn:
        conn.execute(update(store.rows.table).values(summary_metadata=metadata))
    write(store, extract_memory=False)
    row = stored(store)
    assert row["processed"] is True
    assert row["summary_metadata"] == {**(metadata or {}), "extract_memory": False}


@pytest.mark.parametrize("metadata", [True, 1, "bad", [1]])
def test_truthy_nonmapping_metadata_rolls_back_all_conflict_changes(store, metadata):
    write(store)
    with store.rows.engine.begin() as conn:
        conn.execute(update(store.rows.table).values(summary_metadata=metadata))
    before = dict(stored(store))
    store.commits.clear()
    with pytest.raises(TypeError, match="stored summary metadata must be a mapping"):
        write(store, content="must roll back", attachments=[], extract_memory=False)
    assert not store.commits
    assert dict(stored(store)) == before


def test_blank_and_malformed_constraint_never_skip_fresh_auth(store):
    for updates in ({"content": " "}, {"content": ""}, {"session_id": OTHER_SID},
                    {"_native_selected_owner": {"session_id": SID}}):
        with pytest.raises(NativeSelectedOwnerDenied):
            write(store, **updates)
    assert not store.sql and not store.commits
    assert write(store, content=" ", _native_selected_owner=None) is None
    assert not store.sql and not store.commits


def test_postgresql_compilation_retains_inherited_default_and_conflict_target(store, monkeypatch):
    execution = Mock()
    execution.execute.return_value.first.return_value = (None,)
    context = MagicMock()
    context.__enter__.return_value = execution
    context.__exit__.return_value = False
    monkeypatch.setattr(message_store, "Session", lambda engine: context)
    pg_store = MessageRowStore(SimpleNamespace(dialect=postgresql.dialect()), store.rows.table, "partition")
    pg_store.upsert(message_id=MID, role="user", content="hello", timestamp=1.0,
                    session_id=SID,
                    _native_selected_owner=_NativeSelectedOwnerConstraint(SID, BOT, USER, True))
    statement = execution.execute.call_args.args[0]
    sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "ON CONFLICT (bot_id, id) DO UPDATE" in sql
    assert "FROM sessions" in sql and "sessions.bot_id =" in sql and "sessions.user_id =" in sql
    assert "bot_id" not in sql.split(" SELECT ")[0].split("(", 1)[1]
    assert "FOR UPDATE" not in sql.upper() and "FOR KEY SHARE" not in sql.upper()
    assert execution.execute.call_count == 1
    execution.commit.assert_called_once()


def test_unconstrained_sqlite_stays_plain_upsert_without_sessions_read(store):
    assert write(store, _native_selected_owner=None) is None
    assert len(store.sql) == 1
    assert "FROM sessions" not in store.sql[0]
    assert "RETURNING" not in store.sql[0]
    assert "ON CONFLICT (id)" in store.sql[0]


def test_real_tool_private_forwarding_rechecks_source_after_preflight(monkeypatch, signer, store):
    from llm_bawt.mcp_server import server
    from llm_bawt.mcp_server.bot_access import invocation_context

    backend = SimpleNamespace(bot_id=BOT, _message_rows=store.rows)
    backend.add_message = lambda **kwargs: PostgreSQLMessageMixin.add_message(backend, **kwargs)
    storage = MemoryStorage.__new__(MemoryStorage)
    monkeypatch.setattr(storage, "_get_backend", lambda *a, **kw: backend)
    monkeypatch.setattr(server, "_get_storage", lambda: storage)
    guard, _, owner, _ = guards(monkeypatch)
    fn = guard.register("messages_add", server.add_message)
    args = _arguments(role="assistant")
    token = native_envelope(args)
    with envelope(token), invocation_context("messages_add", args):
        asyncio.run(guard.preflight("messages_add", args))
        result = asyncio.run(fn(**args))
        assert result["content"] == "hello"
        assert len(store.sql) == 1
        # A purge between the early guard and the actual write must deny even
        # a retry of the existing id; no fallback UPDATE and no Message success.
        asyncio.run(guard.preflight("messages_add", args))
        with store.rows.engine.begin() as conn:
            conn.execute(store.sessions.delete())
        store.sql.clear()
        store.commits.clear()
        with pytest.raises(BotAccessError, match="^access_denied$"):
            asyncio.run(fn(**args))
    assert len(store.sql) == 1 and not store.commits
    assert current_native_selected_constraint() is None
    owner.assert_not_called()


def test_storage_backend_forwarding_and_sanitized_failure(monkeypatch, store):
    backend = SimpleNamespace(bot_id=BOT, _message_rows=store.rows)
    backend.add_message = lambda **kwargs: PostgreSQLMessageMixin.add_message(backend, **kwargs)
    storage = MemoryStorage.__new__(MemoryStorage)
    monkeypatch.setattr(storage, "_get_backend", lambda *a, **kw: backend)
    constraint = _NativeSelectedOwnerConstraint(SID, BOT, USER, True)
    result = asyncio.run(storage.add_message("assistant", "hello", bot_id=BOT, session_id=SID,
                                            message_id=MID, _native_selected_owner=constraint))
    assert result.id == MID
    with store.rows.engine.begin() as conn:
        conn.execute(store.sessions.delete())
    store.commits.clear()
    with pytest.raises(BotAccessError, match="^access_denied$") as error:
        asyncio.run(storage.add_message("assistant", "stale", bot_id=BOT, session_id=SID,
                                        message_id=MID, _native_selected_owner=constraint))
    assert error.value.__cause__ is None
    assert not store.commits
    assert stored(store)["content"] == "hello"
