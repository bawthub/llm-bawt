"""TASK-1907 corrective owner READ inside native-selected DML on private PG.

The HTTP purge cases reuse the real selected-native route/coordinator/history/MCP
chain. Only URL construction for the purge's separate engine is redirected to the
fixture-owned Unix socket; the settings route, purge SQL and commit are real.
No lock/fence or purge-after-write-snapshot serialization is claimed.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import contextmanager

import pytest
import sqlalchemy
from sqlalchemy import event, text, update
from sqlalchemy.orm import Session

from llm_bawt import task_turn_context
from llm_bawt.bot_access import service as service_codec
from llm_bawt.memory.message_store import NativeSelectedOwnerDenied, _NativeSelectedOwnerConstraint
from llm_bawt.message_authorship import AuthorReference
from tests.test_message_native_upsert import FALSY_METADATA, INVALID_METADATA, capture_sql, one_upsert, row
from tests.test_message_native_upsert_pg import http_runtime, proof_for, real_storage
from tests.test_native_selected_owner_pg import _body, _record, native_runtime, rig

__all__ = ["rig", "real_storage", "http_runtime", "native_runtime"]
pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@contextmanager
def transactions(engine):
    observed = []
    def commit(_conn):
        observed.append("commit")
    def rollback(_conn):
        observed.append("rollback")
    event.listen(engine, "commit", commit)
    event.listen(engine, "rollback", rollback)
    try:
        yield observed
    finally:
        event.remove(engine, "commit", commit)
        event.remove(engine, "rollback", rollback)


def owner_dml(sql, *, private=True):
    one_upsert(sql)
    statement = sql[0]
    assert "FROM sessions" in statement and "sessions.id =" in statement
    assert "sessions.bot_id =" in statement
    assert ("sessions.user_id =" in statement) is private
    assert "RETURNING" in statement and "FOR UPDATE" not in statement


@pytest.fixture(params=[True, False], ids=["fresh-FK", "legacy-no-FK"])
def session_fk(request, rig):
    # This module's DB is disposable and isolated. Never retrofit live schema.
    with rig.engine.begin() as conn:
        exists = conn.execute(text("SELECT count(*) FROM pg_constraint WHERE "
            "conrelid='public.messages'::regclass AND conname='fk_messages_session' ")).scalar_one()
        assert exists == 1
        if not request.param:
            conn.execute(text("ALTER TABLE public.messages DROP CONSTRAINT fk_messages_session"))
    try:
        yield request.param
    finally:
        if not request.param:
            with rig.engine.begin() as conn:
                conn.execute(text("ALTER TABLE public.messages ADD CONSTRAINT fk_messages_session "
                    "FOREIGN KEY (session_id) REFERENCES public.sessions(id) ON DELETE NO ACTION NOT DEFERRABLE"))


@pytest.fixture
def supported_purge(native_runtime, rig, monkeypatch, record_property):
    from llm_bawt.profiles import ProfileManager
    from llm_bawt.prompt_registry import PromptTemplateStore
    from llm_bawt.runtime_settings import RuntimeSettingsStore
    from llm_bawt.service.routes import settings
    from llm_bawt.service.tool_call_store import ToolCallStore

    # Real app-owned schema bootstrap, outside request/write SQL scopes.
    ProfileManager(rig.config)
    PromptTemplateStore(rig.config)
    RuntimeSettingsStore(rig.config)
    ToolCallStore(rig.engine)
    native_runtime.http.app.include_router(settings.router)
    # Preserve native fixture's isolated configuration; supply real private DB
    # configuration at the actual settings-route service dependency.
    for name in ("POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DATABASE"):
        setattr(native_runtime.service.config, name, getattr(rig.config, name))
    real_create = sqlalchemy.create_engine
    results = []

    def purge():
        # Production purge constructs a TCP URL; private PG is socket-only.
        # Redirect only engine construction, never the purge implementation.
        engine = real_create(rig.engine.url)
        with monkeypatch.context() as patch, capture_sql(engine) as sql, transactions(engine) as tx:
            patch.setattr(sqlalchemy, "create_engine", lambda *_a, **_k: engine)
            response = native_runtime.http.post(f"/v1/bots/{rig.bot}/purge-data")
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["slug"] == rig.bot
        assert result["deleted_rows"]["sessions"] == 1
        assert result["deleted_rows"]["messages"] >= 1
        assert tx == ["commit"]
        _record(record_property, "supported_physical_purge_separate_engine", sql)
        record_property("purge_response", json.dumps(result))
        with rig.engine.connect() as conn:
            assert conn.execute(text("SELECT count(*) FROM sessions WHERE bot_id=:bot"), {"bot": rig.bot}).scalar_one() == 0
            assert conn.execute(text("SELECT count(*) FROM messages WHERE bot_id=:bot"), {"bot": rig.bot}).scalar_one() == 0
            assert conn.execute(text("SELECT count(*) FROM bot_profiles WHERE slug=:bot"), {"bot": rig.bot}).scalar_one() == 1
        results.append(result)
        return result

    return purge, results


@pytest.mark.parametrize("phase", ["before-initial", "before-final"])
@pytest.mark.parametrize("stream", [False, True])
def test_supported_physical_purge_after_admission_denies_real_native_append(
        native_runtime, real_storage, rig, session_fk, supported_purge, monkeypatch,
        record_property, phase, stream):
    from llm_bawt.service.session_lifecycle.execution import NativeSelectedHistory

    runtime = native_runtime
    purge, purges = supported_purge
    body = _body(rig, stream=stream)
    proofs, invocations, attempts, transport_errors, terminal_events = [], [], [], [], []
    from llm_bawt.service.turn_stream_worker import TurnStreamWorker
    original_publish = TurnStreamWorker._publish_event_direct

    def publish(worker, payload):
        if payload.get("_type") == "turn_complete":
            terminal_events.append(payload)
        return original_publish(worker, payload)

    monkeypatch.setattr(TurnStreamWorker, "_publish_event_direct", publish)
    original_take = NativeSelectedHistory.take.__func__
    original_call = runtime.memory.client._call_server
    original_upsert = real_storage.get_backend(rig.bot)._message_rows.upsert

    def take(cls, *args, **kwargs):
        selected = original_take(cls, *args, **kwargs)
        proofs.append(selected)
        return selected

    def call(method, arguments):
        if method == "messages_add":
            token = service_codec._runtime_turn.get()
            proof = task_turn_context.open_task_turn_context(token)
            assert (proof.session_id, proof.bot_id, proof.user_id) == (rig.session_id, rig.bot, rig.user)
            invocations.append((arguments["role"], token))
        try:
            return original_call(method, arguments)
        except RuntimeError as error:
            if method == "messages_add":
                transport_errors.append(str(error))
            raise

    def upsert(**kwargs):
        # Transparent observation of actual DML/transaction, including denial.
        with capture_sql(rig.engine) as sql, transactions(rig.engine) as tx:
            try:
                return original_upsert(**kwargs)
            finally:
                attempts.append((kwargs["role"], sql, tx))

    monkeypatch.setattr(NativeSelectedHistory, "take", classmethod(take))
    monkeypatch.setattr(runtime.memory.client, "_call_server", call)
    monkeypatch.setattr(real_storage.get_backend(rig.bot)._message_rows, "upsert", upsert)

    def purge_checked():
        assert len(runtime.admissions) == 1  # real HTTP owned-row admission completed
        assert len(proofs) == 1 and proofs[0].capability
        proof = task_turn_context.open_task_turn_context(proofs[0].capability)
        assert proof.session_id == rig.session_id
        if phase == "before-final":
            assert [role for role, _, _ in runtime.appends] == ["user"]
            with rig.engine.connect() as conn:
                assert conn.execute(text("SELECT count(*) FROM messages WHERE id=:id AND bot_id=:bot"),
                    {"id": body["user_message_id"], "bot": rig.bot}).scalar_one() == 1
        purge()
        # Same original cryptographically valid proof survives the committed
        # physical purge; it is not renewed into a replacement owner authority.
        assert task_turn_context.open_task_turn_context(proofs[0].capability) == proof

    if phase == "before-initial":
        original_add = runtime.memory.history.add_message
        def add(role, *args, **kwargs):
            if role == "user":
                purge_checked()  # after admission AND actual history load, before append
            return original_add(role, *args, **kwargs)
        monkeypatch.setattr(runtime.memory.history, "add_message", add)
    elif stream:
        original_stream = runtime.llm.client.stream_raw
        def model(messages, **kwargs):
            yield from original_stream(messages, **kwargs)
            purge_checked()
        monkeypatch.setattr(runtime.llm.client, "stream_raw", model)
    else:
        original_query = runtime.llm.client.query
        def model(messages, **kwargs):
            result = original_query(messages, **kwargs)
            purge_checked()
            return result
        monkeypatch.setattr(runtime.llm.client, "query", model)

    with capture_sql(rig.engine) as whole:
        response = runtime.http.post("/v1/chat/completions", json=body)
    _record(record_property, "purged_native_whole_http_including_observation_reads", whole)
    assert len(purges) == 1
    assert len(proofs) == 1
    assert {token for _, token in invocations} == {proofs[0].capability}
    if stream:
        assert response.status_code == 200
        frames = [json.loads(line[6:]) for line in response.text.splitlines()
                  if line.startswith("data: ") and line != "data: [DONE]"]
        errors = [frame["error"] for frame in frames if "error" in frame]
        if phase == "before-initial":
            assert errors and "access_denied" in json.dumps(errors), response.text
        else:
            # Existing streaming-finalizer contract reports persistence failure
            # on authoritative turn_complete, NOT on the direct HTTP SSE tail.
            # Keep this limitation explicit; do not claim SSE error propagation.
            assert not errors
            assert any(frame["choices"][0]["finish_reason"] == "stop" for frame in frames)
            assert len(terminal_events) == 1
            terminal = terminal_events[0]
            assert terminal["status"] == "error" and terminal["end_reason"] == "upstream_error"
            assert "access_denied" in terminal["error_text"]
            record_property("streaming_finalizer_terminal", json.dumps(terminal))
            record_property("direct_sse_final_append_error_surfaced", False)
    else:
        assert response.status_code >= 400, response.text
        assert "access_denied" in response.text
        assert "choices" not in response.json()
    assert len(transport_errors) == 1 and "access_denied" in transport_errors[0]
    assert "INSERT INTO" not in transport_errors[0] and rig.session_id not in transport_errors[0]
    assert "INSERT INTO" not in response.text and rig.session_id not in response.text
    assert len(attempts) == (1 if phase == "before-initial" else 2)
    role, denied, tx = attempts[-1]
    assert role == ("user" if phase == "before-initial" else "assistant")
    owner_dml(denied)
    assert tx == ["rollback"]  # no write commit or fabricated Message success
    _record(record_property, "zero_source_denied_append", denied)
    assert runtime.memory.stages == ["preflight", "execution"] * len(attempts)
    assert not runtime.model_calls if phase == "before-initial" else len(runtime.model_calls) == 1
    assert service_codec._runtime_turn.get() is None
    assert service_codec._runtime_native_selected.get() is False
    with rig.engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM sessions WHERE bot_id=:bot"), {"bot": rig.bot}).scalar_one() == 0
        assert conn.execute(text("SELECT count(*) FROM messages WHERE bot_id=:bot"), {"bot": rig.bot}).scalar_one() == 0
        assert conn.execute(text("SELECT count(*) FROM bot_profiles WHERE slug=:bot"), {"bot": rig.bot}).scalar_one() == 1
    record_property("session_fk_present", session_fk)


@pytest.mark.parametrize("private", [True, False], ids=["private", "public"])
@pytest.mark.parametrize("owner", ["matching", "wrong-user", "wrong-bot", "missing"])
def test_actual_memory_client_owner_predicate(http_runtime, real_storage, rig, private, owner, record_property):
    from llm_bawt.bot_access.refresh import PolicyRefresher
    from llm_bawt.runtime_settings import BotProfileStore

    BotProfileStore(rig.config).upsert({"slug": rig.bot, "include_in_global_search": not private})
    assert PolicyRefresher(rig.engine).refresh()
    if owner != "matching":
        with rig.engine.begin() as conn:
            if owner == "wrong-user":
                conn.execute(text("UPDATE sessions SET user_id='other' WHERE id=:sid"), {"sid": rig.session_id})
            elif owner == "wrong-bot":
                conn.execute(text("UPDATE sessions SET bot_id=:bot WHERE id=:sid"), {"bot": http_runtime.other_bot, "sid": rig.session_id})
            else:
                # Actual supported purge is covered above. For this store matrix,
                # a missing source is a distinct valid, freshly minted proof.
                pass
    sid = str(uuid.uuid4()) if owner == "missing" else rig.session_id
    cap = task_turn_context.mint_task_turn_context(session_id=sid,
        turn_id="turn-" + uuid.uuid4().hex, trigger_message_id=rig.message_id,
        bot_id=rig.bot, user_id=rig.user)
    allowed = owner == "matching" or (not private and owner == "wrong-user")
    mid = str(uuid.uuid4())
    with capture_sql(rig.engine) as sql, transactions(rig.engine) as tx:
        with service_codec.bind_service_turn(cap, _native_selected=True):
            if allowed:
                result = http_runtime.client.add_message("assistant", "owner matrix", session_id=sid, message_id=mid)
                assert result.id == mid
            else:
                with pytest.raises(RuntimeError, match="access_denied") as failure:
                    http_runtime.client.add_message("assistant", "owner matrix", session_id=sid, message_id=mid)
                assert "INSERT INTO" not in str(failure.value) and sid not in str(failure.value)
    owner_dml(sql, private=private)
    assert tx == (["commit"] if allowed else ["rollback"])
    assert http_runtime.stages == ["preflight", "execution"]
    _record(record_property, "owner_matrix_append", sql)
    with rig.engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM messages WHERE id=:id AND bot_id=:bot"), {"id": mid, "bot": rig.bot}).scalar_one() == int(allowed)


@pytest.mark.parametrize("content", ["", " \n\t"])
@pytest.mark.parametrize("purged", [False, True])
def test_blank_native_uses_authoritative_fallback_without_dml(
        native_runtime, supported_purge, rig, content, purged, record_property):
    cap = proof_for(rig)
    if purged:
        supported_purge[0]()
    with capture_sql(rig.engine) as sql:
        with service_codec.bind_service_turn(cap, _native_selected=True):
            if purged:
                with pytest.raises(RuntimeError, match="access_denied"):
                    native_runtime.memory.client.add_message("assistant", content, session_id=rig.session_id)
            else:
                result = native_runtime.memory.client.add_message("assistant", content, session_id=rig.session_id)
                assert result.content == content  # legacy blank no-op result retained
    assert not any(s.lstrip().startswith(("INSERT", "UPDATE", "DELETE")) for s in sql)
    assert sum("SELECT bot_id, user_id FROM sessions" in s for s in sql) == (1 if purged else 2)
    assert native_runtime.memory.stages == (["preflight"] if purged else ["preflight", "execution"])
    _record(record_property, "blank_authoritative_fallback", sql)
    with rig.engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM messages WHERE bot_id=:bot"), {"bot": rig.bot}).scalar_one() == (0 if purged else 1)


def constraint(rig, sid=None):
    return _NativeSelectedOwnerConstraint(sid or rig.session_id, rig.bot, rig.user, True)


@pytest.mark.parametrize("stored_role", ["user", "assistant", "system", "summary"])
def test_constrained_real_pg_retry_immutability_and_zero_source_conflict(real_storage, rig, stored_role, record_property):
    store = real_storage.get_backend(rig.bot)._message_rows
    second = str(uuid.uuid4())
    with rig.engine.begin() as conn:
        conn.execute(text("INSERT INTO sessions (id,bot_id,user_id,status) VALUES (:id,:bot,:user,'archived')"),
                     {"id": second, "bot": rig.bot, "user": rig.user})
    def write(**kwargs):
        return store.upsert(**{"message_id": "stable", "role": stored_role, "content": "first", "timestamp": 10,
            "session_id": rig.session_id, "_native_selected_owner": constraint(rig), **kwargs})
    with capture_sql(rig.engine) as sql:
        write(attachments=[{"asset_id": "old"}], reasoning="thought", author=AuthorReference.user("first-author"))
    owner_dml(sql)
    first = row(store)
    with capture_sql(rig.engine) as sql:
        write(role="assistant" if stored_role == "user" else "user", content="retry", timestamp=20,
              session_id=second, _native_selected_owner=constraint(rig, second))
    owner_dml(sql)
    retried = row(store)
    assert retried == {**first, "content": "retry", "timestamp": 10 if stored_role == "user" else 20}
    with capture_sql(rig.engine) as sql:
        write(content="final", timestamp=30, attachments=[], reasoning="", author=AuthorReference.unknown())
    owner_dml(sql)
    final = row(store)
    assert final == {**retried, "content": "final", "timestamp": 10 if stored_role == "user" else 30,
                    "attachments": [], "reasoning": "", "author_entity_type": None, "author_entity_id": None}
    missing = str(uuid.uuid4())
    with capture_sql(rig.engine) as sql, transactions(rig.engine) as tx, pytest.raises(NativeSelectedOwnerDenied):
        write(content="must not update conflict", session_id=missing, _native_selected_owner=constraint(rig, missing), extract_memory=False)
    owner_dml(sql)
    assert tx == ["rollback"] and row(store) == final
    _record(record_property, "zero_source_existing_conflict", sql)


@pytest.mark.parametrize("metadata", FALSY_METADATA + INVALID_METADATA + [{"nested": {"keep": True}, "extract_memory": True}])
def test_constrained_pg_exclusion_metadata_and_rollback(real_storage, rig, metadata):
    store = real_storage.get_backend(rig.bot)._message_rows
    values = dict(message_id="stable", role="assistant", content="first", timestamp=10,
                  session_id=rig.session_id, _native_selected_owner=constraint(rig))
    store.upsert(**values)
    with rig.engine.begin() as conn:
        conn.execute(update(store.table).where(store.table.c.id == "stable").values(summary_metadata=metadata))
    first = row(store)
    invalid = bool(metadata) and not isinstance(metadata, dict)
    with capture_sql(rig.engine) as sql, transactions(rig.engine) as tx:
        if invalid:
            with pytest.raises(TypeError, match="mapping"):
                store.upsert(**{**values, "content": "rollback", "timestamp": 20, "attachments": [], "reasoning": "changed", "extract_memory": False})
        else:
            store.upsert(**{**values, "content": "excluded", "extract_memory": False})
    owner_dml(sql)
    assert tx == (["rollback"] if invalid else ["commit"])
    if invalid:
        assert row(store) == first
    else:
        actual = row(store)
        assert actual["processed"] is True
        assert actual["summary_metadata"] == {**(metadata or {}), "extract_memory": False}
        store.upsert(**values)
        assert row(store)["summary_metadata"] == actual["summary_metadata"]


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
def test_constrained_pg_commit_failure_or_cancellation_preserves_prior_row(real_storage, rig, failure):
    store = real_storage.get_backend(rig.bot)._message_rows
    values = dict(message_id="stable", role="assistant", content="original", timestamp=10,
                  session_id=rig.session_id, _native_selected_owner=constraint(rig))
    store.upsert(**values)
    first = row(store)
    def fail(_session):
        raise failure("precommit failure")
    event.listen(Session, "before_commit", fail)
    try:
        with capture_sql(rig.engine) as sql, transactions(rig.engine) as tx, pytest.raises(failure, match="precommit failure"):
            store.upsert(**{**values, "content": "must rollback", "timestamp": 20, "extract_memory": False})
        owner_dml(sql)
        assert tx == ["rollback"]
    finally:
        event.remove(Session, "before_commit", fail)
    assert row(store) == first
    store.upsert(**{**values, "content": "recovered", "timestamp": 30})
    assert row(store)["content"] == "recovered"
