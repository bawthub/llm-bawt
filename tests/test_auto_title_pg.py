"""Guarded title mutation through real private PG, HTTP and coordinator storage."""
from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest
from sqlalchemy import event, text

from llm_bawt.bot_access import service as service_codec
from llm_bawt.models.message import Message
from llm_bawt.service.schemas import ChatMessage
from llm_bawt.sessions import auto_title
from tests.session_lifecycle_pg_fixtures import rig
from tests.test_message_native_upsert import capture_sql
from tests.test_message_native_upsert_pg import http_runtime, proof_for, real_storage
from tests.test_nonstream_turn_execution import _claimed, _isolate, _request, _service

__all__ = ["rig", "real_storage", "http_runtime", "_isolate"]
pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.fixture(autouse=True)
def clear_memo():
    auto_title._settled_sessions.clear()
    yield
    auto_title._settled_sessions.clear()


def set_metadata(rig, metadata):
    with rig.engine.begin() as conn:
        conn.execute(text("UPDATE sessions SET session_metadata=CAST(:meta AS jsonb) WHERE id=:id"),
                     {"meta": json.dumps(metadata), "id": rig.session_id})


def get_metadata(rig):
    with rig.engine.connect() as conn:
        return conn.execute(text("SELECT session_metadata FROM sessions WHERE id=:id"),
                            {"id": rig.session_id}).scalar_one()


def guarded_update_only(sql):
    assert len(sql) == 1 and sql[0].lstrip().startswith("UPDATE sessions"), sql
    assert "owner_bot" in sql[0] and "owner_user" in sql[0], sql
    assert sql.pool_ping_sql == ["SELECT 1"], sql.all_statements
    assert len(sql.all_statements) == 2


def old_eligible(metadata):
    """Frozen helper's Python eligibility, not a replacement storage backend."""
    meta = metadata or {}
    source = (meta.get("title_source") or "default") if isinstance(meta, dict) else "default"
    try:
        return source == "default" and not (isinstance(meta, dict) and (meta.get("title") or "").strip())
    except (AttributeError, TypeError):
        return False


FALSY = [None, False, 0, 0.0, "", [], {}]
METADATA = [None, False, 0, "", [], {}, "legacy", ["legacy"],
            {"nested": {"keep": 1}, "keep": None},
            *[{"title_source": v, "title": ""} for v in FALSY],
            *[{"title_source": "default", "title": v} for v in FALSY],
            *[{"title_source": v} for v in ["user", "auto", " default ", True, 1, [1], {"x": 1}]],
            *[{"title_source": "default", "title": v} for v in ["Keep", True, 1, [1], {"x": 1}]],
            *[{"title_source": "default", "title": c} for c in auto_title.PYTHON_STRIP_CHARS]]


@pytest.mark.parametrize("metadata", METADATA)
def test_guard_matches_old_metadata_eligibility_and_merge(real_storage, rig, metadata):
    set_metadata(rig, metadata)
    manager = real_storage.get_short_term_manager(rig.bot)
    with capture_sql(rig.engine) as sql:
        changed = manager.try_set_session_auto_title(rig.session_id, "Candidate", user_id=rig.user)
    guarded_update_only(sql)
    assert changed is bool(old_eligible(metadata))
    actual = get_metadata(rig)
    if not changed:
        assert actual == metadata
    elif isinstance(metadata, dict) or metadata is None:
        # JSON null is a non-object operand, just like the existing || update.
        expected = {**metadata, "title": "Candidate", "title_source": "auto"} if isinstance(metadata, dict) else [
            None, {"title": "Candidate", "title_source": "auto"}]
        assert actual == expected
    elif isinstance(metadata, list):
        assert actual == [*metadata, {"title": "Candidate", "title_source": "auto"}]
    else:
        assert actual == [metadata, {"title": "Candidate", "title_source": "auto"}]


def test_sql_null_metadata_keeps_shallow_object_merge(real_storage, rig):
    with rig.engine.begin() as conn:
        conn.execute(text("UPDATE sessions SET session_metadata=NULL WHERE id=:id"), {"id": rig.session_id})
    with capture_sql(rig.engine) as sql:
        assert real_storage.get_short_term_manager(rig.bot).try_set_session_auto_title(
            rig.session_id, "Candidate", user_id=rig.user)
    guarded_update_only(sql)
    assert get_metadata(rig) == {"title": "Candidate", "title_source": "auto"}


@pytest.mark.parametrize("scope", ["wrong-bot", "wrong-user", "missing", "userless", "archived"])
def test_bot_pin_supplied_user_and_legacy_userless_contract(real_storage, rig, monkeypatch, scope):
    manager = real_storage.get_short_term_manager(rig.bot)
    session_id, user = rig.session_id, rig.user
    if scope == "wrong-bot":
        monkeypatch.setattr(manager, "bot_id", rig.bot + "foreign")
    elif scope == "wrong-user":
        user = "different-user"
    elif scope == "missing":
        session_id = str(uuid.uuid4())
    elif scope == "userless":
        user = None  # supported bot-only embedded caller, not a new MCP authority
    else:
        with rig.engine.begin() as conn:
            conn.execute(text("UPDATE sessions SET status='archived' WHERE id=:id"), {"id": session_id})
    with capture_sql(rig.engine) as sql:
        changed = manager.try_set_session_auto_title(session_id, "Candidate", user_id=user)
    guarded_update_only(sql)
    assert changed is (scope in {"userless", "archived"})
    metadata = get_metadata(rig)
    if changed:
        assert metadata["title"] == "Candidate"
    else:
        assert metadata is None


def test_missing_session_can_be_created_then_titled(real_storage, rig):
    sid = str(uuid.uuid4())
    manager = real_storage.get_short_term_manager(rig.bot)
    with capture_sql(rig.engine) as sql:
        auto_title.maybe_auto_title(manager, sid, "first", user_id=rig.user)
    guarded_update_only(sql)
    assert sid not in auto_title._settled_sessions
    with rig.engine.begin() as conn:
        conn.execute(text("INSERT INTO sessions (id, bot_id, user_id, status) "
                          "VALUES (:id, :bot, :user, 'archived')"),
                     {"id": sid, "bot": rig.bot, "user": rig.user})
    with capture_sql(rig.engine) as sql:
        auto_title.maybe_auto_title(manager, sid, "retry", user_id=rig.user)
    guarded_update_only(sql)
    assert sid in auto_title._settled_sessions


def test_userless_embedded_write_keeps_title_behavior(real_storage, rig):
    with capture_sql(rig.engine) as sql:
        result = asyncio.run(real_storage.add_message("user", "Embedded title", bot_id=rig.bot,
                                                     session_id=rig.session_id))
    assert result.content == "Embedded title"
    assert len(sql) == 2 and sql[0].startswith("INSERT") and "UPDATE sessions AS s" in sql[1], sql
    assert sql.pool_ping_sql == ["SELECT 1"] * 2
    assert get_metadata(rig)["title"] == "Embedded title"


@pytest.mark.parametrize("user", [None, "different-user"])
@pytest.mark.parametrize("primed", [False, True], ids=["cold-memo", "settled-memo"])
def test_http_cannot_launder_missing_or_wrong_user_into_title_update(http_runtime, rig, user, primed):
    from llm_bawt.bot_access.service import ServicePrincipal

    if primed:
        set_metadata(rig, {"title": "Existing title", "title_source": "user"})
        http_runtime.client.get_session(rig.session_id)
        assert rig.session_id in auto_title._settled_sessions
    before = get_metadata(rig)
    client = http_runtime.client_for()
    client.user_id = user
    client._service_principal = ServicePrincipal(rig.bot, user)
    with service_codec.bind_service_turn(proof_for(rig)), capture_sql(rig.engine) as sql:
        with pytest.raises(RuntimeError, match="access_denied|caller_unavailable"):
            client.add_message("user", "Must not title", session_id=rig.session_id)
    assert not any(s.lstrip().startswith(("INSERT", "UPDATE")) for s in sql), sql
    assert get_metadata(rig) == before


def test_concurrent_first_titles_preserve_one_winner(real_storage, rig):
    manager = real_storage.get_short_term_manager(rig.bot)
    barrier = threading.Barrier(2)

    def write(title):
        barrier.wait(timeout=10)  # timing-ok: hang guard for real database writers
        return manager.try_set_session_auto_title(rig.session_id, title, user_id=rig.user)

    with capture_sql(rig.engine) as sql, ThreadPoolExecutor(max_workers=2) as executor:
        changed = list(executor.map(write, ["First", "Second"]))
    assert sorted(changed) == [False, True]
    assert len(sql) == 2 and all(s.lstrip().startswith("UPDATE sessions") for s in sql), sql
    assert sql.pool_ping_sql == ["SELECT 1"] * 2
    assert get_metadata(rig)["title"] in {"First", "Second"}


def test_user_rename_committed_before_waiting_auto_update_wins(real_storage, rig):
    manager = real_storage.get_short_term_manager(rig.bot)
    started = threading.Event()

    def before(_conn, _cursor, statement, *_args):
        if "UPDATE sessions AS s" in statement:
            started.set()

    with rig.engine.begin() as conn:
        conn.execute(text("UPDATE sessions SET session_metadata=CAST(:meta AS jsonb) WHERE id=:id"),
                     {"meta": json.dumps({"title": "User rename", "title_source": "user", "keep": 1}),
                      "id": rig.session_id})
        event.listen(rig.engine, "before_cursor_execute", before)
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(manager.try_set_session_auto_title, rig.session_id, "Must not win",
                                     user_id=rig.user)
            assert started.wait(timeout=10)  # timing-ok: real worker startup hang guard
        finally:
            event.remove(rig.engine, "before_cursor_execute", before)
        # Commit and release the row lock, allowing the competing UPDATE's
        # predicate to recheck the committed user title instead of stale state.
    try:
        assert future.result(timeout=10) is False  # timing-ok: DB worker hang guard
    finally:
        executor.shutdown(wait=True)
    assert get_metadata(rig) == {"title": "User rename", "title_source": "user", "keep": 1}


def test_title_failure_rolls_back_without_affecting_message_commit(http_runtime, real_storage, rig):
    runtime = http_runtime
    original = get_metadata(rig)

    def fail_title_commit(conn):
        # SQLAlchemy begin() transaction commit event, after real UPDATE executes.
        if conn.info.pop("fail_title", False):
            raise RuntimeError("cosmetic commit failed")

    def mark_title(conn, _cursor, statement, *_args):
        if "UPDATE sessions AS s" in statement:
            conn.info["fail_title"] = True

    event.listen(rig.engine, "before_cursor_execute", mark_title)
    event.listen(rig.engine, "commit", fail_title_commit)
    message_id = str(uuid.uuid4())
    try:
        with service_codec.bind_service_turn(proof_for(rig)), capture_sql(rig.engine) as sql:
            result = runtime.client.add_message("user", "A title candidate", message_id=message_id,
                                                session_id=rig.session_id)
        assert result.id == message_id
        assert len(sql) == 2 and sql[0].startswith("INSERT") and "UPDATE sessions AS s" in sql[1], sql
    finally:
        event.remove(rig.engine, "before_cursor_execute", mark_title)
        event.remove(rig.engine, "commit", fail_title_commit)
    assert get_metadata(rig) == original
    with rig.engine.connect() as conn:
        assert conn.execute(text("SELECT content FROM messages WHERE bot_id=:bot AND id=:id"),
                            {"bot": rig.bot, "id": message_id}).scalar_one() == "A title candidate"
    assert rig.session_id not in auto_title._settled_sessions


@pytest.mark.parametrize("failure", ["message", "manager"])
def test_cosmetic_manager_error_and_failed_message_boundary(real_storage, rig, monkeypatch, failure):
    if failure == "manager":
        monkeypatch.setattr(real_storage, "get_short_term_manager", Mock(side_effect=RuntimeError("manager failed")))
        result = asyncio.run(real_storage.add_message("user", "A title candidate", bot_id=rig.bot,
                             user_id=rig.user, session_id=rig.session_id))
        assert result.content == "A title candidate"
    else:
        manager = Mock()
        monkeypatch.setattr(real_storage, "get_short_term_manager", manager)
        with pytest.raises(Exception):
            asyncio.run(real_storage.add_message("user", "Must fail", bot_id=rig.bot, user_id=rig.user,
                        session_id=rig.session_id, message_id="x" * 37))
        manager.assert_not_called()
    assert get_metadata(rig) is None


@pytest.mark.parametrize("role,content", [("assistant", "assistant"), ("system", "system"),
                                          ("user", "/new"), ("user", "  \n\t")])
def test_no_title_dml_for_other_roles_slash_or_blank(http_runtime, rig, role, content):
    with service_codec.bind_service_turn(proof_for(rig)), capture_sql(rig.engine) as sql:
        http_runtime.client.add_message(role, content, session_id=rig.session_id)
    assert not any(s.lstrip().startswith("UPDATE") for s in sql), sql
    assert get_metadata(rig) is None


def test_existing_titled_successive_http_writes_have_honest_cost(http_runtime, rig, record_property):
    set_metadata(rig, {"title": "User title", "title_source": "user", "keep": 1})
    samples = []
    with service_codec.bind_service_turn(proof_for(rig)):
        for content in ["First plain message", "Second plain message"]:
            start = time.perf_counter()
            with capture_sql(rig.engine) as sql:
                http_runtime.client.add_message("user", content, session_id=rig.session_id)
            samples.append({"seconds": time.perf_counter() - start, "application_statements": len(sql),
                            "application_selects": sum(s.lstrip().startswith("SELECT") for s in sql),
                            "pool_selects": len(sql.pool_ping_sql), "total_statements": len(sql.all_statements)})
            assert len(sql) == 2 and sql[0].startswith("INSERT") and "UPDATE sessions AS s" in sql[1], sql
            assert sql.pool_ping_sql == ["SELECT 1"] * 2
            assert len(sql.all_statements) == 4
            assert rig.session_id not in auto_title._settled_sessions
    record_property("titled_unknown_successive_write_receipt", json.dumps(samples))
    assert get_metadata(rig) == {"title": "User title", "title_source": "user", "keep": 1}


@pytest.mark.parametrize("titled", [False, True], ids=["untitled", "already-titled"])
def test_actual_agent_coordinator_uses_guard_or_already_fetched_title(http_runtime, real_storage, rig,
                                                                    titled, record_property):
    from llm_bawt.service.session_lifecycle.bridge_reset import BridgeSessionResetMixin

    if titled:
        set_metadata(rig, {"title": "Kept title", "title_source": "user", "keep": {"nested": True}})
    service = _service(Mock(return_value=("answer", "", [])))
    llm = service._get_llm_bawt.return_value
    llm.history_manager = http_runtime.history
    llm.bot.slug, llm.user_id = rig.bot, rig.user
    service._bind_agent_thread = BridgeSessionResetMixin._bind_agent_thread.__get__(service)
    service._resolve_thread_sdk_key = BridgeSessionResetMixin._resolve_thread_sdk_key.__get__(service)
    measured = []

    def persist(role, content):
        start = time.perf_counter()
        with capture_sql(rig.engine) as sql:
            http_runtime.history.add_message(role, content, session_id=llm._session_id_override)
        measured.append((sql, time.perf_counter() - start))

    def prepare(prompt, **kwargs):
        persist("user", prompt)
        return [Message(role="user", content=prompt)]

    llm.prepare_messages_for_query.side_effect = prepare
    service._finalize_turn.side_effect = lambda **kw: persist("assistant", kw["response_text"])
    request = _claimed(_request()).model_copy(update={
        "bot_id": rig.bot, "user": rig.user, "session_id": rig.session_id,
        "user_message_id": str(uuid.uuid4()), "inter_bot_turn_id": "turn-delivery-" + uuid.uuid4().hex,
        "messages": [ChatMessage(role="user", content="A real plain message")],
    })
    with capture_sql(rig.engine) as turn_sql:
        asyncio.run(service.chat_completion(request))
    assert len(measured) == 2
    initial, final = (entry[0] for entry in measured)
    assert len(initial) == (1 if titled else 2), initial
    assert initial[0].startswith("INSERT")
    if not titled:
        assert "UPDATE sessions AS s" in initial[1]
    assert initial.pool_ping_sql == ["SELECT 1"] * len(initial)
    assert len(final) == 1 and final[0].startswith("INSERT"), final
    assert final.pool_ping_sql == ["SELECT 1"]
    assert not any(s.lstrip().startswith("SELECT") for s in initial + final)
    # The SDK's existing binding lookup is shown separately, not moved out of
    # the write capture or newly introduced to populate the cosmetic memo.
    assert sum(s.lstrip().startswith("SELECT") for s in turn_sql) >= 1
    assert rig.session_id in auto_title._settled_sessions
    metadata = get_metadata(rig)
    assert metadata["title"] == ("Kept title" if titled else "A real plain message")
    record_property("coordinator_title_write_receipt", json.dumps([
        {"seconds": elapsed, "application_statements": len(sql),
         "pool_selects": len(sql.pool_ping_sql), "total_statements": len(sql.all_statements)}
        for sql, elapsed in measured]))


def test_existing_session_read_primes_only_real_settled_rows(http_runtime, rig):
    assert rig.session_id not in auto_title._settled_sessions
    with capture_sql(rig.engine) as sql:
        http_runtime.client.get_session(rig.session_id)
    assert rig.session_id not in auto_title._settled_sessions
    set_metadata(rig, {"title": "Owned title", "title_source": "user"})
    with capture_sql(rig.engine) as sql:
        http_runtime.client.get_session(rig.session_id)
    assert rig.session_id in auto_title._settled_sessions
    assert not any(s.lstrip().startswith(("INSERT", "UPDATE")) for s in sql), sql
