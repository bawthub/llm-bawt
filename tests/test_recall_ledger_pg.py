"""Recall evidence ledger and live snippet SQL on real PostgreSQL (TASK-1519).

Run: make test-target NODES=tests/test_recall_ledger_pg.py
"""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import timedelta

import pytest
from sqlalchemy import create_engine, inspect, text

from llm_bawt.recall.contracts import RecallScope
from llm_bawt.recall.drilldown import EvidenceDrilldown, SNIPPET_CHARS
from llm_bawt.recall.ledger import LEDGER_TABLE, PostgresLedger, handle_digest
from llm_bawt.recall.postgres import RecallDatabase
from llm_bawt.recall.task_source import TaskSource
from llm_bawt.utils import db
from tests.pg_isolation import require_test_db_config
from tests.test_isolated_recall_pg import USER, Rig

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.fixture(scope="module")
def engine(private_app_pg):
    engine = create_engine(db.build_postgres_url(require_test_db_config()))
    try:
        yield engine
    finally:
        engine.dispose()


def entry(source, ref, *, source_bot=None, start_char=0):
    return {"source": source, "id": ref, "source_bot": source_bot, "session_id": None, "at": None,
            "speaker": None, "follows": None, "window_kind": "matched", "start_char": start_char,
            "admitted": True, "cited": False}


def test_ledger_round_trip_is_owner_scoped_hashed_and_text_free(engine):
    ledger = PostgresLedger(engine)
    handle = f"handle-{uuid.uuid4().hex}"
    scope = RecallScope("owner-bot", "owner-user", "session-1")
    entries = [entry("messages", "m1", source_bot="peer")]
    ledger.write(handle, scope, {"messages": {"status": "ok", "records": 1}}, entries, 0)

    record = ledger.read(handle, "owner-bot", "owner-user")
    assert (record.session_id, record.entries, record.omitted) == ("session-1", entries, 0)
    assert ledger.read(handle, "other-bot", "owner-user") is None
    assert ledger.read(handle, "owner-bot", "other-user") is None
    assert ledger.read(handle + "x", "owner-bot", "owner-user") is None

    with engine.connect() as conn:
        stored = conn.execute(text(f"SELECT handle_sha256 FROM {LEDGER_TABLE} WHERE bot_id = 'owner-bot'")).scalars().all()
    assert handle_digest(handle) in stored and handle not in stored
    columns = {column["name"] for column in inspect(engine).get_columns(LEDGER_TABLE)}
    assert columns == {"handle_sha256", "bot_id", "user_id", "session_id", "coverage", "entries",
                       "omitted", "created_at", "expires_at"}


def test_expired_records_are_unreadable_and_purged_on_the_next_write(engine):
    expired = PostgresLedger(engine, ttl=timedelta(seconds=-1))
    handle = f"expired-{uuid.uuid4().hex}"
    scope = RecallScope("expiry-bot", "expiry-user", "session-1")
    expired.write(handle, scope, {}, [], 0)
    assert expired.read(handle, "expiry-bot", "expiry-user") is None
    PostgresLedger(engine).write(f"fresh-{uuid.uuid4().hex}", scope, {}, [], 0)
    with engine.connect() as conn:
        left = conn.execute(text(f"SELECT count(*) FROM {LEDGER_TABLE} WHERE handle_sha256 = :d"),
                            {"d": handle_digest(handle)}).scalar_one()
    assert left == 0


def resolve(rig, source, entries, scope):
    drilldown = EvidenceDrilldown(RecallDatabase(rig.engine), PostgresLedger(rig.engine), TaskSource())
    return asyncio.run(drilldown._resolve(source, entries, scope))


def test_message_snippets_are_live_windows_under_current_authority(engine):
    rig = Rig(engine)
    peer = f"peer{uuid.uuid4().hex[:10]}"
    peer_session = rig.add_session(str(uuid.uuid4()), bot=peer)
    other_user = rig.add_session(str(uuid.uuid4()), bot=peer, user="someone-else")
    body = "lead " + "relay background decision " * 20
    good = rig.add(body, at=rig.now - 30, partition=peer, session=peer_session, author=peer, author_type="bot")
    foreign = rig.add(body, at=rig.now - 20, partition=peer, session=other_user, author=peer, author_type="bot")
    scope = RecallScope(rig.bot, USER, rig.session, frozenset(), frozenset({rig.bot, peer}))
    rows = resolve(rig, "messages", [entry("messages", good, source_bot=peer, start_char=5),
                                     entry("messages", foreign, source_bot=peer)], scope)
    assert rows[0]["available"] is True
    assert rows[0]["snippet"] == body[5:5 + SNIPPET_CHARS] and rows[0]["more"] is True
    assert rows[1] == {**rows[1], "available": False} and "snippet" not in rows[1]


def test_memory_and_tool_snippets_keep_their_source_authority(engine):
    rig = Rig(engine)
    owned = rig.add("relay memory source", at=rig.now - 60)
    stranger = rig.add_session(str(uuid.uuid4()), user="someone-else")
    foreign = rig.add("someone else's turn", at=rig.now - 50, session=stranger)
    good_memory, bad_memory = f"mem-{uuid.uuid4().hex}", f"mem-{uuid.uuid4().hex}"
    turn = f"turn-{uuid.uuid4().hex}"
    with engine.begin() as conn:
        for memory_id, sources in ((good_memory, [owned]), (bad_memory, [owned, foreign])):
            conn.execute(text("""INSERT INTO memories (id, bot_id, content, source_message_ids)
                VALUES (:id, :bot, :content, CAST(:sources AS jsonb))"""),
                         {"id": memory_id, "bot": rig.bot, "content": f"Memory {memory_id}",
                          "sources": json.dumps(sources)})
        conn.execute(text("""INSERT INTO turn_logs (id, created_at, path, stream, bot_id, user_id, status,
                                 user_prompt, response_text, trigger_message_id)
            VALUES (:id, now(), '/test', false, :bot, :user, 'ok', '', '', :message)"""),
                     {"id": turn, "bot": rig.bot, "user": USER, "message": owned})
        tool_id = conn.execute(text("""INSERT INTO tool_call_records
                (turn_id, bot_id, user_id, tool_name, iteration, created_at, approval_status, result_text)
            VALUES (:turn, :bot, :user, 'ops_run', 1, now(), 'approved', 'SECRET RESULT') RETURNING id"""),
                               {"turn": turn, "bot": rig.bot, "user": USER}).scalar_one()
    scope = RecallScope(rig.bot, USER, rig.session, frozenset(), frozenset({rig.bot}))
    memories = resolve(rig, "memories", [entry("memories", good_memory), entry("memories", bad_memory)], scope)
    assert memories[0]["snippet"] == f"Memory {good_memory}"
    assert memories[1]["available"] is False
    [tool] = resolve(rig, "tool_evidence", [entry("tool_evidence", str(tool_id))], scope)
    assert tool["snippet"] == "ops_run · approved"
    other = RecallScope(rig.bot, "someone-else", rig.session, frozenset(), frozenset({rig.bot}))
    assert resolve(rig, "tool_evidence", [entry("tool_evidence", str(tool_id))], other)[0]["available"] is False
