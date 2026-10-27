"""``tool_args`` recall and drill-down authority on real PostgreSQL (TASK-1557).

Run: make test-target NODES=tests/test_recall_tool_args_pg.py
"""
from __future__ import annotations

import asyncio
import json
import uuid

import pytest
from sqlalchemy import create_engine, text

from llm_bawt.recall.contracts import RecallScope
from llm_bawt.recall.drilldown import EvidenceDrilldown
from llm_bawt.recall.ledger import PostgresLedger
from llm_bawt.recall.postgres import RecallDatabase, ToolArgSource
from llm_bawt.recall.task_source import TaskSource
from llm_bawt.utils import db
from tests.pg_isolation import require_test_db_config
from tests.test_isolated_recall_pg import USER, Rig

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]

SECRET = "fixture-pass-7"
FACT = "attempted clone repo=https://github.com/example/gizmo-widget dest=/tmp/gw"


@pytest.fixture(scope="module")
def engine(private_app_pg):
    engine = create_engine(db.build_postgres_url(require_test_db_config()))
    try:
        yield engine
    finally:
        engine.dispose()


def add_call(rig, *, bot, trigger, command, user=USER):
    turn = f"turn-{uuid.uuid4().hex}"
    with rig.engine.begin() as conn:
        conn.execute(text("""INSERT INTO turn_logs (id, created_at, path, stream, bot_id, user_id, status,
                                 user_prompt, response_text, trigger_message_id)
            VALUES (:id, now(), '/test', false, :bot, :user, 'ok', '', '', :message)"""),
                     {"id": turn, "bot": bot, "user": user, "message": trigger})
        return str(conn.execute(text("""INSERT INTO tool_call_records
                (turn_id, bot_id, user_id, tool_name, iteration, created_at, arguments_json, result_text)
            VALUES (:turn, :bot, :user, 'Bash', 1, now(), :arguments, 'RESULT BODY') RETURNING id"""),
                                {"turn": turn, "bot": bot, "user": user,
                                 "arguments": json.dumps({"command": command})}).scalar_one())


def peer_trigger(rig, peer, *, user=USER):
    # Archived: a bot/user pair may hold only one active session, and the
    # test adds several triggers per pair.
    session = rig.add_session(str(uuid.uuid4()), bot=peer, user=user, status="archived")
    return rig.add("clone the gizmo widget fork", at=rig.now - 60, partition=peer, session=session,
                   author=peer, author_type="bot")


def test_tool_args_reads_searchable_bots_only_and_admits_only_facts(engine):
    rig = Rig(engine)
    peer, protected = f"peer{uuid.uuid4().hex[:10]}", f"prot{uuid.uuid4().hex[:10]}"
    command = f"cd /tmp && git clone -q https://github.com/example/gizmo-widget gw && curl -u a:{SECRET} x"
    good = add_call(rig, bot=peer, trigger=peer_trigger(rig, peer), command=command)
    # Same command: under an excluded bot, under another user's session, and
    # another user's call whose turn names a message in this user's session.
    add_call(rig, bot=protected, trigger=peer_trigger(rig, protected), command=command)
    add_call(rig, bot=peer, trigger=peer_trigger(rig, peer, user="someone-else"), command=command,
             user="someone-else")
    crossed = add_call(rig, bot=peer, trigger=peer_trigger(rig, peer), command=command, user="someone-else")
    scope = RecallScope(rig.bot, USER, rig.session, frozenset({protected}),
                        frozenset({rig.bot, peer, protected}))
    evidence = asyncio.run(ToolArgSource(RecallDatabase(engine)).search("where was gizmo widget cloned", scope, ()))
    assert [(item.id, item.text, item.source_bot) for item in evidence] == [(good, FACT, peer)]
    assert evidence[0].at is not None and evidence[0].session_id

    drilldown = EvidenceDrilldown(RecallDatabase(engine), PostgresLedger(engine), TaskSource())
    entry = {"source": "tool_args", "id": good, "source_bot": peer}
    [shown] = asyncio.run(drilldown._resolve("tool_args", [entry], scope))
    assert shown["available"] is True and shown["snippet"] == FACT
    assert SECRET not in json.dumps(shown)
    [other_user] = asyncio.run(drilldown._resolve(
        "tool_args", [{"source": "tool_args", "id": crossed, "source_bot": peer}], scope))
    assert other_user["available"] is False and "snippet" not in other_user
    stranger = RecallScope(rig.bot, "someone-else", rig.session, frozenset(), frozenset({rig.bot, peer}))
    [hidden] = asyncio.run(drilldown._resolve("tool_args", [entry], stranger))
    assert hidden["available"] is False and "snippet" not in hidden
