"""Private-PG proof of the approval-continuation authorship backfill (TASK-1807).

Continuation turns written before the ``system`` author type were stored as
``user:<requester>``. A row may be re-authored only through the full durable
chain: approval request → its ``turn-<continuation_id>`` turn log → that turn's
trigger message carrying the deterministic continuation id → the message's own
session, all agreeing on (bot, user). Each negative case below breaks exactly
one link and must stay untouched.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text

from llm_bawt.approval_policy_store import ToolApprovalPolicyStore
from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
from llm_bawt.memory.postgresql import ensure_bot_partitions
from llm_bawt.message_author_backfill import (
    AUDIT_TABLE,
    ApprovalContinuationAuthorBackfill,
    MessageAuthorBackfill,
)
from llm_bawt.service.approval_continuations import _continuation_identity
from llm_bawt.service.turn_logs import TurnLogStore
from llm_bawt.utils import db
from tests.pg_isolation import require_test_db_config

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]

USER = "operator"
MCP = ("system", "approval-continuation:mcp")
HARNESS = ("system", "approval-continuation:harness")


@pytest.fixture(scope="module")
def stores(private_app_pg):
    config = require_test_db_config()
    approvals = ToolApprovalPolicyStore(config)  # bootstraps tool_approval_requests
    TurnLogStore(config)  # bootstraps turn_logs
    InterBotDeliveryStore(config)  # bootstraps inter_bot_deliveries (the legacy classify joins it)
    engine = create_engine(db.build_postgres_url(config))
    try:
        yield SimpleNamespace(engine=engine, approvals=approvals)
    finally:
        engine.dispose()


class Chain:
    """Builds continuation chains; ``break_*`` knobs sever exactly one link."""

    def __init__(self, stores):
        self.engine, self.approvals = stores.engine, stores.approvals
        tag = uuid.uuid4().hex[:10]
        self.bot, self.other, self.protected = f"cb{tag}", f"cbo{tag}", f"cbprot{tag}"
        self.sessions = {}

    def session(self, bot, user=USER):
        key = (bot, user)
        if key not in self.sessions:
            session_id = str(uuid.uuid4())
            with self.engine.begin() as conn:
                ensure_bot_partitions(conn, bot)
                conn.execute(text("INSERT INTO sessions (id, bot_id, user_id, status) "
                                  "VALUES (:id, :bot, :user, 'archived')"),
                             {"id": session_id, "bot": bot, "user": user})
            self.sessions[key] = session_id
        return self.sessions[key]

    def message(self, bot, message_id, *, role="user", author=("user", USER), session=None):
        with self.engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO messages (bot_id, id, role, content, timestamp, session_id,
                                      author_entity_type, author_entity_id)
                VALUES (:bot, :id, :role, 'Tool result: ok', :at, :session, :type, :author)
            """), {"bot": bot, "id": message_id, "role": role, "at": time.time(),
                   "session": session or self.session(bot), "type": author[0],
                   "author": author[1]})
        return message_id

    def turn(self, turn_id, *, bot, user, trigger):
        with self.engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO turn_logs (id, created_at, path, stream, status, bot_id, user_id,
                                       trigger_message_id)
                VALUES (:id, :at, '/v1/chat/completions', true, 'ok', :bot, :user, :trigger)
            """), {"id": turn_id, "at": datetime.now(timezone.utc), "bot": bot, "user": user,
                   "trigger": trigger})

    def add(self, *, kind="mcp", bot=None, author=("user", USER), role="user",
            no_turn=False, turn_bot=None, turn_user=None, other_trigger=False,
            turn_triggers_other=False, session_user=None, message_bot=None):
        """One continuation; returns (message_bot, message_id) of its trigger row."""
        bot = bot or self.bot
        request_id = f"req-{uuid.uuid4().hex}"
        continuation_id = f"approval-cont-{uuid.uuid4().hex}"
        self.approvals.record_request(
            request_id=request_id, bot_id=bot, user_id=USER, turn_id=f"turn-{uuid.uuid4().hex}",
            backend="claude-code", tool_name="Bash", tool_arguments={}, subject="x",
            grant_key="g", policy_id=None, severity="medium", prompt="",
        )
        with self.engine.begin() as conn:
            # Continuations exist only on resolved requests (a pending mcp row must be routable).
            conn.execute(text("UPDATE tool_approval_requests SET status = 'approved', "
                              "continuation_id = :c, request_kind = :k WHERE id = :id"),
                         {"c": continuation_id, "k": kind, "id": request_id})
        message_id = _continuation_identity(
            SimpleNamespace(continuation_id=continuation_id))["user_message_id"]
        if other_trigger:
            message_id = str(uuid.uuid4())
        row_bot = message_bot or bot
        self.message(row_bot, message_id, role=role, author=author,
                     session=self.session(row_bot, session_user or USER))
        if not no_turn:
            self.turn("turn-" + continuation_id, bot=turn_bot or bot, user=turn_user or USER,
                      trigger=str(uuid.uuid4()) if turn_triggers_other else message_id)
        return row_bot, message_id

    def author(self, ref):
        bot, message_id = ref
        with self.engine.connect() as conn:
            return tuple(conn.execute(text(
                "SELECT author_entity_type, author_entity_id FROM messages "
                "WHERE bot_id = :bot AND id = :id"), {"bot": bot, "id": message_id}).one())

    def backfill(self, cls=ApprovalContinuationAuthorBackfill):
        chain = self

        class Scoped(cls):
            def _params(self):
                # Other tests share the private database: exclude every partition
                # but this test's own (the protected one stays excluded by rule).
                with self.engine.connect() as conn:
                    others = [r[0] for r in conn.execute(text("SELECT DISTINCT bot_id FROM messages"))
                              if r[0] not in {chain.bot, chain.other, chain.protected}]
                params = super()._params()
                params["excluded"] = sorted(set(params["excluded"]) | set(others))
                return params

        return Scoped(self.engine, excluded_partitions=[self.protected])


@pytest.fixture
def chain(stores):
    return Chain(stores)


def test_only_full_chain_rows_are_reauthored_and_reversible(chain):
    mcp = chain.add(kind="mcp")
    harness = chain.add(kind="harness")
    untouched = {
        "no_turn_log": chain.add(no_turn=True),
        "trigger_not_continuation_id": chain.add(other_trigger=True),
        "turn_triggers_other_message": chain.add(turn_triggers_other=True),
        "turn_bot_mismatch": chain.add(turn_bot=chain.other),
        "turn_user_mismatch": chain.add(turn_user="someone-else"),
        "session_user_mismatch": chain.add(session_user="someone-else"),
        "message_bot_mismatch": chain.add(message_bot=chain.other),
        "author_other_user": chain.add(author=("user", "someone-else")),
        "author_bot": chain.add(author=("bot", chain.other)),
        "author_system": chain.add(author=MCP),
        "author_null": chain.add(author=(None, None)),
        "assistant_role": chain.add(role="assistant"),
        "protected": chain.add(bot=chain.protected),
    }
    before = {name: chain.author(ref) for name, ref in untouched.items()}
    # A client-supplied row reusing the real continuation id in another partition.
    reused = chain.message(chain.other, mcp[1])

    backfill = chain.backfill()
    plan = backfill.plan()
    assert plan.counts == {"approval_mcp": 1, "approval_harness": 1}
    assert plan.writable == 2

    with pytest.raises(RuntimeError, match="count mismatch"):
        backfill.apply(expected=plan.writable + 1)
    assert chain.author(mcp) == ("user", USER)  # mismatch rolled back everything

    batch, updated = backfill.apply(expected=plan.writable)
    assert updated == 2
    assert chain.author(mcp) == MCP
    assert chain.author(harness) == HARNESS
    assert {name: chain.author(ref) for name, ref in untouched.items()} == before
    assert chain.author((chain.other, reused)) == ("user", USER)
    assert backfill.plan().writable == 0  # idempotent: a re-run writes nothing

    with chain.engine.connect() as conn:
        audited = conn.execute(text(
            f"SELECT rule, old_type, old_id, new_type, new_id FROM {AUDIT_TABLE} "
            "WHERE batch_id = :b ORDER BY rule"), {"b": batch}).all()
    assert [tuple(row) for row in audited] == [
        ("approval_harness", "user", USER, *HARNESS),
        ("approval_mcp", "user", USER, *MCP),
    ]

    assert backfill.revert(batch) == 2
    assert chain.author(mcp) == ("user", USER)
    assert chain.author(harness) == ("user", USER)
    assert backfill.revert(batch) == 0


def test_apply_writes_only_rows_still_holding_the_classified_author(chain):
    mcp = chain.add(kind="mcp")
    backfill = chain.backfill()
    classify = backfill._classify_sql
    # Simulate authorship changing between classify and the UPDATE: the audit
    # records an old author the row no longer holds, so the guard must refuse
    # it and the batch's count check must roll everything back.
    backfill._classify_sql = lambda: (
        f"SELECT c.bot_id, c.id, c.rule, c.old_type, 'stale' AS old_id, c.new_type, c.new_id "
        f"FROM ({classify()}) c")
    with pytest.raises(RuntimeError, match=r"audited=1 updated=0"):
        backfill.apply(expected=1)
    assert chain.author(mcp) == ("user", USER)


def test_legacy_backfill_still_writes_only_null_authorship(chain):
    """The shared core keeps the TASK-1210 backfill on NULL rows only."""
    null_row = chain.message(chain.bot, str(uuid.uuid4()), author=(None, None))
    authored = chain.message(chain.bot, str(uuid.uuid4()), author=("user", "someone-else"))
    backfill = chain.backfill(MessageAuthorBackfill)
    batch, updated = backfill.apply(expected=backfill.plan().writable)
    assert updated == 1
    assert chain.author((chain.bot, null_row)) == ("user", USER)
    assert chain.author((chain.bot, authored)) == ("user", "someone-else")
    assert backfill.revert(batch) == 1
    assert chain.author((chain.bot, null_row)) == (None, None)
