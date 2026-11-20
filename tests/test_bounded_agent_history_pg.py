"""TASK-2395: the bounded window read gives the same seed as the full read.

A fixed snapshot in the runner's private PostgreSQL: summaries beyond the
summary limit, summarized rows, plain and tool-evidence system rows, oversized
rows, equal timestamps and another user's rows. For each budget the seed built
from the window must equal the seed built from the full continuous read: same
ordered message IDs and the same content digest.
"""
import hashlib
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from llm_bawt.memory.postgresql import (
    PARENT_TABLES,
    PostgreSQLShortTermManager,
    partition_name,
)
from llm_bawt.utils.history import HistoryManager, raw_history_cap
from tests.pg_isolation import require_test_db_config

pytestmark = pytest.mark.integration

SETTINGS = {"summary_count": 3, "history_tokens": 12000, "history_max_age_hours": 0}


def _rows():
    """(role, content, timestamp, summarized, user) in a deterministic order."""
    rows = []
    ts = 1000.0
    for i in range(6):
        rows.append(("summary", f"summary {i} " + "s" * 400, ts, False, "a"))
        ts += 1
    for i in range(40):  # already rolled into a summary: never raw context
        rows.append(("user" if i % 2 == 0 else "assistant", f"old {i}", ts, True, "a"))
        ts += 1
    for i in range(160):
        role = ("user", "assistant", "system")[i % 3]
        body = f"turn {i} " + "x" * (37 * (i % 11))
        if role == "system":
            body = ("[Tool Results] " if i % 2 else "persona ") + "y" * 900
        if i in (70, 140):
            body = "huge " + "z" * 60000  # one row larger than small budgets
        rows.append((role, body, ts, False, "a"))
        if i % 5 == 0:
            rows.append(("assistant", f"tie {i}", ts, False, "a"))  # equal timestamp
        if i % 9 == 0:
            rows.append(("user", f"other user {i}", ts, False, "b"))
        ts += 1
    return rows


@pytest.fixture
def snapshot(private_app_pg):
    config = require_test_db_config()
    bot = "_t2395_" + uuid.uuid4().hex[:12]
    manager = PostgreSQLShortTermManager(config, bot_id=bot)
    backend = manager._backend
    sessions = {
        "a": manager.get_or_create_active_session(bot_id=bot, user_id="user_a"),
        "b": manager.get_or_create_active_session(bot_id=bot, user_id="user_b"),
    }
    table = backend._messages_table_name
    with backend.engine.begin() as conn:
        for role, content, ts, summarized, user in _rows():
            conn.execute(text(f"""INSERT INTO {table}
                (id, bot_id, role, content, timestamp, session_id, summarized,
                 author_entity_type, author_entity_id)
                VALUES (:id, :bot, :role, :content, :ts, :sid, :summarized, 'bot', :bot)"""),
                {"id": str(uuid.uuid4()), "bot": bot, "role": role, "content": content,
                 "ts": ts, "sid": sessions[user], "summarized": summarized})
    yield manager
    with backend.engine.begin() as conn:
        for p in PARENT_TABLES:
            conn.execute(text(f"DROP TABLE IF EXISTS {partition_name(p, bot)}"))
        conn.execute(text("DELETE FROM sessions WHERE bot_id=:bot"), {"bot": bot})


def _history(manager, settings=SETTINGS):
    return HistoryManager(
        client=None,
        config=SimpleNamespace(HISTORY_FILE=None, SYSTEM_MESSAGE="default system"),
        db_backend=manager, bot_id=manager.bot_id, user_id="user_a",
        settings_getter=lambda key, fallback: settings.get(key, fallback),
    )


def _seed(hm, budget, include_history=True, include_summaries=True):
    payload = hm.build_context_payload(
        include_history=include_history, include_summaries=include_summaries,
        delivery="seed", max_tokens=budget,
    )
    msgs = payload.seed_messages
    digest = hashlib.sha256(
        "\x00".join(f"{m.role}:{m.content}" for m in msgs).encode()
    ).hexdigest()
    return [m.db_id for m in msgs], digest


@pytest.mark.parametrize("budget", [20, 200, 2000, 9000, 16000, 40000, 200000])
@pytest.mark.parametrize("scope", [(True, True), (True, False), (False, True)])
def test_window_seed_matches_full_read(snapshot, budget, scope):
    full = _history(snapshot)
    full.load_history()
    bounded = _history(snapshot)
    cap = raw_history_cap(budget, SETTINGS["history_tokens"])
    bounded.load_history(window=(cap, SETTINGS["summary_count"]))

    assert _seed(bounded, budget, *scope) == _seed(full, budget, *scope)
    # Seed delivery drops system rows, so the window never reads them.
    assert not any(m.role == "system" for m in bounded.messages)
    # Bounded means bounded: a small budget reads a small slice.
    if budget <= 200:
        assert len(bounded.messages) < len(full.messages) // 4
    elif budget <= 2000:
        assert len(bounded.messages) < len(full.messages) // 3


@pytest.mark.parametrize("history_tokens", [300, 1500, 5000])
def test_history_tokens_far_below_budget_bounds_the_read(snapshot, history_tokens):
    # Live failure: a 1M-window agent with history_tokens=20000 passed the
    # physical budget, so the read barely shrank. The cap is history_tokens.
    settings = {**SETTINGS, "history_tokens": history_tokens}
    budget = 900000
    full = _history(snapshot, settings)
    full.load_history()
    bounded = _history(snapshot, settings)
    bounded.load_history(window=(raw_history_cap(budget, history_tokens), 3))

    for scope in [(True, True), (True, False), (False, True)]:
        assert _seed(bounded, budget, *scope) == _seed(full, budget, *scope)
    assert len(bounded.messages) < len(full.messages) // 2


def test_newest_turn_floor_is_read_past_a_tiny_cap(snapshot):
    # history_tokens smaller than the newest turn: the assembler floors the
    # whole turn in, so the window must read back to its user row.
    settings = {**SETTINGS, "history_tokens": 1}
    full = _history(snapshot, settings)
    full.load_history()
    bounded = _history(snapshot, settings)
    bounded.load_history(window=(raw_history_cap(200000, 1), 3))
    assert _seed(bounded, 200000) == _seed(full, 200000)


def test_window_matches_full_order_for_the_rows_it_returns(snapshot):
    full = _history(snapshot)
    full.load_history()
    bounded = _history(snapshot)
    bounded.load_history(window=(9000, 3))
    ids = [m.db_id for m in full.messages]
    window_ids = [m.db_id for m in bounded.messages]
    assert window_ids == [i for i in ids if i in set(window_ids)]
    assert all(m.session_id for m in bounded.messages)
    assert not any(m.content.startswith("other user") for m in bounded.messages)


def test_summary_count_zero_matches(snapshot):
    settings = {**SETTINGS, "summary_count": 0}
    full = _history(snapshot, settings)
    full.load_history()
    bounded = _history(snapshot, settings)
    bounded.load_history(window=(9000, 0))
    assert _seed(bounded, 9000) == _seed(full, 9000)
    assert not any(m.role == "summary" for m in bounded.messages)


def test_small_history_counts_match_for_cold_start(private_app_pg):
    config = require_test_db_config()
    bot = "_t2395s_" + uuid.uuid4().hex[:12]
    manager = PostgreSQLShortTermManager(config, bot_id=bot)
    sid = manager.get_or_create_active_session(bot_id=bot, user_id="user_a")
    table = manager._backend._messages_table_name
    try:
        with manager._backend.engine.begin() as conn:
            for i, body in enumerate(["q " + "z" * 80000, "a", "persona"]):
                conn.execute(text(f"""INSERT INTO {table}
                    (id, bot_id, role, content, timestamp, session_id, summarized,
                     author_entity_type, author_entity_id)
                    VALUES (:id, :bot, :role, :content, :ts, :sid, false, 'bot', :bot)"""),
                    {"id": str(uuid.uuid4()), "bot": bot,
                     "role": ("user", "assistant", "system")[i], "content": body,
                     "ts": 10 + i, "sid": sid})
        full = _history(manager)
        full.load_history()
        bounded = _history(manager)
        bounded.load_history(window=(10, 3))
        # Cold-start priming checks len(messages) <= 3 after the new prompt
        # is appended; the bounded read must land on the same side.
        assert len(full.messages) == 3
        assert len(bounded.messages) == 3
        assert {m.db_id for m in full.messages if m.role != "system"} <= {
            m.db_id for m in bounded.messages}
        for hm in (full, bounded):
            hm.add_message("user", "next prompt")
        assert (len(bounded.messages) <= 3) == (len(full.messages) <= 3)
    finally:
        with manager._backend.engine.begin() as conn:
            for p in PARENT_TABLES:
                conn.execute(text(f"DROP TABLE IF EXISTS {partition_name(p, bot)}"))
            conn.execute(text("DELETE FROM sessions WHERE bot_id=:bot"), {"bot": bot})


def test_window_reads_one_snapshot_across_a_concurrent_summarization(snapshot, monkeypatch):
    # Codex finding 2: a summary committed after the summary query (here,
    # after the first raw page) must not drop both itself and the rows it
    # replaced, or shift later pages. The read reflects the pre-commit state.
    from llm_bawt.utils import token_estimate

    before = [m.db_id for m in snapshot.get_context_window(
        token_budget=9000, summary_limit=3, user_id="user_a", page_size=10,
    )]
    table = snapshot._backend._messages_table_name
    real = token_estimate.estimate_message_tokens
    fired = []

    def estimate_then_summarize(content):
        if not fired:
            fired.append(True)
            with snapshot._backend.engine.begin() as conn:
                sid = conn.execute(text(f"SELECT session_id FROM {table} LIMIT 1")).scalar()
                conn.execute(text(f"UPDATE {table} SET summarized = TRUE WHERE role <> 'summary'"))
                conn.execute(text(f"""INSERT INTO {table}
                    (id, bot_id, role, content, timestamp, session_id, summarized,
                     author_entity_type, author_entity_id)
                    VALUES (:id, :bot, 'summary', 'new summary', 99999, :sid, false, 'bot', :bot)"""),
                    {"id": str(uuid.uuid4()), "bot": snapshot.bot_id, "sid": sid})
        return real(content)

    monkeypatch.setattr(token_estimate, "estimate_message_tokens", estimate_then_summarize)
    during = [m.db_id for m in snapshot.get_context_window(
        token_budget=9000, summary_limit=3, user_id="user_a", page_size=10,
    )]
    assert fired and during == before
    assert len(set(during)) == len(during)
