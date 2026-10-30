"""TASK-1733 PostgreSQL contract: background (maintenance) calls move the balance.

A priced background call charges its provider's counter once; replaying the same
response charges nothing; turn-log repair neither drops nor double-counts it; an
unpriced one makes the balance unknown; per-day spend splits chat from
maintenance.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from sqlalchemy import text

from llm_bawt.service.turn_logs import TurnLogStore
from llm_bawt.service.usage.background_spend import BackgroundSpendRecorder
from llm_bawt.service.usage.balance import ApiBalanceStore, anchored_balance
from tests.pg_isolation import require_test_db_config

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]

PROVIDER = "xai"
PATH = {"key": "xai-chat", "vendor": "xai", "protocol": "chat-completions",
        "auth_mechanism": "api-key", "engine_kind": None}


@pytest.fixture
def rig():
    store = TurnLogStore(require_test_db_config())
    bot = f"testbgs{uuid.uuid4().hex[:12]}"

    def wipe():
        with store.engine.begin() as conn:
            conn.execute(text("DELETE FROM api_balance_ledger"))
            conn.execute(text("DELETE FROM api_balances"))
            conn.execute(text("DELETE FROM turn_logs WHERE bot_id=:bot"), {"bot": bot})

    wipe()
    try:
        yield store, ApiBalanceStore(store.engine), bot
    finally:
        wipe()


def _recorder(rig, pricing=None):
    return BackgroundSpendRecorder({"access_path": "xai-chat", "model_key": "grok-4.3", "pricing": pricing},
                                   engine_getter=lambda: rig[0].engine)


def _response(ticks=None, rid=None):
    usage = {"prompt_tokens": 1000, "completion_tokens": 100}
    if ticks is not None:
        usage["cost_in_usd_ticks"] = ticks
    return {"id": rid or f"resp-{uuid.uuid4().hex}", "usage": usage}


def _chat_turn(rig, cost):
    store, _balances, bot = rig
    turn_id = f"turn-{uuid.uuid4().hex}"
    with store.engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO turn_logs (id, created_at, path, stream, bot_id, user_id, status, user_prompt)
            VALUES (:id, :created, '/test', false, :bot, 'bg-user', 'streaming', 'test')
        """), {"id": turn_id, "bot": bot, "created": datetime.now(timezone.utc)})
    store.update_turn(turn_id=turn_id, status="ok", end_reason="stop",
                      token_usage={"input_tokens": 1, "output_tokens": 1, "access_path": PATH,
                                   "total_cost_usd": cost})


def _counter(rig):
    row = rig[1].read(PROVIDER)
    return row.spent_usd, row.unpriced_turns


def test_background_call_moves_remaining_and_replay_charges_nothing(rig):
    rig[1].set_anchor(PROVIDER, Decimal("50"))
    response = _response(ticks=2_500_000_000)  # $0.25
    _recorder(rig)(response)
    assert _counter(rig) == (Decimal("0.25"), 0)
    _recorder(rig)(response)  # same provider response id
    assert _counter(rig) == (Decimal("0.25"), 0)
    balance, reason = anchored_balance(rig[1].read(PROVIDER), None)
    assert reason is None and balance.remaining_usd == pytest.approx(49.75)
    with rig[0].engine.connect() as conn:
        row = conn.execute(text("SELECT source, access_path_key, usage_json FROM api_balance_ledger")).one()
    assert row.source == "maintenance" and row.access_path_key == "xai-chat"
    assert '"reasoning_tokens"' in row.usage_json and '"provider_reported"' in row.usage_json


def test_catalog_pricing_prices_a_call_without_reported_cost(rig):
    rig[1].set_anchor(PROVIDER, Decimal("50"))
    _recorder(rig, pricing={"input": 1.25, "output": 2.5})(_response())
    assert _counter(rig) == (Decimal("0.00150000"), 0)


def test_repair_keeps_background_entries_without_double_counting(rig):
    rig[1].set_anchor(PROVIDER, Decimal("50"))
    _recorder(rig)(_response(ticks=10_000_000_000))  # $1
    _chat_turn(rig, 0.5)
    assert _counter(rig) == (Decimal("1.5"), 0)
    with rig[0].engine.begin() as conn:
        conn.execute(text("UPDATE api_balances SET spent_usd=999"))
    assert rig[1].repair(PROVIDER).spent_usd == Decimal("1.5")
    assert rig[1].repair(PROVIDER).spent_usd == Decimal("1.5")


def test_unpriced_background_call_is_counted_as_uncounted_not_free(rig):
    rig[1].set_anchor(PROVIDER, Decimal("50"))
    _recorder(rig)(_response())
    assert _counter(rig) == (Decimal(0), 1)
    balance, reason = anchored_balance(rig[1].read(PROVIDER), None)
    assert reason is None and (balance.remaining_usd, balance.unpriced_turns) == (50.0, 1)


def test_daily_spend_splits_chat_from_maintenance(rig):
    since = datetime.now(timezone.utc) - timedelta(days=1)
    _recorder(rig)(_response(ticks=10_000_000_000))
    _recorder(rig)(_response())
    _chat_turn(rig, 0.5)
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    days = rig[1].daily_spend(PROVIDER, since=since, start=today)
    assert len(days) == 1
    assert (days[0].chat_usd, days[0].maintenance_usd, days[0].unpriced_calls) == (0.5, 1.0, 1)
    assert days[0].date == today.strftime("%Y-%m-%d")
