"""Private PostgreSQL contract for the per-provider API balance counter (TASK-1461).

The counter moves only in the transaction that ends a turn (every terminal
writer), by the change in a turn's ledgered cost: replay is a no-op, a
revision charges the difference,
turns before the anchor or on other paths never count, and an unpriced turn
makes the balance unknown instead of charging a guess. Repair, re-anchor,
creation backfill and legacy-setting migration rebuild the same numbers.
"""
from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from sqlalchemy import text

from llm_bawt.service.turn_logs import TurnLogStore
from llm_bawt.service.usage.balance import (
    ApiBalanceStore,
    anchored_balance,
    balance_update_for_turn,
    bootstrap_balance_schema,
)
from tests.pg_isolation import require_test_db_config

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]

PROVIDER = "anthropic_api"
API_PATH = {"key": "anthropic-api", "vendor": "anthropic", "protocol": "anthropic-messages",
            "auth_mechanism": "api-key", "engine_kind": None}


@pytest.fixture
def rig():
    config = require_test_db_config()
    store = TurnLogStore(config)
    bot = f"testbal{uuid.uuid4().hex[:12]}"

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


def _usage(cost=None, path=API_PATH):
    usage = {"input_tokens": 10, "output_tokens": 5}
    if path is not None:
        usage["access_path"] = path
    if cost is not None:
        usage["total_cost_usd"] = cost
    return usage


def _open_turn(rig, created_at=None) -> str:
    store, _balances, bot = rig
    turn_id = f"turn-{uuid.uuid4().hex}"
    with store.engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO turn_logs (id, created_at, path, stream, bot_id, user_id, status, user_prompt)
            VALUES (:id, :created, '/test', false, :bot, 'balance-user', 'streaming', 'test')
        """), {"id": turn_id, "bot": bot,
               "created": created_at or datetime.now(timezone.utc)})
    return turn_id


def _finish(rig, turn_id, usage) -> None:
    rig[0].update_turn(turn_id=turn_id, status="ok", end_reason="stop", token_usage=usage)


def _turn(rig, usage, created_at=None) -> str:
    turn_id = _open_turn(rig, created_at)
    _finish(rig, turn_id, usage)
    return turn_id


def _counter(rig):
    row = rig[1].read(PROVIDER)
    return (row.spent_usd, row.unpriced_turns) if row else None


def _ledger(rig, turn_id):
    with rig[0].engine.connect() as conn:
        return conn.execute(text("SELECT provider, access_path_key, cost_usd FROM api_balance_ledger"
                                 " WHERE turn_id=:t"), {"t": turn_id}).first()


def test_finalize_charges_once_and_a_revision_charges_only_the_difference(rig):
    rig[1].set_anchor(PROVIDER, Decimal("100"))
    turn = _turn(rig, _usage(2.5))
    assert _counter(rig) == (Decimal("2.5"), 0)
    _finish(rig, turn, _usage(2.5))  # replayed finalize
    rig[0].update_turn(turn_id=turn, response_text="late enrichment")  # terminal update, same cost
    assert _counter(rig) == (Decimal("2.5"), 0)
    _finish(rig, turn, _usage("3.10000001"))  # revised cost
    assert _counter(rig) == (Decimal("3.10000001"), 0)
    assert tuple(_ledger(rig, turn)) == (PROVIDER, "anthropic-api", Decimal("3.10000001"))
    balance, reason = anchored_balance(rig[1].read(PROVIDER), None)
    assert reason is None and balance.remaining_usd == pytest.approx(96.89999999)


def test_open_turn_usage_is_not_charged_until_the_turn_ends(rig):
    rig[1].set_anchor(PROVIDER, Decimal("100"))
    turn = _open_turn(rig)
    rig[0].update_turn(turn_id=turn, token_usage=_usage(1.0))  # partial checkpoint
    assert _counter(rig) == (Decimal(0), 0) and _ledger(rig, turn) is None
    _finish(rig, turn, _usage(1.5))
    assert _counter(rig) == (Decimal("1.5"), 0)


def test_only_turns_after_the_anchor_on_a_balance_path_count(rig):
    before = _open_turn(rig, datetime.now(timezone.utc) - timedelta(hours=1))
    rig[1].set_anchor(PROVIDER, Decimal("50"))
    _finish(rig, before, _usage(9))
    oauth = _turn(rig, _usage(7, path={"key": "anthropic-oauth"}))
    unfrozen = _turn(rig, _usage(5, path=None))  # alias intent is not proof of the credential
    counted = _turn(rig, _usage(1.25))
    assert _counter(rig) == (Decimal("1.25"), 0)
    assert _ledger(rig, before).cost_usd == Decimal(9)  # ledgered for spend detail, not counted
    assert _ledger(rig, oauth) is None and _ledger(rig, unfrozen) is None
    assert _ledger(rig, counted).cost_usd == Decimal("1.25")


@pytest.mark.parametrize("bad_cost", [None, -1, "NaN", "Infinity", True, "abc"])
def test_unpriced_turn_fails_closed_until_it_is_priced(rig, bad_cost):
    rig[1].set_anchor(PROVIDER, Decimal("100"))
    _turn(rig, _usage(1))
    turn = _turn(rig, _usage(bad_cost))
    assert _counter(rig) == (Decimal(1), 1)
    balance, reason = anchored_balance(rig[1].read(PROVIDER), None)
    assert balance is None and "no recorded cost" in reason
    _finish(rig, turn, _usage(bad_cost))  # replay stays one unpriced turn
    assert _counter(rig) == (Decimal(1), 1)
    _finish(rig, turn, _usage(0.75))
    assert _counter(rig) == (Decimal("1.75"), 0)


def test_reanchor_restarts_the_count_and_old_turns_never_return(rig):
    rig[1].set_anchor(PROVIDER, Decimal("100"))
    old = _turn(rig, _usage(4))
    _turn(rig, _usage(None))
    rig[1].set_anchor(PROVIDER, Decimal("80"))
    assert _counter(rig) == (Decimal(0), 0)
    _finish(rig, old, _usage(6))  # a late revision of a pre-anchor turn
    _turn(rig, _usage(2))
    assert _counter(rig) == (Decimal(2), 0)
    assert rig[1].repair(PROVIDER).spent_usd == Decimal(2)
    assert rig[1].clear(PROVIDER) and rig[1].read(PROVIDER) is None
    _turn(rig, _usage(3))  # no anchor: ledgered only
    assert rig[1].read(PROVIDER) is None


def test_repair_rebuilds_the_counter_and_ledger_from_turn_logs(rig):
    rig[1].set_anchor(PROVIDER, Decimal("100"))
    _turn(rig, _usage(1.5))
    _turn(rig, _usage(2))
    _turn(rig, _usage(None))
    expected = _counter(rig)
    with rig[0].engine.begin() as conn:
        conn.execute(text("DELETE FROM api_balance_ledger"))
        conn.execute(text("UPDATE api_balances SET spent_usd=999, unpriced_turns=0"))
    repaired = rig[1].repair(PROVIDER)
    assert (repaired.spent_usd, repaired.unpriced_turns) == expected == (Decimal("3.5"), 1)


def test_concurrent_finalizes_never_lose_a_charge(rig):
    rig[1].set_anchor(PROVIDER, Decimal("100"))
    turns = [_open_turn(rig) for _ in range(12)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda t: _finish(rig, t, _usage("0.125")), turns + turns))
    assert _counter(rig) == (Decimal("1.5"), 0)


def test_ledger_spend_windows_need_no_turn_log_scan(rig):
    now = datetime.now(timezone.utc)
    _turn(rig, _usage(1), now - timedelta(days=10))
    _turn(rig, _usage(2), now - timedelta(days=3))
    _turn(rig, _usage(4), now - timedelta(minutes=1))
    starts = (now - timedelta(hours=1), now - timedelta(days=7), now - timedelta(days=30))
    assert rig[1].spend(PROVIDER, since=now - timedelta(days=60), starts=starts) == (4, 6, 7)
    assert rig[1].spend(PROVIDER, since=now - timedelta(days=5), starts=starts) == (4, 6, 6)


def test_turn_complete_update_reads_the_committed_counter(rig, monkeypatch):
    from llm_bawt.service import usage as usage_registry

    monkeypatch.setattr(usage_registry.get_adapter(PROVIDER), "credential_connected_at", lambda: None)
    rig[1].set_anchor(PROVIDER, Decimal("20"))
    turn = _turn(rig, _usage(0.5))
    with rig[0].engine.connect() as conn:
        raw = conn.execute(text("SELECT token_usage_json FROM turn_logs WHERE id=:t"), {"t": turn}).scalar_one()
    update = balance_update_for_turn(rig[0].engine, raw)
    assert update["provider"] == PROVIDER and update["access_path_key"] == "anthropic-api"
    assert update["balance"]["remaining_usd"] == 19.5 and update["balance"]["used_usd"] == 0.5
    assert update["updated_at"] == update["balance"]["updated_at"]


def test_schema_creation_backfills_the_ledger_from_turn_logs(rig):
    before = _turn(rig, _usage(1.5))
    other = _turn(rig, _usage(3, path={"key": "anthropic-oauth"}))
    with rig[0].engine.begin() as conn:
        conn.execute(text("DROP TABLE api_balance_ledger"))
        bootstrap_balance_schema(conn)
    assert _ledger(rig, before).cost_usd == Decimal("1.5")
    assert _ledger(rig, other) is None


def test_legacy_runtime_setting_anchor_migrates_once(rig):
    from llm_bawt.runtime_settings import RuntimeSettingsStore

    settings = RuntimeSettingsStore(require_test_db_config())
    as_of = datetime.now(timezone.utc) - timedelta(hours=2)
    _turn(rig, _usage(9), as_of - timedelta(hours=1))
    _turn(rig, _usage(1.25), as_of + timedelta(hours=1))
    _turn(rig, _usage(None), as_of + timedelta(hours=1))
    settings.set_value("global", "*", "anthropic_api_balance_anchor",
                       {"amount_usd": "157", "as_of": as_of.isoformat()})
    with rig[0].engine.begin() as conn:
        bootstrap_balance_schema(conn)
    row = rig[1].read(PROVIDER)
    assert (row.anchor_usd, row.anchor_at, row.spent_usd, row.unpriced_turns) == (
        Decimal(157), as_of, Decimal("1.25"), 1)
    assert "anthropic_api_balance_anchor" not in settings.get_scope_settings("global", "*")
    rig[1].set_anchor(PROVIDER, Decimal("40"))
    with rig[0].engine.begin() as conn:
        bootstrap_balance_schema(conn)  # nothing left to migrate: the new anchor stands
    assert rig[1].read(PROVIDER).anchor_usd == Decimal(40)


def test_invalid_legacy_anchor_is_dropped_not_trusted(rig):
    from llm_bawt.runtime_settings import RuntimeSettingsStore

    settings = RuntimeSettingsStore(require_test_db_config())
    settings.set_value("global", "*", "anthropic_api_balance_anchor", {"amount_usd": "-5", "as_of": "yesterday"})
    with rig[0].engine.begin() as conn:
        bootstrap_balance_schema(conn)
    assert rig[1].read(PROVIDER) is None
    assert "anthropic_api_balance_anchor" not in settings.get_scope_settings("global", "*")


# -- every terminal writer charges (Loopy r1 Blocker 1) ------------------------


def test_stop_or_recovery_settle_charges_the_checkpointed_cost(rig):
    from sqlmodel import Session, select

    from llm_bawt.service.turn_logs import TurnLog
    from llm_bawt.service.turn_termination import settle_open_turn

    store, balances, _bot = rig
    balances.set_anchor(PROVIDER, Decimal("100"))
    turn = _open_turn(rig)
    store.update_turn(turn_id=turn, token_usage=_usage(1.25))  # partial checkpoint
    with Session(store.engine) as session:
        row = session.exec(select(TurnLog).where(TurnLog.id == turn).with_for_update()).one()
        assert settle_open_turn(session, row, audit_key="recovery", audit={"cause": "orphaned_by_restart"},
                                now=datetime.now(timezone.utc))
        session.commit()
    assert _counter(rig) == (Decimal("1.25"), 0)
    _finish(rig, turn, _usage(1.25))  # the late worker finalize replays: no double charge
    assert _counter(rig) == (Decimal("1.25"), 0)


def test_stale_reap_charges_and_returns_the_balance_update(rig, monkeypatch):
    from llm_bawt.service import usage as usage_registry

    monkeypatch.setattr(usage_registry.get_adapter(PROVIDER), "credential_connected_at", lambda: None)
    store, balances, bot = rig
    balances.set_anchor(PROVIDER, Decimal("100"))
    stale = _open_turn(rig)
    store.update_turn(turn_id=stale, token_usage=_usage(3))
    current = _open_turn(rig, datetime.now(timezone.utc) + timedelta(seconds=1))
    [reaped] = store.reap_other_open_turns(bot_id=bot, current_turn_id=current)
    assert reaped["id"] == stale and _counter(rig) == (Decimal(3), 0)
    assert reaped["balance_update"]["balance"]["remaining_usd"] == 97.0


def test_expired_delivery_settle_charges_the_checkpointed_cost(rig):
    from llm_bawt.inter_bot_delivery import InterBotDeliveryStore

    store, balances, bot = rig
    deliveries = InterBotDeliveryStore(require_test_db_config())
    balances.set_anchor(PROVIDER, Decimal("100"))
    turn = _open_turn(rig)
    store.update_turn(turn_id=turn, token_usage=_usage(0.5))
    delivery = f"bal-{uuid.uuid4().hex}"
    with store.engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO inter_bot_deliveries (id, sender_bot_id, target_bot_id, message, user_message_id,
                turn_id, status, author_entity_id, transport_accepted_at)
            VALUES (:id, 'sender', :bot, 'm', :msg, :turn, 'QUEUED', 'sender', now() - interval '8 days')
        """), {"id": delivery, "bot": bot, "msg": f"msg-{delivery}", "turn": turn})
    try:
        assert delivery in [r.id for r in deliveries.fail_expired_accepted_queued()]
        assert store.get_turn(turn).ended_at is not None
        assert _counter(rig) == (Decimal("0.5"), 0)
    finally:
        with store.engine.begin() as conn:
            conn.execute(text("DELETE FROM inter_bot_deliveries WHERE id=:id"), {"id": delivery})


# -- an anchor write never misses a racing charge (Loopy r1 Blocker 2) ---------


def test_a_turn_finishing_while_the_first_anchor_commits_is_charged(rig, monkeypatch):
    from threading import Event, Thread

    from llm_bawt.service.usage import balance as balance_module

    store, balances, _bot = rig
    inserted, resume, errors = Event(), Event(), []
    real_recount = balance_module._recount

    def paused_recount(conn, provider):
        inserted.set()  # anchor_at taken and the row inserted; the provider lock is held
        if not resume.wait(10):
            raise TimeoutError("anchor did not resume")
        real_recount(conn, provider)

    monkeypatch.setattr(balance_module, "_recount", paused_recount)

    def run(fn):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 — surfaced by the assertion below
            errors.append(exc)

    anchor = Thread(target=run, args=(lambda: balances.set_anchor(PROVIDER, Decimal("100")),))
    anchor.start()
    assert inserted.wait(10)
    turn = _open_turn(rig)  # created after anchor_at, before the anchor row is visible
    finish = Thread(target=run, args=(lambda: _finish(rig, turn, _usage(2)),))
    finish.start()
    finish.join(1)
    assert finish.is_alive()  # waits on the provider lock instead of missing the anchor
    resume.set()
    anchor.join(10)
    finish.join(10)
    assert not errors and not anchor.is_alive() and not finish.is_alive()
    assert _counter(rig) == (Decimal(2), 0)


def test_anchor_counts_a_turn_already_ledgered_after_its_timestamp(rig):
    # A turn stamped after anchor_at (another process's clock) that finished first.
    _turn(rig, _usage(2), datetime.now(timezone.utc) + timedelta(minutes=1))
    row = rig[1].set_anchor(PROVIDER, Decimal("100"))
    assert (row.spent_usd, row.unpriced_turns) == _counter(rig) == (Decimal(2), 0)
