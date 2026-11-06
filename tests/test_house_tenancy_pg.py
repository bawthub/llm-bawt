"""Self-move house tenancy on the real profile store (TASK-2118).

Two bots racing for one house: held behind the house assignment lock, then
exactly one moves in and the other is refused ``occupied``. A manual profile
write of ``house`` waits on the same lock, so a self-move sees it. The write
is audited as the bot itself and never pins. Fixture bots only, isolated PostgreSQL.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import text

from llm_bawt.house_tenancy import HouseMoveRefused, HouseState, move_in, move_out
from tests.pg_lock_wait import wait_blocked_by
from tests.test_bot_tags_pg import rig, upsert  # noqa: F401

pytestmark = pytest.mark.integration


def LIVE():  # noqa: N802
    return HouseState(leases={"h1": {"claude-code": "1"}, "h2": {"claude-code": "1"}},
                      pool={"h1": "claude-code", "h2": "claude-code"})


def agent(store, slug, **fields):
    upsert(store, slug, bot_type="agent", agent_backend="claude-code", **fields)


def outcome(future):
    try:
        return future.result(timeout=10).house
    except HouseMoveRefused as refused:
        return refused.code


def test_two_bots_racing_for_one_house_get_one_winner_and_one_refusal(rig):  # noqa: F811
    store = rig.store()
    agent(store, "fixture-a")
    agent(store, "fixture-b")
    with rig.engine.connect() as holder:
        holder.execute(text("BEGIN"))
        holder.execute(text("SELECT pg_advisory_xact_lock(2118, 1)"))
        pid = holder.execute(text("SELECT pg_backend_pid()")).scalar_one()
        with ThreadPoolExecutor(2) as pool:
            racers = []
            for slug in ("fixture-a", "fixture-b"):
                racers.append(pool.submit(move_in, store, slug, "h1", LIVE))
                wait_blocked_by(rig.engine, pid, racers[-1])
            holder.execute(text("ROLLBACK"))
            outcomes = sorted(outcome(f) for f in racers)
    assert outcomes == ["h1", "occupied"]
    with rig.engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT slug, house_pinned FROM bot_profiles WHERE house = 'h1'")).all()
    assert len(rows) == 1 and rows[0].house_pinned is False
    winner = rows[0].slug
    with rig.engine.connect() as conn:
        audit = conn.execute(text(
            "SELECT changed_by, changes FROM bot_profile_changes "
            "WHERE bot_slug = :s ORDER BY id DESC LIMIT 1"), {"s": winner}).one()
    assert audit.changed_by == f"bot:{winner}"
    assert audit.changes["house"] == {"before": None, "after": "h1"}


def test_any_free_skips_the_occupied_house_and_move_out_frees_it(rig):  # noqa: F811
    store = rig.store()
    agent(store, "fixture-a")
    agent(store, "fixture-b")
    assert move_in(store, "fixture-a", "h1", LIVE).house == "h1"
    with pytest.raises(HouseMoveRefused):
        move_in(store, "fixture-b", "h1", LIVE)
    assert move_in(store, "fixture-b", None, LIVE).house == "h2"
    out = move_out(store, "fixture-a")
    assert (out.vacated, out.vacated_empty) == ("h1", True)
    assert store.get("fixture-a").house is None
    assert store.get("fixture-b").house == "h2"


def test_a_manual_house_write_takes_the_same_lock_and_a_self_move_then_sees_it(rig):  # noqa: F811
    store = rig.store()
    agent(store, "fixture-a")
    agent(store, "fixture-b")
    with rig.engine.connect() as holder:
        holder.execute(text("BEGIN"))
        holder.execute(text("SELECT pg_advisory_xact_lock(2118, 1)"))
        pid = holder.execute(text("SELECT pg_backend_pid()")).scalar_one()
        with ThreadPoolExecutor(1) as pool:
            manual = pool.submit(store.upsert, {"slug": "fixture-a", "house": "h1"},
                                 require_existing=True)
            wait_blocked_by(rig.engine, pid, manual)
            holder.execute(text("ROLLBACK"))
            manual.result(timeout=10)
    # A write that doesn't touch house never takes it.
    with rig.engine.connect() as holder:
        holder.execute(text("BEGIN"))
        holder.execute(text("SELECT pg_advisory_xact_lock(2118, 1)"))
        store.upsert({"slug": "fixture-b", "description": "unrelated"}, require_existing=True)
        holder.execute(text("ROLLBACK"))
    with pytest.raises(HouseMoveRefused) as refused:
        move_in(store, "fixture-b", "h1", LIVE)
    assert refused.value.code == "occupied"


def test_a_pinned_bot_leaves_and_its_pin_goes_with_it(rig):  # noqa: F811
    store = rig.store()
    agent(store, "fixture-a", house="h1", house_pinned=True)
    out = move_out(store, "fixture-a")
    row = store.get("fixture-a")
    assert out.vacated == "h1" and row.house is None and row.house_pinned is False
