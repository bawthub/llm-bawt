"""Count reconciliation reserves empty houses against every assignment path."""
from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest
from sqlalchemy import insert

from llm_bawt.house_reconcile_runtime import operate
from llm_bawt.house_retirement import (
    RESERVATIONS, HouseAssignmentRetiring, assert_assignable, reservations,
)
from llm_bawt.house_tenancy import HouseMoveRefused, list_houses, move_in
from tests.test_house_tenancy import Store, state


def test_reservation_blocks_self_move_and_lists_distinct_retiring_status():
    store = Store(alpha={})
    live = state()
    operate(store, live, "reserve", ["h1"], "run-1")
    with pytest.raises(HouseMoveRefused) as exc:
        move_in(store, "alpha", "h1", live)
    assert exc.value.code == "retiring"
    listing = list_houses(store, "alpha", live, private_bots=frozenset())
    assert next(h for h in listing["houses"] if h["house"] == "h1")["status"] == "retiring"
    assert move_in(store, "alpha", None, live).house == "h2"


def test_assignment_writer_guard_and_no_op_unrelated_house():
    store = Store(alpha={})
    operate(store, state(), "reserve", ["h1"], "run-1")
    with store.engine.begin() as conn:
        with pytest.raises(HouseAssignmentRetiring) as exc:
            assert_assignable(conn, "h1", "alpha")
        assert exc.value.code == "house_retiring"
        assert_assignable(conn, "h2", "alpha")
        assert_assignable(conn, None, "alpha")


@pytest.mark.parametrize("overrides,bots", [
    ({}, {"alpha": {"house": "h1", "pinned": True}}),
    ({"busy_streams": frozenset({"agent:commands:house:h1"})}, {}),
    ({"pool": None}, {}),
])
def test_refused_shrink_does_not_reserve_any_target(overrides, bots):
    store = Store(**bots)
    with pytest.raises(ValueError):
        operate(store, state(**overrides), "reserve", ["h2", "h1"], "run-1")
    with store.engine.connect() as conn:
        assert reservations(conn) == {}


def test_unreadable_state_refuses_without_reserving():
    store = Store()
    def unavailable():
        raise OSError("fixture unavailable")
    with pytest.raises(OSError):
        operate(store, unavailable, "reserve", ["h1"], "run-1")
    with store.engine.connect() as conn:
        assert reservations(conn) == {}


def test_repeated_reservation_is_adopted_only_by_same_reconcile():
    store = Store()
    for _ in range(2):
        operate(store, state(), "reserve", ["h1"], "run-1")
    with pytest.raises(ValueError, match="removal refused"):
        operate(store, state(), "reserve", ["h2", "h1"], "run-2")
    with store.engine.connect() as conn:
        assert reservations(conn) == {"h1": "run-1"}


def test_release_requires_published_smaller_pool_and_matching_owner():
    store = Store()
    operate(store, state(), "reserve", ["h1"], "run-1")
    with pytest.raises(ValueError, match="advertised pool"):
        operate(store, state(), "release", ["h1"], "run-1")
    shrunk = state(pool={"h2": "claude-code"})
    with pytest.raises(ValueError, match="another reconciliation"):
        operate(store, shrunk, "release", ["h1"], "run-2")
    for _ in range(2):
        operate(store, shrunk, "release", ["h1"], "run-1")
    with store.engine.connect() as conn:
        assert reservations(conn) == {}


def test_cancel_requires_restored_live_house_and_preserves_other_reservations():
    store = Store()
    with store.engine.begin() as conn:
        conn.execute(insert(RESERVATIONS), [
            {"house": "h1", "reconcile_id": "run-1"},
            {"house": "h2", "reconcile_id": "run-2"},
        ])
    with pytest.raises(ValueError, match="restore and verify"):
        operate(store, state(leases={}), "cancel", ["h1"], "run-1")
    operate(store, state(), "cancel", ["h1"], "run-1")
    with store.engine.connect() as conn:
        assert reservations(conn) == {"h2": "run-2"}


def test_inspection_is_read_only_and_reports_occupancy():
    store = Store(alpha={"house": "h1"})
    rows = operate(store, state(), "inspect", ["h1"], "run-1")["houses"]
    assert rows[0]["reasons"] == ["assigned"]
    with store.engine.connect() as conn:
        assert reservations(conn) == {}


@pytest.mark.parametrize("houses", [[""], ["../h1"], ["H1"], ["h1", "h1"], ["h1"] * 17])
def test_invalid_targets_are_refused(houses):
    store = Store()
    with pytest.raises(ValueError):
        operate(store, state(), "reserve", houses, "run-1")


@pytest.mark.parametrize("run", [None, [], 1, "", "../bad"])
def test_invalid_reconcile_id_is_a_validation_error(run):
    with pytest.raises(ValueError, match="reconcile id"):
        operate(Store(), state(), "inspect", [], run)


def test_inspect_without_targets_includes_configured_and_reserved_houses():
    store = Store()
    operate(store, state(), "reserve", ["h1"], "run-1")
    rows = operate(store, state(pool={"h2": "claude-code"}), "inspect", [], "run-1")["houses"]
    assert {row["house"] for row in rows} == {"h1", "h2"}
    assert next(row for row in rows if row["house"] == "h1")["reservation"] == "run-1"


def test_runtime_inspection_preserves_configured_preview_origin():
    store = Store()
    snapshot = state(preview_hosts={"h1": "https://h1.example.test"})
    rows = operate(store, snapshot, "inspect", ["h1"], "run-1")["houses"]
    assert rows[0]["preview_host"] == "https://h1.example.test"


@pytest.mark.parametrize("pending", [False, True])
def test_json_cli_reuses_stream_rules_for_a_stopped_house(monkeypatch, capsys, pending):
    import redis
    from llm_bawt.house_reconcile_runtime import main
    from tests.test_house_tenancy import FakeRedis

    store = Store()
    groups = {"agent:commands:house:h3": [{"lag": 0, "pending": 1}]} if pending else {}
    client = FakeRedis(groups=groups)
    client.close = lambda: None
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({
        "action": "reserve", "houses": ["h3"], "reconcile_id": "fixture-run",
    })))
    monkeypatch.setattr("llm_bawt.utils.config.Config",
                        lambda: SimpleNamespace(REDIS_URL="redis://fixture/0"))
    monkeypatch.setattr("llm_bawt.utils.db.get_shared_engine", lambda config: store.engine)
    monkeypatch.setattr(redis.Redis, "from_url", lambda *args, **kwargs: client)
    monkeypatch.setattr("llm_bawt.house_pool_config.configured_pool", lambda: [
        {"id": "h3", "harness": "claude-code", "preview_host": "https://h3.example.test"},
    ])
    assert main() == (1 if pending else 0)
    result = json.loads(capsys.readouterr().out)
    assert ("error" in result) == pending
    with store.engine.connect() as conn:
        assert reservations(conn) == ({} if pending else {"h3": "fixture-run"})


def test_json_cli_bootstraps_house_retirements_before_first_read(monkeypatch, capsys):
    """TASK-2637: on a fresh tenant the host runs this CLI as soon as the app
    reports healthy, before anything has created house_retirements."""
    import redis
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import OperationalError
    from sqlalchemy.pool import StaticPool
    from llm_bawt.house_reconcile_runtime import main
    from tests.test_house_tenancy import FakeRedis

    engine = create_engine("sqlite://", poolclass=StaticPool,
                           connect_args={"check_same_thread": False})
    with engine.begin() as conn:  # bot_profiles exists (frontend migrations); retirements does not
        conn.execute(text("CREATE TABLE bot_profiles (slug TEXT PRIMARY KEY, bot_type TEXT, "
                          "agent_backend TEXT, house TEXT, house_pinned BOOLEAN)"))
    with pytest.raises(OperationalError, match="house_retirements"):
        operate(SimpleNamespace(engine=engine), state(), "reserve", ["h1"], "run-0")

    client = FakeRedis()
    client.close = lambda: None
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({
        "action": "reserve", "houses": ["h3"], "reconcile_id": "fixture-run",
    })))
    monkeypatch.setattr("llm_bawt.utils.config.Config",
                        lambda: SimpleNamespace(REDIS_URL="redis://fixture/0"))
    monkeypatch.setattr("llm_bawt.utils.db.get_shared_engine", lambda config: engine)
    monkeypatch.setattr(redis.Redis, "from_url", lambda *args, **kwargs: client)
    monkeypatch.setattr("llm_bawt.house_pool_config.configured_pool", lambda: [
        {"id": "h3", "harness": "claude-code"},
    ])
    assert main() == 0, capsys.readouterr().out
    with engine.connect() as conn:
        assert reservations(conn) == {"h3": "fixture-run"}


def test_ready_schema_runs_no_ddl():
    """A ready tenant must not take DDL locks on bot_profiles (TASK-2637 review)."""
    from sqlalchemy import event
    from llm_bawt.house_retirement import ensure_ready

    store = Store()
    seen = []
    event.listen(store.engine, "before_cursor_execute",
                 lambda conn, cur, stmt, *a: seen.append(stmt.split()[0].upper()))
    ensure_ready(store.engine)
    assert not {"CREATE", "ALTER", "DROP"} & set(seen)
