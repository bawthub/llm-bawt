"""Isolated snapshot primitive only; no live bots, DB, worker or resolver wiring."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from threading import Barrier
from types import MappingProxyType
from unittest.mock import Mock

import pytest

from llm_bawt.bot_access import BotAccessError, BotPolicyResolver, TrustedCaller
from llm_bawt.bot_access.snapshot import PolicySnapshot, SnapshotUnavailable, policy_snapshot

ROWS = [
    {"slug": "fixture-owner", "include_in_global_search": True},
    {"slug": "fixture-peer", "include_in_global_search": True},
]


def view(*, private=False, caller=None):
    rows = [dict(row) for row in ROWS]
    rows[1]["include_in_global_search"] = not private
    return BotPolicyResolver._compile(rows, caller)


@pytest.mark.parametrize("reason", ["snapshot_cold", "snapshot_busy", "snapshot_timeout",
    "snapshot_database", "snapshot_validation", "snapshot_disconnected", "snapshot_revoked"])
def test_cold_and_invalidated_snapshots_fail_closed_with_bounded_reason(reason):
    state = PolicySnapshot()
    if reason != "snapshot_cold":
        state.invalidate(reason)
    with pytest.raises(SnapshotUnavailable, match=f"^{reason}$"):
        state.get()
    assert state.diagnostics()["state"] == reason


def test_install_shares_immutable_callerless_view():
    state = PolicySnapshot(clock=lambda: 0)
    policy = view()
    assert state.install(policy, 1, 10, state.generation())
    assert state.get() is policy
    with pytest.raises(TypeError):
        state.get().partitions["fixture-peer"] = "wrong"
    with pytest.raises(TypeError):
        state.get().tags.tags["fixture-peer"] = frozenset({"private"})
    with pytest.raises(FrozenInstanceError):
        state.get().caller = None
    caller = TrustedCaller("fixture-owner", "fixture-user", "service")
    with pytest.raises(ValueError, match="snapshot_validation"):
        state.install(view(caller=caller), 2, 10, state.generation())


@pytest.mark.parametrize("deadline", [0, -1, float("nan")])
def test_expired_or_nan_install_is_not_ready(deadline):
    state = PolicySnapshot(clock=lambda: 0)
    assert not state.install(view(), 1, deadline, state.generation())
    with pytest.raises(SnapshotUnavailable):
        state.get()


def test_expiry_does_not_refresh_or_extend_itself():
    now = [0]
    state = PolicySnapshot(clock=lambda: now[0])
    assert state.install(view(), 1, 10, state.generation())
    now[0] = 10
    for _ in range(5):
        with pytest.raises(SnapshotUnavailable, match="snapshot_expired"):
            state.get()
    assert state.diagnostics() == {
        "state": "snapshot_expired", "revision": 1, "lease_remaining_seconds": 0.0,
        "refreshes": 1, "failures": 0,
    }


def test_unchanged_background_refresh_keeps_inflight_authority_current():
    now = [0]
    state = PolicySnapshot(clock=lambda: now[0])
    original = view()
    generation = state.generation()
    assert state.install(original, 1, 5, generation)
    now[0] = 2
    assert state.install(view(), 2, 7, generation)
    state.require_current(original)
    assert state.get() is original
    assert state.diagnostics()["revision"] == 2
    assert state.diagnostics()["lease_remaining_seconds"] == 5


def test_changed_refresh_revokes_inflight_authority_even_without_notification():
    state = PolicySnapshot(clock=lambda: 0)
    original = view()
    assert state.install(original, 1, 5, state.generation())
    assert state.install(view(private=True), 2, 7, state.generation())
    with pytest.raises(SnapshotUnavailable, match="snapshot_revoked"):
        state.require_current(original)


@pytest.mark.parametrize("changed_field", ["tags", "column_private", "catalog", "partitions"])
def test_each_policy_decision_input_change_replaces_identity(changed_field):
    state = PolicySnapshot(clock=lambda: 0)
    original = view()
    assert state.install(original, 1, 5, state.generation())
    if changed_field == "partitions":
        changed = replace(original, partitions=MappingProxyType({
            **original.partitions, "fixture-new": "fixturenew",
        }))
    elif changed_field == "tags":
        changed = replace(original, tags=replace(original.tags, tags=MappingProxyType({
            **original.tags.tags, "fixture-peer": frozenset({"no_global_search"}),
        })))
    elif changed_field == "column_private":
        changed = replace(original, tags=replace(
            original.tags, column_private=frozenset({"fixture-peer"})))
    else:
        changed = replace(original, tags=replace(
            original.tags, catalog=original.tags.catalog | {"fixture-role"}))
    assert state.install(changed, 2, 7, state.generation())
    assert state.get() is changed
    with pytest.raises(SnapshotUnavailable, match="snapshot_revoked"):
        state.require_current(original)


def test_successful_equal_refresh_does_not_revive_expired_authority():
    now = [0]
    state = PolicySnapshot(clock=lambda: now[0])
    original = view()
    assert state.install(original, 1, 5, state.generation())
    now[0] = 5
    with pytest.raises(SnapshotUnavailable, match="snapshot_expired"):
        state.require_current(original)
    assert state.install(view(), 2, 10, state.generation())
    with pytest.raises(SnapshotUnavailable, match="snapshot_revoked"):
        state.require_current(original)


def test_notification_does_not_restore_old_authority_even_for_equal_policy():
    state = PolicySnapshot(clock=lambda: 0)
    original = view()
    assert state.install(original, 1, 5, state.generation())
    state.invalidate("snapshot_revoked")
    assert state.install(view(), 2, 7, state.generation())
    with pytest.raises(SnapshotUnavailable, match="snapshot_revoked"):
        state.require_current(original)


def test_late_refresh_cannot_restore_revoked_generation():
    state = PolicySnapshot(clock=lambda: 0)
    generation = state.generation()
    assert state.install(view(), 1, 10, generation)
    state.invalidate("snapshot_revoked")
    assert not state.install(view(), 2, 20, generation)
    with pytest.raises(SnapshotUnavailable):
        state.get()
    assert state.install(view(private=True), 2, 20, state.generation())
    with pytest.raises(BotAccessError, match="access_denied"):
        state.get().require_target("fixture-peer")
    assert not state.install(view(), 1, 30, state.generation())


def test_revocation_during_refresh_refuses_late_install_without_timing_assumptions():
    state = PolicySnapshot(clock=lambda: 0)
    ready = Barrier(2)
    revoked = Barrier(2)

    def refresh():
        generation = state.generation()
        policy = view()
        ready.wait()
        revoked.wait()
        return state.install(policy, 1, 10, generation)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(refresh)
        ready.wait()
        state.invalidate("snapshot_revoked")
        revoked.wait()
        assert future.result() is False


def test_sixty_four_concurrent_memory_lookups_do_not_use_db_or_four_slot_worker():
    engine = Mock()
    state = policy_snapshot(engine)
    state.install(view(), 1, float("inf"), state.generation())
    start = Barrier(8)

    def burst(_):
        start.wait()
        return [state.get().require_target("fixture-peer") for _ in range(8)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(burst, range(8)))
    assert results == [["fixture-peer"] * 8] * 8
    assert engine.mock_calls == []
    assert policy_snapshot(engine) is state
    assert policy_snapshot(Mock()) is not state


def test_diagnostics_and_logs_never_echo_unknown_reason(caplog):
    state = PolicySnapshot()
    state.invalidate("secret or backend detail")
    assert "secret or backend detail" not in caplog.text
    assert state.diagnostics()["state"] == "snapshot_unavailable"
    with pytest.raises(SnapshotUnavailable, match="^snapshot_unavailable$"):
        state.get()
