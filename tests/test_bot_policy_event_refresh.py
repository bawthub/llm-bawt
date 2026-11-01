"""Event-only worker/admin contract, synthetic engines and deterministic wakeups."""
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from llm_bawt.bot_access import BotAccessError, BotPolicyResolver
from llm_bawt.bot_access import refresh as module
from llm_bawt.bot_access.snapshot import SnapshotUnavailable, policy_snapshot
from llm_bawt.bot_policy_events import POLICY_CHANNEL

ROWS = [{"slug": "fixture_owner", "include_in_global_search": True},
        {"slug": "fixture_peer", "include_in_global_search": True}]


def worker(monkeypatch, *, fail=False):
    engine = MagicMock()
    listener = engine.connect.return_value.execution_options.return_value.__enter__.return_value
    listener.execute.return_value.scalar_one.return_value = "fixture_schema"
    raw = listener.connection.driver_connection
    raw.notifies = []
    owner = module.PolicyRefresher(engine)
    now = [0.0]
    owner.snapshot._clock = lambda: now[0]
    read = Mock(side_effect=ValueError("sensitive detail") if fail else None, return_value=ROWS)
    monkeypatch.setattr(module, "read_snapshot_rows", read)
    return owner, raw, now, read


@pytest.mark.parametrize("fail", [False, True])
def test_idle_worker_has_no_timeout_reads_rebuilds_or_hidden_expiry(monkeypatch, fail):
    owner, _raw, now, read = worker(monkeypatch, fail=fail)
    wakes = [0]

    # Exactly three args: adding a periodic select timeout fails this test.
    def wait(_readers, _writers, _errors):
        wakes[0] += 1
        now[0] += 86400
        assert read.call_count == 1
        if fail:
            with pytest.raises(BotAccessError, match="policy_unavailable"):
                BotPolicyResolver(owner.engine).resolve_sync()
        else:
            assert owner.snapshot.get().require_target("fixture_peer")
        if wakes[0] == 4:
            owner._stop.set()
        return [], [], []

    monkeypatch.setattr(module.select, "select", wait)
    owner._run()
    assert read.call_count == 1
    assert owner.snapshot.diagnostics()["state"] == "snapshot_stopped"
    assert owner._wake is None


def test_racing_change_rereads_once_without_a_timer(monkeypatch):
    owner, raw, _now, read = worker(monkeypatch)
    calls = [0]

    def poll():
        calls[0] += 1
        if calls[0] == 1:
            raw.notifies.append(SimpleNamespace(channel=POLICY_CHANNEL, payload="fixture_schema"))

    raw.poll.side_effect = poll

    def wait(*_):
        assert read.call_count == 2
        assert owner.snapshot.get().require_target("fixture_peer")
        owner._stop.set()
        return [], [], []

    monkeypatch.setattr(module.select, "select", wait)
    owner._run()
    assert read.call_count == 2


def test_listener_failure_invalidates_before_reconnect_startup_load(monkeypatch):
    owner, _raw, _now, read = worker(monkeypatch)
    wakes = [0]

    def wait(*_):
        wakes[0] += 1
        if wakes[0] == 1:
            raise OSError("sensitive listener detail")
        assert read.call_count == 2
        assert owner.snapshot.get().require_target("fixture_peer")
        owner._stop.set()
        return [], [], []

    def reconnect_wait(delay):
        assert delay == 0.5
        with pytest.raises(SnapshotUnavailable, match="snapshot_database"):
            owner.snapshot.get()
        return False

    monkeypatch.setattr(module.select, "select", wait)
    monkeypatch.setattr(owner._stop, "wait", reconnect_wait)
    owner._run()
    assert owner.engine.connect.call_count == 2
    assert read.call_count == 2


def test_stop_wakes_blocked_listener_without_periodic_check(monkeypatch):
    owner, _raw, _now, _read = worker(monkeypatch)
    joined = Mock()
    monkeypatch.setattr(owner._thread, "join", joined)

    def wait(readers, _writers, _errors):
        owner.stop()
        wake = readers[1]
        assert wake.recv(1) == b"\0"
        return [wake], [], []

    monkeypatch.setattr(module.select, "select", wait)
    owner._run()
    joined.assert_called_once_with(timeout=3)
    assert owner.snapshot.diagnostics()["state"] == "snapshot_stopped"


@pytest.mark.parametrize("error", [RuntimeError("sensitive detail"), TimeoutError("sensitive detail"),
                                 module.PoolTimeout("sensitive detail")])
def test_transient_load_failure_retries_only_until_recovered(monkeypatch, error):
    owner, _raw, _now, read = worker(monkeypatch)
    read.side_effect = [error, ROWS]
    delays = []

    def backoff(delay):
        delays.append(delay)
        with pytest.raises(SnapshotUnavailable):
            owner.snapshot.get()
        return False

    def wait(_readers, _writers, _errors):
        assert read.call_count == 2
        assert owner.snapshot.get().require_target("fixture_peer")
        owner._stop.set()
        return [], [], []

    monkeypatch.setattr(owner._stop, "wait", backoff)
    monkeypatch.setattr(module.select, "select", wait)
    owner._run()
    assert delays == [0.1]
    assert read.call_count == 2


def test_retry_budget_exhausts_and_renews_only_for_a_new_event(monkeypatch):
    owner, raw, _now, read = worker(monkeypatch)
    read.side_effect = RuntimeError("sensitive detail")
    delays = []
    wakes = [0]

    def backoff(delay):
        delays.append(delay)
        with pytest.raises(SnapshotUnavailable, match="snapshot_database"):
            owner.snapshot.get()
        return False

    def wait(_readers, _writers, _errors):
        wakes[0] += 1
        assert read.call_count == wakes[0] * 3
        if wakes[0] == 1:
            raw.notifies.append(SimpleNamespace(channel=POLICY_CHANNEL, payload="fixture_schema"))
            return [raw], [], []
        owner._stop.set()
        return [], [], []

    monkeypatch.setattr(owner._stop, "wait", backoff)
    monkeypatch.setattr(module.select, "select", wait)
    owner._run()
    assert delays == [0.1, 0.5, 0.1, 0.5]
    assert read.call_count == 6


def test_stop_interrupts_transient_retry_without_another_load(monkeypatch):
    owner, _raw, _now, read = worker(monkeypatch)
    read.side_effect = RuntimeError("sensitive detail")

    def backoff(_delay):
        owner._stop.set()
        return True

    monkeypatch.setattr(owner._stop, "wait", backoff)
    monkeypatch.setattr(module.select, "select", Mock(side_effect=AssertionError("unexpected idle wait")))
    owner._run()
    assert read.call_count == 1


def test_local_completion_distinguishes_current_failed_pending_and_revoked_loads(monkeypatch):
    owner, _raw, _now, read = worker(monkeypatch)
    assert owner.wait_for_refresh(0, 0) == {"status": "timeout"}
    assert owner.refresh()
    assert owner.wait_for_refresh(0, 0) == {"status": "installed"}
    owner.snapshot.invalidate("snapshot_revoked")
    generation = owner.snapshot.generation()
    assert owner.wait_for_refresh(0, 0) == {"status": "failed", "reason": "snapshot_revoked"}
    assert owner.wait_for_refresh(generation, 0) == {"status": "timeout"}
    read.side_effect = ValueError("sensitive detail")
    assert not owner.refresh()
    assert owner.wait_for_refresh(generation, 0) == {"status": "failed", "reason": "snapshot_validation"}
    read.side_effect = None
    assert owner.refresh()
    assert owner.wait_for_refresh(generation, 0) == {"status": "installed"}
    monkeypatch.setattr(owner._thread, "join", Mock())
    owner.stop()
    assert owner.wait_for_refresh(generation, 0) == {"status": "failed", "reason": "snapshot_stopped"}


def test_admin_waits_for_its_generation_after_commit_outside_lifecycle_lock(monkeypatch):
    owner, _raw, _now, read = worker(monkeypatch)
    conn = owner.engine.begin.return_value.__enter__.return_value
    conn.dialect.name = "postgresql"
    monkeypatch.setitem(module._owners, owner.engine, (owner, 1))
    committed = []
    owner.engine.begin.return_value.__exit__.side_effect = lambda *_: committed.append(True)

    def wait(generation, timeout):
        assert committed == [True]
        assert module._owners_lock.acquire(blocking=False)
        module._owners_lock.release()
        assert generation == owner.snapshot.generation()
        assert timeout == 0
        assert read.call_count == 0  # no policy I/O by the request
        return {"status": "timeout"}

    monkeypatch.setattr(owner, "wait_for_refresh", wait)
    assert module.request_policy_refresh(owner.engine, timeout=0) == {"status": "timeout"}
    assert read.call_count == 0


def test_manual_admin_hint_invalidates_local_and_commits_without_policy_read(monkeypatch):
    engine = MagicMock()
    conn = engine.begin.return_value.__enter__.return_value
    conn.dialect.name = "postgresql"
    owner = module.PolicyRefresher(engine)
    read = Mock(return_value=ROWS)
    monkeypatch.setattr(module, "read_snapshot_rows", read)
    assert owner.refresh()
    engine.reset_mock()
    assert module.request_policy_refresh(engine) == {"status": "unavailable", "reason": "snapshot_cold"}
    with pytest.raises(SnapshotUnavailable, match="snapshot_revoked"):
        policy_snapshot(engine).get()
    engine.begin.assert_called_once()
    conn.execute.assert_called_once()
    statement, params = conn.execute.call_args.args
    assert "pg_notify" in str(statement)
    assert params == {"channel": POLICY_CHANNEL}
    assert engine.begin.return_value.__exit__.call_args.args[0] is None
    assert read.call_count == 1


@pytest.mark.parametrize("error,reason", [(RuntimeError("sensitive detail"), "snapshot_database"),
                                        (TimeoutError("sensitive detail"), "snapshot_timeout")])
def test_failed_admin_publication_is_sanitized_and_keeps_policy_unavailable(error, reason):
    engine = MagicMock()
    engine.begin.side_effect = error
    with pytest.raises(SnapshotUnavailable, match=f"^{reason}$"):
        module.request_policy_refresh(engine)
    assert policy_snapshot(engine).diagnostics()["state"] == reason
    with pytest.raises(SnapshotUnavailable):
        policy_snapshot(engine).get()


def test_existing_admin_reload_reports_requested_refresh_and_publication_failure(monkeypatch):
    from fastapi import HTTPException
    from llm_bawt.service.routes import settings

    engine = MagicMock()
    engine.begin.return_value.__enter__.return_value.dialect.name = "postgresql"
    service = SimpleNamespace(config=object())
    monkeypatch.setattr(settings, "get_service", lambda: service)
    monkeypatch.setattr(settings, "get_bot_profile_store", lambda _: SimpleNamespace(engine=engine))
    monkeypatch.setattr(settings, "_reload_bot_registry", Mock())
    monkeypatch.setattr(settings, "_reload_service_model_catalog", Mock())
    monkeypatch.setattr(settings, "_invalidate_all_instance_cache", lambda _: 2)
    monkeypatch.setattr(settings, "_clear_session_model_overrides", lambda _: 3)
    request = Mock(return_value={"status": "installed"})
    monkeypatch.setattr(module, "request_policy_refresh", request)
    result = settings.reload_bots()
    assert result == {"status": "reloaded", "bot_access_policy": "refresh_requested",
                      "bot_access_policy_app": {"status": "installed"},
                      "cleared_instances": 2, "cleared_session_model_overrides": 3}
    for status, code in (("failed", 503), ("unavailable", 503), ("timeout", 504)):
        request.return_value = {"status": status}
        with pytest.raises(HTTPException) as exc:
            settings.reload_bots()
        assert exc.value.status_code == code
        assert exc.value.detail["bot_access_policy"] == "refresh_requested"
        assert exc.value.detail["bot_access_policy_app"] == {"status": status}
    request.side_effect = SnapshotUnavailable("snapshot_database")
    with pytest.raises(HTTPException) as exc:
        settings.reload_bots()
    assert exc.value.status_code == 503
    assert exc.value.detail == "Bot access policy refresh unavailable"
