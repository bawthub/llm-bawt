"""TASK-1704 slice B: backup maintenance model on spending limit."""
from __future__ import annotations

import re
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llm_bawt.service import maintenance_fallback as mf
from llm_bawt.service.maintenance_fallback import (
    Backup, FallbackClient, MemoryExhaustionStore, StateHealth, fallback_checks, served_model,
)
from llm_bawt.service.maintenance_health import MaintenanceHealth, MonitoredClient

PRIMARY, PRIMARY_PROVIDER = "primary-model", "prov-a"
BACKUP, BACKUP_PROVIDER = "backup-model", "prov-b"


class _StatusError(Exception):
    def __init__(self, status: int, text: str = "", headers: dict | None = None):
        super().__init__(text or f"Error code: {status}")
        self.status_code = status
        self.headers = headers or {}


def _spend(headers=None):
    return _StatusError(403, "personal-team-blocked:spending-limit", headers)


class Clock:
    def __init__(self, now: float = 1_000_000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


class Rig:
    def __init__(self, *, backup: bool = True, store=None, cooldown: float = 600.0,
                 backup_methods=("query", "query_bounded")):
        self.clock = Clock()
        self.health = MaintenanceHealth()
        self.store = store or MemoryExhaustionStore(clock=self.clock)
        self.state = StateHealth(self.health)
        self.primary_inner = MagicMock(spec=["query", "query_bounded"])
        self.backup_inner = MagicMock(spec=list(backup_methods))
        self.backup_enabled = backup
        self.incompatible: list[tuple] = []
        self.cooldown = cooldown

    def open_backup(self, _provider):
        if not self.backup_enabled:
            self.incompatible.append((None, "not set"))
            return None
        return Backup(client=MonitoredClient(self.backup_inner, job="job", model=BACKUP,
                                             provider=BACKUP_PROVIDER, credential=None,
                                             health=self.health),
                      model=BACKUP, provider=BACKUP_PROVIDER)

    def client(self) -> FallbackClient:
        primary = MonitoredClient(self.primary_inner, job="job", model=PRIMARY,
                                  provider=PRIMARY_PROVIDER, credential=None, health=self.health)
        return FallbackClient(primary, job="job", model=PRIMARY, provider=PRIMARY_PROVIDER,
                              open_backup=self.open_backup, store=self.store,
                              cooldown_seconds=lambda: self.cooldown, state_health=self.state,
                              on_incompatible=lambda a, w: self.incompatible.append((a, w)),
                              clock=self.clock)


def test_spending_limit_serves_backup_and_marks_provider():
    rig = Rig()
    rig.primary_inner.query.side_effect = _spend()
    rig.backup_inner.query.return_value = "from backup"
    client = rig.client()
    assert client.query(messages=[]) == "from backup"
    assert (client.last_served_model, client.last_served_provider, client.last_via_fallback) == (
        BACKUP, BACKUP_PROVIDER, True)
    assert served_model(client, PRIMARY) == BACKUP
    record = rig.store.get(PRIMARY_PROVIDER)
    assert record is not None and record.until == rig.clock.now + 600 and record.hint_source == "setting"


@pytest.mark.parametrize("error", [_StatusError(429, "slow down"), _StatusError(401),
                                   TimeoutError("read timed out"), _StatusError(503)])
def test_other_failures_never_use_backup(error):
    rig = Rig()
    rig.primary_inner.query.side_effect = error
    with pytest.raises(type(error)):
        rig.client().query(messages=[])
    rig.backup_inner.query.assert_not_called()
    assert rig.store.get(PRIMARY_PROVIDER) is None


def test_spending_limit_error_string_also_triggers():
    rig = Rig()
    rig.primary_inner.query.return_value = "ERROR: API Error - Error code: 402"
    rig.backup_inner.query.return_value = "ok"
    assert rig.client().query(messages=[]) == "ok"


def test_unset_backup_fails_visibly_with_primary_error():
    rig = Rig(backup=False)
    rig.primary_inner.query.side_effect = _spend()
    with pytest.raises(_StatusError):
        rig.client().query(messages=[])
    assert rig.incompatible == [(None, "not set")]


def test_backup_that_cannot_serve_the_method_is_rejected_visibly():
    rig = Rig(backup_methods=("query",))
    rig.primary_inner.query_bounded.side_effect = _spend()
    with pytest.raises(_StatusError):
        rig.client().query_bounded(messages=[], timeout_seconds=5)
    assert rig.incompatible and "query_bounded" in rig.incompatible[0][1]


def test_exhausted_provider_skips_primary_until_cooldown():
    rig = Rig()
    rig.primary_inner.query.side_effect = _spend()
    rig.backup_inner.query.return_value = "b"
    client = rig.client()
    client.query(messages=[])
    rig.primary_inner.query.reset_mock()
    rig.clock.now += 300
    for _ in range(3):
        assert client.query(messages=[]) == "b"
    rig.primary_inner.query.assert_not_called()


def test_provider_reset_hint_overrides_setting():
    rig = Rig()
    rig.primary_inner.query.side_effect = _spend({"retry-after": "42"})
    rig.backup_inner.query.return_value = "b"
    rig.client().query(messages=[])
    record = rig.store.get(PRIMARY_PROVIDER)
    assert record.until == rig.clock.now + 42 and record.hint_source == "provider"


def _exhaust(rig):
    rig.primary_inner.query.side_effect = _spend()
    rig.backup_inner.query.return_value = "b"
    rig.client().query(messages=[])
    rig.primary_inner.query.reset_mock()


def test_record_outlives_until_and_exactly_one_concurrent_probe():
    rig = Rig()
    _exhaust(rig)
    rig.clock.now += 10 * 86400  # first call long after the cooldown
    gate = threading.Event()
    calls = []

    def slow_primary(**_):
        calls.append(1)
        gate.wait(5)
        return "primary"

    rig.primary_inner.query.side_effect = slow_primary
    results: list = []
    probe = threading.Thread(target=lambda: results.append(rig.client().query(messages=[])))
    probe.start()
    while not calls:
        pass
    others = [rig.client().query(messages=[]) for _ in range(5)]
    gate.set()
    probe.join()
    assert len(calls) == 1 and others == ["b"] * 5 and results == ["primary"]
    assert rig.store.get(PRIMARY_PROVIDER) is None  # successful probe cleared it


def test_failed_probe_is_served_by_backup_and_rearms_cooldown():
    rig = Rig()
    _exhaust(rig)
    since = rig.store.get(PRIMARY_PROVIDER).since
    rig.clock.now += 700
    rig.primary_inner.query.side_effect = _spend()
    client = rig.client()
    assert client.query(messages=[]) == "b" and client.last_via_fallback
    record = rig.store.get(PRIMARY_PROVIDER)
    assert record.since == since and record.until == rig.clock.now + 600
    rig.primary_inner.query.reset_mock()
    rig.clock.now += 300
    rig.client().query(messages=[])
    rig.primary_inner.query.assert_not_called()  # rearmed: no probe inside the new cooldown
    rig.clock.now += 400
    rig.primary_inner.query.side_effect = None
    rig.primary_inner.query.return_value = "back"
    assert rig.client().query(messages=[]) == "back"  # probe claim was released by rearm


def test_crashed_probe_claim_expires():
    rig = Rig()
    _exhaust(rig)
    rig.clock.now += 700
    assert rig.store.claim_probe(PRIMARY_PROVIDER, mf.PROBE_CLAIM_SECONDS)  # crashed holder
    assert rig.client().query(messages=[]) == "b"
    rig.primary_inner.query.assert_not_called()
    rig.clock.now += mf.PROBE_CLAIM_SECONDS + 1
    rig.primary_inner.query.side_effect = None
    rig.primary_inner.query.return_value = "back"
    assert rig.client().query(messages=[]) == "back"


def test_state_down_falls_through_per_call_with_one_alert_row():
    rig = Rig()
    rig.store.down = True
    rig.primary_inner.query.side_effect = _spend()
    rig.backup_inner.query.return_value = "b"
    client = rig.client()
    for _ in range(3):
        assert client.query(messages=[]) == "b"
        assert client.last_via_fallback
    assert rig.primary_inner.query.call_count == 3  # no skip-the-primary without state
    rows = [r for r in rig.health.checks_payload()["checks"] if r["name"] == mf.STATE_JOB]
    assert len(rows) == 1 and rows[0]["status"] == "down"
    rig.store.down = False
    rig.client().query(messages=[])
    rows = [r for r in rig.health.checks_payload()["checks"] if r["name"] == mf.STATE_JOB]
    assert rows[0]["status"] == "ok"


def test_backup_also_failing_is_loud_and_secret_free():
    rig = Rig()
    rig.primary_inner.query.side_effect = _spend()
    rig.backup_inner.query.side_effect = _StatusError(500, "sk-secret-token leaked?")
    with pytest.raises(_StatusError):
        rig.client().query(messages=[])
    rows = {r["name"]: r for r in rig.health.checks_payload()["checks"]}
    assert rows["job"]["status"] == "down"
    assert "sk-secret" not in str(rows)


def test_success_on_primary_reports_primary():
    rig = Rig()
    rig.primary_inner.query.return_value = "p"
    client = rig.client()
    assert client.query(messages=[]) == "p"
    assert served_model(client, PRIMARY) == PRIMARY and not client.last_via_fallback


def _recover(rig):
    rig.clock.now += 700
    rig.primary_inner.query.side_effect = None
    rig.primary_inner.query.return_value = "back"
    assert rig.client().query(messages=[]) == "back"


def test_poll_rows_are_stable_never_ok_and_episode_outlives_recovery():
    rig = Rig()
    assert fallback_checks(rig.store) == []
    _exhaust(rig)
    first = fallback_checks(rig.store)
    assert [(r["name"], r["status"]) for r in first] == [(f"maintenance_fallback:{PRIMARY_PROVIDER}", "degraded")]
    assert fallback_checks(rig.store) == first == fallback_checks(rig.store)
    _recover(rig)
    back = fallback_checks(rig.store)
    assert len(back) == 1 and back[0]["status"] == "degraded"
    # A different row name, so the bell's incident id changes and the recovery is a new notice.
    assert back[0]["name"] == f"maintenance_fallback_recovered:{PRIMARY_PROVIDER}" != first[0]["name"]
    assert "1 episode)" in back[0]["reason"] and "back on primary" in back[0]["reason"]
    assert fallback_checks(rig.store) == back
    _exhaust(rig)  # re-exhaustion: the live record wins, still one row
    again = fallback_checks(rig.store)
    assert len(again) == 1 and again[0]["name"] == first[0]["name"]
    _recover(rig)
    merged = fallback_checks(rig.store)
    assert len(merged) == 1 and "2 episodes" in merged[0]["reason"]
    rig.clock.now += mf.RECOVERED_ROW_SECONDS + 1
    assert fallback_checks(rig.store) == []
    assert all(r["status"] != "ok" for r in first + back + again + merged)


def test_switch_and_recover_between_polls_is_still_shown():
    rig = Rig()
    _exhaust(rig)
    _recover(rig)  # no poll happened in between
    rows = fallback_checks(rig.store)
    assert len(rows) == 1 and rows[0]["status"] == "degraded"


def test_late_probe_after_lease_expiry_changes_nothing():
    rig = Rig()
    _exhaust(rig)
    record = rig.store.get(PRIMARY_PROVIDER)
    rig.clock.now += 700
    stale = rig.store.claim_probe(PRIMARY_PROVIDER, mf.PROBE_CLAIM_SECONDS)
    rig.clock.now += mf.PROBE_CLAIM_SECONDS + 1  # stale lease expired mid-call
    fresh = rig.store.claim_probe(PRIMARY_PROVIDER, mf.PROBE_CLAIM_SECONDS)
    assert fresh and fresh != stale
    assert rig.store.rearm(PRIMARY_PROVIDER, record.since, fresh, rig.clock.now + 600)
    rearmed = rig.store.get(PRIMARY_PROVIDER)
    # The stale holder's late outcome is fenced off either way.
    assert not rig.store.clear(PRIMARY_PROVIDER, record.since, stale)
    assert not rig.store.rearm(PRIMARY_PROVIDER, record.since, stale, rig.clock.now + 5)
    assert rig.store.get(PRIMARY_PROVIDER) == rearmed
    assert not rig.store.clear(PRIMARY_PROVIDER, record.since, fresh)  # re-arm released it


def test_concurrent_edge_crossing_marks_once():
    rig = Rig()
    rig.primary_inner.query.side_effect = _spend()
    rig.backup_inner.query.return_value = "b"
    threads = [threading.Thread(target=lambda: rig.client().query(messages=[])) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(rig.store.all()) == 1 and len(fallback_checks(rig.store)) == 1


def test_explicit_model_disables_fallback_at_call_sites():
    root = Path(mf.__file__).resolve().parent
    sources = {
        "utility_completion.py": "allow_fallback=not request.model",
        "routes/history_summaries.py": "allow_fallback=not model",
    }
    for rel, needle in sources.items():
        assert needle in (root / rel).read_text(), rel
    bg = (root / "background_tasks.py").read_text()
    assert bg.count('allow_fallback=not task.payload.get("model")') == 3


def test_get_background_client_default_has_no_fallback():
    import inspect

    from llm_bawt.service.instance_manager import InstanceManagerMixin as M  # noqa: N814
    assert inspect.signature(M._get_background_client).parameters["allow_fallback"].default is False


def test_no_model_literals_in_module():
    text = Path(mf.__file__).read_text()
    assert not re.search(r"\b(gpt-|grok-|claude-|gemini-|llama)", text)
