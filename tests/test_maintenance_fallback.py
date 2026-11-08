"""TASK-1704 slice B: backup maintenance model on spending limit."""
from __future__ import annotations

import re
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llm_bawt.service import maintenance_fallback as mf
from llm_bawt.service.maintenance_fallback import (
    Backup, Exhaustion, FallbackClient, MemoryExhaustionStore, StateHealth, fallback_checks, served_model,
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
                                   _StatusError(503), _StatusError(504, "gateway timed out"),
                                   ValueError("bad payload")])
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


def test_each_edge_row_carries_a_distinct_episode_stable_across_polls():
    rig = Rig()
    _exhaust(rig)
    first = fallback_checks(rig.store)
    assert first[0]["episode"] and fallback_checks(rig.store) == first
    _recover(rig)
    back = fallback_checks(rig.store)
    rig.clock.now += 60
    _exhaust(rig)
    again = fallback_checks(rig.store)
    _recover(rig)
    merged = fallback_checks(rig.store)
    episodes = [r[0]["episode"] for r in (first, back, again, merged)]
    assert len(set(episodes)) == 4  # every edge is a new bell incident (Loopy r1 B1)


def test_edges_within_one_instant_still_get_distinct_episodes():
    """Loopy r2 B1: Retry-After 0 lets switch, recover and re-switch share one instant."""
    rig = Rig()
    store = rig.store
    seen = []
    for _ in range(2):
        assert store.mark(Exhaustion(provider=PRIMARY_PROVIDER, since=rig.clock.now,
                                     until=rig.clock.now, hint_source="hint"))
        seen.append(fallback_checks(store)[0]["episode"])
        record = store.get(PRIMARY_PROVIDER)
        assert store.clear(PRIMARY_PROVIDER, record.since, store.claim_probe(PRIMARY_PROVIDER, 30))
        seen.append(fallback_checks(store)[0]["episode"])
    assert len(set(seen)) == 4  # the clock never moved


def test_lost_edge_counter_never_reuses_a_dismissed_episode():
    """Byte r1: eviction or a data-less Redis restart must not restart ids at 1."""
    rig = Rig()
    store = rig.store
    assert store.mark(Exhaustion(provider=PRIMARY_PROVIDER, since=rig.clock.now,
                                 until=rig.clock.now, hint_source="hint"))
    first = int(fallback_checks(store)[0]["episode"])
    store._edges.clear()  # counter evicted
    record = store.get(PRIMARY_PROVIDER)
    assert store.clear(PRIMARY_PROVIDER, record.since, store.claim_probe(PRIMARY_PROVIDER, 30))
    rig.clock.now += 1
    store._edges.clear()
    assert store.mark(Exhaustion(provider=PRIMARY_PROVIDER, since=rig.clock.now,
                                 until=rig.clock.now, hint_source="hint"))
    assert int(fallback_checks(store)[0]["episode"]) > first


class _ClaimDownStore(MemoryExhaustionStore):
    def claim_probe(self, provider, ttl):
        raise mf.StateUnavailable("redis gone mid-call")


def test_claim_probe_failure_after_good_get_is_stateless_per_call():
    rig = Rig()
    rig.store = _ClaimDownStore(clock=rig.clock)
    _exhaust(rig)
    rig.clock.now += 700  # cooldown over: this call must claim the probe
    rig.primary_inner.query.reset_mock()
    rig.primary_inner.query.side_effect = _spend()
    rig.backup_inner.query.return_value = "b"
    client = rig.client()
    assert client.query(messages=[]) == "b" and client.last_via_fallback
    assert rig.primary_inner.query.call_count == 1  # primary tried first (Loopy r1 B2)
    rig.primary_inner.query.side_effect = None
    rig.primary_inner.query.return_value = "p"
    assert rig.client().query(messages=[]) == "p"
    rows = [r for r in rig.health.checks_payload()["checks"] if r["name"] == mf.STATE_JOB]
    assert len(rows) == 1 and rows[0]["status"] == "down"


# -- TASK-1716: unreachable primary --------------------------------------------

class ConnectError(Exception):
    """Stands in for httpx.ConnectError (matched by type name, no import)."""


def _wrapped_connect() -> Exception:
    outer = RuntimeError("Connection error.")
    outer.__cause__ = ConnectError("[Errno 111] refused")
    return outer


@pytest.mark.parametrize("error", [
    ConnectError("[Errno 111] Connection refused"),
    _wrapped_connect(),
    TimeoutError("read timed out"),
    ConnectionRefusedError(111, "refused"),
])
def test_unreachable_primary_switches_with_cause(error):
    rig = Rig()
    rig.primary_inner.query.side_effect = error
    rig.backup_inner.query.return_value = "from backup"
    client = rig.client()
    assert client.query(messages=[]) == "from backup"
    record = rig.store.get(PRIMARY_PROVIDER)
    assert record is not None and record.cause == mf.UNREACHABLE
    assert record.until == rig.clock.now + 600
    [row] = fallback_checks(rig.store)
    assert "prov-a unreachable since" in row["reason"]


@pytest.mark.parametrize("reply", ["ERROR: OpenAI API Error - Connection error.",
                                   "ERROR: OpenAI API Error - Request timed out."])
def test_unreachable_error_string_also_triggers(reply):
    rig = Rig()
    rig.primary_inner.query.return_value = reply
    rig.backup_inner.query.return_value = "ok"
    assert rig.client().query(messages=[]) == "ok"
    assert rig.store.get(PRIMARY_PROVIDER).cause == mf.UNREACHABLE


def test_http_reply_is_never_unreachable():
    assert not mf.is_unreachable(_StatusError(502, "connection error upstream"))
    assert mf.switch_cause(_StatusError(503)) is None


def test_announce_brings_probe_forward_and_real_call_switches_back():
    rig = Rig()
    rig.primary_inner.query.side_effect = ConnectError("refused")
    rig.backup_inner.query.return_value = "b"
    client = rig.client()
    client.query(messages=[])
    rig.primary_inner.query.reset_mock()
    rig.clock.now += 10
    client.query(messages=[])  # inside cooldown: primary untouched
    rig.primary_inner.query.assert_not_called()

    assert mf.announce_available(rig.store, now=rig.clock.now) == [PRIMARY_PROVIDER]
    assert rig.store.get(PRIMARY_PROVIDER) is not None  # announce alone clears nothing
    assert mf.announce_available(rig.store, now=rig.clock.now) == []  # idempotent

    rig.primary_inner.query.side_effect = None
    rig.primary_inner.query.return_value = "primary"
    assert client.query(messages=[]) == "primary"
    assert rig.store.get(PRIMARY_PROVIDER) is None
    [row] = fallback_checks(rig.store)
    assert row["name"].startswith("maintenance_fallback_recovered:")
    assert "prov-a unreachable: maintenance was on the backup" in row["reason"]


def test_early_announce_costs_one_fallen_through_probe():
    rig = Rig()
    rig.primary_inner.query.side_effect = ConnectError("refused")
    rig.backup_inner.query.return_value = "b"
    client = rig.client()
    client.query(messages=[])
    mf.announce_available(rig.store, now=rig.clock.now)
    assert client.query(messages=[]) == "b"  # probe fails, backup serves
    assert rig.store.get(PRIMARY_PROVIDER).until == rig.clock.now + 600  # rearmed


def test_announce_never_touches_a_spending_limit_record():
    rig = Rig()
    rig.primary_inner.query.side_effect = _spend()
    rig.backup_inner.query.return_value = "b"
    rig.client().query(messages=[])
    until = rig.store.get(PRIMARY_PROVIDER).until
    assert mf.announce_available(rig.store, now=rig.clock.now) == []
    assert rig.store.get(PRIMARY_PROVIDER).until == until


def _sdk_error(status: int | None, body: str = "") -> Exception:
    """The real OpenAI SDK's error for a gateway reply (or a refused connect)."""
    import httpx
    import openai

    def handler(request):
        if status is None:
            raise httpx.ConnectError("[Errno 111] Connection refused", request=request)
        return httpx.Response(status, text=body, headers={"content-type": "text/plain"})

    sdk = openai.OpenAI(api_key="x", base_url="http://model.invalid/v1", max_retries=0,
                        http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    try:
        sdk.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
    except openai.OpenAIError as error:
        return error
    raise AssertionError("no error")


@pytest.mark.parametrize("status, body", [(503, "upstream connection refused"),
                                          (504, "upstream timed out"),
                                          (502, "Connection error.")])
def test_gateway_5xx_never_unreachable_as_exception_or_client_string(status, body):
    error = _sdk_error(status, body)
    assert mf.switch_cause(error) is None
    # OpenAIClient's real serialization keeps the status in the string.
    from llm_bawt.clients.base import api_error_text as _api_error_text
    reply = f"ERROR: OpenAI API Error - {_api_error_text(error)}"
    assert mf.switch_cause(reply) is None
    rig = Rig()
    rig.primary_inner.query.return_value = reply
    rig.client().query(messages=[])
    rig.backup_inner.query.assert_not_called()
    assert rig.store.get(PRIMARY_PROVIDER) is None


def test_real_sdk_refused_connect_is_unreachable_both_forms():
    error = _sdk_error(None)
    assert mf.switch_cause(error) == mf.UNREACHABLE
    from llm_bawt.clients.base import api_error_text as _api_error_text
    assert mf.switch_cause(f"ERROR: OpenAI API Error - {_api_error_text(error)}") == mf.UNREACHABLE


def _probe_fails_with(rig, client, error):
    rig.clock.now = rig.store.get(PRIMARY_PROVIDER).until + 1
    rig.primary_inner.query.side_effect = error
    client.query(messages=[])
    return rig.store.get(PRIMARY_PROVIDER)


def test_failed_probe_records_its_own_cause_both_directions():
    rig = Rig()
    rig.backup_inner.query.return_value = "b"
    client = rig.client()
    rig.primary_inner.query.side_effect = ConnectError("refused")
    client.query(messages=[])
    record = _probe_fails_with(rig, client, _spend())
    assert record.cause == mf.SPENDING_LIMIT
    assert mf.announce_available(rig.store, now=rig.clock.now) == []
    record = _probe_fails_with(rig, client, ConnectError("refused"))
    assert record.cause == mf.UNREACHABLE
    assert mf.announce_available(rig.store, now=rig.clock.now) == [PRIMARY_PROVIDER]


def _responses_reply(status: int | None, body: str = "") -> str:
    """The real ResponsesClient.query reply over a mocked transport."""
    import httpx

    from llm_bawt.clients.responses_client import ResponsesClient
    from llm_bawt.utils.config import Config

    def handler(request):
        if status is None:
            raise httpx.ConnectError("[Errno 111] Connection refused", request=request)
        if status == 0:
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(status, text=body, headers={"content-type": "text/plain"})

    client = ResponsesClient("m", config=Config(), base_url="http://model.invalid/v1", api_key="x")
    import openai
    client.client = openai.OpenAI(api_key="x", base_url="http://model.invalid/v1", max_retries=0,
                                  http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    return client.query([], plaintext_output=True, stream=False)


@pytest.mark.parametrize("status", [None, 0])
def test_responses_client_no_reply_is_unreachable(status):
    reply = _responses_reply(status)
    assert reply.startswith("ERROR: Responses API Error -"), reply
    assert mf.switch_cause(reply) == mf.UNREACHABLE


@pytest.mark.parametrize("status, body", [(503, "upstream connection refused"),
                                          (504, "upstream timed out"),
                                          (502, "Connection error.")])
def test_responses_client_gateway_5xx_never_unreachable(status, body):
    reply = _responses_reply(status, body)
    assert reply.startswith("ERROR: Responses API Error -"), reply
    assert mf.switch_cause(reply) is None
