"""TASK-2080: a new session ends a house tenancy; the janitor evicts idle tenants."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from llm_bawt import house_lease
from llm_bawt.message_authorship import AuthorReference
from llm_bawt.service.jobs.idle_context_candidates import IdleContextCandidates
from llm_bawt.service.routes.bot_profile_validation import _validate_profile_payload
from llm_bawt.service.session_lifecycle import lifecycle as lifecycle_module
from llm_bawt.service.session_lifecycle.intent import (
    Admission, Continuity, NewSessionIntent, Origin,
)
from llm_bawt.service.session_lifecycle.lifecycle import NewSessionLifecycle
from tests.bot_tag_fixtures import static_tags

NOW = datetime(2026, 11, 6, tzinfo=timezone.utc)


def candidate(**overrides):
    return {
        "bot_id": "disposable", "session_id": "session", "user_id": "owner",
        "usage_turn_id": "turn", "latest_turn_id": "turn",
        "ended_at": NOW - timedelta(minutes=16),
        "last_completed_at": NOW - timedelta(minutes=16),
        "token_usage_json": json.dumps({"resident_tokens": 5}),
        "house": None, "house_pinned": False, **overrides,
    }


# ------------------------------------------------------------------ janitor


@pytest.mark.parametrize("changes,expected", [
    ({"house": "h1"}, None),  # idle tenant: no context threshold
    ({"house": "h1", "token_usage_json": "{"}, None),
    ({"house": "h1", "house_pinned": True}, "resident context within threshold"),
    ({}, "resident context within threshold"),  # mansion bot: unchanged rule
    ({"house": "h1", "last_completed_at": NOW - timedelta(minutes=5)}, "idle less than threshold"),
    ({"house": "h1", "latest_turn_id": "later"}, "newer bot activity invalidated measurement"),
])
def test_idle_house_tenant_is_evicted_without_a_context_threshold(changes, expected):
    assert IdleContextCandidates.skip_reason(candidate(**changes), now=NOW) == expected


class Conn:
    def execute(self, *_args, **_kwargs):
        return SimpleNamespace(first=lambda: None)


def _scheduled(row, monkeypatch):
    monkeypatch.setattr(lifecycle_module, "BotTags", static_tags())
    monkeypatch.setattr(IdleContextCandidates, "read_one", lambda self, conn, bot: row)
    monkeypatch.setattr(lifecycle_module, "classify_target",
                        lambda conn, bot, exclude_turn_id: SimpleNamespace(idle=True))
    monkeypatch.setattr(lifecycle_module, "datetime",
                        SimpleNamespace(now=lambda tz=None: NOW))
    intent = NewSessionIntent(
        bot_id="disposable", user_id="owner", author=AuthorReference.bot("system"),
        continuity=Continuity.CONFIGURED, admission=Admission.SKIP,
        origin=Origin.SCHEDULER, idempotency_key="k", job_id="j", run_id="r",
        expected_session_id="session", expected_usage_turn_id="turn",
    )
    return NewSessionLifecycle.scheduled_skip_reason(Conn(), intent)


def test_scheduled_recheck_admits_an_idle_tenant_with_unbounded_measurement(monkeypatch):
    usage = json.dumps({"input_tokens": 9})  # no provider window
    assert _scheduled(candidate(house="h1", token_usage_json=usage), monkeypatch) is None
    pinned = candidate(house="h1", house_pinned=True,
                       token_usage_json=json.dumps({"resident_tokens": 200000}))
    assert _scheduled(pinned, monkeypatch) is None  # pinned: ordinary size rule applies
    assert _scheduled(candidate(house="h1", house_pinned=True), monkeypatch) == (
        "resident context within threshold")


# ------------------------------------------------------------------ intent


def _api_intent(**over):
    return NewSessionIntent(
        bot_id="b", user_id="u", author=AuthorReference.user("u"),
        continuity=Continuity.CLEAN, admission=Admission.SKIP,
        origin=over.pop("origin", Origin.CONTEXT_API), idempotency_key="k", **over,
    )


def test_keep_house_is_an_api_boolean_persisted_only_when_set():
    assert "keep_house" not in NewSessionLifecycle._encode(_api_intent())
    kept = NewSessionLifecycle._encode(_api_intent(keep_house=True))
    assert kept["keep_house"] is True
    assert NewSessionLifecycle._decode(kept).keep_house is True
    with pytest.raises(ValueError):
        _api_intent(keep_house="yes")
    with pytest.raises(ValueError):
        NewSessionIntent(
            bot_id="b", user_id="u", author=AuthorReference.user("u"),
            continuity=Continuity.CONFIGURED, admission=Admission.WAIT,
            origin=Origin.CHAT_COMMAND, idempotency_key="k", keep_house=True,
        )


def test_profile_pin_validation(monkeypatch):
    monkeypatch.setattr(
        "llm_bawt.service.routes.bot_profile_validation._validate_model_endpoint",
        lambda payload: None)
    _validate_profile_payload({"house": "h1", "house_pinned": True})
    _validate_profile_payload({"house_pinned": True})  # house unchanged on PUT
    for bad in ({"house": None, "house_pinned": True}, {"house_pinned": None}):
        with pytest.raises(HTTPException):
            _validate_profile_payload(dict(bad))


# ------------------------------------------------------------------ eviction


class Store:
    def __init__(self, house="h1", pinned=False):
        self.row = SimpleNamespace(house=house, house_pinned=pinned)
        self.writes = []

    def get(self, slug):
        return self.row

    def upsert(self, payload, **kwargs):
        self.writes.append((payload, kwargs))


class Deliveries:
    def __init__(self, intent=None):
        self.intent = intent

    def get(self, delivery_id):
        return SimpleNamespace(metadata={"new_session_intent": self.intent} if self.intent else {})


def _rig(monkeypatch, *, store, others=(), intent=None, cached="h1"):
    calls = SimpleNamespace(recycled=[], invalidated=[], published=[])
    monkeypatch.setattr("llm_bawt.service.dependencies.get_bot_profile_store",
                        lambda config: store)
    monkeypatch.setattr(house_lease, "tenants", lambda config, house: list(others))
    monkeypatch.setattr(house_lease, "request_recycle",
                        lambda config, house: calls.recycled.append(house) if not others else None)
    monkeypatch.setattr(house_lease, "_publish",
                        lambda config, action: calls.published.append(action))
    monkeypatch.setattr("llm_bawt.bots.invalidate_bots_cache", lambda: None)
    monkeypatch.setattr("llm_bawt.bots._check_reload", lambda: None)
    service = SimpleNamespace(
        config=SimpleNamespace(), invalidate_bot_instances=calls.invalidated.append,
        _inter_bot_dispatcher=SimpleNamespace(store=Deliveries(intent)),
    )
    llm = SimpleNamespace(bot=SimpleNamespace(house=cached),
                          client=SimpleNamespace(_bot_config={"house": cached, "bot_id": "b"}))
    return service, llm, calls


def test_new_session_vacates_the_house_and_recycles_it(monkeypatch):
    store = Store()
    service, llm, calls = _rig(monkeypatch, store=store)
    assert house_lease.evict_for_reset(service, llm, "B", delivery_id="d") == "h1"
    (payload, kwargs), = store.writes
    assert payload["house"] is None and payload["house_pinned"] is False
    assert payload["expected_fields"] == {"house": "h1", "house_pinned": False}
    assert payload["_changed_by"] == house_lease.SYSTEM_ACTOR and kwargs == {"require_existing": True}
    assert "house" not in llm.client._bot_config  # the reset turn runs in the mansion
    assert calls.invalidated == ["b"] and len(calls.published) == 1
    assert calls.recycled == ["h1"]


def test_shared_house_is_not_recycled_while_another_tenant_remains(monkeypatch):
    store = Store()
    service, llm, calls = _rig(monkeypatch, store=store, others=["other"])
    assert house_lease.evict_for_reset(service, llm, "b", delivery_id=None) == "h1"
    assert store.writes and calls.recycled == []


@pytest.mark.parametrize("store,intent,cached", [
    (Store(pinned=True), None, "h1"),
    (Store(), {"keep_house": True}, "h1"),
    (Store(house=None), None, "h1"),
    (Store(), None, None),  # mansion bot: no database read at all
])
def test_kept_pinned_or_mansion_bots_are_untouched(monkeypatch, store, intent, cached):
    service, llm, calls = _rig(monkeypatch, store=store, intent=intent, cached=cached)
    assert house_lease.evict_for_reset(service, llm, "b", delivery_id="d") is None
    assert store.writes == [] and calls.recycled == [] and calls.invalidated == []


def test_pin_landing_after_the_read_keeps_the_tenancy(monkeypatch):
    """A pin PATCH between the read and the bot lock refuses the eviction."""
    from llm_bawt.bot_profile_changes import ProfileFieldsChanged

    store = Store()

    def upsert(payload, **kwargs):
        store.writes.append(payload)
        raise ProfileFieldsChanged("b", {"house": "h1", "house_pinned": True}, "pinned")

    store.upsert = upsert
    service, llm, calls = _rig(monkeypatch, store=store)
    assert house_lease.evict_for_reset(service, llm, "b", delivery_id=None) is None
    assert store.writes[0]["expected_fields"]["house_pinned"] is False
    assert llm.client._bot_config["house"] == "h1"
    assert calls.recycled == [] and calls.invalidated == [] and calls.published == []


def test_tenancy_read_failure_after_eviction_does_not_fail_reset(monkeypatch):
    recycle = house_lease.request_recycle
    store = Store()
    service, llm, calls = _rig(monkeypatch, store=store)
    monkeypatch.setattr(house_lease, "request_recycle", recycle)
    def fail(*args):
        raise RuntimeError("database unavailable")
    monkeypatch.setattr(house_lease, "tenants", fail)
    monkeypatch.setattr("llm_bawt.utils.db.get_shared_engine", fail)
    assert house_lease.evict_for_reset(service, llm, "b", delivery_id=None) == "h1"
    assert store.writes and calls.recycled == []
    assert "house" not in llm.client._bot_config


def test_failed_eviction_never_fails_the_reset(monkeypatch):
    store = Store()
    store.upsert = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db"))
    service, llm, calls = _rig(monkeypatch, store=store)
    assert house_lease.evict_for_reset(service, llm, "b", delivery_id=None) is None
    assert llm.client._bot_config["house"] == "h1" and calls.recycled == []


@pytest.mark.parametrize("intent, reason", [
    ({"origin": "scheduler"}, "idle_evict"),     # the idle janitor
    ({"origin": "chat_command"}, "new_session"),  # a manual /new
    (None, "new_session"),
])
def test_eviction_names_its_campus_reason(monkeypatch, intent, reason):
    store = Store()
    service, llm, _ = _rig(monkeypatch, store=store, intent=intent)
    house_lease.evict_for_reset(service, llm, "b", delivery_id="d")
    assert store.writes[0][0]["_location_reason"] == reason
