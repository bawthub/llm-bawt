"""TASK-1501: a finished turn stores the quota report cached for its route."""

from datetime import datetime, timedelta, timezone
import json
import time
from types import SimpleNamespace

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from llm_bawt.service import usage
from llm_bawt.service.turn_logs import TurnLog, TurnLogStore
from llm_bawt.service.turn_quota_snapshot import (
    LATE_REPLAY_SECONDS,
    MAX_READING_AGE_SECONDS,
    record_turn_quota,
)
from llm_bawt.service.usage.canonical import STATUS_OK, ProviderUsage, UsageBalance, UsageLimit

PATH = "test-path"
USAGE = {"input_tokens": 10, "output_tokens": 5, "access_path": {"key": PATH}}


@pytest.fixture
def cached(monkeypatch):
    """One adapter owning PATH; tests fill or leave its cache slot."""
    adapter = SimpleNamespace(provider="test-provider", account_scope="fixture-account")
    adapter.report_identity = lambda: adapter.account_scope
    monkeypatch.setattr(usage, "_ACCESS_PATH_ADAPTERS", {PATH: adapter})
    monkeypatch.setattr(usage, "_REGISTRY", {adapter.provider: adapter})
    monkeypatch.setattr(usage, "_cache", {})
    monkeypatch.setattr(usage, "_cache_identities", {})

    def put(**overrides):
        fields = dict(
            provider="test-provider", display_name="Test", available=True, status=STATUS_OK,
            fetched_at=int(time.time()), raw={"secret": "native"},
            limits=[UsageLimit(id="w1", label="Window", used_pct=42.0, resets_at=1793003600, window="5h")],
        )
        snap = ProviderUsage(**{**fields, **overrides})
        usage._cache["test-provider"] = (0.0, snap)
        usage._cache_identities[adapter.provider] = adapter.report_identity()
        return snap
    put.adapter = adapter
    return put


def _store():
    engine = create_engine("sqlite://")
    from llm_bawt.service.tool_call_store import ToolCallRecord
    SQLModel.metadata.create_all(engine, tables=[TurnLog.__table__, ToolCallRecord.__table__])
    store = TurnLogStore.__new__(TurnLogStore)
    store.engine = engine
    store.config = SimpleNamespace(model_catalog=None)
    return store


def _finish(store, turn_id, token_usage=USAGE):
    with Session(store.engine) as session:
        session.add(TurnLog(id=turn_id, bot_id="b", user_id="u", status="streaming"))
        session.commit()
    store.update_turn(turn_id=turn_id, status="ok", token_usage=token_usage)
    with Session(store.engine) as session:
        return session.exec(select(TurnLog).where(TurnLog.id == turn_id)).first()


def test_cached_report_is_stored_on_the_terminal_write(cached):
    snap = cached(balance=UsageBalance(amount_usd=None, remaining_usd=3.5, used_usd=None, as_of=None,
                                       source="provider_reported"))
    row = _finish(_store(), "t-cached")
    stored = json.loads(row.quota_snapshot_json)
    assert stored["provider"] == "test-provider" and stored["fetched_at"] == snap.fetched_at
    assert stored["limits"][0]["used_pct"] == 42.0 and stored["limits"][0]["resets_at"] == 1793003600
    assert stored["balance"]["remaining_usd"] == 3.5
    assert "raw" not in stored  # provider-native payloads stay out of history


def test_account_change_never_stamps_the_previous_accounts_quota(cached):
    cached()
    cached.adapter.account_scope = "replacement-account"
    assert _finish(_store(), "t-other-account").quota_snapshot_json is None


def test_no_cached_report_stores_null(cached):
    assert _finish(_store(), "t-none").quota_snapshot_json is None


def test_turn_without_frozen_route_stores_null(cached):
    cached()
    assert _finish(_store(), "t-noroute", token_usage={"input_tokens": 1}).quota_snapshot_json is None


def test_unavailable_cached_report_stores_null(cached):
    cached(fetched_at=None)
    assert _finish(_store(), "t-unavail").quota_snapshot_json is None


def test_first_reading_is_kept_on_later_terminal_updates(cached):
    cached()
    store = _store()
    _finish(store, "t-keep")
    cached(limits=[UsageLimit(id="w1", label="Window", used_pct=90.0)])
    store.update_turn(turn_id="t-keep", status="ok", token_usage=USAGE)
    with Session(store.engine) as session:
        row = session.exec(select(TurnLog).where(TurnLog.id == "t-keep")).first()
    assert json.loads(row.quota_snapshot_json)["limits"][0]["used_pct"] == 42.0


def test_late_replay_does_not_stamp_a_reading_taken_after_the_fact(cached):
    ended = datetime(2026, 11, 8, 12, 0, tzinfo=timezone.utc)
    cached(fetched_at=int(ended.timestamp()) - 30)
    row = SimpleNamespace(id="t-late", ended_at=ended, quota_snapshot_json=None,
                          token_usage_json=json.dumps(USAGE))
    record_turn_quota(row, now=ended + timedelta(seconds=LATE_REPLAY_SECONDS + 1))
    assert row.quota_snapshot_json is None
    record_turn_quota(row, now=ended + timedelta(seconds=5))
    assert row.quota_snapshot_json is not None


def test_reading_older_than_the_age_limit_is_not_stored(cached):
    ended = datetime(2026, 11, 8, 12, 0, tzinfo=timezone.utc)
    row = SimpleNamespace(id="t-old", ended_at=ended, quota_snapshot_json=None,
                          token_usage_json=json.dumps(USAGE))
    cached(fetched_at=int(ended.timestamp()) - MAX_READING_AGE_SECONDS - 1)
    record_turn_quota(row, now=ended)
    assert row.quota_snapshot_json is None
    cached(fetched_at=int(ended.timestamp()) - MAX_READING_AGE_SECONDS)
    record_turn_quota(row, now=ended)
    assert row.quota_snapshot_json is not None


def test_cache_failure_never_fails_the_turn(cached, monkeypatch):
    def boom(_key):
        raise RuntimeError("registry unavailable")
    monkeypatch.setattr(usage, "usage_adapter_for_access_path", boom)
    row = _finish(_store(), "t-boom")
    assert row.status == "ok" and row.ended_at is not None and row.quota_snapshot_json is None
